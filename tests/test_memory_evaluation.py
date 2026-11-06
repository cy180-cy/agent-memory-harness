"""Small three-strategy, export and replay regressions; guarded Mock HTTP only."""

from copy import deepcopy
import csv
import json
import uuid

import pytest

import memory_support as support
import portable_support as portable
from memory_comparison import evaluation, execution, results, spec
from run_memory_comparison import live_preflight
from test_comparison_user_output import synthetic_builder, frame
from test_comparison_runtime_fix import complete_frames


@pytest.fixture(scope="module")
def frozen():
    repo = spec.read_json(support.RUNTIME[1] / "repository_state.json")
    return evaluation.make_suite("offline_three_strategy_test", repo, evaluation.task_manifest(("34",)))


@pytest.fixture
def folder():
    assert support.RUNTIME is not None, "Use the guarded --development-tests evaluation entry"
    p = support.RUNTIME[1] / ("evaluation_" + uuid.uuid4().hex[:10])
    p.mkdir()
    return p


def test_three_strategies_share_actual_loop_and_isolated_state(frozen, folder, synthetic_builder, monkeypatch):
    build = execution.build_group
    constructed = []
    def remember(group):
        orch = build(group)
        constructed.append(orch)
        return orch
    monkeypatch.setattr(execution, "build_group", remember)
    frames = {g["group"]: complete_frames(g["group"]) for g in frozen["groups"]}
    value = evaluation.run_suite(frozen, folder, support.RUNTIME[0], frames_by_group=frames)
    assert [g["status"] for g in value["groups"]] == ["completed"] * 3
    assert all(g["official_reward"] == 1 for g in value["groups"])
    assert value["real_api_requests"] == 0 and value["mock_http_requests"] == 12
    assert all(g["allowed"] for g in value["gates"])
    assert [o.agent.memory_strategy.name for o in constructed] == list(evaluation.STRATEGIES)
    for path in ("agent", "agent_state", "user", "user_state"):
        assert len({id(getattr(o, path)) for o in constructed}) == 3
    assert len({id(o.environment.tools.db) for o in constructed}) == 3
    assert len({id(o.agent_state.memory) for o in constructed}) == 3
    for orch, g in zip(constructed, frozen["groups"]):
        assert not hasattr(orch.agent, "task")
        assert orch.agent_state.memory is not None
        local = spec.read_json(folder / g["group"] / "request_ledger.json")
        assert len(local["events"]) == 4
        for e in local["events"]:
            body = e["parameters"]
            assert body["max_completion_tokens"] == (3072 if e["role"] == "agent" else 1024)
            assert body.get("thinking_budget") == (1536 if e["role"] == "agent" else None)
            assert "max_tokens" not in body and e["sdk"]["max_retries"] == 0
    # Raw request evidence remains authoritative even if a displayed summary is wrong.
    first = folder / frozen["groups"][0]["group"]
    summary = spec.read_json(first / "summary.json")
    summary["usage"]["known_subtotal"]["prompt_tokens"] = 999999
    spec.atomic_json(first / "summary.json", summary)
    report = results.export_results([folder], folder / "exports")
    assert len(report["rows"]) == 3 and all(r["requests"] == 4 for r in report["rows"])
    assert all(r["known_token_subtotal"]["prompt_tokens"] == 80 for r in report["rows"])
    structured = report["rows"][2]
    assert structured["stored_facts"] > 0 and structured["source_kind"] == "mock"
    assert structured["task_scope"] == "synthetic_fixture" and structured["actual_task_id"] == "synthetic-user-output-no-retail"
    text = results.render_demo(folder, "34")
    assert all(name in text for name in evaluation.STRATEGIES)
    assert "结构化事实" in text and "message_index" in text and "get_users" in text
    assert "<scenario>" not in text and "Mock" in text
    with (folder / "exports/metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[2]["agent_input_tokens"] == "40" and rows[2]["cost_cny"] == ""
    with pytest.raises(FileExistsError):
        evaluation.run_suite(frozen, folder, support.RUNTIME[0], frames_by_group=frames)
    assert len(spec.read_json(folder / "comparison_request_ledger.json")["attempts"]) == 12


@pytest.mark.parametrize("change", ["model", "output", "policy", "scoring", "memory_budget"])
def test_mismatched_conditions_refused(frozen, change):
    altered = deepcopy(frozen)
    g = altered["groups"][2]
    if change == "model":
        g["models"]["agent"] = "openai/other"
    elif change == "output":
        g["role_args"]["user"]["max_completion_tokens"] = 512
    elif change == "policy":
        g["inputs"]["policy_sha256"] = "wrong"
    elif change == "scoring":
        g["evaluation"]["type"] = "ENV"
    else:
        g["memory"]["config"]["max_input_tokens"] = 9999
    with pytest.raises(ValueError):
        evaluation.validate_suite(altered)


def test_resource_stop_pauses_rest_of_suite_and_preserves_attempt(frozen, folder, synthetic_builder):
    frames = {g["group"]: complete_frames(g["group"]) for g in frozen["groups"]}
    first = frozen["groups"][0]["group"]
    frames[first] = [frame("user", "over the total output budget", completion=1025)]
    value = evaluation.run_suite(frozen, folder, support.RUNTIME[0], frames_by_group=frames)
    assert [g["status"] for g in value["groups"]] == ["resource_stop", "not_executed", "not_executed"]
    assert synthetic_builder == [first] and value["mock_http_requests"] == 1
    assert not value["gates"][0]["allowed"]
    report = results.export_results([folder], folder / "exports")
    assert [r["outcome_class"] for r in report["rows"]] == ["resource_stop", "not_executed", "not_executed"]
    assert all(r["official_reward"] is None for r in report["rows"])
    assert [r["strategy"] for r in report["rows"]] == list(evaluation.STRATEGIES)
    assert len(spec.read_json(folder / "comparison_request_ledger.json")["attempts"]) == 1


def test_no_suite_authorization_no_credential_read_or_construction(frozen, folder, monkeypatch):
    calls = []
    monkeypatch.setattr(execution, "run_group", lambda *a, **k: calls.append("constructed"))
    with pytest.raises(PermissionError):
        live_preflight(frozen, None, folder, {"tracked_patch_sha256": frozen["groups"][0]["code"]["tracked_patch_sha256"]},
                       lambda name: calls.append("key_read"))
    with pytest.raises(PermissionError):
        evaluation.run_suite(frozen, folder, support.RUNTIME[0], live=True)
    assert calls == [] and not (folder / "execution_started.lock").exists()


def test_score_classification_unknown_usage_and_reasoning_not_double_counted(folder):
    from test_comparison_gate import normal
    summary = normal(0)
    summary.update(status="completed", scoring_validity="official_score_recorded", pause_comparison=False,
                   pause_reasons=[], mode="mock", official_reward=0, db_result={"db_reward": 0, "db_match": False})
    config = {"memory": {"name": "structured_memory"}, "task_id": "34", "group": "C"}
    spec.write_new(folder / "summary.json", summary)
    spec.write_new(folder / "official_reward.json", {"reward": 0, "db_check": summary["db_result"]})
    spec.write_new(folder / "request_ledger.json", {"synthetic_no_http": True, "actual_api_requests": 0,
        "mock_http_requests": 0, "events": [{"ordinal": 1, "role": "agent", "http_status": 200}]})
    raw = {"usage": {"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120,
                     "completion_tokens_details": {"reasoning_tokens": 70}}}
    spec.write_new(folder / "model_responses.json", [{"evidence": {"raw_response": raw}}])
    row = results.group_metrics(folder, config)
    assert row["outcome_class"] == "business_failure" and row["official_reward"] == 0
    assert row["actual_tokens"]["total_tokens"] == 120
    assert row["roles"]["agent"]["reasoning_tokens_known_subtotal"] == 70
    raw["usage"]["completion_tokens"] = "bad"
    spec.atomic_json(folder / "model_responses.json", [{"evidence": {"raw_response": raw}}])
    row = results.group_metrics(folder, config)
    assert row["actual_tokens"] is None and row["unknown_usage_attempts"] == 1
    assert row["roles"]["agent"]["actual_tokens"] is None and row["cost_cny"] is None
    counts = results.aggregate([row])[0]
    assert counts["valid_scores"] == counts["business_failures"] == 1
    assert results.classification({"status": "execution_error"}, False) == "execution_error"
    assert results.classification({"status": "completed"}, False) == "evaluation_invalid"
    assert results.classification({"status": "execution_error", "output_truncated": 1}, False) == "resource_stop"


def test_legacy_real_run_is_not_relabeled_as_memory_strategy(folder):
    path = portable.stage_result_excerpt(folder / "legacy", "legacy_metrics_excerpt.json")
    row = results.collect(path)[0]
    assert row["strategy"] == "legacy_llm_agent" and row["source_kind"] == "historical_real"
    assert row["requests"] == 17 and row["known_token_subtotal"]["total_tokens"] == 90122
    assert row["config_version"] == "legacy-max-tokens" and row["scoring_valid"]
    assert "legacy_llm_agent" in results.render_demo(path, "34")


def test_structured_compaction_evidence_can_be_replayed_without_hidden_user_prompt(folder):
    path = portable.stage_result_excerpt(folder / "structured_demo", "structured_demo_excerpt.json")
    row = results.collect(path)[0]
    assert row["trimmed_generations"] == 2 and row["stored_facts"] > 0
    text = results.render_demo(path, "34")
    assert "message_index" in text and "structured_memory" in text and "本地估算" in text
    assert "<scenario>" not in text


def test_descriptive_differences_exclude_mock_partial_and_different_conditions():
    # Synthetic metric rows test the arithmetic, not real model performance.
    def row(strategy, tokens):
        return {"strategy": strategy, "source_kind": "real", "comparison_id": "synthetic_arithmetic_only",
            "task_id": "34", "experiment_condition_sha256": "same", "scoring_valid": True, "status": "completed",
            "official_reward": 1, "actual_tokens": {"prompt_tokens": tokens, "completion_tokens": 10},
            "request_count_matches": True, "tool_results_missing": 0, "requests": 4, "total_seconds": 1,
            "roles": {"agent": {"actual_tokens": {"prompt_tokens": tokens}}}}
    rows = [row(s, t) for s, t in zip(evaluation.STRATEGIES, (100, 80, 75))]
    diffs = results.observed_differences(rows)
    assert len(diffs) == 2 and diffs[0]["observations"]["agent_input_tokens"]["difference"] == -20
    rows[2]["actual_tokens"] = None
    assert results.observed_differences(rows) == []
    rows[2] = row("structured_memory", 75)
    rows[2]["source_kind"] = "mock"
    assert results.observed_differences(rows) == []
    rows[2]["source_kind"] = "real"
    rows[2]["experiment_condition_sha256"] = "different"
    assert results.observed_differences(rows) == []


def test_same_clock_tick_never_reuses_an_output_directory(folder, monkeypatch):
    import run_memory_comparison as entry
    from types import SimpleNamespace
    monkeypatch.setattr(entry, "ROOT", folder)
    monkeypatch.setattr(entry, "datetime", SimpleNamespace(now=lambda zone: SimpleNamespace(
        strftime=lambda pattern: "20000101T000000000000+0800")))
    first, second = entry.fresh_output(), entry.fresh_output()
    assert first != second and first.is_dir() and second.is_dir()
