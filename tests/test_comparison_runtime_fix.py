"""Mode gates and Agent output profile only; no credentials or real HTTP."""

from copy import deepcopy
import json
import uuid

import pytest

import memory_support as support
import portable_support as portable
import memory_comparison.execution as execution
from memory_comparison import spec
from run_memory_comparison import live_preflight
from test_comparison_gate import normal, blocked
from test_comparison_user_output import synthetic_builder, frame





@pytest.fixture
def plan():
    return portable.synthetic_plan()


@pytest.fixture
def folder():
    if support.RUNTIME is None:
        pytest.skip("Use run_memory_comparison.py --tests --runtime-fix-only")
    path = support.RUNTIME[1] / ("synthetic_runtime_fix_" + uuid.uuid4().hex[:12])
    path.mkdir()
    return path


def test_mode_names_are_explicit_and_unknown_values_never_fall_back():
    assert [spec.normalize_mode(m) for m in ("live", "real", "mock")] == ["live", "live", "mock"]
    for value in (None, "LIVE", "offline", "", True):
        with pytest.raises(ValueError):
            spec.normalize_mode(value)
    with pytest.raises(ValueError):
        spec.execution_mode(live="real")
    with pytest.raises(ValueError):
        spec.execution_mode(live=True, mode="mock")


@pytest.mark.parametrize("mode", ["live", "real"])
def test_both_live_names_require_actual_approval_before_key_or_construction(plan, folder, monkeypatch, mode):
    calls = []
    monkeypatch.setattr(execution, "run_group", lambda *a, **k: calls.append("constructed"))
    with pytest.raises(PermissionError):
        execution.run_pair(plan, folder, support.RUNTIME[0], mode=mode, authorization=None)
    repo = {"tracked_patch_sha256": plan["groups"][0]["code"]["tracked_patch_sha256"]}
    with pytest.raises(PermissionError):
        live_preflight(plan, None, folder, repo, lambda n: calls.append("key_read"), mode=mode)
    assert calls == [] and not (folder / "execution_started.lock").exists()


def synthetic_gate_executor(monkeypatch, a_result, mode):
    """Historical real-event/live-ledger schema with NO HTTP or live admission."""
    calls, admission = [], []
    canonical = spec.normalize_mode(mode)

    def test_admission(plan, authorization, *, mode):
        # No approved authorization file is made. This stub exists only inside
        # a deterministic test of run_pair's already-admitted control flow.
        assert authorization == "SYNTHETIC_TEST_STUB_NO_LIVE_APPROVAL"
        assert mode == "live"
        admission.append(mode)
        return {"synthetic_test_admission_only": True}

    monkeypatch.setattr(execution, "require_authorization", test_admission)

    def execute(group, path, ledger, safety, **kwargs):
        calls.append(group["group"])
        value = deepcopy(a_result if group["group"] == "A" else normal())
        value.update(group=group["group"], run_id=group["run_id"], mode=canonical)
        value.update(real_api_requests=value["requests"] if canonical == "live" else 0,
                     mock_http_requests=value["requests"] if canonical == "mock" else 0)
        path.mkdir()
        events = []
        for ordinal in range(1, value["requests"] + 1):
            event = {"ordinal": ordinal, "role": "user" if ordinal == 1 else "agent",
                     "mode": "real" if canonical == "live" else "mock",
                     "status": "reserved_before_io", "parameters": {"source": "synthetic_no_http"}}
            ledger.sync(group["group"], {"events": [event]})
            event.update(status="response_received", http_status=200)
            ledger.sync(group["group"], {"events": [event]})
            events.append(event)
        required = ("trajectory.json", "agent_state.json", "context_audit.json", "model_responses.json",
                    "official_reward.json", "simulation.json")
        for name in required:
            spec.atomic_json(path / name, {"source": "synthetic_no_http", "kind": name})
        spec.atomic_json(path / "official_reward.json", {"reward": value["official_reward"], "db_check": value["db_result"]})
        spec.atomic_json(path / "request_ledger.json", {
            "actual_api_requests": value["real_api_requests"], "mock_http_requests": value["mock_http_requests"],
            "events": events, "blocked": [], "synthetic_no_http": True})
        value["evidence_hashes"] = {name: spec.sha(path / name) for name in (*required, "request_ledger.json")}
        spec.atomic_json(path / "summary.json", value)
        ledger.finish_group(group["group"], value)
        return value

    monkeypatch.setattr(execution, "run_group", execute)
    return calls, admission


@pytest.mark.parametrize("mode", ["mock", "live", "real"])
@pytest.mark.parametrize("reward", [0, 1])
def test_actual_pair_gate_accepts_complete_real_alias_records_and_reward_zero(
        plan, folder, monkeypatch, mode, reward):
    calls, admission = synthetic_gate_executor(monkeypatch, normal(reward), mode)
    result = execution.run_pair(plan, folder, support.RUNTIME[0], mode=mode,
                               authorization="SYNTHETIC_TEST_STUB_NO_LIVE_APPROVAL")
    assert calls == ["A", "B"] and result["a_to_b_gate"]["allowed"] is True
    assert result["groups"][0]["official_reward"] == reward
    assert admission == ([] if mode == "mock" else ["live"])
    assert spec.read_json(folder / "a_to_b_gate.json")["allowed"] is True
    before = (folder / "comparison_request_ledger.json").read_bytes()
    with pytest.raises(FileExistsError):
        execution.run_pair(plan, folder, support.RUNTIME[0], mode=mode,
                           authorization="SYNTHETIC_TEST_STUB_NO_LIVE_APPROVAL")
    assert calls == ["A", "B"] and before == (folder / "comparison_request_ledger.json").read_bytes()


@pytest.mark.parametrize("cause", ["output_budget_exceeded", "budget_verification_unknown", "evaluation_error", "invalid_gold", "at_request_cap"])
def test_real_alias_does_not_release_B_after_invalid_A(plan, folder, monkeypatch, cause):
    value = blocked(cause)
    calls, admission = synthetic_gate_executor(monkeypatch, value, "real")
    result = execution.run_pair(plan, folder, support.RUNTIME[0], mode="real",
                               authorization="SYNTHETIC_TEST_STUB_NO_LIVE_APPROVAL")
    assert calls == ["A"] and admission == ["live"]
    assert not result["a_to_b_gate"]["allowed"] and result["groups"][1]["requests"] == 0
    assert spec.read_json(folder / "a_to_b_gate.json")["allowed"] is False
    assert result["groups"][0]["stop_reason"] == value["stop_reason"]


def test_unknown_or_mismatched_ledger_mode_is_rejected_without_reserving(folder):
    with pytest.raises(ValueError):
        spec.RequestLedger(folder / "unknown", "synthetic", mode="production")
    ledger = spec.RequestLedger(folder / "known", "synthetic", mode="real")
    assert ledger.data["mode"] == "live"
    for mode in ("production", "mock"):
        with pytest.raises(ValueError):
            ledger.sync("A", {"events": [{"ordinal": 1, "status": "reserved_before_io", "mode": mode}]})
    assert spec.read_json(ledger.path)["attempts"] == []


def test_factory_uses_new_profile_and_keeps_explicit_legacy_fixture_meaning(plan, monkeypatch):
    group = plan["groups"][0]
    monkeypatch.setattr(spec, "code_identity", lambda repo: deepcopy(group["code"]))
    monkeypatch.setattr(spec, "task_identity", lambda task, env: deepcopy(group["inputs"]))
    calibration = {"selection": {"recommended_max_input_tokens": 7168, "safety_margin_tokens": 128}}
    new = spec.make_pair("synthetic-profile-new", {}, calibration, None, None)
    legacy = spec.make_pair("synthetic-profile-old", {}, calibration, None, None,
                            output_config_version=spec.LEGACY_OUTPUT_CONFIG)
    assert all(g["role_args"]["agent"]["max_completion_tokens"] == 3072 for g in new["groups"])
    assert all(g["role_args"]["agent"]["extra_body"] == {"thinking_budget": 1536} for g in new["groups"])
    assert all(g["role_args"]["agent"]["max_completion_tokens"] == 1024 for g in legacy["groups"])
    assert all(g["role_args"]["user"]["max_completion_tokens"] == 512 for g in legacy["groups"])


def tool_frame(*, completion=2048, finish="tool_calls", ident="synthetic-budget-tool"):
    body = support.response(calls=[(ident, "get_users", {})], completion=completion)
    body["choices"][0]["finish_reason"] = finish
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1536, "text_tokens": completion}
    return {"role": "agent", "body": body, "provenance": "synthetic_mock_usage_no_live_model"}


def complete_frames(ident):
    from tau2.user.user_simulator_base import STOP
    return [frame("user", "SYNTHETIC_USER", completion=600), tool_frame(ident=ident),
            frame("agent", "SYNTHETIC_TOOL_COMPLETED"), frame("user", STOP)]


def assert_actual_wire(path):
    ledger = spec.read_json(path / "request_ledger.json")
    assert ledger["actual_api_requests"] == 0
    for event in ledger["events"]:
        body = event["parameters"]
        assert body["max_completion_tokens"] == (3072 if event["role"] == "agent" else 1024)
        assert "max_tokens" not in body and "extra_body" not in body
        assert "enable_thinking" not in body and "reasoning_effort" not in body
        if event["role"] == "agent":
            assert body["thinking_budget"] == 1536
        else:
            assert "thinking_budget" not in body
        assert event["mode"] == "mock"
        assert all(event["sdk"][k] == 0 for k in ("max_retries", "http_retries", "task_retries"))
    return ledger


def test_actual_litellm_both_strategies_send_new_budgets_and_execute_tool_loop(
        plan, folder, synthetic_builder):
    import qianwen_adapter as adapter
    original_args, original_caps = adapter.role_args, adapter.LIMITS
    result = execution.run_pair(plan, folder, support.RUNTIME[0], frames_by_group={
        "A": complete_frames("synthetic-A-tool"), "B": complete_frames("synthetic-B-tool")})
    assert synthetic_builder == ["A", "B"] and result["a_to_b_gate"]["allowed"]
    for group in ("A", "B"):
        local = assert_actual_wire(folder / group)
        assert len(local["events"]) == 4
        tools = spec.read_json(folder / group / "tool_events.json")
        assert len(tools) == 1 and tools[0]["result"]["error"] is False
        assert tools[0]["call"]["name"] == "get_users"
        assert any(m.get("tool_call_id") == tools[0]["call"]["id"]
                   for m in local["events"][2]["parameters"]["messages"])
    assert adapter.role_args is original_args and adapter.LIMITS is original_caps
    assert result["mock_http_requests"] == 8 and result["real_api_requests"] == 0


def test_thinking_parameter_loss_is_blocked_before_mock_transport_dispatch(
        plan, folder, synthetic_builder, monkeypatch):
    import httpx

    original = httpx.Client.send
    changed = []

    def remove_parameter(client, request, *args, **kwargs):
        body = json.loads(request.content)
        if "thinking_budget" in body:
            del body["thinking_budget"]
            request._content = json.dumps(body).encode("utf-8")
            request.headers["content-length"] = str(len(request._content))
            changed.append(True)
        return original(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", remove_parameter)
    result = execution.run_pair(plan, folder, support.RUNTIME[0], frames_by_group={
        "A": complete_frames("blocked-before-dispatch"), "B": complete_frames("unused")})
    assert changed == [True] and synthetic_builder == ["A"]
    assert result["mock_http_requests"] == 1 and result["real_api_requests"] == 0
    assert result["groups"][0]["tool_calls"] == 0 and result["groups"][1]["requests"] == 0


@pytest.mark.parametrize("source", ["captured-old-agent1024", "synthetic-length3072-with-tool"])
def test_agent_truncation_remains_blocked_before_tools_or_B(plan, folder, synthetic_builder, source):
    bad = tool_frame(completion=3072, finish="length")
    if source == "captured-old-agent1024":
        record = portable.fixture("truncation_responses.json")["agent1024"]
        bad = {"role": "agent", "body": deepcopy(record["body"]),
               "provenance": "captured_old_response_only_not_evidence_of_new_supplier_behavior"}
    result = execution.run_pair(plan, folder, support.RUNTIME[0], frames_by_group={
        "A": [frame("user", "SYNTHETIC_USER"), bad], "B": complete_frames("unused")})
    assert synthetic_builder == ["A"] and not (folder / "B").exists()
    assert result["groups"][0]["tool_calls"] == 0 and result["groups"][0]["output_truncated"] == 1
    assert result["groups"][0]["official_reward"] is None and result["groups"][1]["requests"] == 0
    records = spec.read_json(folder / "A/model_responses.json")
    assert records[-1]["evidence"]["raw_response"] == bad["body"]
    assert records[-1]["converted"]["litellm_response"]["choices"][0]["finish_reason"] == "length"
    assert records[-1]["classification"]["adapter_compatibility_check"]["status"] == "failed"
    assert_actual_wire(folder / "A")


@pytest.mark.parametrize("failure", ["over-budget", "missing-usage", "invalid-usage"])
def test_agent_resource_stop_saves_response_and_blocks_tool_and_B(plan, folder, synthetic_builder, failure):
    bad = tool_frame(completion=3073)
    if failure == "missing-usage":
        del bad["body"]["usage"]
    elif failure == "invalid-usage":
        bad["body"]["usage"]["completion_tokens"] = "3073"
    result = execution.run_pair(plan, folder, support.RUNTIME[0], frames_by_group={
        "A": [frame("user", "SYNTHETIC_USER"), bad], "B": complete_frames("unused")})
    expected = "output_budget_exceeded" if failure == "over-budget" else "budget_verification_unknown"
    assert synthetic_builder == ["A"] and result["groups"][0]["stop_reason"] == expected
    assert result["groups"][0]["tool_calls"] == 0 and result["groups"][0]["official_reward"] is None
    assert result["groups"][1]["requests"] == 0
    records = spec.read_json(folder / "A/model_responses.json")
    assert records[-1]["evidence"]["raw_response"] == bad["body"]
    assert records[-1]["resource_stop"]["status"] == expected
    assert_actual_wire(folder / "A")


def test_request_caps_persist_and_uncertain_attempts_are_never_refunded(folder):
    from test_memory_comparison import test_group_and_global_ledger_caps_are_durable_and_never_refund
    test_group_and_global_ledger_caps_are_durable_and_never_refund(folder)


def test_existing_transport_stops_before_second_dispatch_and_global_count_is_durable(folder):
    from test_memory_comparison import (
        test_existing_transport_still_enforces_limit_before_mock_dispatch,
        test_global_reservation_is_durable_before_existing_transport_dispatch,
    )
    one, two = folder / "local", folder / "global"
    one.mkdir(); two.mkdir()
    test_existing_transport_still_enforces_limit_before_mock_dispatch(one, support.RUNTIME)
    test_global_reservation_is_durable_before_existing_transport_dispatch(two, support.RUNTIME)


def test_old_profiles_and_consumed_live_records_remain_unchanged(plan, folder):
    old = portable.synthetic_plan(spec.USER_OUTPUT_CONFIG)
    spec.validate_pair(old)
    assert spec.configured_role_args(spec.USER_OUTPUT_CONFIG)["agent"]["max_completion_tokens"] == 1024
    assert "extra_body" not in spec.configured_role_args(spec.USER_OUTPUT_CONFIG)["agent"]
    assert spec.configured_role_args(spec.LEGACY_OUTPUT_CONFIG)["user"]["max_completion_tokens"] == 512
    consumed = portable.fixture("historical_status_excerpt.json")["agent1024"]
    assert consumed["actual_requests"] == 9 and consumed["may_restart_or_resume"] is False
    assert consumed["status"] == "consumed_stopped_after_A"
    auth = portable.stale_authorization(folder)
    with pytest.raises(PermissionError):
        spec.require_authorization(plan, auth, mode="real")
