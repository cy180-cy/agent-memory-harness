"""Targeted output-profile tests: real local LiteLLM, synthetic Mock domain only."""

from copy import deepcopy
import uuid

import pytest

import memory_support as support
import portable_support as portable
import memory_comparison.execution as execution
from memory_comparison import spec





@pytest.fixture
def plan():
    return portable.synthetic_plan(spec.USER_OUTPUT_CONFIG)


@pytest.fixture
def folder():
    if support.RUNTIME is None:
        pytest.skip("Use run_memory_comparison.py --tests --user-output-only")
    path = support.RUNTIME[1] / ("synthetic_user_output_" + uuid.uuid4().hex[:12])
    path.mkdir()
    return path


@pytest.fixture
def synthetic_builder(monkeypatch):
    """Replace only Retail construction; keep actual agents, loop and scorer."""
    from tau2.data_model.tasks import Task
    from tau2.domains.mock.environment import get_environment
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.user.user_simulator import UserSimulator

    constructed = []

    def build(group):
        constructed.append(group["group"])
        env = get_environment()
        task = Task.model_validate({
            "id": "synthetic-user-output-no-retail",
            "user_scenario": {"instructions": "Synthetic offline greeting, then finish."},
            "evaluation_criteria": {"reward_basis": ["DB"], "nl_assertions": []},
        })
        agent = registry.get_agent_factory("memory_agent")(
            env.get_tools(), env.get_policy(), llm=group["models"]["agent"],
            llm_args=deepcopy(group["role_args"]["agent"]),
            memory_strategy=group["memory"]["name"], memory_config=group["memory"]["config"])
        user = UserSimulator(llm=group["models"]["user"],
                             llm_args=deepcopy(group["role_args"]["user"]),
                             instructions=str(task.user_scenario), tools=None)
        return Orchestrator(domain="mock", agent=agent, user=user, environment=env,
                            task=task, max_steps=30, max_errors=1, seed=42,
                            simulation_id=group["run_id"], timeout=300,
                            validate_communication=True)

    monkeypatch.setattr(execution, "build_group", build)
    return constructed


def frame(role, content, *, completion=32, finish="stop"):
    body = support.response(content, completion=completion)
    body["model"] = "qwen3.8-flash" if role == "user" else "qwen3.7-plus"
    body["choices"][0]["finish_reason"] = finish
    body["usage"]["completion_tokens_details"] = {
        "reasoning_tokens": min(344, completion), "text_tokens": completion}
    return {"role": role, "body": body, "provenance": "synthetic_mock_usage"}


def normal_frames():
    from tau2.user.user_simulator_base import STOP
    return [frame("user", "SYNTHETIC_COMPLETE_USER", completion=600),
            frame("agent", "SYNTHETIC_AGENT_REPLY"), frame("user", STOP)]


def assert_wire_and_journal(path, expected):
    local = spec.read_json(path / "request_ledger.json")
    assert local["actual_api_requests"] == 0
    assert [event["role"] for event in local["events"]] == expected
    for event in local["events"]:
        body = event["parameters"]
        assert body["max_completion_tokens"] == 1024 and "max_tokens" not in body
        assert body["temperature"] == 0 and body.get("stream", False) is False
        assert "enable_thinking" not in body and "thinking_budget" not in body
        assert event["sdk"]["max_retries"] == event["sdk"]["http_retries"] == event["sdk"]["task_retries"] == 0
    return local


@pytest.mark.parametrize("group_index", [0, 1], ids=["A-full-history", "B-budget-window"])
def test_both_roles_send_new_budget_and_complete_user_reaches_agent(
        plan, folder, synthetic_builder, group_index):
    import qianwen_adapter as adapter
    from output_budget import TOTAL_OUTPUT_LIMITS

    group = plan["groups"][group_index]
    original = adapter.LIMITS
    ledger = spec.RequestLedger(folder, plan["identity_sha256"])
    result = execution.run_group(group, folder / group["group"], ledger,
                                 support.RUNTIME[0], frames=normal_frames())
    assert result["status"] == "completed" and result["output_truncated"] == 0
    local = assert_wire_and_journal(folder / group["group"], ["user", "agent", "user"])
    assert "SYNTHETIC_COMPLETE_USER" in str(local["events"][1]["parameters"]["messages"])
    assert len(ledger.data["attempts"]) == 3 and result["tool_calls"] == 0
    assert adapter.LIMITS is original and TOTAL_OUTPUT_LIMITS == {"agent": 1024, "user": 512}


@pytest.mark.parametrize("source", ["historical-512", "synthetic-1024"])
def test_actual_pair_truncation_never_releases_partial_user_or_constructs_B(
        plan, folder, synthetic_builder, source):
    import qianwen_adapter as adapter

    original = adapter.LIMITS
    bad = frame("user", "SYNTHETIC_TRUNCATED_USER", completion=1024, finish="length")
    if source == "historical-512":
        record = portable.fixture("truncation_responses.json")["user512"]
        bad = {"role": "user", "body": deepcopy(record["body"]),
               "provenance": "captured_historical_response_replayed_offline_new_request_profile"}
    frames = normal_frames()[:2] + [bad]
    result = execution.run_pair(plan, folder, support.RUNTIME[0],
                               frames_by_group={"A": frames, "B": normal_frames()})
    assert synthetic_builder == ["A"] and not (folder / "B").exists()
    assert result["mock_http_requests"] == 3 and result["real_api_requests"] == 0
    assert result["groups"][0]["output_truncated"] == 1
    assert result["groups"][0]["official_reward"] is None
    assert result["groups"][1]["requests"] == 0 and not result["a_to_b_gate"]["allowed"]
    assert "A_output_incomplete" in result["a_to_b_gate"]["reasons"]
    local = assert_wire_and_journal(folder / "A", ["user", "agent", "user"])
    records = spec.read_json(folder / "A/model_responses.json")
    assert records[-1]["evidence"]["raw_response"] == bad["body"]
    assert records[-1]["converted"]["litellm_response"]["choices"][0]["finish_reason"] == "length"
    assert records[-1]["classification"]["adapter_compatibility_check"]["status"] == "failed"
    partial = bad["body"]["choices"][0]["message"]["content"]
    trajectory = spec.read_json(folder / "A/trajectory.json")
    agent_history = spec.read_json(folder / "A/agent_state.json")["messages"]
    assert all(message.get("content") != partial for message in trajectory + agent_history)
    durable = folder / "comparison_request_ledger.json"
    before = durable.read_bytes()
    assert len(spec.read_json(durable)["attempts"]) == len(local["events"]) == 3
    with pytest.raises(FileExistsError):
        execution.run_pair(plan, folder, support.RUNTIME[0], frames_by_group={"A": frames})
    assert before == durable.read_bytes() and synthetic_builder == ["A"]
    assert adapter.LIMITS is original


@pytest.mark.parametrize("failure", ["over-budget", "missing-usage", "invalid-usage"])
def test_user_resource_pause_keeps_response_and_blocks_agent_and_B(
        plan, folder, synthetic_builder, failure):
    bad = frame("user", "RESOURCE_PAUSED_USER", completion=1025)
    if failure == "missing-usage":
        del bad["body"]["usage"]
    elif failure == "invalid-usage":
        bad["body"]["usage"]["completion_tokens"] = "1025"
    expected = "output_budget_exceeded" if failure == "over-budget" else "budget_verification_unknown"
    result = execution.run_pair(plan, folder, support.RUNTIME[0],
                               frames_by_group={"A": [bad], "B": normal_frames()})
    assert synthetic_builder == ["A"] and not (folder / "B").exists()
    assert result["groups"][0]["stop_reason"] == expected
    assert result["groups"][0]["official_reward"] is None
    assert result["groups"][0]["tool_calls"] == 0
    assert result["groups"][1]["requests"] == 0
    assert_wire_and_journal(folder / "A", ["user"])
    records = spec.read_json(folder / "A/model_responses.json")
    assert records[0]["evidence"]["raw_response"] == bad["body"]
    assert records[0]["resource_stop"]["status"] == expected
    assert result["mock_http_requests"] == 1 and result["real_api_requests"] == 0


def test_old_profile_remains_512_and_scoped_caps_restore(plan, folder, synthetic_builder):
    import qianwen_adapter as adapter

    old = portable.synthetic_plan(spec.LEGACY_OUTPUT_CONFIG, omit_legacy_version=True)
    spec.validate_pair(old)
    assert "output_config_version" not in old["groups"][0]
    assert spec.group_output_limits(old["groups"][0]) == {"agent": 1024, "user": 512}
    original = adapter.LIMITS
    frames = normal_frames()
    frames[0] = frame("user", "LEGACY_COMPLETE_USER", completion=400)
    result = execution.run_group(old["groups"][0], folder / "legacy",
                                 spec.RequestLedger(folder, "synthetic-legacy-profile"),
                                 support.RUNTIME[0], frames=frames)
    assert result["status"] == "completed"
    events = spec.read_json(folder / "legacy/request_ledger.json")["events"]
    assert [e["parameters"]["max_completion_tokens"] for e in events] == [512, 1024, 512]
    assert adapter.LIMITS is original and adapter.role_args("user")["max_completion_tokens"] == 512


@pytest.mark.parametrize("mutation", ["mismatch", "both-wrong", "max-tokens", "version"])
def test_profile_rejects_mismatch_or_unapproved_parameters(plan, mutation):
    if mutation == "mismatch":
        plan["groups"][1]["role_args"]["user"]["max_completion_tokens"] = 512
    else:
        for group in plan["groups"]:
            if mutation == "both-wrong":
                group["role_args"]["user"]["max_completion_tokens"] = 2048
            elif mutation == "max-tokens":
                group["role_args"]["user"]["max_tokens"] = 1024
            else:
                group["output_config_version"] = "unapproved-version"
    with pytest.raises(ValueError):
        spec.validate_pair(plan)


def test_group_and_total_slots_remain_durable_without_refunds(folder):
    from test_memory_comparison import test_group_and_global_ledger_caps_are_durable_and_never_refund
    test_group_and_global_ledger_caps_are_durable_and_never_refund(folder)


def test_frozen_historical_records_and_consumed_authorization_stay_intact():
    consumed = portable.fixture("historical_status_excerpt.json")["user512"]
    ledger = {"attempts": [{"group": g} for g in consumed["attempt_groups"]],
              "groups": {"B": {"actually_executed": consumed["B_actually_executed"]}}}
    assert len(ledger["attempts"]) == 12 and all(e["group"] == "A" for e in ledger["attempts"])
    assert consumed["status"] == "consumed_stopped_after_A" and consumed["may_restart_or_resume"] is False
    assert ledger["groups"]["B"]["actually_executed"] is False
    assert "AMH_QIANWEN_API_KEY" not in list(__import__("os").environ)
