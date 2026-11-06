"""Synthetic A-result contracts and actual run_pair control-flow tests."""

from copy import deepcopy
from pathlib import Path
import uuid

import pytest

import memory_support as support
import portable_support as portable
import memory_comparison.execution as execution
import memory_comparison.spec as spec


@pytest.fixture
def plan():
    return portable.synthetic_plan(spec.LEGACY_OUTPUT_CONFIG)


@pytest.fixture
def folder():
    if support.RUNTIME is None:
        pytest.skip("Use validation/run_memory_comparison.py --tests --gate-only")
    path = support.RUNTIME[1]/("synthetic_gate_"+uuid.uuid4().hex[:12])
    path.mkdir()
    return path


def normal(reward=1, requests=2):
    return {"actually_executed":True,"status":"completed","termination_reason":"user_stop",
        "scoring_validity":"official_score_recorded","official_reward":reward,
        "db_result":{"db_match":bool(reward),"db_reward":reward},"pause_comparison":False,"stop_reason":None,
        "mode":"mock","requests":requests,"step_count":4,"total_seconds":1.0,
        "real_api_requests":0,"mock_http_requests":requests,
        "usage":{"status":"known","unknown_attempts":0,"known_subtotal":{
            "prompt_tokens":20*requests,"completion_tokens":32*requests,"total_tokens":52*requests}},
        "output_truncated":0,"usage_verification_unknown":0,
        "output_resource_decisions":[{"status":"budget_verified"} for _ in range(requests)],
        "execution_checks":{k:True for k in ("model_calls_complete","tool_calls_complete","trajectory_complete",
            "safety_verified","artifacts_complete")},
        "role_counts":{"user":{"requests":1,"http_success":1,"failed":0,"timeouts":0},
            "agent":{"requests":requests-1,"http_success":requests-1,"failed":0,"timeouts":0}}}


BLOCKS = ("output_budget_exceeded","budget_verification_unknown","context_budget_exceeded",
    "context_budget_unknown","request_limit_reached","max_steps","task_timeout",
    "authentication_or_permission_error","rate_limited","shared_transport_error",
    "evaluation_error","invalid_gold","incomplete_model","incomplete_tool","incomplete_trajectory",
    "safety_error","missing_checks","unknown_state","not_executed","missing_result",
    "at_request_cap","old_reward_after_limit","usage_fields_missing","score_mismatch")


def blocked(case):
    value = normal()
    if case == "missing_result":
        return None
    if case == "at_request_cap":
        return normal(requests=20)
    if case == "usage_fields_missing":
        value["usage"]["known_subtotal"] = None
    elif case == "score_mismatch":
        value["db_result"] = {"db_match":False,"db_reward":0}
    elif case in ("incomplete_model","incomplete_tool","incomplete_trajectory","safety_error"):
        key = {"incomplete_model":"model_calls_complete","incomplete_tool":"tool_calls_complete",
            "incomplete_trajectory":"trajectory_complete","safety_error":"safety_verified"}[case]
        value["execution_checks"][key] = False
    elif case == "missing_checks":
        value["execution_checks"] = None
    elif case == "unknown_state":
        value["status"] = "unknown"
    elif case == "not_executed":
        value["actually_executed"] = False
    elif case == "invalid_gold":
        value.update(scoring_validity="warning_requires_review",stop_reason="gold_replay_warning")
    else:
        value.update(status="resource_stop",stop_reason=case,official_reward=None)
        if case == "old_reward_after_limit":
            value.update(stop_reason="output_budget_exceeded",official_reward=1)
        if case in ("authentication_or_permission_error","rate_limited","shared_transport_error","evaluation_error"):
            value.update(status="evaluation_error" if case == "evaluation_error" else "execution_error",
                error_type="OriginalSyntheticError")
    return value


@pytest.mark.parametrize("reward", [0,1])
def test_valid_completion_contract_allows_both_business_scores(plan,reward):
    assert execution.completion_reasons(normal(reward),plan["groups"][0]) == []


@pytest.mark.parametrize("case", BLOCKS)
def test_incomplete_or_unknown_result_is_never_valid(plan,case):
    assert execution.completion_reasons(blocked(case),plan["groups"][0])


def fake_executor(monkeypatch, a_result, *, after_A=None, raise_A=False):
    calls = []

    def execute(group, path, ledger, safety, **kwargs):
        # This is the construction/request entry spy. B must never reach it when blocked.
        calls.append(group["group"])
        value = deepcopy(a_result if group["group"] == "A" else normal())
        if value is None:
            return None
        path.mkdir()
        value.update(group=group["group"],run_id=group["run_id"])
        events = []
        for ordinal in range(1,value["requests"]+1):
            event = {"ordinal":ordinal,"role":"user" if ordinal == 1 else "agent","mode":"mock",
                "status":"reserved_before_io","parameters":{"source":"synthetic_no_http"}}
            ledger.sync(group["group"],{"events":[event]})
            event.update(status="response_received",http_status=200)
            ledger.sync(group["group"],{"events":[event]})
            events.append(event)
        required = ("trajectory.json","agent_state.json","context_audit.json","model_responses.json",
            "official_reward.json","simulation.json")
        for name in required:
            spec.atomic_json(path/name,{"source":"synthetic_no_http","kind":name})
        spec.atomic_json(path/"official_reward.json",{"reward":value["official_reward"],"db_check":value["db_result"]})
        spec.atomic_json(path/"request_ledger.json",{"actual_api_requests":0,"mock_http_requests":len(events),
            "events":events,"blocked":[],"synthetic_no_http":True})
        value["evidence_hashes"] = {name:spec.sha(path/name) for name in (*required,"request_ledger.json")}
        spec.atomic_json(path/"summary.json",value)
        ledger.finish_group(group["group"],value)
        if group["group"] == "A" and after_A is not None:
            after_A(path,ledger)
        if group["group"] == "A" and raise_A:
            raise RuntimeError("synthetic failure after A evidence saved")
        return value

    monkeypatch.setattr(execution,"run_group",execute)
    return calls


@pytest.mark.parametrize("reward", [0,1])
def test_actual_run_pair_constructs_B_for_valid_reward_zero_or_one(plan,folder,monkeypatch,reward):
    calls = fake_executor(monkeypatch,normal(reward))
    result = execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A","B"]
    assert result["a_to_b_gate"]["allowed"] is True
    assert result["groups"][0]["official_reward"] == reward
    assert spec.read_json(folder/"a_to_b_gate.json")["allowed"] is True
    assert len(spec.read_json(folder/"comparison_request_ledger.json")["attempts"]) == 4


@pytest.mark.parametrize("case", BLOCKS)
def test_actual_run_pair_never_enters_B_on_invalid_A(plan,folder,monkeypatch,case):
    original = blocked(case)
    calls = fake_executor(monkeypatch,original)
    result = execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"]  # No B construction, credential reader or request entry.
    assert result["a_to_b_gate"]["allowed"] is False
    assert result["groups"][1]["actually_executed"] is False
    assert result["groups"][1]["requests"] == 0
    durable = spec.read_json(folder/"comparison_request_ledger.json")
    assert durable["a_to_b_gate"] == spec.read_json(folder/"a_to_b_gate.json")
    assert durable["shared_stop"]
    assert all(e["group"] == "A" for e in durable["attempts"])
    if original is not None:
        assert result["groups"][0]["stop_reason"] == original["stop_reason"]
        assert result["groups"][0].get("error_type") == original.get("error_type")
        assert spec.read_json(folder/"A/summary.json") == result["groups"][0]
        assert len(durable["attempts"]) == original["requests"]
    before = (folder/"comparison_request_ledger.json").read_bytes()
    with pytest.raises(FileExistsError):execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"]
    assert (folder/"comparison_request_ledger.json").read_bytes() == before


def test_historical_block_record_prevents_A_even_without_lock(plan,folder,monkeypatch):
    ledger = spec.RequestLedger(folder,plan["identity_sha256"])
    ledger.finish_group("A",{"status":"blocked_before_start","pause_comparison":True,
        "stop_reason":"historical_block","actually_executed":False})
    before = ledger.path.read_bytes()
    calls = fake_executor(monkeypatch,normal())
    with pytest.raises(FileExistsError):execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == [] and ledger.path.read_bytes() == before


def test_unexpected_A_exception_does_not_overwrite_original_A_error(plan,folder,monkeypatch):
    original = blocked("authentication_or_permission_error")
    calls = fake_executor(monkeypatch,original,raise_A=True)
    result = execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"]
    assert result["groups"][0]["error_type"] == "OriginalSyntheticError"
    assert result["groups"][0]["stop_reason"] == original["stop_reason"]
    assert result["pair_errors"] == [{"group":"A","stage":"group_execution","error_type":"RuntimeError"}]
    assert "A_execution_raised" in result["a_to_b_gate"]["reasons"]


@pytest.mark.parametrize("damage", ["summary","artifact","local_ledger","global_ledger"])
def test_bad_persistence_blocks_B_and_preserves_unreadable_global_ledger(plan,folder,monkeypatch,damage):
    def alter(path,ledger):
        if damage == "summary":
            value = spec.read_json(path/"summary.json");value["status"] = "unknown"
            spec.atomic_json(path/"summary.json",value)
        elif damage == "artifact":
            spec.atomic_json(path/"trajectory.json",{"changed":"synthetic"})
        elif damage == "local_ledger":
            value = spec.read_json(path/"request_ledger.json");value["mock_http_requests"] = 0
            spec.atomic_json(path/"request_ledger.json",value)
        else:
            ledger.path.write_text("unreadable synthetic ledger",encoding="utf-8")
    calls = fake_executor(monkeypatch,normal(),after_A=alter)
    result = execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"] and not result["a_to_b_gate"]["allowed"]
    assert result["groups"][1]["requests"] == 0
    if damage == "global_ledger":
        assert (folder/"comparison_request_ledger.json").read_text(encoding="utf-8") == "unreadable synthetic ledger"


def test_gate_write_failure_never_allows_B(plan,folder,monkeypatch):
    calls = fake_executor(monkeypatch,normal())
    real_write = spec.atomic_json
    def fail_gate(path,value):
        if Path(path).name == "a_to_b_gate.json":
            raise OSError("synthetic gate write failure")
        return real_write(path,value)
    monkeypatch.setattr(spec,"atomic_json",fail_gate)
    result = execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"] and result["groups"][1]["requests"] == 0
    assert "gate_persistence_error" in result["a_to_b_gate"]["reasons"]
    assert spec.read_json(folder/"comparison_request_ledger.json")["a_to_b_gate"]["allowed"] is False


def test_total_persistence_failure_raises_without_entering_B(plan,folder,monkeypatch):
    calls = fake_executor(monkeypatch,normal())
    real_finish = spec.RequestLedger.finish_group
    def fail_after_A(self,group,result):
        real_finish(self,group,result)
        if group == "A":
            monkeypatch.setattr(spec,"atomic_json",lambda *args,**kwargs:(_ for _ in ()).throw(OSError("synthetic disk failure")))
    monkeypatch.setattr(spec.RequestLedger,"finish_group",fail_after_A)
    with pytest.raises(OSError):execution.run_pair(plan,folder,support.RUNTIME[0])
    assert calls == ["A"]
    assert spec.read_json(folder/"A/summary.json")["official_reward"] == 1
