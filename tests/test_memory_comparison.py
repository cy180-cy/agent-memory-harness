"""Retail calibration and the shared, bounded comparison entry (offline only)."""

from copy import deepcopy
import os
import uuid

import pytest

import memory_support as support
import portable_support as portable
from memory_comparison.calibration import calibrate, captured_history
from memory_comparison.execution import build_group, mock_frames, retail_task, run_group, run_pair
from memory_comparison.spec import (
    HISTORY, LEGACY_OUTPUT_CONFIG, RequestLedger, RequestLimitStop, check_identity,
    atomic_json, digest, make_pair, read_json, validate_pair, write_new,
)
from run_memory_comparison import live_preflight
from tau2.agent.memory_agent import MemoryAgent
from tau2.data_model.message import UserMessage
from tau2.runner.build import build_environment
from tau2.utils.llm_utils import to_litellm_messages


@pytest.fixture(scope="module")
def runtime():
    if support.RUNTIME is None:
        pytest.skip("Use validation/run_memory_comparison.py --tests for guarded tests")
    return support.RUNTIME


@pytest.fixture(scope="module")
def material(runtime):
    task,env = retail_task(),build_environment("retail")
    report = calibrate(task,env)
    repo = read_json(runtime[1] / "repository_state.json")
    return task,env,report,repo


@pytest.fixture
def plan(material):
    task,env,report,repo = material
    return make_pair("test_"+uuid.uuid4().hex[:12],repo,report,task,env,
        output_config_version=LEGACY_OUTPUT_CONFIG)


@pytest.fixture
def folder(runtime):
    path = runtime[1]/("comparison_case_"+uuid.uuid4().hex[:12])
    path.mkdir()
    return path


def refreeze(plan):
    """Explicit synthetic test configurations; never a paid authorization."""
    plan["identity_sha256"] = digest({k:v for k,v in plan.items() if k != "identity_sha256"})
    return plan


@pytest.fixture(scope="module")
def completed_pair(runtime, material):
    task,env,report,repo = material
    plan = make_pair("test_trim_"+uuid.uuid4().hex[:12],repo,report,task,env,
        output_config_version=LEGACY_OUTPUT_CONFIG)
    folder = runtime[1]/plan["comparison_id"]
    folder.mkdir()
    return plan,folder,run_pair(plan,folder,runtime[0])


def test_retail_calibration_uses_real_input_fields_and_safe_fixed_budget(material):
    _,_,report,_ = material
    selection = report["selection"]
    assert len(report["rows"]) == 11
    assert selection["safe_lower_bound"] == max(r["minimum_estimate"] for r in report["rows"])+128
    assert selection["recommended_max_input_tokens"] == 7168
    assert selection["trimmed_agent_positions"] == [8,9,10,11]
    assert report["counter"]["is_estimate"] is True
    for row in report["rows"]:
        assert row["input_source"] == "captured" and row["count_view_source"] == "reconstructed"
        assert row["fixed_input_estimate"] == 4273
        assert row["budget_window"]["estimated_input_tokens_after"]+128 <= 7168
        assert row["minimum_estimate"]+128 <= 7168


def test_offline_child_never_inherits_private_credential_name():
    assert os.environ.get("AMH_COMPARISON_OFFLINE_CHILD") == "1"
    assert "AMH_QIANWEN_API_KEY" not in list(os.environ)  # Names only; never read its value.


def test_reconstructed_prefix_matches_only_captured_visible_past(material):
    from tau2.data_model.simulation import SimulationRun
    simulation = SimulationRun.model_validate(read_json(HISTORY/"trajectory.json"))
    events = {e["ordinal"]:e for e in read_json(HISTORY/"captured_agent_inputs.json")}
    for row in material[2]["rows"]:
        event = events[row["historical_request_ordinal"]]
        restored = captured_history(event["parameters"]["messages"][1:])
        end = row["trajectory_response_index"]
        assert row["visible_prefix_range"] == [0,end]
        assert to_litellm_messages(restored) == to_litellm_messages(simulation.messages[:end])
        assert len(restored) == end  # Response at end and all future messages excluded.


@pytest.mark.parametrize("mutation", ["model","output","policy","scorer","user"])
def test_pair_rejects_non_strategy_difference(plan,mutation):
    b = plan["groups"][1]
    if mutation == "model": b["models"]["agent"] = "openai/unapproved"
    elif mutation == "output": b["role_args"]["agent"]["max_completion_tokens"] = 999
    elif mutation == "policy": b["inputs"]["policy_sha256"] = "different policy"
    elif mutation == "scorer": b["evaluation"]["strict_replay"] = False
    else: b["user_persona_config"] = {"different":"user"}
    with pytest.raises(ValueError,match="Non-strategy"):
        validate_pair(plan)


def test_frozen_identity_rejects_even_shared_condition_change(plan):
    for group in plan["groups"]:
        group["inputs"]["policy_sha256"] = "same but changed"
    validate_pair(plan)
    with pytest.raises(ValueError,match="Frozen comparison changed"):
        check_identity(plan)


def test_actual_builders_and_all_session_database_objects_are_isolated(plan):
    a,b = [build_group(g) for g in plan["groups"]]
    assert type(a.agent) is type(b.agent) is MemoryAgent
    assert a.agent.memory_strategy.name == "full_history" and b.agent.memory_strategy.name == "budget_window"
    assert a.task is not b.task and a.task == b.task
    assert a.environment.tools.db is not b.environment.tools.db
    assert a.user is not b.user and a.agent is not b.agent
    a.initialize();b.initialize()
    assert a.agent_state is not b.agent_state and a.user_state is not b.user_state
    assert a.agent_state.memory is not b.agent_state.memory
    assert a.agent_state.context_selections is not b.agent_state.context_selections
    b_db = b.environment.get_db_hash()
    b_user = b.user_state.model_dump(mode="json")
    a.agent_state.messages.append(UserMessage(role="user",content="only A"))
    a.agent._build_model_context(a.agent_state)
    next(iter(a.environment.tools.db.orders.values())).address.city = "synthetic isolation mutation"
    assert b.environment.get_db_hash() == b_db
    assert b.user_state.model_dump(mode="json") == b_user
    assert "only A" not in str(b.agent_state.model_dump(mode="json"))
    fresh = b.agent.get_init_state()
    assert fresh.memory.messages == [] and fresh.context_selections == []
    assert not hasattr(a.agent,"task") and not hasattr(b.agent,"task")


def test_actual_retail_pair_trims_and_preserves_tool_links(completed_pair,material):
    plan,folder,result = completed_pair
    assert result["real_api_requests"] == 0 and result["mock_http_requests"] == 34
    assert result["quality_comparison_valid"] is False
    for group in result["groups"]:
        assert group["status"] == "completed" and group["requests"] == 17
        assert group["tool_calls"] == 6 and group["tool_errors"] == 0
        assert group["usage"]["source"] == "mock" and group["cost"]["amount_cny"] is None
        assert group["official_reward"] == 1 and group["db_result"]["db_match"]
        assert group["scoring_validity"] == "official_score_recorded"
    audits = {g:read_json(folder/g/"context_audit.json") for g in ("A","B")}
    a_agents = [a for a in audits["A"] if a["role"] == "agent"]
    b_agents = [a for a in audits["B"] if a["role"] == "agent"]
    assert [a["local_estimated_input"] for a in a_agents] == [r["full_history_estimate"] for r in material[2]["rows"]]
    assert [i+1 for i,a in enumerate(b_agents) if a["selection"]["status"] == "trimmed"] == [8,9,10,11]
    assert result["groups"][1]["trimming"] == {"triggered":True,"trimmed_generations":4}
    for entry in b_agents:
        calls,results = [],[]
        for message in entry["model_visible_context"]["messages"]:
            calls += [c["id"] for c in message.get("tool_calls") or []]
            if message["role"] == "tool": results.append(message["tool_call_id"])
        assert sorted(calls) == sorted(results) and len(calls) == len(set(calls))
        assert entry["local_estimated_input"]+128 <= 7168
        assert entry["selection"]["tool_pairing"]["selected_context"]["status"] == "passed"
    for g in ("A","B"):
        wire = read_json(folder/g/"request_ledger.json")
        for event in wire["events"]:
            body = event["parameters"]
            assert body["max_completion_tokens"] == (1024 if event["role"] == "agent" else 512)
            assert "max_tokens" not in body and "seed" not in body
            assert event["sdk"]["max_retries"] == event["sdk"]["http_retries"] == event["sdk"]["task_retries"] == 0
    assert plan["groups"][0]["inputs"] == plan["groups"][1]["inputs"]


def test_large_budget_pair_matches_every_actual_http_request(plan,folder,runtime):
    plan["groups"][1]["memory"]["config"]["max_input_tokens"] = 100000
    refreeze(plan)
    result = run_pair(plan,folder,runtime[0])
    assert all(g["status"] == "completed" for g in result["groups"])
    left = read_json(folder/"A/request_ledger.json")["events"]
    right = read_json(folder/"B/request_ledger.json")["events"]
    assert [e["parameters"] for e in left] == [e["parameters"] for e in right]
    assert all(d["removed_block_count"] == 0 for d in result["groups"][1]["context_selections"])
    assert result["groups"][1]["trimming"]["triggered"] is False


def test_business_failure_is_not_shared_fault_and_does_not_rerun(plan,folder,runtime):
    frames = mock_frames()
    frames[0]["body"]["choices"][0]["message"] = {"role":"assistant","content":"###STOP###"}
    frames[0]["body"]["choices"][0]["finish_reason"] = "stop"
    result = run_pair(plan,folder,runtime[0],frames_by_group={"A":frames[:1],"B":mock_frames()})
    a,b = result["groups"]
    assert a["status"] == "completed" and a["business_outcome"] == "business_failure"
    assert a["official_reward"] == 0 and a["requests"] == 1
    assert a["pause_comparison"] is False
    assert b["status"] == "completed" and b["requests"] == 17


def test_full_history_context_limit_is_resource_stop_and_blocks_B(plan,folder,runtime):
    frames = mock_frames()
    frames[1] = {"role":"agent","status":400,"body":{"error":{
        "code":"context_length_exceeded","type":"invalid_request_error","message":"synthetic context limit"}}}
    result = run_pair(plan,folder,runtime[0],frames_by_group={"A":frames[:2],"B":mock_frames()})
    a,b = result["groups"]
    assert a["status"] == "resource_stop" and a["stop_reason"] == "context_window_exceeded"
    assert a["official_reward"] is None and a["pause_comparison"] is True
    assert a["requests"] == 2 and b["actually_executed"] is False and b["requests"] == 0


def test_minimum_budget_stops_agent_before_send_without_reward(plan,folder,runtime):
    group = deepcopy(plan["groups"][1]);group["memory"]["config"]["max_input_tokens"] = 1
    ledger = RequestLedger(folder,"synthetic-minimum-test")
    result = run_group(group,folder/"B",ledger,runtime[0])
    assert result["status"] == "resource_stop" and result["stop_reason"] == "context_budget_exceeded"
    assert result["role_counts"]["agent"]["requests"] == 0
    assert result["role_counts"]["user"]["requests"] == 1  # First User turn precedes Agent's budget check.
    assert result["tool_calls"] == 0 and result["official_reward"] is None
    assert result["scoring_validity"] == "not_scored"
    assert (folder/"B/trajectory.json").is_file() and (folder/"B/agent_state.json").is_file()


@pytest.mark.parametrize("unknown", [False,True])
def test_output_v2_pause_blocks_business_tools_and_next_model(plan,folder,runtime,unknown):
    frames = mock_frames()
    tool_body = next(f["body"] for f in frames if any(c["function"]["name"]=="modify_pending_order_address"
        for c in f["body"]["choices"][0]["message"].get("tool_calls") or []))
    frames[1]["body"] = deepcopy(tool_body)
    if unknown:
        frames[1]["body"].pop("usage")
    else:
        frames[1]["body"]["usage"] = {"prompt_tokens":20,"completion_tokens":1025,"total_tokens":1045}
    ledger = RequestLedger(folder,"synthetic-output-test")
    result = run_group(plan["groups"][1],folder/"B",ledger,runtime[0],frames=frames)
    assert result["status"] == "resource_stop"
    assert result["stop_reason"] == ("budget_verification_unknown" if unknown else "output_budget_exceeded")
    assert result["requests"] == 2 and result["tool_calls"] == 0 and result["official_reward"] is None
    assert result["db_initial_hash"] == result["db_final_hash"]
    saved = read_json(folder/"B/model_responses.json")
    assert saved[1]["evidence"]["raw_response"]["choices"][0]["message"]["tool_calls"]


@pytest.mark.parametrize("fault", ["unauthorized","timeout"])
def test_shared_fault_pauses_B_and_does_not_refund_attempt(plan,folder,runtime,fault):
    frames = mock_frames()
    if fault == "unauthorized":
        frames[0] = {"role":"user","status":401,"body":{"error":{"message":"synthetic auth error"}}}
    else:
        frames[0]["transport_error"] = "timeout"
    result = run_pair(plan,folder,runtime[0],frames_by_group={"A":frames,"B":mock_frames()})
    a,b = result["groups"]
    assert a["status"] == "execution_error" and a["requests"] == 1
    assert b["actually_executed"] is False and b["requests"] == 0
    reopened = RequestLedger(folder,plan["identity_sha256"])
    assert len(reopened.data["attempts"]) == 1
    assert a["role_counts"]["user"]["failed"] == 1
    assert a["role_counts"]["user"]["timeouts"] == (1 if fault == "timeout" else 0)
    with pytest.raises(FileExistsError):
        run_pair(plan,folder,runtime[0])
    assert (folder/"A/trajectory.json").is_file()


def test_group_and_global_ledger_caps_are_durable_and_never_refund(folder):
    ledger = RequestLedger(folder,"synthetic-cap-test")
    ledger.start_once()
    for group in ("A","B"):
        for ordinal in range(1,21):
            event = {"ordinal":ordinal,"status":"reserved_before_io","role":"agent","mode":"mock"}
            ledger.sync(group,{"events":[event]})
            event["status"] = "transport_error"
            ledger.sync(group,{"events":[event]})
    assert len(ledger.data["attempts"]) == 40
    with pytest.raises(RequestLimitStop):
        ledger.sync("A",{"events":[{"ordinal":21,"status":"reserved_before_io"}]})
    with pytest.raises(RequestLimitStop):
        ledger.sync("C",{"events":[{"ordinal":1,"status":"reserved_before_io"}]})
    reopened = RequestLedger(folder,"synthetic-cap-test")
    assert len(reopened.data["attempts"]) == 40
    with pytest.raises(FileExistsError): reopened.start_once()
    with pytest.raises(ValueError): RequestLedger(folder,"different-identity")


def test_existing_transport_still_enforces_limit_before_mock_dispatch(folder,runtime):
    import httpx
    import qianwen_adapter as adapter
    from budget_transport import CountedTransport
    from output_budget import BudgetSession
    session = BudgetSession()
    sent = []
    body = support.response("synthetic text")
    def handler(request):
        sent.append(request)
        return httpx.Response(200,json=body)
    transport = CountedTransport.create(httpx.MockTransport(handler),safety=runtime[0],max_requests=1,
        caps=adapter.LIMITS,budget_session=session,
        persist=lambda value:atomic_json(folder/"request_ledger.json",value))
    transport.role = "agent"
    saved = []
    with httpx.Client(transport=transport,timeout=30,trust_env=False,follow_redirects=False) as client:
        def invoke():
            return adapter.invoke("agent",[UserMessage(role="user",content="synthetic cap test")],
                client=client,key_reader=lambda name:"MOCK-COMPARISON-NOT-A-REAL-KEY",budget_session=session,
                persist_evidence=saved.append)
        invoke()
        with pytest.raises(Exception): invoke()
    assert len(sent) == len(transport.events) == 1
    assert transport.blocked[-1]["reason"] == "Hard model request count reached"


def test_global_reservation_is_durable_before_existing_transport_dispatch(folder,runtime):
    import httpx
    import qianwen_adapter as adapter
    from budget_transport import CountedTransport
    from output_budget import BudgetSession
    ledger = RequestLedger(folder,"synthetic-dispatch-persistence",total=1)
    session = BudgetSession()
    sent = []
    def handler(request):
        persisted = read_json(ledger.path)
        assert len(persisted["attempts"]) == 1
        assert persisted["attempts"][0]["status"] == "reserved_before_io"
        sent.append(request)
        return httpx.Response(200,json=support.response("synthetic response"))
    def persist(value):
        # The admitted global journal and the local journal both precede I/O.
        ledger.sync("A",value)
        atomic_json(folder/"request_ledger.json",value)
    transport = CountedTransport.create(httpx.MockTransport(handler),safety=runtime[0],max_requests=20,
        caps=adapter.LIMITS,budget_session=session,persist=persist)
    transport.role = "agent"
    with httpx.Client(transport=transport,timeout=30,trust_env=False,follow_redirects=False) as client:
        def invoke():
            return adapter.invoke("agent",[UserMessage(role="user",content="synthetic global cap")],
                client=client,key_reader=lambda name:"MOCK-COMPARISON-NOT-A-REAL-KEY",budget_session=session,
                persist_evidence=lambda evidence:None)
        invoke()
        with pytest.raises(Exception):invoke()
    assert len(sent) == 1 and ledger.last_refusal == "persistent_request_limit_reached"
    assert len(read_json(ledger.path)["attempts"]) == 1
    assert read_json(folder/"request_ledger.json")["mock_http_requests"] == 1


def test_step_limit_stops_pending_tool_without_business_reward(plan,folder,runtime):
    group = deepcopy(plan["groups"][1]);group["limits"]["max_steps"] = 2
    frames = mock_frames()
    tool_body = next(f["body"] for f in frames if any(c["function"]["name"]=="modify_pending_order_address"
        for c in f["body"]["choices"][0]["message"].get("tool_calls") or []))
    frames[1]["body"] = deepcopy(tool_body)
    ledger = RequestLedger(folder,"synthetic-step-boundary")
    result = run_group(group,folder/"B",ledger,runtime[0],frames=frames)
    assert result["status"] == "resource_stop" and result["stop_reason"] == "max_steps"
    assert result["step_count"] == 2 and result["requests"] == 2
    assert result["tool_calls"] == 0 and result["official_reward"] is None


@pytest.mark.parametrize("auth_kind", ["missing","pending","old"])
def test_new_authorization_gate_precedes_any_key_or_network(plan,folder,material,auth_kind):
    called = []
    if auth_kind == "missing": auth = None
    elif auth_kind == "old": auth = portable.stale_authorization(folder)
    else:
        auth = folder/"pending.json"
        write_new(auth,{"approved":False,"phase":"Phase1-memory-comparison-live",
            "comparison_id":plan["comparison_id"],"identity_sha256":plan["identity_sha256"]})
    with pytest.raises(PermissionError):
        live_preflight(plan,auth,folder,material[3],lambda name:called.append(name))
    assert called == []


def test_changed_source_manifest_is_rejected_before_key(plan,folder,material):
    source = next(iter(plan["groups"][0]["code"]["source_hashes"]))
    for g in plan["groups"]:g["code"]["source_hashes"][source] = "incorrect hash"
    refreeze(plan)
    called = []
    with pytest.raises(ValueError,match="Source identity changed"):
        live_preflight(plan,None,folder,material[3],lambda name:called.append(name))
    assert called == []


def test_outputs_cannot_overwrite_previous_group_or_preparation(plan,folder,runtime):
    target = folder/"sentinel.json"
    write_new(target,{"original":"preserve"})
    with pytest.raises(FileExistsError):write_new(target,{"overwrite":True})
    assert read_json(target) == {"original":"preserve"}
    group_folder = folder/"A";group_folder.mkdir()
    sentinel = group_folder/"old.txt";sentinel.write_text("preserve",encoding="utf-8")
    ledger = RequestLedger(folder,"synthetic-old-folder")
    with pytest.raises(FileExistsError):run_group(plan["groups"][0],group_folder,ledger,runtime[0])
    assert sentinel.read_text(encoding="utf-8") == "preserve"


def test_historical_real_usage_is_not_relabelled_as_mock_usage():
    records = read_json(HISTORY/"source.json")["usage_evidence"]
    original = deepcopy(records)
    frames = mock_frames()
    assert records == original
    assert any(r["usage"].get("completion_tokens",0)>1024 for r in records if r["role"]=="agent")
    assert all(f["body"]["usage"]["completion_tokens"]==32 and "mock_usage" in f["provenance"] for f in frames)
