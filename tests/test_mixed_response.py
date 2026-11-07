"""Guarded offline adapter checks; private capture stays outside Git."""

from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import uuid

import pytest

import memory_support as support
from memory_comparison import spec
from memory_comparison.fees import FeeBudget, PRICE_POLICY, token_cost

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "validation/phase1_comparison/20261106T174245439114+0800_8e7d5cc84c"


@pytest.fixture(autouse=True)
def profile(monkeypatch):
    import qianwen_adapter as adapter
    if support.RUNTIME is None:
        pytest.skip("Use run_memory_comparison.py --tests --development-tests mixed")
    monkeypatch.setattr(adapter, "LIMITS", {"agent": 3072, "user": 1024})
    monkeypatch.setattr(adapter, "role_args", lambda role: spec.configured_role_args()[role])


@pytest.fixture
def folder():
    path = support.RUNTIME[1] / ("mixed_" + uuid.uuid4().hex[:10])
    path.mkdir()
    return path


def client_for(folder, bodies, session):
    """Actual counted transport + existing fee journal; all HTTP is Mock."""
    import httpx
    import qianwen_adapter as adapter
    from budget_transport import CountedTransport
    ledger = spec.RequestLedger(folder, "synthetic-mixed-replay")
    ledger.fees = FeeBudget(ledger, PRICE_POLICY)
    wire, queue = [], list(deepcopy(bodies))
    def handler(request):
        assert queue, "No implicit extra request is allowed"
        wire.append(json.loads(request.content))
        body = queue.pop(0)
        spec.write_new(folder / f"received_{len(wire)}.json", {"source": "mock_transport", "body": body})
        ledger.fees.settle("replay", len(wire), "agent", body.get("usage"), 200)
        return httpx.Response(200, json=body, headers={"x-request-id": "synthetic-offline-request"})
    def persist(value):
        spec.atomic_json(folder / "request_ledger.json", value)
        ledger.sync("replay", value)
    transport = CountedTransport.create(httpx.MockTransport(handler), safety=support.RUNTIME[0],
        max_requests=20, caps=adapter.LIMITS, budget_session=session, persist=persist,
        reserve_cost=lambda n, role, body: ledger.fees.reserve("replay", n, role, body))
    return httpx.Client(transport=transport, timeout=30, trust_env=False, follow_redirects=False), wire, ledger


def mixed_body(text="Let me check the information.", calls=None):
    result = support.response(calls=calls or [("synthetic-call", "echo_value", {"value": "alpha"})])
    result["choices"][0]["message"]["content"] = text
    return result


def invoke_once(body, folder, *, env=None, messages=None):
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession
    from tau2.agent import llm_agent
    from tau2.data_model.message import UserMessage
    env = env or support.make_environment()
    session, records = BudgetSession(), []
    client, wire, ledger = client_for(folder, [body], session)
    with client, bind_roles(client, lambda name: "MOCK-MIXED-NOT-A-KEY", records,
            budget_session=session, persist=lambda rows: spec.atomic_json(folder / "model_records.json", rows)):
        reply = llm_agent.generate(model=adapter.MODELS["agent"],
            messages=messages or [UserMessage(role="user", content="Synthetic lookup request")],
            tools=env.get_tools(), **adapter.role_args("agent"))
    return reply, records, wire, ledger


def test_saved_fourth_response_actual_agent_tool_loop_and_fees(folder):
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession
    from tau2.agent.memory_agent import MemoryAgent
    from tau2.data_model.message import AssistantMessage, UserMessage
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.environment import get_environment
    from tau2.orchestrator.orchestrator import Orchestrator, Role
    from tau2.user.user_simulator import UserSimulator

    captured = HISTORY / "live_execution/t69_full_history/received_responses/0004.json"
    if not captured.is_file():
        pytest.skip("Private captured response is local-only; synthetic tests remain portable")
    protected = [captured, HISTORY / "authorization.approved.json",
                 HISTORY / "live_execution/comparison_request_ledger.json",
                 HISTORY / "live_execution/suite_summary.json"]
    before = {str(p): spec.sha(p) for p in protected}
    capture = spec.read_json(captured)
    expected_hash = spec.read_json(ROOT / "docs/experiments/phase1_07_20261106/metrics.json")["evidence"]["fourth_response_sha256"]
    assert spec.sha(captured) == expected_hash
    raw = capture["raw_response"]
    sent = spec.read_json(captured.parents[1] / "request_ledger.json")["events"][3]["parameters"]
    visible = [m for m in sent["messages"] if m["role"] != "system"]
    history = [(UserMessage if m["role"] == "user" else AssistantMessage)(role=m["role"], content=m["content"])
               for m in visible]
    env = get_environment()
    agent = MemoryAgent(env.get_tools(), env.get_policy(), adapter.MODELS["agent"],
                        llm_args=adapter.role_args("agent"), memory_strategy="full_history")
    state = agent.get_init_state(history[:-1])
    # Not a Retail task run: only the saved response's read-only identity tool.
    task = Task.model_validate({"id": "synthetic-offline-mixed-loop",
        "user_scenario": {"instructions": "Offline tool protocol check only."},
        "evaluation_criteria": {"reward_basis": ["DB"], "nl_assertions": []}})
    orch = Orchestrator("retail", agent, UserSimulator(llm=adapter.MODELS["user"], instructions="Synthetic; never generated"), env, task)
    db_before = env.get_db_hash()
    executed = []
    actual_get_response = env.get_response
    def get_response(call):
        executed.append(call.model_dump(mode="json"))
        return actual_get_response(call)
    env.get_response = get_response
    session, records = BudgetSession(), []
    synthetic_end = support.response("SYNTHETIC_END: offline tool result received.")
    client, wire, ledger = client_for(folder, [raw, synthetic_end], session)
    with client, bind_roles(client, lambda name: "MOCK-MIXED-NOT-A-KEY", records,
            budget_session=session, persist=lambda rows: spec.atomic_json(folder / "model_records.json", rows)):
        reply, state = agent.generate_next_message(history[-1], state)
        orch.message, orch.from_role = reply, Role.AGENT
        orch._check_communication_error()  # Actual upstream half-duplex contract.
        results = orch._execute_tool_calls(reply.tool_calls)
        assert len(results) == 1 and not results[0].error
        final, state = agent.generate_next_message(orch._wrap_tool_results(results), state)
    original = raw["choices"][0]["message"]
    call = original["tool_calls"][0]
    assert len(executed) == 1 and len(wire) == 2
    assert executed[0]["id"] == results[0].id == call["id"]
    assert executed[0]["name"] == call["function"]["name"]
    assert executed[0]["arguments"] == json.loads(call["function"]["arguments"])
    assert reply.content is None and reply.raw_data["choices"][0]["message"]["content"] == original["content"]
    assert records[0]["evidence"]["raw_response"] == raw
    assert records[0]["evidence"]["usage"] == raw["usage"]
    audit = records[0]["evidence"]["mixed_content_normalization"]
    assert audit["status"] == "normalized" and audit["original_content"] == original["content"]
    assert records[0]["classification"]["raw_protocol_check"]["status"] == "failed"
    assert records[0]["classification"]["adapter_compatibility_check"]["status"] == "passed"
    next_tools = [m for m in wire[1]["messages"] if m["role"] == "tool"]
    assert next_tools == [{"role": "tool", "content": results[0].content, "tool_call_id": call["id"]}]
    assert original["content"] not in json.dumps(wire[1]["messages"], ensure_ascii=False)
    assert final.content == synthetic_end["choices"][0]["message"]["content"]
    assert env.get_db_hash() == db_before
    assert len(ledger.data["attempts"]) == 2 and all(e["mode"] == "mock" for e in ledger.data["attempts"])
    assert all(e["sdk"]["max_retries"] == e["sdk"]["http_retries"] == e["sdk"]["task_retries"] == 0 for e in ledger.data["attempts"])
    first_fee = ledger.fees.data["entries"][0]
    assert first_fee["raw_usage"] == raw["usage"]
    assert Decimal(first_fee["estimated_cost_cny"]) == token_cost(PRICE_POLICY, "agent", raw["usage"]["prompt_tokens"], raw["usage"]["completion_tokens"])
    assert all(spec.sha(Path(p)) == h for p, h in before.items())
    spec.write_new(support.RUNTIME[1] / "mixed_replay_summary.json", {
        "source": "captured_fourth_response_replayed_via_actual_litellm_mock_transport",
        "source_sha256": expected_hash, "raw_protocol_check": "failed_mixed_text_and_tools",
        "adapter_compatibility_check": "passed", "tool_executions": 1,
        "tool_result_id_matches": True, "next_model_input_contains_tool_result": True,
        "audit_text_not_in_next_model_input": True, "original_usage_unchanged": True,
        "history_hashes_unchanged": True, "db_unchanged": True, "mock_http_requests": 2,
        "real_api_requests": 0, "second_response_source": "synthetic", "official_reward": None})


@pytest.mark.parametrize("shape", ["text", "pure_tools", "mixed_multiple_tools"])
def test_ordinary_and_pure_tool_behavior_and_order_are_preserved(shape, folder):
    calls = [("first-id", "echo_value", {"value": "alpha"}), ("second-id", "upper_value", {"value": "beta"})]
    body = support.response("SYNTHETIC_TEXT") if shape == "text" else support.response(calls=calls)
    if shape == "mixed_multiple_tools":
        body["choices"][0]["message"]["content"] = "Let me check the information."
    reply, records, wire, ledger = invoke_once(body, folder)
    assert len(wire) == len(ledger.data["attempts"]) == 1
    assert records[0]["evidence"]["raw_response"] == body
    if shape == "text":
        assert reply.content == "SYNTHETIC_TEXT" and not reply.tool_calls
    else:
        assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == calls
        assert not reply.has_text_content()
    assert ("mixed_content_normalization" in records[0]["evidence"]) == (shape == "mixed_multiple_tools")


@pytest.mark.parametrize("case", ["answer", "confirmation", "new_fact_after_preamble", "invalid_json", "missing_id", "duplicate_ids", "truncated", "filtered"])
def test_unsafe_or_invalid_mixed_messages_still_stop(case, folder):
    body = mixed_body()
    message = body["choices"][0]["message"]
    if case == "answer":
        message["content"] = "Your refund has been approved."
    elif case == "confirmation":
        message["content"] = "Do you authorize this cancellation?"
    elif case == "new_fact_after_preamble":
        message["content"] += " Your balance is 50."
    elif case == "invalid_json":
        message["tool_calls"][0]["function"]["arguments"] = "not-json"
    elif case == "missing_id":
        message["tool_calls"][0]["id"] = ""
    elif case == "duplicate_ids":
        message["tool_calls"] *= 2
    else:
        body["choices"][0]["finish_reason"] = "length" if case == "truncated" else "content_filter"
    with pytest.raises(Exception):
        invoke_once(body, folder)
    journal = spec.read_json(folder / "request_ledger.json")
    assert journal["actual_api_requests"] == 0 and journal["mock_http_requests"] == 1
    records = spec.read_json(folder / "model_records.json")
    assert records[0]["evidence"]["raw_response"] == body
    assert records[0].get("error_type") or records[0].get("resource_stop")


def test_write_tool_preamble_cannot_bypass_consent(folder):
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
    class Tools(ToolKitBase):
        @is_tool(ToolType.WRITE)
        def change_value(self, value: str) -> str:
            """Synthetic write operation; must never be executed in this test."""
            raise AssertionError("Unexpected tool execution")
    env = Environment(domain_name="mock", policy="Do not write without consent.", tools=Tools())
    body = mixed_body(calls=[("write-id", "change_value", {"value": "alpha"})])
    with pytest.raises(Exception):
        invoke_once(body, folder, env=env)
    audit = spec.read_json(folder / "model_records.json")[0]["evidence"]["mixed_content_normalization"]
    assert audit["status"] == "blocked" and audit["reason"] == "mixed_non_read_tool_requires_explicit_text_handling"


def test_budget_unknown_still_pauses_before_normalization(folder):
    body = mixed_body()
    body.pop("usage")
    with pytest.raises(Exception):
        invoke_once(body, folder)
    journal = spec.read_json(folder / "comparison_request_ledger.json")
    assert len(journal["attempts"]) == 1 and journal["currency_budget"]["entries"][0]["status"] == "unknown"
    assert journal["shared_stop"] == "cost_verification_unknown"
