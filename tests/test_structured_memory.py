"""Focused deterministic facts, provenance, budgets and real local Agent wiring."""

from copy import deepcopy
import json

import pytest

import memory_support as support
import portable_support as portable
from tau2.agent.budget_window import ContextBudgetError
from tau2.agent.memory_agent import MemoryAgent, MemoryAgentState
from tau2.agent.structured_memory import StructuredMemory, StructuredMemoryState
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.registry import registry


def user(value):
    return UserMessage(role="user", content=value if isinstance(value, str) else json.dumps(value))


def assistant(value):
    return AssistantMessage(role="assistant", content=value if isinstance(value, str) else json.dumps(value))


def call(ident, tool="lookup", **arguments):
    return AssistantMessage(role="assistant", content="", tool_calls=[ToolCall(id=ident, name=tool, arguments=arguments)])


def result(ident, value, error=False):
    return ToolMessage(role="tool", id=ident, content=json.dumps(value), error=error)


def current(state, category, path=None):
    return [f for f in state.facts if f.category == category and f.status == "active" and (path is None or f.path == path)]


class CharacterCounter:
    """Synthetic deterministic units; not vendor tokens."""
    def count_request(self, request):
        return len(json.dumps(request, ensure_ascii=False))

    def describe(self):
        return {"name": "test-json-characters", "is_estimate": True}


def configured(history, budget=6000, counter=None):
    strategy = StructuredMemory(max_input_tokens=budget, safety_margin_tokens=10,
                                counter=counter or CharacterCounter())
    agent = MemoryAgent(tools=[], domain_policy="fixed policy", llm="offline",
                        memory_strategy=strategy)
    return agent, agent.get_init_state(history)


def test_facts_update_with_sources_and_entity_scope():
    strategy = StructuredMemory(max_input_tokens=6000, counter=CharacterCounter())
    state = strategy.initialize([user({"goal": "inspect my request", "constraints": {"currency": "USD"}}),
        call("one", order_id="O1"), result("one", {"order_id": "O1", "status": "pending"}),
        call("two", order_id="O2"), result("two", {"order_id": "O2", "status": "shipped"}),
        call("three", order_id="O1"), result("three", {"order_id": "O1", "status": "cancelled"})])
    facts = current(state, "entities", "/status")
    assert {(f.scope["order_id"], f.value) for f in facts} == {("O1", "cancelled"), ("O2", "shipped")}
    updated = next(f for f in facts if f.scope["order_id"] == "O1")
    assert [s["message_index"] for s in updated.sources] == [5, 6]
    assert all(s["call_id"] == "three" for s in updated.sources)
    assert any(f.value == "pending" and f.status == "superseded" for f in state.facts)
    assert current(state, "user_constraints")[0].value == "USD"


def test_conflicting_same_batch_is_preserved_then_later_observation_resolves():
    strategy = StructuredMemory(max_input_tokens=6000, counter=CharacterCounter())
    calls = AssistantMessage(role="assistant", content="", tool_calls=[
        ToolCall(id=i, name="lookup", arguments={"order_id": "O1"}) for i in ("a", "b")])
    state = strategy.initialize([user("check"), calls,
        result("a", {"order_id": "O1", "status": "pending"}),
        result("b", {"order_id": "O1", "status": "cancelled"})])
    assert {f.value for f in state.facts if f.path == "/status" and f.status == "conflict"} == {"pending", "cancelled"}
    strategy.observe(state, call("c", order_id="O1"))
    strategy.observe(state, result("c", {"order_id": "O1", "status": "cancelled"}))
    assert [f.value for f in current(state, "entities", "/status")] == ["cancelled"]


def test_mismatched_entity_error_and_orphan_do_not_create_verified_facts():
    strategy = StructuredMemory(max_input_tokens=6000, counter=CharacterCounter())
    state = strategy.initialize([user("check"), call("wrong", order_id="O1"),
        result("wrong", {"order_id": "O2", "status": "cancelled"}),
        call("failed", order_id="O3"), result("failed", {"order_id": "O3", "status": "cancelled"}, True)])
    strategy.observe(state, result("unknown", {"order_id": "O4"}))
    assert not current(state, "entities")
    assert any(f.path == "entity_mismatch" for f in state.facts)
    assert any(f.category == "tool_operations" and f.status == "failed" for f in state.facts)
    with pytest.raises(ContextBudgetError) as error:
        strategy.build_context(state, request_builder=lambda h: {"messages": h})
    assert error.value.stop_reason == "context_protocol_error"


def test_yes_is_not_permission_and_explicit_confirmation_is_exact_and_single_use():
    strategy = StructuredMemory(max_input_tokens=6000, counter=CharacterCounter())
    operation = {"tool": "update_record", "arguments": {"record_id": "R1", "value": 1}}
    state = strategy.initialize([user("update my record"), assistant({"proposed_action": operation}), user("Yes")])
    assert not current(state, "confirmations")
    strategy.observe(state, assistant({"proposed_action": operation}))
    strategy.observe(state, user({"confirmation": {**operation, "approved": True}}))
    grant = current(state, "confirmations")[0]
    assert [s["message_index"] for s in grant.sources] == [3, 4]
    strategy.observe(state, call("other", "update_record", record_id="R2", value=1))
    strategy.observe(state, result("other", "done"))
    assert state.calls["other"].confirmation_id is None and grant.status == "active"
    strategy.observe(state, call("wrong-type", "update_record", record_id="R1", value=True))
    strategy.observe(state, result("wrong-type", "done"))
    assert state.calls["wrong-type"].confirmation_id is None
    strategy.observe(state, call("matching", "update_record", record_id="R1", value=1))
    strategy.observe(state, result("matching", "done"))
    assert state.calls["matching"].confirmation_id == grant.id and grant.status == "consumed"


def test_new_user_turn_invalidates_unused_confirmation_and_scope_mismatch_stays_unresolved():
    strategy = StructuredMemory(max_input_tokens=6000, counter=CharacterCounter())
    operation = {"tool": "update_record", "arguments": {"record_id": "R1"}}
    state = strategy.initialize([user("update"), assistant({"proposed_action": operation}),
        user({"confirmation": {**operation, "approved": True}}), user("Wait, change the plan")])
    assert not current(state, "confirmations")
    assert any(f.status == "needs_reconfirmation" for f in state.facts)
    strategy.observe(state, assistant({"proposed_action": operation}))
    strategy.observe(state, user({"confirmation": {"tool": "update_record", "arguments": {"record_id": "R2"}, "approved": True}}))
    assert not current(state, "confirmations")
    assert any(f.value == "unmatched_confirmation" for f in state.facts)


def compactable_history():
    return [user({"goal": "check records", "facts": {"reference": "R1"}}),
        call("old", record_id="R1"), result("old", {"record_id": "R1", "status": "pending"}),
        assistant("old narrative " * 3000), user("recent request"), assistant("recent answer"), user("current request")]


def test_budgeted_injection_keeps_provenance_and_original_history_unchanged():
    history = compactable_history()
    original = deepcopy(history)
    agent, state = configured(history)
    before = state.model_dump(mode="json")
    context = agent._build_model_context(state)
    diagnostic = state.context_selections[-1]
    assert diagnostic["status"] == "structured" and diagnostic["removed_block_count"] == 1
    payload = json.loads(context[1].content.split("\n", 1)[1])
    assert any(f["category"] == "entities" and f["path"] == "/status" and f["value"] == "pending" for f in payload["facts"])
    for fact in payload["facts"]:
        original_fact = next(f for f in state.memory.facts if f.id == fact["source_id"])
        assert diagnostic["memory_sources"][fact["source_id"]] == original_fact.model_dump(mode="json")
        assert diagnostic["memory_sources"][fact["source_id"]]["sources"]
    assert diagnostic["estimated_input_tokens_after"] + 10 <= 6000
    assert state.messages == history == original and state.memory.messages == history
    context[-1].content = "mutation of returned copy"
    assert state.messages == original
    assert before["messages"] == state.model_dump(mode="json")["messages"]


def test_task_trial_isolation_and_state_json_roundtrip():
    agent, first = configured(compactable_history())
    second = agent.get_init_state([user("independent task")])
    assert isinstance(second.memory, StructuredMemoryState)
    assert not current(second.memory, "entities")
    restored = MemoryAgentState.model_validate_json(first.model_dump_json())
    assert isinstance(restored.memory, StructuredMemoryState) and restored.memory.facts == first.memory.facts
    first.memory.facts[0].value = "changed only in first state"
    assert second.memory.facts[0].value != first.memory.facts[0].value
    assert restored.memory.facts[0].value != first.memory.facts[0].value


def test_budget_too_small_or_counter_unknown_pauses_before_generation():
    agent, state = configured(compactable_history(), budget=10)
    with pytest.raises(ContextBudgetError) as error:
        agent._build_model_context(state)
    assert error.value.stop_reason == "context_budget_exceeded" and error.value.official_reward is None
    class BrokenCounter(CharacterCounter):
        def count_request(self, request):
            raise ValueError("synthetic counter failure")
    agent, state = configured([user("request")], counter=BrokenCounter())
    with pytest.raises(ContextBudgetError) as error:
        agent._build_model_context(state)
    assert error.value.stop_reason == "context_budget_unknown"


def test_real_local_token_counter_includes_memory_note_and_fixed_inputs():
    agent = MemoryAgent(tools=[], domain_policy="fixed policy", llm="openai/qwen3.7-plus",
                        memory_strategy="structured_memory", memory_config={"max_input_tokens": 4096, "safety_margin_tokens": 128})
    state = agent.get_init_state(compactable_history())
    original_system = deepcopy(state.system_messages)
    context = agent._build_model_context(state)
    diagnostic = state.context_selections[-1]
    assert diagnostic["status"] == "structured"
    assert diagnostic["estimated_input_tokens_after"] + 128 <= 4096
    assert context[:len(original_system)] == original_system
    assert "Task-local memory" in context[len(original_system)].content


def test_structured_agent_keeps_existing_output_pause_before_tool_release():
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession, ResourceStop
    from offline_safety import dump

    env = support.make_environment()
    agent = registry.get_agent_factory("memory_agent")(env.get_tools(), env.get_policy(),
        llm=adapter.MODELS["agent"], llm_args=adapter.role_args("agent"), memory_strategy="structured_memory",
        memory_config={"max_input_tokens": 4096})
    state = agent.get_init_state()
    session = BudgetSession()
    client, wire, folder = support.mock_client("structured-output-pause", [
        support.response(calls=[("blocked", "echo_value", {"value": "never execute"})], completion=1025)], session)
    records = []
    with client, bind_roles(client, lambda n: "MOCK-STRUCTURED-NOT-A-KEY", records,
                           budget_session=session, persist=lambda rows: dump(folder/"model_records.json", rows)):
        with pytest.raises(ResourceStop):
            agent.generate_next_message(user("request"), state)
    assert len(wire) == 1 and env.tools.executions == []
    assert session.stop_decision["status"] == "output_budget_exceeded"
    assert records[0]["evidence"]["raw_response"]["usage"]["completion_tokens"] == 1025


def test_same_registry_entry_selects_three_strategies_without_hidden_task_input():
    factory = registry.get_agent_factory("memory_agent")
    for name in ("full_history", "budget_window", "structured_memory"):
        args = {} if name == "full_history" else {"max_input_tokens": 6000}
        agent = factory([], "policy", llm="offline", memory_strategy=name, memory_config=args,
                        task={"hidden_gold": "must not be forwarded"})
        assert agent.memory_strategy.name == name and not hasattr(agent, "task")


def test_actual_agent_http_mock_tool_loop_uses_injected_memory():
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession
    from offline_safety import dump

    env = support.make_environment()
    agent = registry.get_agent_factory("memory_agent")(env.get_tools(), env.get_policy(),
        llm=adapter.MODELS["agent"], llm_args=adapter.role_args("agent"), memory_strategy="structured_memory",
        memory_config={"max_input_tokens": 6000, "counter": CharacterCounter()})
    state = agent.get_init_state(compactable_history())
    session = BudgetSession()
    client, wire, folder = support.mock_client("structured-tool-loop", [
        support.response(calls=[("new-tool", "echo_value", {"value": "ok"})]), support.response("finished")], session)
    records = []
    with client, bind_roles(client, lambda n: "MOCK-STRUCTURED-NOT-A-KEY", records,
                           budget_session=session, persist=lambda rows: dump(folder/"model_records.json", rows)):
        reply, state = agent.generate_next_message(user("continue"), state)
        result_message = env.get_response(reply.tool_calls[0])
        agent.generate_next_message(result_message, state)
    assert env.tools.executions == [["echo_value", "ok"]] and len(wire) == 2
    assert any("Task-local memory" in (m.get("content") or "") for m in wire[0]["messages"])
    assert any(m.get("tool_call_id") == "new-tool" for m in wire[1]["messages"])
    assert len(state.messages) == len(compactable_history()) + 4


def test_existing_full_history_snapshot_and_budget_window_shared_interface():
    from test_memory_agent import FIXTURE
    from test_budget_window import TestBudgetWindowNodes

    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert support.run_scenario("memory_agent", "single_tool") == frozen["cases"]["single_tool"]
    TestBudgetWindowNodes().test_exact_boundary_oldest_blocks_only_and_full_recounts()


def test_shared_executor_persists_structured_state_and_compaction_count(monkeypatch):
    from memory_comparison import spec
    from memory_comparison import execution
    from tau2.data_model.tasks import InitialState
    from test_comparison_runtime_fix import complete_frames
    from test_comparison_user_output import synthetic_builder

    # Reuse the existing fixture's upstream Mock-domain builder, then give it
    # ordinary visible initial dialogue. No Retail task or hidden answer is used.
    builder_fixture = synthetic_builder.__wrapped__(monkeypatch)
    original = execution.build_group
    def with_visible_history(group):
        orch = original(group)
        orch.task.initial_state = InitialState(message_history=[
            user({"goal": "check records", "facts": {"reference": "R1"}}),
            assistant("old visible narrative " * 6000), user("past question"), assistant("past answer"),
            user("previous request"), assistant("previous answer")])
        return orch
    monkeypatch.setattr(execution, "build_group", with_visible_history)
    group = deepcopy(portable.synthetic_plan()["groups"][1])
    group["memory"] = {"name": "structured_memory", "version": "1", "config": {
        "max_input_tokens": 7168, "safety_margin_tokens": 128}}
    folder = support.RUNTIME[1]/"structured_executor"
    folder.mkdir()
    ledger = spec.RequestLedger(folder, "synthetic-structured-executor")
    result = execution.run_group(group, folder/"B", ledger, support.RUNTIME[0], frames=complete_frames("structured-executor-tool"))
    assert builder_fixture == ["B"] and result["status"] == "completed"
    assert result["tool_calls"] == 1 and result["trimming"]["triggered"] is True
    assert result["trimming"]["trimmed_generations"] > 0
    saved = spec.read_json(folder/"B/agent_state.json")
    assert saved["memory"]["kind"] == "structured_memory" and saved["memory"]["facts"]
    assert len(list((folder/"B/received_responses").glob("*.json"))) == result["requests"] == 4
    assert spec.read_monitor_snapshot(folder)["requests"] == 4
