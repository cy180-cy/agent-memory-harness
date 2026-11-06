"""Exact saved-prefix replay and compact, provenance-preserving serialization."""
from copy import deepcopy
import json

import pytest
import memory_support as support
import portable_support as portable
from memory_comparison import spec
from tau2.agent.budget_window import ContextBudgetError, interaction_blocks
from tau2.agent.input_tokens import LocalJsonTokenCounter
from tau2.agent.memory_agent import MemoryAgentState
from tau2.data_model.message import AssistantMessage
from tau2.agent.structured_memory import StructuredMemory
from test_structured_memory import user, assistant, call, result, configured, compactable_history

OLD = portable.FIXTURES / "structured_obstruction.json"


def replay_obstruction():
    """Reuse exact saved visible state and the same upstream request-view boundary."""
    from tau2.agent.memory_agent import MemoryAgent
    from tau2.runner.build import build_environment
    group = portable.fixture("structured_obstruction.json")["config"]
    env = build_environment("retail")
    agent = MemoryAgent(env.get_tools(), env.get_policy(), llm=group["models"]["agent"],
                        llm_args=group["role_args"]["agent"], memory_strategy="structured_memory",
                        memory_config=group["memory"]["config"])
    state = MemoryAgentState.model_validate(portable.fixture("structured_obstruction.json")["state"])
    counts = []
    counter = LocalJsonTokenCounter()
    class RecordingCounter:
        def count_request(self, request):
            value = counter.count_request(request)
            counts.append({"tokens": value, "request": deepcopy(request)})
            return value
        def describe(self):
            return counter.describe()
    agent.memory_strategy.counter = RecordingCounter()
    snapshot = state.model_dump(mode="json")
    try:
        context = agent._build_model_context(state)
        status = "passed"
    except ContextBudgetError as error:
        context, status = None, error.stop_reason
    assert state.messages == MemoryAgentState.model_validate(snapshot).messages
    assert state.memory.model_dump(mode="json") == snapshot["memory"]
    diagnostic = state.context_selections[-1]
    cut = diagnostic.get("raw_message_start", 0)
    recent = state.memory.messages[cut:]
    variants = {"fixed_only": [], "fixed_and_recent": recent}
    measurements = {name: counter.count_request(agent._model_input_view(state, history)) for name, history in variants.items()}
    goal = next(f for f in state.memory.facts if f.id == "f1")
    measurements["fixed_recent_and_bare_goal_quote_NOT_VALID_CONTEXT"] = counter.count_request(agent._model_input_view(
        state, [AssistantMessage(role="assistant", content=goal.value["verbatim"])] + recent))
    fixed = agent._model_input_view(state, [])
    fixed["tools"], fixed["tool_choice"] = None, None
    measurements["fixed_system_without_tool_definitions"] = counter.count_request(fixed)
    last = counts[-1]["request"]
    note = last["messages"][len(state.system_messages)]["content"]
    measurement = {"source": str(OLD), "boundary": "MemoryAgent._model_input_view, complete JSON local estimate",
        "status": status, "state_fact_count": len(state.memory.facts), "state_message_count": len(state.messages),
        "candidate_complete_counts": [x["tokens"] for x in counts], "component_complete_request_probes": measurements,
        "note": note, "diagnostic": diagnostic, "final_request": last,
        "guardrail": "Probe differences are explanatory, not independently summed token counts"}
    return measurement, state, context


def payload(context):
    return json.loads(context[1].content.split("\n", 1)[1])["facts"]


def test_saved_obstruction_recount_preserves_required_goal_and_recent_interactions():
    before = portable.fixture("structured_obstruction.json")["before"]
    after, state, context = replay_obstruction()
    assert before["candidate_complete_counts"][-1] == 7174
    assert after["candidate_complete_counts"][-1] < 7174
    assert after["component_complete_request_probes"]["fixed_and_recent"] == 6975
    raw = portable.fixture("structured_obstruction.json")["state"]
    assert state.messages == MemoryAgentState.model_validate(raw).messages
    request = after["final_request"]
    original = before["final_request"]
    assert request["tools"] == original["tools"] and request["tool_choice"] == original["tool_choice"]
    assert request["messages"][0] == original["messages"][0]
    assert request["messages"][2:] == original["messages"][2:]
    facts = json.loads(after["note"].split("\n", 1)[1])["facts"]
    goal = next(f for f in facts if f["category"] == "task_goal")
    assert goal["value"] == next(f for f in state.memory.facts if f.id == "f1").value
    assert goal["confidence"] == "verbatim_unparsed" and goal["status"] == "active"
    assert after["diagnostic"]["memory_sources"][goal["source_id"]]["sources"] == [{"message_index": 1, "role": "user"}]
    assert after["diagnostic"]["raw_message_start"] == 3
    within = after["candidate_complete_counts"][-1] + 128 <= 7168
    assert (context is not None) == within
    assert after["status"] == ("passed" if within else "context_budget_exceeded")
    assert after["diagnostic"]["official_reward"] is None
    spec.write_new(support.RUNTIME[1] / "obstruction_after.json", after)


def test_old_entity_injected_with_exact_version_source_map_without_mutating_history():
    history = [user({"goal": "inspect", "constraints": {"currency": "USD"}}),
        call("old", record_id="R1"), result("old", {"record_id": "R1", "status": "pending"}),
        call("updated", record_id="R1"), result("updated", {"record_id": "R1", "status": "shipped"}),
        assistant("irrelevant narrative " * 3000), user("recent"), assistant("reply"), user("current")]
    agent, state = configured(history)
    frozen = state.memory.model_dump(mode="json")
    context = agent._build_model_context(state)
    facts = payload(context)
    shipped = next(f for f in facts if f["category"] == "entities" and f["path"] == "/status")
    assert shipped["value"] == "shipped" and shipped["status"] == "active"
    assert shipped["scope"] == {"record_id": "R1"}
    assert not any(f["value"] == "pending" for f in facts)
    diag = state.context_selections[-1]
    record = diag["memory_sources"][shipped["source_id"]]
    assert [s["message_index"] for s in record["sources"]] == [3, 4]
    assert all(s["message_index"] < diag["raw_message_start"] for s in record["sources"])
    assert context[2:] == history[diag["raw_message_start"]:]
    assert state.memory.model_dump(mode="json") == frozen and state.messages == history
    assert any(f.status == "superseded" and f.value == "pending" for f in state.memory.facts)
    # Selection provenance is a snapshot, not a mutable alias of later state.
    next(f for f in state.memory.facts if f.id == shipped["source_id"]).sources.append({"message_index": 99})
    assert [s["message_index"] for s in record["sources"]] == [3, 4]
    spec.write_new(support.RUNTIME[1] / "compact_memory_example.json", {"fact": shipped, "audit_source": record, "diagnostic": diag})


def test_unresolved_conflict_alternatives_both_survive_compact_serialization():
    from tau2.data_model.message import ToolCall
    calls = AssistantMessage(role="assistant", content="", tool_calls=[
        ToolCall(id=i, name="lookup", arguments={"order_id": "O9"}) for i in ("a", "b")])
    history = [user("check"), calls, result("a", {"order_id": "O9", "status": "pending"}),
        result("b", {"order_id": "O9", "status": "cancelled"}), assistant("old narration " * 3000),
        user("recent"), assistant("answer"), user("current")]
    agent, state = configured(history)
    facts = payload(agent._build_model_context(state))
    conflicting = [f for f in facts if f.get("path") == "/status"]
    assert {f["value"] for f in conflicting} == {"pending", "cancelled"}
    assert {f["status"] for f in conflicting} == {"conflict"}
    assert len({f["source_id"] for f in conflicting}) == 2


def test_dedup_requires_whole_retained_same_value_tool_evidence():
    history = [user("inspect"), call("old", record_id="R1"), result("old", {"record_id": "R1", "status": "pending"}),
        assistant("long old reply " * 3000), user("recent"), call("new", record_id="R1"),
        result("new", {"record_id": "R1", "status": "pending"}), assistant("answer"), user("current")]
    agent, state = configured(history)
    context = agent._build_model_context(state)
    diag = state.context_selections[-1]
    fact = next(f for f in state.memory.facts if f.category == "entities" and f.path == "/status")
    assert fact.id in diag["deduplicated_fact_ids"]
    assert fact.id not in diag["memory_fact_ids"]
    assert context[2:] == history[4:]
    _, pairing = interaction_blocks(context[1:])
    assert pairing["calls"] == pairing["results"] == 1
    assert not agent.memory_strategy._already_in_recent(fact, state.memory, 6)  # partial retained pair
    conflicting = fact.model_copy(update={"status": "conflict"}, deep=True)
    assert not agent.memory_strategy._already_in_recent(conflicting, state.memory, 4)


def test_real_input_counter_includes_compact_note_and_tiny_budget_blocks_generate(monkeypatch):
    from tau2.agent import llm_agent
    agent, state = configured(compactable_history(), budget=4096, counter=LocalJsonTokenCounter())
    context = agent._build_model_context(state)
    after = agent.memory_strategy.counter.count_request(agent._model_input_view(state, context[1:]))
    assert after == state.context_selections[-1]["estimated_input_tokens_after"]
    assert after + 10 <= 4096 and payload(context)
    calls = []
    monkeypatch.setattr(llm_agent, "generate", lambda **kwargs: calls.append(kwargs))
    tiny, tiny_state = configured(compactable_history(), budget=1, counter=LocalJsonTokenCounter())
    with pytest.raises(ContextBudgetError) as error:
        tiny.generate_next_message(user("continue"), tiny_state)
    assert error.value.stop_reason == "context_budget_exceeded" and error.value.official_reward is None
    assert calls == []


def test_semantic_open_item_paths_and_confirmation_boundaries_are_not_diagnostics():
    strategy = StructuredMemory(max_input_tokens=6000)
    op = {"tool": "update_record", "arguments": {"record_id": "R7", "value": 1}}
    state = strategy.initialize([user("inspect"), assistant({"proposed_action": op})])
    waiting = next(f for f in state.facts if f.category == "open_items")
    assert strategy._context_fact(waiting)["path"] == "awaiting_confirmation"
    strategy.observe(state, user({"confirmation": {**op, "approved": False}}))
    denied = next(f for f in state.facts if f.category == "confirmations")
    rendered = strategy._context_fact(denied)
    assert rendered["status"] == "denied" and rendered["value"] is False
    assert rendered["scope"] == op and rendered["path"] == "permission"
    strategy.observe(state, call("wrong", record_id="R7"))
    strategy.observe(state, result("wrong", {"record_id": "R8"}))
    mismatch = next(f for f in state.facts if f.path == "entity_mismatch")
    assert strategy._context_fact(mismatch)["path"] == "entity_mismatch"


def test_cannot_claim_structured_context_after_dropping_every_supported_old_fact():
    history = [assistant("old question? " * 3000), user("recent"), assistant("answer"), user("current")]
    agent, state = configured(history, budget=3000)
    with pytest.raises(ContextBudgetError) as error:
        agent._build_model_context(state)
    assert error.value.stop_reason == "context_budget_exceeded"
    assert error.value.diagnostics["detail"] == "no_supported_memory_record_fits"
    assert error.value.diagnostics["memory_fact_ids"] == []
    assert state.messages == history and state.memory.facts
