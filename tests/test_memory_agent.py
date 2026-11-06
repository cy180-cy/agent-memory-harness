"""Memory equivalence, isolation and existing output-budget integration."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import pytest

import memory_support as support
from tau2.agent import llm_agent
from tau2.agent.llm_agent import LLMAgent
from tau2.agent.memory import FullHistory, MemoryState, MemoryStrategy
from tau2.agent.memory_agent import MemoryAgent, MemoryAgentState, create_memory_agent
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.registry import registry
from tau2.runner.build import build_agent
from tau2.utils.llm_utils import to_litellm_messages

FIXTURE = Path(__file__).parent / "fixtures/memory/full_history_fc0055dc.json"
FIXTURE_SHA256 = "e4e2edcd9cc1ea659c5c01527bd8bb9ea839f47d16991287b790b5a6c69578ee"
SCENARIO_SHA256 = "c00996ea8bd1775c7ccaa381cd26d2cde832c0b2e2fa0815c945213323065a7c"


@pytest.fixture
def offline_runtime():
    if support.RUNTIME is None:
        pytest.skip("Run validation/run_memory_tests.py for guarded HTTP integration")
    return support.RUNTIME


@pytest.fixture
def requests(monkeypatch):
    captured = []
    def generate(**kwargs):
        captured.append(deepcopy(kwargs))
        return AssistantMessage(role="assistant", content=f"reply-{len(captured)}")
    monkeypatch.setattr(llm_agent, "generate", generate)
    return captured


def agent(**kwargs):
    return MemoryAgent(tools=[], domain_policy="fixed policy", llm="offline-model", **kwargs)


def test_frozen_fixture_integrity():
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    assert hashlib.sha256(Path(support.__file__).read_bytes()).hexdigest() == SCENARIO_SHA256
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert frozen["head"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    assert sum(case["request_count"] for case in frozen["cases"].values()) == 10


@pytest.mark.parametrize("agent_name", ["llm_agent", "memory_agent"])
@pytest.mark.parametrize("scenario", support.SCENARIOS)
def test_each_request_matches_independent_head_snapshot(offline_runtime, agent_name, scenario):
    frozen = json.loads(FIXTURE.read_text(encoding="utf-8"))
    # Compare all generation arguments, full wire JSON, messages, IDs and tool order.
    assert support.run_scenario(agent_name, scenario) == frozen["cases"][scenario]


def test_strategy_returns_independent_copies_and_reset():
    strategy = FullHistory()
    history = [UserMessage(role="user",content="seed"),
        AssistantMessage(role="assistant",content="",tool_calls=[ToolCall(id="seed-call",name="echo_value",arguments={"nested":{"values":[1]}})]),
        ToolMessage(role="tool",id="seed-call",content="ok")]
    before = deepcopy(history)
    first = strategy.initialize(history)
    second = strategy.initialize()
    context = strategy.build_context(first)
    context[0].content = "changed"
    context[1].tool_calls[0].arguments["nested"]["values"].append(2)
    context[2].id = "wrong"
    assert history == before
    assert first.messages == before
    assert second.messages == []
    assert not hasattr(strategy, "messages")


def test_agent_context_cannot_mutate_raw_history_or_policy(requests):
    a = agent()
    history = [UserMessage(role="user",content="seed")]
    original = deepcopy(history)
    state = a.get_init_state(history)
    history[0].content = "caller mutation"
    assert state.messages == original
    _, state = a.generate_next_message(UserMessage(role="user",content="next"),state)
    raw = deepcopy(state.model_dump(mode="json"))
    context = a._build_model_context(state)
    context[0].content = "rewritten policy"
    context[1].content = "rewritten history"
    assert state.model_dump(mode="json") == raw
    assert a.domain_policy == "fixed policy"
    assert state.messages is not state.memory.messages
    assert all(m is not n for m,n in zip(state.messages,state.memory.messages))


def test_tasks_trials_reset_and_interleaved_states(requests):
    a = agent()
    first = a.get_init_state([UserMessage(role="user",content="task-A")])
    second = a.get_init_state([UserMessage(role="user",content="task-B")])
    a.generate_next_message(UserMessage(role="user",content="A1"),first)
    a.generate_next_message(UserMessage(role="user",content="B1"),second)
    a.generate_next_message(UserMessage(role="user",content="A2"),first)
    assert [m.content for m in requests[0]["messages"]][1:] == ["task-A","A1"]
    assert [m.content for m in requests[1]["messages"]][1:] == ["task-B","B1"]
    assert [m.content for m in requests[2]["messages"]][1:] == ["task-A","A1","reply-1","A2"]
    saved = deepcopy(first.model_dump(mode="json"))
    reset = a.get_init_state()
    a.generate_next_message(UserMessage(role="user",content="fresh trial"),reset)
    assert [m.content for m in requests[-1]["messages"]][1:] == ["fresh trial"]
    assert first.model_dump(mode="json") == saved
    assert reset.memory is not first.memory and first.memory is not second.memory
    restored = MemoryAgentState.model_validate_json(first.model_dump_json())
    assert restored.memory_cursor == len(restored.messages)
    assert to_litellm_messages(restored.memory.messages) == to_litellm_messages(first.memory.messages)


def test_strategy_lifecycle_observes_each_visible_event_once(requests):
    class Recording(FullHistory):
        def __init__(self):
            self.events = []
        def initialize(self, history=()):
            self.events.append(("initialize", [m.role for m in history]))
            return super().initialize(history)
        def observe(self,state,message):
            self.events.append(("observe",message.role))
            super().observe(state,message)
        def build_context(self,state):
            self.events.append(("build",None))
            return super().build_context(state)
    strategy = Recording()
    a = agent(memory_strategy=strategy)
    state = a.get_init_state([UserMessage(role="user",content="seed")])
    a.generate_next_message(UserMessage(role="user",content="next"),state)
    assert strategy.events == [("initialize",["user"]),("observe","user"),("build",None),("observe","assistant")]
    assert state.memory_cursor == len(state.messages) == len(state.memory.messages) == 3
    a._build_model_context(state)
    assert len([e for e in strategy.events if e[0] == "observe"]) == 2
    assert all(not isinstance(m,SystemMessage) for m in state.memory.messages)


def test_strategy_return_value_is_the_actual_context_extension_point(requests):
    class ContextProbe(FullHistory):
        # Test double only, not a production window/summary implementation.
        name = "context_probe"
        def build_context(self,state):
            return [state.messages[-1].model_copy(deep=True)]
    a = agent(memory_strategy=ContextProbe())
    state = a.get_init_state([UserMessage(role="user",content="retained raw history")])
    a.generate_next_message(UserMessage(role="user",content="visible current"),state)
    assert [m.content for m in requests[0]["messages"]][1:] == ["visible current"]
    assert [m.content for m in state.messages] == ["retained raw history","visible current","reply-1"]


@pytest.mark.parametrize("bad", ["unknown", "summary_memory", None, {}])
def test_invalid_strategy_rejected_before_generation(requests, bad):
    with pytest.raises((ValueError,TypeError)):
        registry.get_agent_factory("memory_agent")([], "fixed policy", llm="offline-model", memory_strategy=bad)
    assert requests == []


def test_strategy_config_not_forwarded_as_model_kwargs(requests):
    with pytest.raises(ValueError,match="not an LLM argument"):
        agent(llm_args={"memory_strategy":"unknown"})
    assert requests == []


def test_memory_cannot_replace_system_policy(requests):
    class PolicyInjection(FullHistory):
        def build_context(self,state):
            return [SystemMessage(role="system",content="injected policy")]
    a = agent(memory_strategy=PolicyInjection())
    state = a.get_init_state()
    with pytest.raises(ValueError,match="agent-visible"):
        a.generate_next_message(UserMessage(role="user",content="next"),state)
    assert requests == []
    assert a.domain_policy == "fixed policy"


def test_registry_default_agent_and_memory_selection():
    env = support.make_environment()
    ordinary = build_agent("llm_agent",env,llm="offline-model")
    memory = build_agent("memory_agent",env,llm="offline-model")
    assert type(ordinary) is LLMAgent
    assert type(memory) is MemoryAgent
    assert isinstance(memory.memory_strategy,MemoryStrategy)
    assert memory.memory_strategy.name == "full_history"
    assert memory.get_init_state().strategy_version == "1"


def test_memory_contract_has_no_hidden_task_input():
    for method in (MemoryStrategy.initialize,MemoryStrategy.observe,MemoryStrategy.build_context):
        assert "task" not in inspect.signature(method).parameters
    assert "task" not in inspect.signature(MemoryAgent).parameters
    assert set(MemoryState.model_fields) == {"messages"}
    class HiddenTask:
        def __getattribute__(self,name):
            raise AssertionError("Hidden Task was read: " + name)
    a = create_memory_agent([],"fixed policy",llm="offline-model",task=HiddenTask())
    assert not hasattr(a,"task")


def test_hidden_evaluation_changes_do_not_change_context(offline_runtime):
    left = support.run_scenario("memory_agent","empty_history",task={"hidden":"GOLD_A","actions":["answer-A"]})
    right = support.run_scenario("memory_agent","empty_history",task={"hidden":"GOLD_B","actions":["answer-B"]})
    assert left == right
    assert "GOLD_" not in json.dumps(left)


@pytest.mark.parametrize("agent_name", ["llm_agent","memory_agent"])
@pytest.mark.parametrize("stop", ["output_budget_exceeded","budget_verification_unknown"])
def test_memory_does_not_bypass_existing_total_output_pause(offline_runtime,agent_name,stop):
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession, ResourceStop
    from offline_safety import dump
    env = support.make_environment()
    a = build_agent(agent_name,env,llm=adapter.MODELS["agent"],llm_args=adapter.role_args("agent"))
    body = support.response(calls=[("blocked-call","echo_value",{"value":"must not execute"})],completion=1025)
    if stop == "budget_verification_unknown":
        body.pop("usage")
    session = BudgetSession()
    client,wire,folder = support.mock_client(agent_name+"_"+stop,[body],session)
    records = []
    state = a.get_init_state()
    with client, bind_roles(client,lambda name:"MOCK-MEMORY-NOT-A-REAL-KEY",records,budget_session=session,
                            persist=lambda rows:dump(folder/"model_records.json",rows)):
        with pytest.raises(ResourceStop) as caught:
            message,state = a.generate_next_message(UserMessage(role="user",content="attempt"),state)
            if message.is_tool_call():
                env.get_response(message.tool_calls[0])
        assert caught.value.stop_reason == stop
        with pytest.raises(ResourceStop):
            a.generate_next_message(UserMessage(role="user",content="must remain paused"),state)
    assert len(wire) == 1
    assert env.tools.executions == []
    assert records[0]["official_reward"] is None
    assert records[0]["business_outcome"] == "not_evaluated_resource_pause"
