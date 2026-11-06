"""Whole-turn input budgeting; deterministic boundaries and guarded integration."""
from copy import deepcopy
import json

import pytest

import memory_support as support
from tau2.agent.budget_window import BudgetWindow, ContextBudgetError, interaction_blocks
from tau2.agent.input_tokens import LocalJsonTokenCounter
from tau2.agent.memory_agent import MemoryAgent
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.registry import registry
from tau2.utils.llm_utils import to_litellm_messages


def user(text):
    return UserMessage(role="user", content=text)


def assistant(text):
    return AssistantMessage(role="assistant", content=text)


def call(*ids):
    return AssistantMessage(role="assistant", content="", tool_calls=[ToolCall(id=i, name="echo_value", arguments={"value":i}) for i in ids])


def result(ident):
    return ToolMessage(role="tool", id=ident, content="result " + ident)


class WeightCounter:
    """Synthetic test units, explicitly not provider token counts."""
    def __init__(self):
        self.seen = []

    def count_request(self, request):
        self.seen.append(deepcopy(request))
        return 10 + 10 * len(request["messages"]) + 5 * len(request["tools"] or [])

    def describe(self):
        return {"name":"deterministic-test-weights","version":"1","tokenizer":"none; synthetic units","is_estimate":True}


def configured(history, budget=50, *, margin=0, counter=None, tools=None):
    counter = WeightCounter() if counter is None else counter
    strategy = BudgetWindow(max_input_tokens=budget, safety_margin_tokens=margin, counter=counter)
    a = MemoryAgent(tools=tools or [], domain_policy="fixed policy", llm="openai/qwen3.7-plus", memory_strategy=strategy)
    state = a.get_init_state(history)
    return a, state, counter


@pytest.fixture
def offline_runtime():
    if support.RUNTIME is None:
        pytest.skip("Use validation/run_budget_window_tests.py for guarded HTTP tests")
    return support.RUNTIME


class TestBudgetWindowNodes:
    def test_exact_boundary_oldest_blocks_only_and_full_recounts(self):
        history = [user("old1"),assistant("a1"),user("old2"),assistant("a2"),user("previous"),assistant("confirm?"),user("Yes")]
        a,state,counter = configured(history,50)
        before = deepcopy(state.messages)
        chosen = a._build_model_context(state)
        diag = state.context_selections[-1]
        assert [m.content for m in chosen[1:]] == ["previous","confirm?","Yes"]
        assert [c["estimated_input_tokens"] for c in diag["candidate_counts"]] == [90,70,50]
        assert diag["removed_block_count"] == 2 and diag["removed_message_count"] == 4
        assert diag["retained_blocks"][0]["start"] == 4
        assert diag["retained_blocks"][-1]["end"] == 7
        assert len(counter.seen) == 3 and all(len(r["messages"]) >= 4 for r in counter.seen)
        assert state.messages == before and state.memory.messages == before

    def test_margin_is_independent_and_minimum_not_removed(self):
        a,state,_ = configured([user("previous"),assistant("question"),user("current")],50,margin=1)
        with pytest.raises(ContextBudgetError) as error:
            a._build_model_context(state)
        assert error.value.stop_reason == "context_budget_exceeded"
        assert error.value.official_reward is None
        diag = state.context_selections[-1]
        assert diag["estimated_input_tokens_after"] == 50 and diag["safety_margin_tokens"] == 1
        assert diag["retained_block_count"] == 2 and diag["removed_block_count"] == 0

    @pytest.mark.parametrize("answer", ["Yes","No","a different response"])
    def test_two_block_retention_does_not_depend_on_user_text(self,answer):
        a,state,_ = configured([user("old"),assistant("old answer"),user("previous"),assistant("question"),user(answer)],50)
        assert [m.content for m in a._build_model_context(state)[1:]] == ["previous","question",answer]

    def test_preamble_is_a_separate_complete_block(self):
        history=[assistant("greeting"),user("previous"),assistant("answer"),user("current")]
        blocks,pairs=interaction_blocks(history)
        assert blocks[0].kind == "preamble" and blocks[0].closed_by_next_user
        a,state,_=configured(history,50)
        chosen=a._build_model_context(state)
        assert [m.content for m in chosen[1:]] == ["previous","answer","current"]
        assert state.context_selections[-1]["removed_block_count"] == 1
        assert pairs["status"] == "passed"

    def test_initial_preamble_is_protected_when_immediately_previous(self):
        a,state,_=configured([assistant("greeting"),user("first")],40)
        assert [m.content for m in a._build_model_context(state)[1:]] == ["greeting","first"]
        assert state.context_selections[-1]["removed_block_count"] == 0

    def test_multi_tool_block_removed_as_one_unit(self):
        history=[user("old"),call("a","b"),result("b"),result("a"),assistant("done"),user("previous"),assistant("question"),user("current")]
        a,state,_=configured(history,50)
        chosen=a._build_model_context(state)
        assert all(not getattr(m,"tool_calls",None) and not isinstance(m,ToolMessage) for m in chosen)
        diag=state.context_selections[-1]
        assert diag["removed_message_count"] == 5
        assert diag["tool_pairing"]["full_history"]["calls"] == 2
        assert diag["tool_pairing"]["selected_context"]["calls"] == 0

    def test_consecutive_tool_chain_retained_in_previous_block(self):
        history=[user("old"),assistant("old answer"),user("previous"),call("a"),result("a"),call("b"),result("b"),assistant("done"),user("current")]
        a,state,_=configured(history,90)
        chosen=a._build_model_context(state)[1:]
        assert [m.id for m in chosen if isinstance(m,ToolMessage)] == ["a","b"]
        assert len(chosen) == 7
        assert state.context_selections[-1]["removed_message_count"] == 2

    @pytest.mark.parametrize("bad", [
        [user("old"),result("orphan")],
        [user("old"),call("a","a"),result("a")],
        [user("old"),call("a"),result("a"),result("a")],
        [user("old"),call("a"),result("wrong")],
        [user("old"),call("a"),user("new too soon")],
    ])
    def test_invalid_old_tool_relationship_cannot_be_hidden_by_removal(self,bad):
        a,state,counter=configured(bad+[user("previous"),assistant("question"),user("current")],50)
        with pytest.raises(ContextBudgetError) as error:
            a._build_model_context(state)
        assert error.value.stop_reason == "context_protocol_error"
        assert counter.seen == []
        assert state.context_selections[-1]["tool_pairing"]["status"] == "failed"

    def test_partial_multi_tool_results_wait_without_fabrication(self):
        a,state,counter=configured([user("current"),call("a","b"),result("a")],1000)
        with pytest.raises(ContextBudgetError) as error:
            a._build_model_context(state)
        assert error.value.stop_reason == "context_incomplete_tools" and counter.seen == []
        state.messages.append(result("b"))
        chosen=a._build_model_context(state)
        assert [m.id for m in chosen if isinstance(m,ToolMessage)] == ["a","b"]

    def test_repeated_build_and_interleaved_states_are_non_destructive(self):
        a,first,_=configured([user("oldA"),assistant("a"),user("prevA"),assistant("a"),user("nowA")],50)
        second=a.get_init_state([user("prevB"),assistant("b"),user("nowB")])
        raw_first=deepcopy(first.messages)
        one=a._build_model_context(first)
        other=a._build_model_context(second)
        again=a._build_model_context(first)
        assert to_litellm_messages(one) == to_litellm_messages(again)
        assert "nowA" not in str(to_litellm_messages(other))
        one[0].content="mutated policy";one[1].content="mutated context"
        assert first.messages == raw_first and first.memory.messages == raw_first
        assert first.memory_cursor == len(first.messages)
        assert first.context_selections is not second.context_selections
        fresh=a.get_init_state()
        assert fresh.messages == fresh.memory.messages == [] and fresh.context_selections == []

    @pytest.mark.parametrize("budget,margin", [(0,0),(-1,0),(True,0),(1.5,0),(100,-1),(100,True)])
    def test_invalid_limits_rejected(self,budget,margin):
        with pytest.raises(ValueError):
            BudgetWindow(max_input_tokens=budget,safety_margin_tokens=margin)

    @pytest.mark.parametrize("value", [0,None,True,-1])
    def test_invalid_counter_value_pauses_not_zero(self,value):
        class Broken(WeightCounter):
            def count_request(self,request): return value
        a,state,_=configured([user("current")],100,counter=Broken())
        with pytest.raises(ContextBudgetError) as error:a._build_model_context(state)
        assert error.value.stop_reason == "context_budget_unknown"
        assert state.context_selections[-1]["estimated_input_tokens_after"] is None

    def test_counter_exception_pauses(self):
        class Broken(WeightCounter):
            def count_request(self,request): raise OSError("offline tokenizer unavailable")
        a,state,_=configured([user("current")],100,counter=Broken())
        with pytest.raises(ContextBudgetError,match="context_budget_unknown"):a._build_model_context(state)

    def test_registered_configuration_requires_explicit_input_budget(self):
        factory=registry.get_agent_factory("memory_agent")
        with pytest.raises(TypeError):factory([],"policy",llm="offline",memory_strategy="budget_window")
        with pytest.raises(ValueError):factory([],"policy",llm="offline",memory_strategy="unknown")
        with pytest.raises(ValueError):factory([],"policy",llm="offline",memory_strategy="full_history",memory_config={"max_input_tokens":100})

    def test_installed_offline_counter_counts_fixed_policy_tools_and_ids(self):
        counter=LocalJsonTokenCounter()
        env=support.make_environment()
        a=MemoryAgent(tools=env.get_tools(),domain_policy="policy",llm="openai/qwen3.7-plus")
        state=a.get_init_state([user("hello")])
        request=a._model_input_view(state,state.messages)
        base=counter.count_request(request)
        longer=deepcopy(request);longer["tools"][0]["function"]["description"] += " extra tool description"*100
        longer_policy=deepcopy(request);longer_policy["messages"][0]["content"] += " extra policy"*100
        calls=deepcopy(request);calls["messages"] += [{"role":"assistant","tool_calls":[{"id":"x","type":"function","function":{"name":"echo_value","arguments":'{"value":"long argument text"}'}}]},{"role":"tool","tool_call_id":"x","content":"long result text"}]
        assert counter.count_request(longer)>base and counter.count_request(longer_policy)>base and counter.count_request(calls)>base
        assert counter.describe()["is_estimate"] is True

    def test_only_tool_description_change_can_cause_budget_stop(self):
        counter=LocalJsonTokenCounter();env=support.make_environment()
        base=MemoryAgent(tools=env.get_tools(),domain_policy="policy",llm="openai/qwen3.7-plus")
        state=base.get_init_state([user("current")])
        limit=counter.count_request(base._model_input_view(state,state.messages))
        a=MemoryAgent(tools=env.get_tools(),domain_policy="policy",llm="openai/qwen3.7-plus",memory_strategy="budget_window",memory_config={"max_input_tokens":limit})
        a._build_model_context(a.get_init_state([user("current")]))
        a.tools[0].long_desc += " extra schema description"*200
        with pytest.raises(ContextBudgetError,match="context_budget_exceeded"):
            a._build_model_context(a.get_init_state([user("current")]))


def trimming_agent(*, counter=None):
    """Fixed synthetic conversation; compute an explicit offline test budget."""
    import qianwen_adapter as adapter

    env = support.make_environment()
    history = [user("REMOVED_OLD1 " * 500), assistant("old answer"),
        user("REMOVED_OLD2 " * 500), call("historical-id"), result("historical-id"), assistant("old complete"),
        user("previous request"), assistant("Please confirm?")]
    incoming = user("Yes")
    expected_call = AssistantMessage(role="assistant", content="", tool_calls=[
        ToolCall(id="current-id", name="echo_value", arguments={"value":"alpha"})])
    base = MemoryAgent(tools=env.get_tools(), domain_policy=env.get_policy(),
        llm=adapter.MODELS["agent"], llm_args=adapter.role_args("agent"))
    base_state = base.get_init_state(history)
    local = LocalJsonTokenCounter()
    minimum = history[-2:] + [incoming]
    after_tool = minimum + [expected_call, ToolMessage(role="tool", id="current-id", content="alpha")]
    # This number is fixed before the request and deliberately includes the tool follow-up.
    limit = max(local.count_request(base._model_input_view(base_state, candidate))
        for candidate in (minimum, after_tool)) + 8
    config = {"max_input_tokens":limit, "safety_margin_tokens":8}
    if counter is not None:
        config["counter"] = counter
    a = registry.get_agent_factory("memory_agent")(env.get_tools(), env.get_policy(),
        llm=adapter.MODELS["agent"], llm_args=adapter.role_args("agent"),
        memory_strategy="budget_window", memory_config=config)
    return env, a, a.get_init_state(history), incoming, limit


class TestBudgetWindowIntegration:
    @pytest.mark.parametrize("scenario", support.SCENARIOS)
    def test_large_budget_each_request_matches_full_history(self, offline_runtime, monkeypatch, scenario):
        expected = support.run_scenario("memory_agent", scenario)
        original = registry.get_agent_factory("memory_agent")
        selected_agents = []

        def factory(*args, **kwargs):
            a = original(*args, **kwargs, memory_strategy="budget_window",
                memory_config={"max_input_tokens":100000, "safety_margin_tokens":8})
            selected_agents.append(a)
            return a

        monkeypatch.setitem(registry._agent_factories, "memory_agent", factory)
        actual = support.run_scenario("memory_agent", scenario)
        assert actual == expected
        assert len(selected_agents) == 1 and isinstance(selected_agents[0].memory_strategy, BudgetWindow)

    def test_actual_trim_and_tool_loop_through_litellm(self, offline_runtime):
        from budget_bindings import bind_roles
        from offline_safety import dump
        from output_budget import BudgetSession
        from tau2.agent import llm_agent

        env, a, state, incoming, limit = trimming_agent()
        initial_history = deepcopy(state.messages)
        bodies = [support.response(calls=[("current-id","echo_value",{"value":"alpha"})]),
            support.response("tool finished")]
        # Independent FullHistory run of the specified retained turns supplies a
        # strict reference at BOTH boundaries, including LiteLLM's serialization.
        # No filtering of null fields or updating of the frozen HEAD fixture.
        reference_env = support.make_environment()
        reference = MemoryAgent(tools=reference_env.get_tools(), domain_policy=a.domain_policy,
            llm=a.llm, llm_args=deepcopy(a.llm_args))
        reference_state = reference.get_init_state(initial_history[-2:])
        reference_session = BudgetSession()
        reference_client, reference_wire, reference_folder = support.mock_client(
            "full_history_retained_turns", bodies, reference_session)
        reference_requests = []
        with reference_client, bind_roles(reference_client,lambda name:"MOCK-BUDGET-WINDOW-NOT-A-KEY",[],
                budget_session=reference_session,persist=lambda rows:dump(reference_folder / "model_records.json",rows)):
            reference_generate = llm_agent.generate

            def capture_reference(**request):
                reference_requests.append(support.canonical_request(request))
                return reference_generate(**request)

            llm_agent.generate = capture_reference
            message, reference_state = reference.generate_next_message(incoming.model_copy(deep=True),reference_state)
            response = reference_env.get_response(message.tool_calls[0])
            reference.generate_next_message(response,reference_state)
        session = BudgetSession()
        client, wire, folder = support.mock_client("budget_window_actual_trim", bodies, session)
        records, requests = [], []
        with client, bind_roles(client, lambda name:"MOCK-BUDGET-WINDOW-NOT-A-KEY", records,
                budget_session=session, persist=lambda rows:dump(folder / "model_records.json",rows)):
            original = llm_agent.generate

            def capture(**request):
                requests.append(support.canonical_request(request))
                return original(**request)

            llm_agent.generate = capture
            first, state = a.generate_next_message(incoming, state)
            assert first.is_tool_call() and len(first.tool_calls) == 1
            tool_message = env.get_response(first.tool_calls[0])
            assert not tool_message.error and tool_message.id == "current-id"
            final, state = a.generate_next_message(tool_message, state)
        assert final.content == "tool finished" and not final.is_tool_call()
        assert len(wire) == len(requests) == 2
        assert requests == reference_requests and wire == reference_wire
        assert env.tools.executions == [["echo_value","alpha"]]
        assert env.tools.executions == reference_env.tools.executions
        assert state.messages[:len(initial_history)] == initial_history
        assert state.memory.messages == state.messages and state.memory_cursor == len(state.messages)
        assert len(state.context_selections) == 2
        counter = LocalJsonTokenCounter()
        comparison = []
        for request, sent, diag in zip(requests, wire, state.context_selections):
            assert diag["status"] == "trimmed" and diag["removed_block_count"] == 2
            assert diag["removed_message_count"] == 6 and diag["retained_block_count"] == 2
            assert diag["retained_blocks"][0]["start"] == 6
            assert diag["tool_pairing"]["full_history"]["status"] == "passed"
            assert diag["tool_pairing"]["selected_context"]["status"] == "passed"
            assert diag["estimated_input_tokens_after"] + 8 <= limit
            assert "REMOVED_OLD" not in json.dumps(sent)
            assert sent["tools"] == request["tools"] == [t.openai_schema for t in env.get_tools()]
            assert sent["tool_choice"] == request.get("tool_choice", "auto")
            assert sent["messages"][0] == to_litellm_messages(state.system_messages)[0]
            assert sent["max_completion_tokens"] == 1024 and "max_tokens" not in sent
            # FullHistory and the selected suffix use precisely the same estimator.
            visible_count = len(initial_history) + (1 if len(comparison) == 0 else 3)
            full = state.messages[:visible_count]
            full_count = counter.count_request(a._model_input_view(state, full))
            selected_view = {k:deepcopy(request[k]) for k in ("model","messages","tools")}
            selected_view["tool_choice"] = request.get("tool_choice", "auto")
            assert counter.count_request(selected_view) == diag["estimated_input_tokens_after"]
            assert full_count == diag["estimated_input_tokens_before"]
            comparison.append({"full_history_estimate":full_count,
                "budget_window_estimate":diag["estimated_input_tokens_after"]})
        second = wire[1]["messages"]
        assert [m["tool_call_id"] for m in second if m["role"] == "tool"] == ["current-id"]
        assert [c["id"] for m in second for c in m.get("tool_calls",[])] == ["current-id"]
        dump(folder / "trim_example.json", {"max_input_tokens":limit,"safety_margin_tokens":8,
            "counter":counter.describe(),"comparison":comparison,"context_selections":state.context_selections,
            "tool_executions":env.tools.executions,"mock_http_requests":len(wire),"real_api_requests":0,
            "raw_history_preserved":True,"removed_content_not_in_wire":True})

    def test_minimum_over_budget_prevents_request_and_tools(self, offline_runtime):
        import qianwen_adapter as adapter
        from budget_bindings import bind_roles
        from offline_safety import dump
        from output_budget import BudgetSession

        env = support.make_environment()
        a = registry.get_agent_factory("memory_agent")(env.get_tools(),env.get_policy(),
            llm=adapter.MODELS["agent"],llm_args=adapter.role_args("agent"),
            memory_strategy="budget_window",memory_config={"max_input_tokens":1})
        state = a.get_init_state([user("previous request"),assistant("Please confirm?")])
        session = BudgetSession()
        client, wire, folder = support.mock_client("minimum_over_budget", [], session)
        records = []
        with client, bind_roles(client,lambda name:"MOCK-BUDGET-WINDOW-NOT-A-KEY",records,
                budget_session=session,persist=lambda rows:dump(folder / "model_records.json",rows)):
            with pytest.raises(ContextBudgetError) as error:
                message, state = a.generate_next_message(user("Yes"),state)
                if message.is_tool_call():
                    env.get_response(message.tool_calls[0])
        assert error.value.stop_reason == "context_budget_exceeded" and error.value.official_reward is None
        assert len(wire) == 0 and records == [] and env.tools.executions == []
        assert [m.content for m in state.messages] == ["previous request","Please confirm?","Yes"]
        diag = state.context_selections[-1]
        assert diag["removed_block_count"] == 0 and diag["retained_block_count"] == 2
        assert diag["estimated_input_tokens_after"] > 1
        dump(folder / "blocked_example.json", {"diagnostics":diag,"mock_http_requests":0,
            "real_api_requests":0,"tool_executions":[],"official_reward":error.value.official_reward})

    def test_counter_failure_prevents_actual_request(self, offline_runtime):
        from budget_bindings import bind_roles
        from offline_safety import dump
        from output_budget import BudgetSession

        class Broken(LocalJsonTokenCounter):
            def count_request(self, request):
                raise OSError("local counter unavailable")

        env, a, state, incoming, _ = trimming_agent(counter=Broken())
        session = BudgetSession()
        client, wire, folder = support.mock_client("counter_unknown", [], session)
        with client, bind_roles(client,lambda name:"MOCK-BUDGET-WINDOW-NOT-A-KEY",[],
                budget_session=session,persist=lambda rows:dump(folder / "model_records.json",rows)):
            with pytest.raises(ContextBudgetError,match="context_budget_unknown"):
                a.generate_next_message(incoming,state)
        assert wire == [] and env.tools.executions == []
        assert state.context_selections[-1]["estimated_input_tokens_before"] is None

    @pytest.mark.parametrize("stop", ["output_budget_exceeded","budget_verification_unknown"])
    def test_trim_does_not_bypass_unchanged_total_output_pause(self, offline_runtime, stop):
        from budget_bindings import bind_roles
        from offline_safety import dump
        from output_budget import BudgetSession, ResourceStop

        env, a, state, incoming, _ = trimming_agent()
        body = support.response(calls=[("blocked-call","echo_value",{"value":"must not execute"})],completion=1025)
        if stop == "budget_verification_unknown":
            body.pop("usage")
        session = BudgetSession()
        client, wire, folder = support.mock_client("trim_"+stop,[body],session)
        records = []
        with client, bind_roles(client,lambda name:"MOCK-BUDGET-WINDOW-NOT-A-KEY",records,
                budget_session=session,persist=lambda rows:dump(folder / "model_records.json",rows)):
            with pytest.raises(ResourceStop) as error:
                message, state = a.generate_next_message(incoming,state)
                if message.is_tool_call():
                    env.get_response(message.tool_calls[0])
            assert error.value.stop_reason == stop
            with pytest.raises(ResourceStop):
                a.generate_next_message(user("paused"),state)
        assert state.context_selections[0]["status"] == "trimmed"
        assert len(wire) == 1 and env.tools.executions == []
        assert records[0]["official_reward"] is None
        assert records[0]["business_outcome"] == "not_evaluated_resource_pause"
