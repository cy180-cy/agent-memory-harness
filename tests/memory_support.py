"""Fixed offline scenarios shared by the pre-change capture and regressions."""
from copy import deepcopy
import json
from pathlib import Path
import uuid

RUNTIME = None
HTTP_MOCK_REQUESTS = 0
SCENARIOS = (
    "empty_history", "initial_history", "dialogue", "single_tool",
    "multi_tool", "initial_tool_history",
)


def configure(safety, artifacts):
    global RUNTIME
    RUNTIME = (safety, Path(artifacts))


def make_environment():
    from tau2.environment.environment import Environment
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

    class MemoryTools(ToolKitBase):
        def __init__(self):
            super().__init__()
            self.executions = []

        @is_tool(ToolType.READ)
        def echo_value(self, value: str) -> str:
            """Echo a value in memory.

            Args:
                value: Value to echo.
            Returns:
                The unchanged value.
            """
            self.executions.append(["echo_value", value])
            return value

        @is_tool(ToolType.READ)
        def upper_value(self, value: str) -> str:
            """Uppercase a value in memory.

            Args:
                value: Value to uppercase.
            Returns:
                The uppercase value.
            """
            self.executions.append(["upper_value", value])
            return value.upper()

    return Environment(domain_name="mock", policy="Only echo or uppercase the supplied value. Preserve this policy.", tools=MemoryTools())


def initial_history(scenario):
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
    if scenario == "initial_history":
        return [UserMessage(role="user", content="prior user"), AssistantMessage(role="assistant", content="prior assistant")]
    if scenario == "initial_tool_history":
        return [UserMessage(role="user", content="prior tool request"),
            AssistantMessage(role="assistant", content="", tool_calls=[ToolCall(id="history-call", name="echo_value", arguments={"value":"seed"})]),
            ToolMessage(role="tool", id="history-call", content="seed", requestor="assistant")]
    return []


def response(content="ok", calls=None, *, completion=32):
    msg = {"role":"assistant", "content":content}
    if calls:
        msg.update(content="", tool_calls=[{"id":ident,"type":"function","function":{"name":name,"arguments":json.dumps(args)}} for ident,name,args in calls])
    return {"id":"fixed-response", "object":"chat.completion", "created":1, "model":"qwen3.7-plus",
        "choices":[{"index":0,"finish_reason":"tool_calls" if calls else "stop","message":msg}],
        "usage":{"prompt_tokens":20,"completion_tokens":completion,"total_tokens":20+completion}}


def frames(scenario):
    if scenario == "dialogue":
        return [response("first answer"), response("second answer"), response("third answer")]
    if scenario == "single_tool":
        return [response(calls=[("single-call","echo_value",{"value":"alpha"})]), response("tool finished")]
    if scenario == "multi_tool":
        return [response(calls=[("multi-first","echo_value",{"value":"alpha"}), ("multi-second","upper_value",{"value":"beta"})]), response("both tools finished")]
    return [response("answer " + scenario)]


def canonical_request(request):
    from tau2.utils.llm_utils import to_litellm_messages
    return {k: to_litellm_messages(v) if k == "messages" else [t.openai_schema for t in v] if k == "tools" and v is not None else deepcopy(v) for k,v in request.items()}


def mock_client(label, bodies, session):
    import httpx
    import qianwen_adapter as adapter
    from budget_transport import CountedTransport
    from offline_safety import dump
    if RUNTIME is None:
        raise RuntimeError("Use validation/run_memory_tests.py for guarded HTTP tests")
    safety, artifacts = RUNTIME
    folder = artifacts / (label + "_" + uuid.uuid4().hex[:8])
    folder.mkdir()
    remaining, wire = list(bodies), []

    def handler(request):
        global HTTP_MOCK_REQUESTS
        if not remaining:
            raise AssertionError("Unexpected extra model request")
        HTTP_MOCK_REQUESTS += 1
        wire.append(json.loads(request.content))
        return httpx.Response(200, json=deepcopy(remaining.pop(0)), headers={"x-request-id":"fixed-offline-request-id"})

    transport = CountedTransport.create(httpx.MockTransport(handler), safety=safety, max_requests=20,
        caps=adapter.LIMITS, budget_session=session,
        persist=lambda value: dump(folder / "request_ledger.json",value))
    return httpx.Client(transport=transport, timeout=30, trust_env=False, follow_redirects=False), wire, folder


def run_scenario(agent_name, scenario, *, task=None):
    import qianwen_adapter as adapter
    from budget_bindings import bind_roles
    from output_budget import BudgetSession
    from offline_safety import dump
    from tau2.agent import llm_agent
    from tau2.data_model.message import MultiToolMessage, UserMessage
    from tau2.runner.build import build_agent
    from tau2.utils.llm_utils import to_litellm_messages

    env = make_environment()
    args = {**adapter.role_args("agent"), "tool_choice":"auto"}
    agent = build_agent(agent_name, env, llm=adapter.MODELS["agent"], llm_args=args, task=task)
    state = agent.get_init_state(initial_history(scenario))
    initialized = {"system_messages":to_litellm_messages(state.system_messages), "messages":to_litellm_messages(state.messages), "tools":[t.openai_schema for t in agent.tools]}
    session = BudgetSession()
    client, wire, folder = mock_client(agent_name + "_" + scenario, frames(scenario), session)
    records, requests = [], []
    with client, bind_roles(client, lambda name:"MOCK-MEMORY-NOT-A-REAL-KEY", records,
                           budget_session=session, persist=lambda rows:dump(folder / "model_records.json",rows)):
        budget_generate = llm_agent.generate
        def capture(**request):
            requests.append(canonical_request(request))
            return budget_generate(**request)
        llm_agent.generate = capture
        first, state = agent.generate_next_message(UserMessage(role="user",content="new visible request"),state)
        if scenario == "dialogue":
            for text in ("second visible request","third visible request"):
                _, state = agent.generate_next_message(UserMessage(role="user",content=text),state)
        elif first.is_tool_call():
            results = [env.get_response(call) for call in first.tool_calls]
            if any(result.error for result in results):
                raise AssertionError("Offline tool error")
            incoming = results[0] if len(results) == 1 else MultiToolMessage(role="tool",tool_messages=results)
            _, state = agent.generate_next_message(incoming,state)
    return {"initialized":initialized,"generate_requests":requests,"http_requests":wire,
        "final_messages":to_litellm_messages(state.messages),"tool_executions":env.tools.executions,
        "request_count":len(wire)}
