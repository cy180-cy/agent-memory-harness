"""Minimal copy of Phase0-06 Agent/User binding with the total-output gate."""
from contextlib import contextmanager
import qianwen_adapter as adapter
from compatibility import assess
from offline_safety import require
from output_budget import ResourceStop


@contextmanager
def bind_roles(client, key_reader, records, *, budget_session, persist):
    from tau2.agent import llm_agent
    from tau2.user import user_simulator
    old_agent, old_user = llm_agent.generate, user_simulator.generate

    def invoke(role, **request):
        budget_session.before_request()
        require(request.get("model") == adapter.MODELS[role], "Unexpected role/model; Judge and Review disabled")
        expected = adapter.role_args(role)
        require(all(request.get(k) == v for k, v in expected.items()) and "max_tokens" not in request, "Role arguments differ from total-output draft")
        client._transport.role = role
        observed, actual = {}, []
        entry = {"role": role, "real_model_request": client._transport.live}
        records.append(entry)

        def save(evidence):
            entry["evidence"] = evidence
            persist(records)

        try:
            converted = adapter.invoke(role, request["messages"], client=client,
                key_reader=key_reader, tools=request.get("tools"), tool_choice=request.get("tool_choice"),
                evidence=observed, message_observer=actual.append,
                budget_session=budget_session, persist_evidence=save)
            classification = assess(observed.get("raw_response"), converted, request.get("tools") or [], tool_loop_verified=True)
            entry.update(converted=converted, classification=classification)
            require(classification["adapter_compatibility_check"]["status"] == "passed", "Unsafe response; business tool blocked")
            require(len(actual) == 1, "Actual upstream message missing")
            return actual[0]
        except ResourceStop as error:
            entry.update(resource_stop=error.decision, official_reward=None, business_outcome="not_evaluated_resource_pause")
            raise
        except Exception as error:
            client._transport.stopped = True
            entry["error_type"] = type(error).__name__
            raise
        finally:
            persist(records)

    llm_agent.generate = lambda **request: invoke("agent", **request)
    user_simulator.generate = lambda **request: invoke("user", **request)
    try:
        yield
    finally:
        llm_agent.generate, user_simulator.generate = old_agent, old_user
