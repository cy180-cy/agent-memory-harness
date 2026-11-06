"""Thin orchestration over upstream builders/runner and the existing v2 client.

No Agent loop, model client, tool implementation or scorer is reimplemented.
"""

import contextlib
import json
import math
import time
from copy import deepcopy

from .spec import (
    HISTORY, PROXY, RequestLedger, atomic_json, check_identity, read_json,
    task_identity, validate_pair, write_new,
    require_authorization, sha, group_output_limits, normalize_mode, execution_mode,
)


def completion_reasons(result, group):
    """Positive execution-validity contract, independent of business success."""
    if not isinstance(result, dict):
        return ["A_result_missing"]
    reasons = []
    if result.get("stop_reason"):
        reasons.append(str(result["stop_reason"]))
    if result.get("actually_executed") is not True:
        reasons.append("A_not_executed")
    if result.get("status") != "completed":
        reasons.append("A_not_completed")
    if result.get("termination_reason") not in ("user_stop", "agent_stop"):
        reasons.append("A_termination_not_normal")
    if result.get("pause_comparison") is True or result.get("error_type"):
        reasons.append("A_pause_or_error_recorded")
    if result.get("scoring_validity") != "official_score_recorded":
        reasons.append("A_scoring_not_valid")
    reward = result.get("official_reward")
    if type(reward) not in (int, float) or not math.isfinite(reward) or not 0 <= reward <= 1:
        reasons.append("A_reward_unverifiable")
    db = result.get("db_result")
    if (not isinstance(db, dict) or type(db.get("db_match")) is not bool
            or type(db.get("db_reward")) not in (int,float)
            or not math.isfinite(db["db_reward"]) or not 0 <= db["db_reward"] <= 1):
        reasons.append("A_DB_result_missing")
    elif db["db_reward"] != reward or db["db_match"] != (db["db_reward"] == 1):
        reasons.append("A_score_components_inconsistent")
    requests = result.get("requests")
    if type(requests) is not int or requests <= 0:
        reasons.append("A_request_count_unknown")
    elif requests >= group["limits"]["group_requests"]:
        reasons.append("request_limit_reached")
    steps = result.get("step_count")
    if type(steps) is not int or not 0 <= steps <= group["limits"]["max_steps"]:
        reasons.append("A_step_count_unverifiable")
    usage = result.get("usage")
    if not isinstance(usage, dict) or usage.get("status") != "known" or usage.get("unknown_attempts") != 0:
        reasons.append("budget_verification_unknown")
    counts = usage.get("known_subtotal") if isinstance(usage,dict) else None
    if (not isinstance(counts,dict) or any(type(counts.get(k)) is not int or counts[k] < 0
            for k in ("prompt_tokens","completion_tokens","total_tokens"))
            or counts["prompt_tokens"] + counts["completion_tokens"] != counts["total_tokens"]):
        reasons.append("budget_verification_unknown")
    decisions = result.get("output_resource_decisions")
    if not isinstance(decisions, list) or len(decisions) != requests or any(
            d.get("status") != "budget_verified" for d in decisions if isinstance(d, dict)) or any(
            not isinstance(d, dict) for d in decisions):
        reasons.append("A_output_verification_incomplete")
    if result.get("output_truncated") != 0 or result.get("usage_verification_unknown") != 0:
        reasons.append("A_output_incomplete")
    checks = result.get("execution_checks")
    checks = checks if isinstance(checks, dict) else {}
    for name in ("model_calls_complete", "tool_calls_complete", "trajectory_complete",
                 "safety_verified", "artifacts_complete"):
        if checks.get(name) is not True:
            reasons.append("A_" + name + "_not_verified")
    elapsed = result.get("total_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0:
        reasons.append("A_time_unknown")
    elif elapsed >= group["limits"]["task_timeout_seconds"]:
        reasons.append("task_timeout")
    return list(dict.fromkeys(reasons))


def evaluate_a_to_b(result, plan, durable, folder, *, group_id="A", completed_prefix=()):
    """Reconcile A with persisted evidence before constructing any B resources."""
    group = next(g for g in plan["groups"] if g["group"] == group_id)
    group_folder = folder / group_id
    reasons = completion_reasons(result, group)
    try:
        if durable["identity"] != plan["identity_sha256"]:
            reasons.append("ledger_identity_mismatch")
        if durable.get("shared_stop"):
            reasons.append(str(durable["shared_stop"]))
        if durable["groups"].get(group_id) != result:
            reasons.append("A_result_not_durably_recorded")
        attempts = durable["attempts"]
        entries = [e for e in attempts if e["group"] == group_id]
        if any(e["group"] not in (*completed_prefix, group_id) for e in attempts):
            reasons.append("unexpected_pre_B_attempt")
        if len(attempts) >= plan["groups"][0]["limits"]["comparison_requests"]:
            reasons.append("comparison_request_limit_reached")
        if not isinstance(result, dict) or len(entries) != result.get("requests"):
            reasons.append("request_ledger_count_mismatch")
        else:
            if [e["ordinal"] for e in entries] != list(range(1, len(entries) + 1)):
                reasons.append("request_ordinal_mismatch")
            mode = normalize_mode(durable["mode"])
            if mode != normalize_mode(result.get("mode")) or any(
                    normalize_mode(e.get("mode")) != mode or e.get("status") != "response_received"
                    or type(e.get("http_status")) is not int or not 200 <= e["http_status"] < 300
                    for e in entries):
                reasons.append("model_attempt_not_complete")
            for role in ("agent", "user"):
                count = sum(e.get("role") == role for e in entries)
                counts = result.get("role_counts", {}).get(role, {})
                if (counts.get("requests"), counts.get("http_success"), counts.get("failed"),
                        counts.get("timeouts")) != (count, count, 0, 0):
                    reasons.append("role_request_count_mismatch")
            if any(e.get("role") not in ("agent", "user") for e in entries):
                reasons.append("unapproved_model_role")
            summary = group_folder / "summary.json"
            if read_json(summary) != result:
                reasons.append("A_summary_readback_mismatch")
            score = read_json(group_folder / "official_reward.json")
            if score.get("reward") != result.get("official_reward") or score.get("db_check") != result.get("db_result"):
                reasons.append("A_official_score_readback_mismatch")
            local = read_json(group_folder / "request_ledger.json")
            local_events = [{**e, "mode": normalize_mode(e.get("mode"))} for e in local.get("events", [])]
            expected_events = [{**{k:v for k,v in e.items() if k not in ("key","group")},
                                "mode": normalize_mode(e.get("mode"))} for e in entries]
            if (local_events != expected_events
                    or local.get("actual_api_requests") != (len(entries) if mode == "live" else 0)
                    or local.get("mock_http_requests") != (len(entries) if mode == "mock" else 0)):
                reasons.append("local_global_request_ledger_mismatch")
            required = {"trajectory.json", "agent_state.json", "context_audit.json",
                        "model_responses.json", "request_ledger.json", "official_reward.json",
                        "simulation.json"}
            hashes = result.get("evidence_hashes", {})
            if (not required <= hashes.keys() or not hashes.keys() <= required | {"tool_events.json"}
                    or any(sha(group_folder / name) != h for name, h in hashes.items())):
                reasons.append("A_evidence_missing_or_changed")
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        reasons.append("A_persisted_evidence_unverifiable")
    reasons = list(dict.fromkeys(reasons))
    return {"schema":"A-to-B-gate-v1", "allowed":not reasons,
        "primary_reason":reasons[0] if reasons else None, "reasons":reasons,
        "A_status":result.get("status") if isinstance(result, dict) else None,
        "A_stop_reason":result.get("stop_reason") if isinstance(result, dict) else None,
        "business_reward_is_not_the_gate":True}


def retail_task(task_id="34", split="train"):
    from tau2.domains.retail.environment import get_tasks
    tasks = [t for t in get_tasks(split) if t.id == str(task_id)]
    if len(tasks) != 1:
        raise ValueError("Retail task missing or ambiguous: " + str(task_id))
    return tasks[0].model_copy(deep=True)


def build_group(group):
    """Use upstream low-level builders, registry factory and Orchestrator."""
    from tau2.agent.memory_agent import MemoryAgent
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.runner.build import build_environment, build_user

    task = retail_task(group.get("task_id", "34"), group.get("split", "train"))
    env = build_environment("retail")
    if task_identity(task,env) != group["inputs"]:
        raise ValueError("Effective task/policy/tools/data differ from frozen identity")
    basis = [x.value for x in task.evaluation_criteria.reward_basis]
    if (basis != group["evaluation"]["reward_basis"] or not set(basis) <= {"DB", "NL_ASSERTION"}
            or task.evaluation_criteria.nl_assertions):
        raise ValueError("Unexpected Judge requirement")
    agent = registry.get_agent_factory("memory_agent")(
        env.get_tools(),env.get_policy(),llm=group["models"]["agent"],
        llm_args=deepcopy(group["role_args"]["agent"]),
        memory_strategy=group["memory"]["name"],memory_config=group["memory"]["config"])
    user = build_user("user_simulator",env,task,llm=group["models"]["user"],
        llm_args=deepcopy(group["role_args"]["user"]),persona_config=None)
    if type(agent) is not MemoryAgent or hasattr(agent,"task"):
        raise ValueError("Ordinary MemoryAgent without hidden task required")
    return Orchestrator(domain="retail",agent=agent,user=user,environment=env,task=task,
        max_steps=group["limits"]["max_steps"],max_errors=group["limits"]["max_errors"],
        seed=group["seed_rule"]["orchestrator_seed"],simulation_id=group["run_id"],
        validate_communication=True,timeout=group["limits"]["task_timeout_seconds"])


def mock_frames():
    """Synthetic transport responses derived from historical visible content.

    Historical usage is NEVER relabelled as usage from a new v2/live run.
    Fixed fake usage only exercises plumbing; it cannot establish task quality.
    """
    frames = []
    for record in read_json(HISTORY / "mock_frames.json"):
        body = deepcopy(record["evidence"]["raw_response"])
        body["usage"] = {"prompt_tokens":20,"completion_tokens":32,"total_tokens":52}
        frames.append({"role":record["role"],"body":body,
            "provenance":"synthetic_fixed_response_with_mock_usage; historical content reused"})
    return frames


class ExecutionResourceStop(RuntimeError):
    pass


@contextlib.contextmanager
def capture_model_inputs(orchestrator, folder, audit, save_progress):
    """Observe the common generate boundary; never add diagnostics to messages."""
    from tau2.agent import llm_agent
    from tau2.agent.budget_window import ContextBudgetError
    from tau2.agent.input_tokens import LocalJsonTokenCounter
    from tau2.user import user_simulator
    from tau2.utils.llm_utils import to_litellm_messages

    original_agent,original_user = llm_agent.generate,user_simulator.generate
    counter = LocalJsonTokenCounter()

    def observed(role, original, request):
        view = {"model":request["model"],"messages":to_litellm_messages(request["messages"]),
            "tools":[t.openai_schema for t in request.get("tools") or []] or None,
            "tool_choice":request.get("tool_choice")}
        if view["tools"] and view["tool_choice"] is None:
            view["tool_choice"] = "auto"
        entry = {"ordinal":len(audit)+1,"role":role,"source":"captured",
            "boundary":"upstream generate before LiteLLM HTTP normalization","model_visible_context":view,
            "provider_seed_parameter":"not_forwarded_by_existing_adapter","status":"before_dispatch"}
        audit.append(entry)
        if role == "agent":
            state = orchestrator.agent_state
            try:
                entry["local_estimated_input"] = counter.count_request(view)
                entry["full_history_estimated_input"] = counter.count_request(
                    orchestrator.agent._model_input_view(state,state.messages))
            except Exception as error:
                entry.update(status="context_budget_unknown",error_type=type(error).__name__)
                atomic_json(folder/"context_audit.json",audit)
                raise ContextBudgetError("context_budget_unknown",entry) from None
            entry["raw_history_message_count"] = len(state.messages)
            entry["selection"] = deepcopy(state.context_selections[-1]) if state.context_selections else {
                "strategy":"full_history","status":"untrimmed_reference","input_budget":None,
                "retained_range":[0,len(state.messages)]}
        atomic_json(folder/"context_audit.json",audit)
        try:
            result = original(**request)
            entry["status"] = "response_released"
            return result
        except BaseException as error:
            entry.update(status="stopped",error_type=type(error).__name__)
            raise
        finally:
            atomic_json(folder/"context_audit.json",audit)
            save_progress()

    llm_agent.generate = lambda **request:observed("agent",original_agent,request)
    user_simulator.generate = lambda **request:observed("user",original_user,request)
    try:
        yield
    finally:
        llm_agent.generate,user_simulator.generate = original_agent,original_user


def run_group(group, folder, ledger, safety, *, frames=None, live=False, secret=None):
    """Bind this sequential group's explicit caps to the unchanged v2 client.

    Both existing bindings and adapter.invoke validate adapter.LIMITS. Rebind
    only the process-local alias, never mutate the historical defaults, and
    restore it even when construction, persistence or generation fails.
    """
    import qianwen_adapter as adapter

    mode = execution_mode(live=live)
    if normalize_mode(ledger.data["mode"]) != mode:
        raise ValueError("Group mode differs from comparison ledger")
    if live and not safety.live_authorized:
        raise PermissionError("Live group requires the existing authorized entry")
    caps = group_output_limits(group)
    original, original_args = adapter.LIMITS, adapter.role_args
    role_args = deepcopy(group["role_args"])

    def bound_args(role):
        if role not in role_args:
            raise ValueError("Only configured Agent/User roles are enabled")
        return deepcopy(role_args[role])

    try:
        adapter.LIMITS = caps
        adapter.role_args = bound_args
        return _run_group(group, folder, ledger, safety,
                          frames=frames, live=live, secret=secret)
    finally:
        adapter.LIMITS = original
        adapter.role_args = original_args


def _run_group(group, folder, ledger, safety, *, frames=None, live=False, secret=None):
    """Execute the official runner once, preserving partials on every stop."""
    started = time.perf_counter()
    import httpx
    import qianwen_adapter as adapter
    import budget_transport
    from budget_bindings import bind_roles
    from loguru import logger
    from output_budget import BudgetSession, ResourceStop
    from tau2.agent.budget_window import ContextBudgetError
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import simulation as runner

    folder.mkdir(exist_ok=False)
    write_new(folder/"effective_group.json",group)
    orchestrator = build_group(group)
    records,audit,tools,warnings,supplier_errors = [],[],[],[],[]
    safety_before = {k:len(v) for k,v in safety.events["attempts"].items()}
    session = BudgetSession()
    queue = deepcopy(mock_frames() if frames is None else frames)
    stage = ["conversation"]

    # Upstream allows an outstanding environment step to finish beyond max_steps.
    # Enforce this comparison's boundary BEFORE every step/tool, without replacing
    # the upstream loop or changing its task termination/scoring implementation.
    original_step = orchestrator.step
    def bounded_step():
        if orchestrator.step_count >= group["limits"]["max_steps"]:
            raise ExecutionResourceStop("max_steps")
        if time.perf_counter()-started >= group["limits"]["task_timeout_seconds"]:
            raise ExecutionResourceStop("task_timeout")
        session.before_request()
        return original_step()
    orchestrator.step = bounded_step

    def save(name,value):
        atomic_json(folder/name,adapter.redact(value,secret))

    def progress():
        save("trajectory.json",[m.model_dump(mode="json") for m in orchestrator.trajectory])
        state = orchestrator.agent_state
        if state is not None:
            save("agent_state.json",state.model_dump(mode="json"))

    original_tool = orchestrator.environment.get_response
    def tool_response(call):
        response = original_tool(call)
        tools.append({"call":call.model_dump(mode="json"),"result":response.model_dump(mode="json")})
        save("tool_events.json",tools)
        progress()
        return response
    orchestrator.environment.get_response = tool_response

    def persist_transport(value):
        value = deepcopy(value)
        for event in value["events"]:
            raw_mode = event["mode"]
            event["mode"] = normalize_mode(raw_mode)
            if raw_mode != event["mode"]:
                event["transport_mode"] = raw_mode
        save("request_ledger.json",value)
        ledger.sync(group["group"],value)  # Both ledgers durable BEFORE any I/O.
        progress()

    def handler(request):
        if not queue:
            raise AssertionError("Unexpected extra Mock model request")
        frame = queue.pop(0)
        if frame["role"] != transport.role:
            raise AssertionError("Mock role order changed")
        if frame.get("transport_error") == "timeout":
            raise httpx.ReadTimeout("synthetic timeout",request=request)
        return httpx.Response(frame.get("status",200),json=deepcopy(frame["body"]),
            headers={"x-request-id":"synthetic-comparison-request"})

    def response_error(response):
        # Curated error codes only; never serialize headers or arbitrary messages.
        if response.status_code < 400:
            return
        response.read()
        try:
            error = response.json().get("error",{})
        except (ValueError,AttributeError):
            error = {}
        code = error.get("code") if isinstance(error,dict) else None
        kind = error.get("type") if isinstance(error,dict) else None
        supplier_errors.append({"http_status":response.status_code,"code":code,"type":kind,
            "context_resource":code in ("context_length_exceeded","context_window_exceeded","max_context_length_exceeded")})
        save("supplier_error_codes.json",supplier_errors)

    inner = httpx.HTTPTransport(proxy=PROXY,retries=0,trust_env=False) if live else httpx.MockTransport(handler)
    received = folder / "received_responses"
    received.mkdir()
    original_handle = inner.handle_request

    def capture_received_response(request):
        # This runs INSIDE the existing counted transport, before its finally
        # journal update can fail. It never sends or retries another request.
        response = original_handle(request)
        response.read()
        try:
            body = response.json()
        except ValueError:
            body = {}
        allowed = ("id", "object", "created", "model", "choices", "usage")
        raw = {k: body[k] for k in allowed if isinstance(body, dict) and k in body}
        evidence = {"ordinal": transport.events[-1]["ordinal"], "role": transport.role,
            "source_mode": "real" if live else "mock", "http_status": response.status_code,
            "raw_response": raw,
            "supplier_request_ids": {k: response.headers[k] for k in
                ("x-request-id", "x-dashscope-request-id", "request-id", "trace-id") if k in response.headers}}
        target = received / f"{evidence['ordinal']:04d}.json"
        if target.exists():
            raise FileExistsError("Received response evidence must not be overwritten")
        atomic_json(target, adapter.redact(evidence, secret))
        return response

    inner.handle_request = capture_received_response
    # This historical v2 transport used PROXY only on its unexecuted live branch.
    # Supply the already-pinned path locally; no historical file is changed.
    budget_transport.PROXY = PROXY
    transport = budget_transport.CountedTransport.create(inner,safety=safety,
        max_requests=group["limits"]["group_requests"],caps=group_output_limits(group),live=live,
        secret=secret or "",budget_session=session,persist=persist_transport)
    transport.deadline = started + group["limits"]["task_timeout_seconds"]

    def verify_output_request(request):
        # Check the actual SDK JSON before the counted transport can dispatch.
        body = json.loads(request.content)
        expected = group["role_args"][transport.role]
        thinking = expected.get("extra_body", {}).get("thinking_budget")
        if (body.get("max_completion_tokens") != expected["max_completion_tokens"]
                or "max_tokens" in body or body.get("thinking_budget") != thinking
                or "extra_body" in body):
            transport.stopped = True
            raise ValueError("Output parameters missing or changed at HTTP boundary")
    outcome = {"group":group["group"],"run_id":group["run_id"],"mode":"live" if live else "mock",
        "actually_executed":True,"status":"running","official_reward":None,"db_result":None,
        "scoring_validity":"not_scored","pause_comparison":False,"stop_reason":None,
        "usage_source":"supplier_http_body" if live else "mock",
        "cost":{"status":"unknown" if live else "not_applicable_mock","amount_cny":None,
            "basis":"supplier price/billing unavailable","hard_currency_cap":False}}
    save("summary.json",outcome)
    before_db = orchestrator.environment.get_db_hash()
    original_eval = runner.evaluate_simulation
    def evaluate(**kwargs):
        stage[0] = "evaluation"
        save("simulation_unscored.json",kwargs["simulation"].model_dump(mode="json"))
        reason = kwargs["simulation"].termination_reason.value
        if reason in ("max_steps","timeout","context_window_exceeded"):
            raise ExecutionResourceStop(reason)
        reward = original_eval(**kwargs)
        save("official_reward.json",reward.model_dump(mode="json"))
        return reward
    runner.evaluate_simulation = evaluate

    def warning(message):
        warnings.append({"stage":stage[0],"message":adapter.redact(message.record["message"],secret)})
    sink = logger.add(warning,level="WARNING")
    try:
        with httpx.Client(transport=transport,timeout=30,trust_env=False,follow_redirects=False,
                event_hooks={"request":[verify_output_request],"response":[response_error]}) as client:
            with bind_roles(client,lambda name:secret if live else "MOCK-COMPARISON-NOT-A-REAL-KEY",records,
                    budget_session=session,persist=lambda rows:save("model_responses.json",rows)):
                with capture_model_inputs(orchestrator,folder,audit,progress):
                    simulation = runner.run_simulation(orchestrator,evaluation_type=EvaluationType.ALL)
            save("simulation.json",simulation.model_dump(mode="json"))
            reward = simulation.reward_info
            evaluation_warnings = any(w["stage"] == "evaluation" for w in warnings)
            outcome.update(status="completed",termination_reason=simulation.termination_reason.value,
                official_reward=reward.reward,db_result=reward.db_check.model_dump(mode="json") if reward.db_check else None,
                scoring_validity="warning_requires_review" if evaluation_warnings else "official_score_recorded",
                business_outcome="success" if reward.reward == 1 else "business_failure")
            if evaluation_warnings:
                outcome.update(pause_comparison=True,stop_reason="evaluation_warning",business_outcome="unconfirmed")
    except BaseException as error:
        reason = getattr(error,"stop_reason",None) or ledger.last_refusal
        blocked = transport.blocked[-1]["reason"] if transport.blocked else None
        if isinstance(error,ExecutionResourceStop):
            reason = str(error)
        if blocked and ("request count" in blocked or "deadline" in blocked):
            reason = "request_limit_reached" if "request count" in blocked else "task_timeout"
        statuses = [e.get("http_status") for e in transport.events]
        shared = any(s is not None and s >= 400 for s in statuses) or any(e.get("status") == "transport_error" for e in transport.events)
        if any(s in (401,403) for s in statuses):
            reason = "authentication_or_permission_error"
        elif any(e.get("status") == "transport_error" for e in transport.events):
            reason = "shared_transport_error"
        if any(e["context_resource"] for e in supplier_errors):
            reason,shared = "context_window_exceeded",False
        resource = isinstance(error,(ContextBudgetError,ResourceStop,ExecutionResourceStop)) or reason in ("request_limit_reached","task_timeout","persistent_request_limit_reached")
        resource = resource or reason == "context_window_exceeded"
        outcome.update(status="evaluation_error" if stage[0] == "evaluation" and not resource else
                       "resource_stop" if resource and not shared else "execution_error",
            stop_reason=reason or "shared_execution_or_protocol_error",error_type=type(error).__name__,
            pause_comparison=True,
            business_outcome="not_evaluated",official_reward=None,db_result=None)
    finally:
        runner.evaluate_simulation = original_eval
        logger.remove(sink)
        progress()
        save("warnings.json",warnings)
        attempts = transport.events
        by_role = {}
        for role in adapter.MODELS:
            entries = [e for e in attempts if e["role"] == role]
            by_role[role] = {"requests":len(entries),"http_success":sum(e.get("http_status",999)<400 for e in entries),
                "failed":sum(e.get("http_status",0)>=400 or e.get("status")=="transport_error" for e in entries),
                "timeouts":sum("Timeout" in e.get("error_type","") for e in entries)}
        usages = [r.get("evidence",{}).get("usage") for r in records]
        known = [u for u in usages if isinstance(u,dict) and all(type(u.get(k)) is int for k in ("prompt_tokens","completion_tokens","total_tokens"))]
        selections = deepcopy(getattr(orchestrator.agent_state,"context_selections",[]))
        outcome.update(role_counts=by_role,requests=len(attempts),real_api_requests=len(attempts) if live else 0,
            mock_http_requests=0 if live else len(attempts),step_count=orchestrator.step_count,
            tool_calls=len(tools),tool_errors=sum(t["result"]["error"] for t in tools),
            total_seconds=time.perf_counter()-started,
            usage={"source":outcome["usage_source"],"status":"known" if len(known)==len(attempts) and attempts else "partial_or_unknown",
                "known_subtotal":{k:sum(u[k] for u in known) for k in ("prompt_tokens","completion_tokens","total_tokens")} if known else None,
                "unknown_attempts":len(attempts)-len(known),"raw_records":"model_responses.json"},
            output_truncated=sum(r.get("evidence",{}).get("finish_reason")=="length" for r in records),
            output_resource_decisions=deepcopy(session.decisions),
            context_selections=selections,
            trimming={"triggered":any(d.get("status") in ("trimmed", "structured") for d in selections),
                "trimmed_generations":sum(d.get("status") in ("trimmed", "structured") for d in selections)},
            usage_verification_unknown=sum(d["status"]=="budget_verification_unknown" for d in session.decisions),
            db_initial_hash=before_db,db_final_hash=orchestrator.environment.get_db_hash(),
            judge_calls=0,review_calls=0,retries={"sdk":0,"http":0,"task":0})
        if outcome["status"] != "completed" and not outcome.get("termination_reason"):
            outcome["termination_reason"] = getattr(orchestrator.termination_reason,"value",None)
        from tau2.agent.budget_window import interaction_blocks, ToolHistoryError
        try:
            interaction_blocks(orchestrator.trajectory)
            paired = True
        except ToolHistoryError:
            paired = False
        required = ["trajectory.json", "agent_state.json", "context_audit.json",
                    "model_responses.json", "request_ledger.json", "official_reward.json", "simulation.json"]
        if tools:
            required.append("tool_events.json")
        outcome["evidence_hashes"] = {name:sha(folder/name) for name in required if (folder/name).is_file()}
        outcome["execution_checks"] = {
            "model_calls_complete":len(records) == len(audit) == len(attempts) and bool(attempts)
                and all(r.get("converted",{}).get("status") == "received" and not r.get("error_type")
                    and r.get("classification",{}).get("adapter_compatibility_check",{}).get("status") == "passed"
                    for r in records) and all(a.get("status") == "response_released" for a in audit),
            "tool_calls_complete":paired and len(tools) == sum(len(m.tool_calls or [])
                for m in orchestrator.trajectory if hasattr(m,"tool_calls")),
            "trajectory_complete":orchestrator.done is True and outcome.get("termination_reason") in ("user_stop","agent_stop"),
            "safety_verified":all(len(v) == safety_before[k] for k,v in safety.events["attempts"].items())
                and not any(r.get("evidence",{}).get("credential_echo_detected") for r in records),
            "artifacts_complete":set(outcome["evidence_hashes"]) == set(required),
        }
        outcome["pause_reasons"] = completion_reasons(outcome,group)
        outcome["pause_comparison"] = bool(outcome["pause_reasons"])
        if outcome["pause_comparison"] and not outcome.get("stop_reason"):
            outcome["stop_reason"] = outcome["pause_reasons"][0]
        save("summary.json",outcome)
        ledger.finish_group(group["group"],outcome)
    return outcome


def run_pair(plan, folder, safety, *, frames_by_group=None, live=False, mode=None, secret=None, authorization=None):
    """Sequential groups; shared faults pause the second group without retries."""
    started = time.perf_counter()
    mode = execution_mode(live=live, mode=mode)
    live = mode == "live"
    validate_pair(plan)
    check_identity(plan)
    if live:
        require_authorization(plan,authorization,mode=mode)
    ledger = RequestLedger(folder,plan["identity_sha256"],mode=mode)
    ledger.start_once()
    pair_errors = []

    def execute(group):
        try:
            value = run_group(group,folder/group["group"],ledger,safety,
                frames=None if frames_by_group is None else frames_by_group[group["group"]],live=live,secret=secret)
        except BaseException as error:
            # Keep A's raw error/score untouched; record orchestration errors separately.
            pair_errors.append({"group":group["group"],"stage":"group_execution","error_type":type(error).__name__})
            try:
                value = read_json(folder/group["group"]/"summary.json")
            except (OSError,ValueError):
                value = ledger.data["groups"].get(group["group"])
        if isinstance(value,dict):
            return value
        return {"group":group["group"],"run_id":group["run_id"],"actually_executed":None,
            "status":"result_missing","stop_reason":"A_result_missing" if group["group"] == "A" else "B_result_missing",
            "official_reward":None,"requests":sum(e["group"] == group["group"] for e in ledger.data["attempts"]),
            "real_api_requests":0,"mock_http_requests":0}

    a = execute(plan["groups"][0])
    journal_verified = False
    try:
        durable = read_json(ledger.path)
        journal_verified = durable == ledger.data
        gate = evaluate_a_to_b(a,plan,durable,folder)
    except (OSError,ValueError,TypeError):
        gate = {"schema":"A-to-B-gate-v1","allowed":False,
            "primary_reason":"request_ledger_unverifiable","reasons":["request_ledger_unverifiable"]}
    if not journal_verified:
        gate["reasons"].append("request_ledger_unverifiable")
    if pair_errors:
        gate["reasons"].append("A_execution_raised")
    if gate["reasons"]:
        gate.update(allowed=False,primary_reason=gate["reasons"][0])
    try:
        if journal_verified:
            ledger.record_gate(gate)
        else:
            # Preserve an unreadable or externally changed ledger as evidence.
            atomic_json(folder/"a_to_b_gate.json",gate)
    except BaseException as error:
        pair_errors.append({"stage":"gate_persistence","error_type":type(error).__name__})
        gate = {**gate,"allowed":False,"primary_reason":gate.get("primary_reason") or "gate_persistence_error",
            "reasons":[*gate["reasons"],"gate_persistence_error"]}
        ledger.data["a_to_b_gate"] = deepcopy(gate)
        ledger.data["shared_stop"] = ledger.data["shared_stop"] or gate["primary_reason"]
    if gate["allowed"]:
        b = execute(plan["groups"][1])
    else:
        b = {"group":"B","run_id":plan["groups"][1]["run_id"],"actually_executed":False,
            "status":"not_executed_A_gate","stop_reason":gate["primary_reason"],
            "official_reward":None,"requests":0,"real_api_requests":0,"mock_http_requests":0}
        if journal_verified:
            ledger.finish_group("B",b)
    results = [a,b]
    result = {"comparison_id":plan["comparison_id"],"identity_sha256":plan["identity_sha256"],
        "mode":"live" if live else "mock","groups":results,
        "real_api_requests":len(ledger.data["attempts"]) if live else 0,
        "mock_http_requests":0 if live else len(ledger.data["attempts"]),
        "a_to_b_gate":gate,"pair_errors":pair_errors,
        "total_seconds":time.perf_counter()-started,
        "interpretation":plan["groups"][0]["comparison_interpretation"],
        "quality_comparison_valid":False if not live else "single_trial_observation_only"}
    atomic_json(folder/"comparison_summary.json",result)
    return result
