"""A small fixed Retail matrix, using the existing single-group execution path."""

from copy import deepcopy
from pathlib import Path
import time

from memory_comparison import execution, spec

STRATEGIES = ("full_history", "budget_window", "structured_memory")
TASKS = {
    "34": "待处理订单地址修改；已有离线参考路径验证",
    "69": "待处理订单取消与礼品卡退款；已有离线参考路径验证",
    "73": "已送达订单退货申请；不使用有缺陷的商品变体修改工具",
    "93": "已送达订单单商品换货；覆盖变体选择及付款方式",
}


def reference_preflight(task):
    """Fail on the first reference error, outside every ordinary Agent context.

    This only checks reference tools on a fresh in-memory DB. It is not an
    autonomous run, nor evidence that a memory strategy can solve the task.
    """
    from tau2.runner.build import build_environment
    env = build_environment("retail")
    initial = task.initial_state
    env.set_state(initialization_data=initial.initialization_data if initial else None,
                  initialization_actions=initial.initialization_actions if initial else None,
                  message_history=(initial.message_history or []) if initial else [], strict=True)
    before = env.get_db_hash()
    calls = []
    for action in task.evaluation_criteria.actions or []:
        env.make_tool_call(tool_name=action.name, requestor=action.requestor, **action.arguments)
        calls.append(action.name)
    return {"kind": "offline_reference_tools_only", "completed": True,
            "actions": calls, "action_count": len(calls),
            "db_changed": before != env.get_db_hash(), "real_api_requests": 0}


def task_manifest(task_ids=tuple(TASKS)):
    """Select by operation coverage and defects, never by model success rate."""
    from tau2.domains.retail.environment import get_tasks
    ids = tuple(str(i) for i in task_ids)
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= TASKS.keys():
        raise ValueError("Choose a nonempty unique subset of the fixed Retail train task list")
    selected, excluded = [], []
    for task in get_tasks("train"):
        criteria = task.evaluation_criteria
        names = [a.name for a in criteria.actions or []]
        reasons = []
        if "modify_pending_order_items" in names:
            reasons.append("未修复的商品修改变体复用缺陷；保守排除使用此工具的参考任务")
        if criteria.nl_assertions:
            reasons.append("非空NL断言需要Judge，本方案禁止额外模型评审")
        if not names:
            reasons.append("合法空动作任务；本次小样本聚焦数据库写操作，DB评分区分度不足")
        if task.id in ids:
            if reasons:
                raise ValueError("Selected task is excluded: " + task.id + ": " + "; ".join(reasons))
            selected.append({"task_id": task.id, "selection_basis": TASKS[task.id],
                "reference_tool_names": names, "reward_basis": [b.value for b in criteria.reward_basis],
                "nl_assertion_count": len(criteria.nl_assertions or []),
                "reference_preflight": reference_preflight(task)})
        elif reasons:
            excluded.append({"task_id": task.id, "reasons": reasons})
    selected.sort(key=lambda row: ids.index(row["task_id"]))
    if len(selected) != len(ids):
        raise ValueError("Fixed tasks missing from the local train split")
    return {"selection_version": "retail-operations-four-v1", "domain": "retail", "split": "train",
            "task_ids": list(ids), "num_trials": 1, "selected": selected, "excluded": excluded,
            "selection_uses_model_scores": False,
            "empty_nl_rule": "官方NL评分器在空断言时直接返回，不生成Judge请求",
            "limitations": "探索性便利样本，每策略每任务一次；不支持稳定成功率或统计显著性结论"}


def make_suite(comparison_id, repo, manifest):
    """Freeze three strategies per task, without introducing a second client."""
    from tau2.runner.build import build_environment
    calibration = {"selection": {"recommended_max_input_tokens": 7168, "safety_margin_tokens": 128},
                   "provenance": "frozen Phase1-03 input budget, not recalibrated from future scores"}
    base = spec.make_pair(comparison_id, repo, calibration,
                          execution.retail_task(), build_environment("retail"))["groups"][0]
    groups = []
    total = len(manifest["task_ids"]) * len(STRATEGIES) * 20
    for task_id in manifest["task_ids"]:
        task = execution.retail_task(task_id)
        inputs = spec.task_identity(task, build_environment("retail"))
        for strategy in STRATEGIES:
            g = deepcopy(base)
            name = "t" + task_id + "_" + strategy
            g.update(group=name, run_id=comparison_id + "_" + name, task_id=task_id, inputs=deepcopy(inputs))
            g["evaluation"]["reward_basis"] = [b.value for b in task.evaluation_criteria.reward_basis]
            g["limits"]["comparison_requests"] = total
            g["memory"] = {"name": strategy, "version": "1", "config": {} if strategy == "full_history"
                           else {"max_input_tokens": 7168, "safety_margin_tokens": 128}}
            g["comparison_interpretation"] = "全历史参考组与受输入预算约束的两种策略的工程取舍；并非相同输入预算"
            groups.append(g)
    plan = {"schema": "memory-suite-v1", "comparison_id": comparison_id, "groups": groups,
            "task_manifest": manifest, "budget_target_cny": None, "price_status": "unknown",
            "currency_hard_cap": False, "authorization_status": "pending_not_approved",
            "live_executed": False, "trial_count": 1, "retry_policy": "no_retries_or_resume"}
    validate_suite(plan)
    plan["identity_sha256"] = spec.digest(plan)
    return plan


def validate_suite(plan):
    """Keep pair defaults intact; check all effective non-strategy conditions."""
    if plan.get("schema") != "memory-suite-v1" or plan.get("trial_count") != 1:
        raise ValueError("Unknown suite schema or repetition count")
    manifest = plan["task_manifest"]
    ids = manifest["task_ids"]
    if not ids or len(ids) != len(set(ids)) or not set(ids) <= TASKS.keys():
        raise ValueError("Invalid frozen task selection")
    selected = {x["task_id"]: x for x in manifest["selected"]}
    if set(selected) != set(ids):
        raise ValueError("Task selection evidence missing")
    groups = plan["groups"]
    total = len(ids) * 3 * 20
    if len(groups) != len(ids) * 3:
        raise ValueError("Exactly three strategies per task required")
    canonical = None
    for pos, task_id in enumerate(ids):
        triple = groups[pos * 3:pos * 3 + 3]
        expected_names = ["t" + task_id + "_" + s for s in STRATEGIES]
        if [g["group"] for g in triple] != expected_names:
            raise ValueError("Task/strategy order changed")
        evidence = selected[task_id]
        if (evidence["nl_assertion_count"] != 0 or not evidence["reference_preflight"]["completed"]
                or not set(evidence["reward_basis"]) <= {"DB", "NL_ASSERTION"}):
            raise ValueError("Unverified reference or Judge requirement")
        for g in triple:
            if (g["comparison_id"] != plan["comparison_id"] or g["task_id"] != task_id
                    or g["run_id"] != plan["comparison_id"] + "_" + g["group"]):
                raise ValueError("Group identity changed")
        # Reuse the existing complete pair validator; project just run identities.
        pair = {"comparison_id": plan["comparison_id"], "groups": deepcopy(triple[:2])}
        for label, g in zip(("A", "B"), pair["groups"]):
            g.update(group=label, run_id=plan["comparison_id"] + "_" + label)
        spec.validate_pair(pair, task_ids=ids, total_requests=total, reward_basis=evidence["reward_basis"])
        common = [{k: v for k, v in g.items() if k not in ("group", "run_id", "memory")} for g in triple]
        if not common[0] == common[1] == common[2]:
            raise ValueError("Non-strategy conditions differ within a task")
        for strategy, g in zip(STRATEGIES, triple):
            expected = {"name": strategy, "version": "1", "config": {} if strategy == "full_history"
                        else {"max_input_tokens": 7168, "safety_margin_tokens": 128}}
            if g["memory"] != expected or g["output_config_version"] != spec.OUTPUT_CONFIG_VERSION:
                raise ValueError("Frozen memory/output configuration changed")
        global_common = {k: v for k, v in common[0].items() if k not in ("task_id", "inputs", "evaluation")}
        evaluation_common = {k: v for k, v in common[0]["evaluation"].items() if k != "reward_basis"}
        shared_inputs = {k: v for k, v in common[0]["inputs"].items() if k != "task_sha256"}
        current = (global_common, evaluation_common, shared_inputs)
        if canonical is not None and current != canonical:
            raise ValueError("Non-task conditions differ across tasks")
        canonical = current
    return {"status": "passed", "groups": len(groups), "total_request_cap": total,
            "only_strategy_differences_within_each_task": True}


def synthetic_frames(task_id):
    """A non-solving negative control for new tasks; NEVER a Retail effect result."""
    if task_id == "34":
        return execution.mock_frames()
    from tau2.user.user_simulator_base import STOP
    def frame(role, text):
        return {"role": role, "provenance": "synthetic_mock_negative_control_no_task_answer",
                "body": {"id": "synthetic-suite", "object": "chat.completion", "created": 1,
                "model": "qwen3.7-plus" if role == "agent" else "qwen3.8-flash",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 32, "total_tokens": 52}}}
    return [frame("user", "This is a synthetic offline plumbing check."),
            frame("agent", "This offline check did not perform the business operation."), frame("user", STOP)]


def run_suite(plan, folder, safety, *, frames_by_group=None, live=False, secret=None, authorization=None):
    """Thin sequential orchestration; existing group, journal and strict gates."""
    started = time.perf_counter()
    folder = Path(folder)
    validate_suite(plan)
    spec.check_identity(plan)
    if live:
        spec.require_authorization(plan, authorization)
    total = plan["groups"][0]["limits"]["comparison_requests"]
    ledger = spec.RequestLedger(folder, plan["identity_sha256"], total=total, mode="live" if live else "mock")
    ledger.start_once()
    spec.write_new(folder / "effective_plan.json", plan)
    rows, gates, completed, errors = [], [], [], []
    stop = None
    for index, group in enumerate(plan["groups"]):
        name = group["group"]
        if stop:
            rows.append({"group": name, "run_id": group["run_id"], "actually_executed": False,
                         "status": "not_executed", "stop_reason": stop, "requests": 0,
                         "official_reward": None, "real_api_requests": 0, "mock_http_requests": 0})
            continue
        raised = False
        try:
            frames = None if live else (frames_by_group[name] if frames_by_group is not None else synthetic_frames(group["task_id"]))
            value = execution.run_group(group, folder / name, ledger, safety, frames=frames, live=live, secret=secret)
        except BaseException as error:
            raised = True
            errors.append({"group": name, "stage": "group_execution", "error_type": type(error).__name__})
            try:
                value = spec.read_json(folder / name / "summary.json")
            except (OSError, ValueError):
                value = {"group": name, "status": "result_missing", "official_reward": None,
                         "requests": sum(e["group"] == name for e in ledger.data["attempts"])}
        rows.append(value)
        # Use exactly the repaired A-to-B evidence checks, including readback,
        # strict (<20) request boundary, resource pauses and valid reward=0.
        if index + 1 < len(plan["groups"]):
            try:
                durable = spec.read_json(ledger.path)
                if durable != ledger.data or raised:
                    raise RuntimeError("Execution journal not verified")
                gate = execution.evaluate_a_to_b(value, plan, durable, folder,
                                                  group_id=name, completed_prefix=completed)
                gate.update(from_group=name, to_group=plan["groups"][index + 1]["group"])
                ledger.record_gate(gate, key="gate_" + str(index + 1))
            except (OSError, ValueError, RuntimeError, TypeError) as error:
                gate = {"allowed": False, "primary_reason": "execution_or_gate_persistence_error",
                        "from_group": name, "error_type": type(error).__name__}
            gates.append(gate)
            if not gate["allowed"]:
                stop = gate["primary_reason"]
        completed.append(name)
        partial = {"comparison_id": plan["comparison_id"], "mode": "live" if live else "mock",
                   "groups": rows, "gates": gates, "errors": errors, "stop_reason": stop}
        spec.atomic_json(folder / "suite_progress.json", partial)
    result = {"schema": "memory-suite-results-v1", "comparison_id": plan["comparison_id"],
              "identity_sha256": plan["identity_sha256"], "mode": "live" if live else "mock",
              "groups": rows, "gates": gates, "errors": errors, "stop_reason": stop,
              "real_api_requests": len(ledger.data["attempts"]) if live else 0,
              "mock_http_requests": 0 if live else len(ledger.data["attempts"]),
              "total_seconds": time.perf_counter() - started,
              "quality_comparison_valid": False if not live else "exploratory_single_trial_only"}
    spec.atomic_json(folder / "suite_summary.json", result)
    return result
