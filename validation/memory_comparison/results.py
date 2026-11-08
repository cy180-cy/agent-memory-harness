"""Recompute metrics from existing evidence; export and replay without models."""

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path

from memory_comparison.spec import digest, read_json, write_new

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
RESOURCE_WORDS = ("budget", "context_window", "request_limit", "max_steps", "task_timeout", "output_truncated")


def load(folder, name, default=None):
    path = Path(folder) / name
    return read_json(path) if path.is_file() else default


def valid_usage(value):
    if not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0 for k in TOKEN_FIELDS):
        return False
    return value["prompt_tokens"] + value["completion_tokens"] == value["total_tokens"]


def classification(summary, valid_score):
    status = summary.get("status", "unknown")
    reason = str(summary.get("stop_reason") or "")
    if status.startswith("not_executed") or summary.get("actually_executed") is False:
        return "not_executed"
    if status == "resource_stop" or summary.get("output_truncated", 0) > 0 or any(word in reason for word in RESOURCE_WORDS):
        return "resource_stop"
    if status == "evaluation_error" or (status == "completed" and not valid_score):
        return "evaluation_invalid"
    if valid_score:
        return "completed_success" if summary.get("official_reward", summary.get("reward_info", {}).get("reward")) == 1 else "business_failure"
    return "execution_error"


def group_metrics(folder, group=None, fallback=None):
    """Never turn a missing score/usage/cost into zero or an optimization win."""
    folder = Path(folder)
    config = group or load(folder, "effective_group.json", {})
    legacy = not bool(config.get("memory"))
    old_config = load(folder, "effective_config.json", {}) if legacy else {}
    summary = load(folder, "summary.json", fallback or {})
    journal = load(folder, "request_ledger.json", {})
    events = journal.get("events", [])
    responses = load(folder, "model_responses.json", [])
    score = load(folder, "official_reward.json", summary.get("reward_info", {})) or {}
    raw_reward = score.get("reward", summary.get("official_reward"))
    score_valid = (summary.get("status") == "completed" and type(raw_reward) in (int, float)
                   and math.isfinite(raw_reward) and 0 <= raw_reward <= 1)
    score_valid &= ((summary.get("scoring_validity") == "official_score_recorded") if not legacy
                    else summary.get("evaluation_basis_warning") is False)
    if not legacy and not (folder / "official_reward.json").is_file():
        score_valid = False
    if any(w.get("stage") == "evaluation" for w in load(folder, "warnings.json", summary.get("warnings", []))):
        score_valid = False
    db = score.get("db_check") or summary.get("db_result") or {}
    score_valid &= (type(db.get("db_reward")) in (int, float) and math.isfinite(db["db_reward"])
                    and 0 <= db["db_reward"] <= 1 and type(db.get("db_match")) is bool
                    and db["db_match"] == (db["db_reward"] == 1))
    if summary.get("pause_comparison") or summary.get("pause_reasons"):
        score_valid = False
    source = "mock" if (summary.get("mode") == "mock" or summary.get("usage_source") == "mock"
                         or journal.get("mock_http_requests", 0) > 0) else "historical_real" if legacy else "real"
    if not events and summary.get("actually_executed") is False:
        source = "not_executed"
    by_role = {}
    evidence_usage = []
    for i, event in enumerate(events):
        record = responses[i] if i < len(responses) else {}
        raw = record.get("evidence", {}).get("raw_response") or {}
        evidence_source = "model_responses.json"
        if not valid_usage(raw.get("usage")):
            early = load(folder, "received_responses/" + f"{event['ordinal']:04d}.json", {})
            raw = early.get("raw_response") or raw
            evidence_source = "received_responses"
        usage = raw.get("usage")
        evidence_usage.append({"ordinal": event["ordinal"], "role": event["role"],
                               "usage": usage, "valid": valid_usage(usage), "evidence_source": evidence_source})
    for role in ("agent", "user"):
        selected = [(e, u) for e, u in zip(events, evidence_usage) if e.get("role") == role]
        known = [u["usage"] for _, u in selected if u["valid"]]
        subtotal = {k: sum(u[k] for u in known) for k in TOKEN_FIELDS}
        unknown = len(selected) - len(known)
        by_role[role] = {"requests": len(selected), "successful_http": sum(
            type(e.get("http_status")) is int and 200 <= e["http_status"] < 300 for e, _ in selected),
            "failed": sum(e.get("status") == "transport_error" or (e.get("http_status") or 0) >= 400 for e, _ in selected),
            "timeouts": sum("Timeout" in e.get("error_type", "") for e, _ in selected),
            "unknown_usage_attempts": unknown, "known_subtotal": subtotal,
            "actual_tokens": None if unknown else subtotal,
            "reasoning_tokens_known_subtotal": sum(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                for u in known if type(u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)) is int)}
    unknown = sum(v["unknown_usage_attempts"] for v in by_role.values())
    subtotal = {k: sum(v["known_subtotal"][k] for v in by_role.values()) for k in TOKEN_FIELDS}
    trajectory = load(folder, "trajectory.json", [])
    tools = load(folder, "tool_events.json")
    if tools is None:
        calls = [c for m in trajectory for c in m.get("tool_calls") or []]
        results = {m.get("id"): m for m in trajectory if m.get("role") == "tool"}
        tools = [{"call": c, "result": results.get(c.get("id"), {})} for c in calls]
    state = load(folder, "agent_state.json", {})
    selections = state.get("context_selections", summary.get("context_selections", []))
    facts = state.get("memory", {}).get("facts", [])
    audit = load(folder, "context_audit.json", [])
    name = config.get("memory", {}).get("name", "legacy_llm_agent")
    task_id = str(config.get("task_id", (old_config.get("task_ids") or ["unknown"])[0]))
    simulation = load(folder, "simulation.json", load(folder, "simulation_unscored.json", {}))
    actual_task_id = simulation.get("task_id")
    task_scope = "synthetic_fixture" if actual_task_id is not None and str(actual_task_id) != task_id else "configured_retail"
    cost = summary.get("cost", {})
    amount = cost.get("amount_cny")
    # A null price remains unknown even if usage is complete.
    cost_known = source != "mock" and cost.get("status") == "known" and type(amount) in (int, float) and math.isfinite(amount) and amount >= 0
    row = {"task_id": task_id, "strategy": name,
        "actual_task_id": actual_task_id, "task_scope": task_scope,
        "comparison_id": config.get("comparison_id"), "group": config.get("group", "legacy"),
        "run_id": config.get("run_id", summary.get("run_id", folder.name)), "source_kind": source,
        "config_version": config.get("output_config_version", "legacy-max-tokens"),
        "status": summary.get("status", "unknown"), "stop_reason": summary.get("stop_reason"),
        "raw_official_reward": raw_reward, "official_reward": raw_reward if score_valid else None,
        "db_reward": db.get("db_reward") if score_valid else None, "scoring_valid": bool(score_valid),
        "roles": by_role, "requests": len(events),
        "reported_requests": summary.get("requests", summary.get("actual_api_requests")),
        "request_count_matches": summary.get("requests", summary.get("actual_api_requests", len(events))) == len(events),
        "usage_source": "mock" if source == "mock" else "supplier_raw_response",
        "usage_status": "known" if events and not unknown else "partial_or_unknown",
        "unknown_usage_attempts": unknown, "known_token_subtotal": subtotal,
        "actual_tokens": subtotal if events and not unknown else None,
        "tool_calls": len(tools), "tool_errors": sum(t.get("result", {}).get("error") is True for t in tools),
        "tool_results_missing": sum(not bool(t.get("result")) for t in tools),
        "raw_finish_reasons": [r.get("evidence", {}).get("finish_reason") for r in responses],
        "output_resource_decisions": summary.get("output_resource_decisions", []),
        "trimmed_generations": sum(s.get("status") in ("trimmed", "structured") for s in selections),
        "removed_blocks": sum(s.get("removed_block_count", len(s.get("removed_blocks", []))) for s in selections),
        "stored_facts": len(facts), "active_facts": sum(f.get("status") == "active" for f in facts),
        "injected_fact_counts": [len(s.get("memory_fact_ids", [])) for s in selections],
        "local_context_estimates": [{"generation": i + 1, **{k: s.get(k) for k in (
            "status", "estimated_input_tokens_before", "estimated_input_tokens_after", "removed_block_count",
            "retained_blocks", "memory_fact_ids", "tool_pairing", "counter")}} for i, s in enumerate(selections)],
        "captured_context_count": sum(a.get("role") == "agent" and a.get("source") == "captured" for a in audit),
        "agent_contexts": [{"ordinal": a.get("ordinal"), "provenance": a.get("source"),
            "boundary": a.get("boundary"), "full_history_local_estimate": a.get("full_history_estimated_input"),
            "selected_local_estimate": a.get("local_estimated_input"), "raw_history_messages": a.get("raw_history_message_count")}
            for a in audit if a.get("role") == "agent"],
        "total_seconds": summary.get("total_seconds"), "cost_status": "not_applicable_mock" if source == "mock" else "known" if cost_known else "unknown",
        "cost_cny": amount if cost_known else None, "cost_basis": cost.get("basis", "无可信的同平台价格或账单"),
        "evidence_folder": str(folder.resolve()), "metrics_origin": "recomputed_from_raw_evidence",
        "experiment_condition_sha256": digest({k: v for k, v in config.items() if k not in ("group", "run_id", "memory")}) if config else None,
        "notes": ["思考Token为输出细分，不再次相加", "本地输入估算与供应商usage分开；Mock不属于真实实验结果"]}
    normalized = {**summary, "official_reward": raw_reward}
    row["outcome_class"] = classification(normalized, score_valid)
    if row["outcome_class"] == "not_executed":
        row.update(official_reward=None, db_reward=None, scoring_valid=False, actual_tokens=None)
    return row


def collect(folder):
    """Read a suite, old A/B, a single group, or the original plain-Agent run."""
    folder = Path(folder).resolve()
    plan = load(folder, "comparison_plan.json")
    if plan:
        execution = folder / ("mock_execution" if (folder / "mock_execution").is_dir() else "live_execution")
        return collect(execution)
    parent_plan = load(folder, "effective_plan.json", load(folder.parent, "comparison_plan.json", {}))
    groups = {g["group"]: g for g in parent_plan.get("groups", [])}
    result = load(folder, "suite_summary.json", load(folder, "comparison_summary.json"))
    if result:
        return [group_metrics(folder / r["group"], groups.get(r["group"]), r) for r in result["groups"]]
    if (folder / "summary.json").is_file():
        return [group_metrics(folder)]
    # Tests can contain a single actual executor run in a named subfolder.
    found = sorted(folder.glob("*/effective_group.json"))
    if found:
        return [group_metrics(p.parent) for p in found]
    raise ValueError("No supported saved results at " + str(folder))


def aggregate(rows):
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["source_kind"], row["strategy"], row["config_version"], row["task_scope"])].append(row)
    result = []
    for (source, strategy, version, task_scope), members in buckets.items():
        counts = Counter(r["outcome_class"] for r in members)
        result.append({"source_kind": source, "strategy": strategy, "config_version": version,
                       "task_scope": task_scope,
                       "runs": len(members), "valid_scores": sum(r["scoring_valid"] for r in members),
                       "business_failures": counts["business_failure"], "execution_errors": counts["execution_error"],
                       "resource_stops": counts["resource_stop"], "invalid_evaluations": counts["evaluation_invalid"],
                       "not_executed": counts["not_executed"], "requests": sum(r["requests"] for r in members),
                       "known_token_subtotal": {k: sum(r["known_token_subtotal"][k] for r in members) for k in TOKEN_FIELDS},
                       "unknown_usage_attempts": sum(r["unknown_usage_attempts"] for r in members)})
    return result


def observed_differences(rows):
    """Only complete, comparable REAL runs get descriptive differences."""
    buckets = defaultdict(list)
    for row in rows:
        if row["source_kind"] == "real":
            buckets[(row["comparison_id"], row["task_id"], row["experiment_condition_sha256"])].append(row)
    differences = []
    for (comparison, task, _), members in buckets.items():
        if len(members) != 3 or {r["strategy"] for r in members} != {"full_history", "budget_window", "structured_memory"}:
            continue
        if not all(r["scoring_valid"] and r["status"] == "completed" and r["actual_tokens"] is not None
                   and r["request_count_matches"] and not r["tool_results_missing"] for r in members):
            continue
        base = next(r for r in members if r["strategy"] == "full_history")
        for other in (r for r in members if r is not base):
            fields = {"agent_input_tokens": (base["roles"]["agent"]["actual_tokens"]["prompt_tokens"], other["roles"]["agent"]["actual_tokens"]["prompt_tokens"]),
                      "input_tokens": (base["actual_tokens"]["prompt_tokens"], other["actual_tokens"]["prompt_tokens"]),
                      "output_tokens": (base["actual_tokens"]["completion_tokens"], other["actual_tokens"]["completion_tokens"]),
                      "requests": (base["requests"], other["requests"]),
                      "seconds": (base["total_seconds"], other["total_seconds"])}
            differences.append({"comparison_id": comparison, "task_id": task, "strategy": other["strategy"],
                "versus": "full_history", "same_observed_reward": base["official_reward"] == other["official_reward"],
                "observations": {k: {"difference": b-a, "percent_of_reference": (b-a)*100/a if a else None}
                    for k, (a,b) in fields.items() if type(a) in (int,float) and type(b) in (int,float)},
                "claim_limit": "单任务一次观测差异；相同Reward不证明同等过程质量，不支持稳定收益或统计显著性"})
    return differences


def export_results(folders, output):
    """New output directory only; historical source evidence is never rewritten."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [r for folder in folders for r in collect(folder)]
    report = {"schema": "memory-metrics-v1", "rows": rows, "aggregates": aggregate(rows),
              "real_completed_observed_differences": observed_differences(rows),
              "quality_claim": "Mock验证工程链路；未完成运行不支持同等质量Token节省结论；单次实验不支持统计显著性",
              "real_usage_is_not_local_estimate": True}
    write_new(output / "metrics.json", report)
    columns = ["task_id", "actual_task_id", "task_scope", "strategy", "source_kind", "config_version", "outcome_class", "official_reward", "db_reward",
               "scoring_valid", "requests", "agent_requests", "user_requests", "agent_input_tokens", "agent_output_tokens",
               "user_input_tokens", "user_output_tokens", "known_input_subtotal", "known_output_subtotal", "unknown_usage_attempts",
               "tool_calls", "tool_errors", "agent_failed", "user_failed", "agent_timeouts", "user_timeouts",
               "trimmed_generations", "removed_blocks", "stored_facts", "active_facts",
               "total_seconds", "cost_status", "cost_cny", "stop_reason", "run_id", "evidence_folder"]
    with (output / "metrics.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            values = {k: row.get(k) for k in columns}
            for role in ("agent", "user"):
                metrics = row["roles"][role]
                values[role + "_requests"] = metrics["requests"]
                values[role + "_failed"] = metrics["failed"]
                values[role + "_timeouts"] = metrics["timeouts"]
                tokens = metrics["actual_tokens"] or {}
                values[role + "_input_tokens"] = tokens.get("prompt_tokens")
                values[role + "_output_tokens"] = tokens.get("completion_tokens")
            values.update(known_input_subtotal=row["known_token_subtotal"]["prompt_tokens"],
                          known_output_subtotal=row["known_token_subtotal"]["completion_tokens"])
            writer.writerow(values)
    lines = ["# 三策略实验结果汇总", "", report["quality_claim"], "",
             "费用无可信依据时为unknown；未评分保留为空。Token为供应商原始usage或明确标注的Mock，思考Token不重复相加。", "",
             "|任务|策略|数据来源|分类|Reward/DB|请求|已知输入/输出Token|usage缺失|裁剪|事实|费用|",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        tokens = r["known_token_subtotal"]
        lines.append(f"|{r['task_id']}|{r['strategy']}|{r['source_kind']}|{r['outcome_class']}|"
                     f"{r['official_reward'] if r['scoring_valid'] else '未评分'}/{r['db_reward'] if r['scoring_valid'] else '—'}|"
                     f"{r['requests']}|{tokens['prompt_tokens']}/{tokens['completion_tokens']}|{r['unknown_usage_attempts']}|"
                     f"{r['trimmed_generations']}|{r['active_facts']}|{r['cost_status']}|")
    lines += ["", "汇总按策略、数据来源及输出配置版本分组，旧普通Agent结果单列，不合并成新策略成绩。", "",
              "|策略/来源/配置|有效评分|业务失败|执行异常|资源停止|评分异常|未执行|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for a in report["aggregates"]:
        lines.append(f"|{a['strategy']} / {a['source_kind']} / {a['config_version']}|{a['valid_scores']}|"
                     f"{a['business_failures']}|{a['execution_errors']}|{a['resource_stops']}|{a['invalid_evaluations']}|{a['not_executed']}|")
    lines += ["",
              "本次文件中的少量记录不自动生成性能提升结论；未完成/Mock记录不能用于同等质量成本比较。"]
    (output / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _render_public_demo(path, task_id=None):
    """Display published derived metrics without needing private request ledgers."""
    report = read_json(path)
    if report.get("schema") != "phase1-07-2-sanitized-evaluation-v1":
        raise ValueError("Unsupported public Demo summary schema")
    rows = report["rows"]
    if not rows or any(r.get("source_kind") not in ("real", "not_executed") for r in rows):
        raise ValueError("Public real-result Demo cannot relabel Mock or unknown sources")
    tasks = list(dict.fromkeys(r["task_id"] for r in rows))
    selected = str(task_id) if task_id is not None else tasks[0]
    if selected not in tasks:
        raise ValueError("任务不在结果中，可选: " + ", ".join(tasks))
    lines = ["Agent Memory Harness｜公开离线Demo",
             "来源：已保存真实实验的脱敏派生结果；本次仅展示，不重新计算官方评分。",
             "不读取API Key，不调用模型；没有Mock结果混入。",
             "可选任务: " + ", ".join(tasks), "当前任务: " + selected]
    labels = {"completed_success": "业务成功", "business_failure": "有效业务失败",
              "resource_stop": "资源停止，未评分", "not_executed": "未执行"}
    for row in (r for r in rows if r["task_id"] == selected):
        score = row["official_reward"] if row["scoring_valid"] else "未评分"
        db = row["db_reward"] if row["scoring_valid"] else "未评分"
        lines += ["", "策略: " + row["strategy"],
                  f"状态: {labels.get(row['outcome_class'], row['outcome_class'])}；Reward={score}；DB={db}"]
        if row["status"] == "not_executed":
            lines.append("前序门控阻止启动；请求=0；Token/费用不适用，不能记为业务失败。")
            continue
        if row["strategy"] == "full_history":
            lines.append("上下文：保留完整Agent可见历史，没有窗口裁剪。")
        for event in row.get("context_selection_events", []):
            before = event["estimated_input_tokens_before"]
            after = event["estimated_input_tokens_after"]
            lines.append(f"  Agent上下文#{event['agent_generation']}: {before} → {after}（本地估算）；"
                         f"{event['status']}；删除块={event['removed_block_count']}；注入事实={event['injected_fact_count']}")
            if event["removed_block_count"] or event["pause_reason"]:
                spans = [(b["start"], b["end"]) for b in event.get("retained_blocks", [])]
                lines.append(f"  保留原始消息区间[start,end): {spans}；工具配对={event['tool_pairing']}")
        facts = row.get("source_backed_facts")
        if facts:
            lines.append(f"结构化状态：事实={facts['fact_count']}；均有来源={facts['all_facts_have_sources']}；"
                         f"版本状态={facts['fact_status_counts']}（不等于全部注入提示词）。")
        lines.append("工具轨迹（仅名称）: " + " → ".join(row["tool_sequence"]))
        usage = row.get("actual_tokens")
        if usage:
            lines.append(f"供应商usage：输入={usage['prompt_tokens']}；输出={usage['completion_tokens']}；"
                         f"总计={usage['total_tokens']}；请求={row['requests']}；工具错误={row['tool_errors']}")
        lines.append(f"耗时={row['total_seconds']:.2f}秒；费用估算=¥{row['estimated_cost_cny']}；供应商账单=unknown")
        if row.get("stop_reason"):
            lines.append("停止原因: " + row["stop_reason"])
    lines += ["", "读图提示：train69两次裁剪均在取消成功之后；train73未证明节省；train93有资源停止和未执行。",
              "原始轨迹不随此公开摘要分发；完整分析见docs/final_experiment_report_zh.md。",
              "每组一次的小样本；未完成任务的低Token不是优化收益。本次真实API请求=0。"]
    return "\n".join(lines)


def render_demo(folder, task_id=None):
    """Read-only Chinese CLI replay. Never display the hidden user prompt."""
    if Path(folder).is_file():
        return _render_public_demo(folder, task_id)
    rows = collect(folder)
    tasks = list(dict.fromkeys(r["task_id"] for r in rows))
    selected = str(task_id) if task_id is not None else tasks[0]
    if selected not in tasks:
        raise ValueError("任务不在结果中，可选: " + ", ".join(tasks))
    lines = ["Agent Memory Harness｜离线结果回放", "可选任务: " + ", ".join(tasks), "当前任务: " + selected,
             "展示已保存证据；不调用模型。Mock结果仅用于演示链路。"]
    for row in (r for r in rows if r["task_id"] == selected):
        p = Path(row["evidence_folder"])
        lines += ["", f"策略: {row['strategy']}｜来源: {row['source_kind']}｜配置: {row['config_version']}",
                  f"配置任务: {row['task_id']}；实际任务: {row['actual_task_id']}；范围: {row['task_scope']}",
                  f"状态: {row['outcome_class']}；Reward={row['official_reward']}；DB={row['db_reward']}"]
        for item in row["local_context_estimates"]:
            lines.append(f"  上下文#{item['generation']}: {item.get('estimated_input_tokens_before')} → "
                         f"{item.get('estimated_input_tokens_after')}（本地估算）；{item.get('status')}；"
                         f"删除块={item.get('removed_block_count')}；保留={item.get('retained_blocks')}；注入事实ID={item.get('memory_fact_ids')}")
        if not row["local_context_estimates"]:
            for a in row["agent_contexts"]:
                lines.append(f"  上下文#{a['ordinal']}: {a['full_history_local_estimate']} → {a['selected_local_estimate']}"
                             f"（本地估算；{a['provenance']}；未裁剪历史）")
        agent_inputs = [a for a in load(p, "context_audit.json", []) if a.get("role") == "agent"]
        if agent_inputs:
            for msg in agent_inputs[-1].get("model_visible_context", {}).get("messages", [])[-2:]:
                lines.append(f"  最后模型可见片段[{msg.get('role')}]: {str(msg.get('content'))[:180]}")
        facts = load(p, "agent_state.json", {}).get("memory", {}).get("facts", [])
        for fact in [f for f in facts if f.get("status") in ("active", "conflict")][:8]:
            view = {k: fact.get(k) for k in ("id", "category", "scope", "path", "confidence", "status", "sources")}
            view["value_excerpt"] = json.dumps(fact.get("value"), ensure_ascii=False)[:200]
            lines.append("  结构化事实: " + json.dumps(view, ensure_ascii=False))
        # Only ordinary Agent-visible tool events/trajectory; no User system prompt.
        events = load(p, "tool_events.json")
        if events is None:
            events = [{"call": c, "result": {}} for m in load(p, "trajectory.json", []) for c in m.get("tool_calls") or []]
        for event in events[:15]:
            call = event["call"]
            lines.append(f"  工具: {call.get('name')} id={call.get('id')} 参数={json.dumps(call.get('arguments'), ensure_ascii=False)[:180]} "
                         f"error={event.get('result', {}).get('error', '未单独记录')}")
        tokens = row["known_token_subtotal"]
        lines += [f"  指标: 请求={row['requests']}；已知输入/输出={tokens['prompt_tokens']}/{tokens['completion_tokens']}；"
                  f"usage未知={row['unknown_usage_attempts']}；工具={row['tool_calls']}；裁剪={row['trimmed_generations']}；"
                  f"保留事实={row['active_facts']}；费用={row['cost_status']}"]
    lines.append("\n未完成任务的低Token不是优化收益；历史普通Agent不是StructuredMemory实验。")
    return "\n".join(lines)
