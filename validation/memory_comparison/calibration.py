"""Calibrate on past Agent-visible prefixes, without generating any response."""

import json
import hashlib
import math
from copy import deepcopy

from .spec import HISTORY, digest, read_json, sha


def captured_history(messages):
    """Restore captured HTTP messages to typed history at the generate boundary.

    This is explicitly a reconstruction, not a claim that the historical HTTP
    body contained the extra/null fields produced by to_litellm_messages.
    """
    from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
    restored = []
    for raw in messages:
        if raw["role"] == "user":
            restored.append(UserMessage(role="user",content=raw["content"]))
        elif raw["role"] == "assistant":
            calls = [ToolCall(id=c["id"],name=c["function"]["name"],
                arguments=json.loads(c["function"]["arguments"])) for c in raw.get("tool_calls") or []]
            restored.append(AssistantMessage(role="assistant",content=raw.get("content"),tool_calls=calls or None))
        elif raw["role"] == "tool":
            restored.append(ToolMessage(role="tool",id=raw["tool_call_id"],content=raw["content"],requestor="assistant"))
        else:
            raise ValueError("Only Agent-visible conversation messages may be restored")
    return restored


def calibrate(task, environment, *, safety_margin=128):
    """Choose a fixed budget from the intersection of safe prefix constraints."""
    import qianwen_adapter as adapter
    from tau2.agent.budget_window import BudgetWindow, ContextBudgetError, interaction_blocks
    from tau2.agent.input_tokens import LocalJsonTokenCounter
    from tau2.agent.memory_agent import MemoryAgent
    from tau2.data_model.simulation import SimulationRun
    from tau2.utils.llm_utils import to_litellm_messages

    captured_inputs = read_json(HISTORY / "captured_agent_inputs.json")
    simulation = SimulationRun.model_validate(read_json(HISTORY / "trajectory.json"))
    fingerprint = read_json(HISTORY / "source.json")
    # The historical policy/tools/data must still match before reconstruction.
    for rel, expected in fingerprint["portable_data_files"].items():
        from .spec import ROOT
        # Git may materialize the unchanged public text data as LF or CRLF.
        # Preserve historical raw-byte hashes in source.json; compare a separately
        # labelled LF digest here, without changing data or token count inputs.
        normalized = (ROOT/rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(normalized).hexdigest() != expected:
            raise ValueError("Historical Retail data no longer matches")
    if simulation.task_id != task.id or task.id != "34" or simulation.policy != environment.get_policy():
        raise ValueError("Historical task/policy provenance mismatch")
    agent = MemoryAgent(tools=environment.get_tools(),domain_policy=simulation.policy,
        llm=adapter.MODELS["agent"],llm_args=adapter.role_args("agent"))
    counter = LocalJsonTokenCounter()
    events = [e for e in captured_inputs if e["role"] == "agent"]
    positions = [i for i,m in enumerate(simulation.messages) if m.role == "assistant" and i > 0]
    if len(events) != len(positions):
        raise ValueError("Cannot uniquely align historical Agent generation positions")
    rows, contexts = [], []
    for event, index in zip(events, positions):
        captured = event["parameters"]
        if not all(k in captured for k in ("model","messages","tools","tool_choice")):
            raise ValueError("Historical capture incomplete; reconstruction cannot be marked captured")
        restored = captured_history(captured["messages"][1:])
        prefix = simulation.messages[:index]  # Never includes the current response or future messages.
        if to_litellm_messages(restored) != to_litellm_messages(prefix):
            raise ValueError("Captured visible prefix differs from trajectory prefix")
        state = agent.get_init_state(restored)
        view = agent._model_input_view(state,state.messages)
        if (captured["messages"][0]["content"] != state.system_messages[0].content or
                captured["tools"] != view["tools"] or captured["tool_choice"] != view["tool_choice"]):
            raise ValueError("Captured fixed inputs differ from current count boundary")
        blocks,pairing = interaction_blocks(state.messages)
        start = blocks[max(0,len(blocks)-2)].start if blocks else 0
        minimum = state.messages[start:]
        captured_view = {k:deepcopy(captured[k]) for k in ("model","messages","tools","tool_choice")}
        rows.append({"agent_position":len(rows)+1,"historical_request_ordinal":event["ordinal"],
            "trajectory_response_index":index,"visible_prefix_range":[0,index],
            "input_source":"captured","count_view_source":"reconstructed",
            "capture_scope":"allowlisted input fields; not a complete raw HTTP body",
            "count_boundary":"Phase1-02 MemoryAgent._model_input_view before LiteLLM HTTP normalization",
            "captured_http_boundary_estimate":counter.count_request(captured_view),
            "full_history_estimate":counter.count_request(view),
            "fixed_input_estimate":counter.count_request(agent._model_input_view(state,[])),
            "minimum_estimate":counter.count_request(agent._model_input_view(state,minimum)),
            "removable_old_blocks":max(0,len(blocks)-2),"tool_pairing":pairing,
            "count_view_sha256":digest(view)})
        contexts.append(state)
    lower = max(r["minimum_estimate"] for r in rows) + safety_margin
    upper = max((r["full_history_estimate"]+safety_margin-1 for r in rows if r["removable_old_blocks"]),default=-1)
    rounded = math.ceil(lower/128)*128
    suitable = lower <= upper
    budget = rounded if rounded <= upper else lower
    for row,state in zip(rows,contexts):
        diagnostics = []
        strategy = BudgetWindow(max_input_tokens=budget,safety_margin_tokens=safety_margin)
        try:
            strategy.build_context(state.memory,
                request_builder=lambda h:agent._model_input_view(state,h),record_selection=diagnostics.append)
        except ContextBudgetError:
            pass
        row["budget_window"] = diagnostics[-1]
    if any(r["budget_window"]["pause_reason"] for r in rows):
        raise ValueError("Selected budget does not fit all known minimum sets")
    trimmed = [r["agent_position"] for r in rows if r["budget_window"]["removed_block_count"]]
    return {"schema":"train34-input-calibration-v1","real_api_requests":0,
        "source_run":fingerprint["source_run"],"source_hashes":{name:sha(HISTORY/name) for name in
            ("captured_agent_inputs.json","trajectory.json","source.json")},
        "counter":counter.describe(),"rows":rows,
        "selection":{"rule":"max(minimum)+margin; round up to 128 if still in trim interval, otherwise use exact lower bound",
            "safety_margin_tokens":safety_margin,"margin_basis":"explicit local conservative allowance; not provider-calibrated",
            "safe_lower_bound":lower,"trim_upper_bound":upper,
            "recommended_max_input_tokens":budget,"suitable_for_known_prefix_trimming":suitable and bool(trimmed),
            "trimmed_agent_positions":trimmed,"frozen_before_future_live_run":True},
        "limitations":["JSON cl100k estimate, not native Qwen tokens",
            "Known historical prefixes only; new conversations may differ or stop",
            "Historical max_tokens run is not total-output-v2 supplier validation",
            "No score or semantic-quality conclusion from fixed-response replay"]}


def markdown_table(report):
    """Compact calibration table with both evidence and count-boundary labels."""
    lines = ["| Agent位置 / 历史请求 | 来源 / 计数视图 | 固定 | FullHistory | 最低集合 | 可移除块 | 预算下删除块 | 裁剪后估算 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        diag = row["budget_window"]
        lines.append(f"| {row['agent_position']} / {row['historical_request_ordinal']} | captured / reconstructed | {row['fixed_input_estimate']} | {row['full_history_estimate']} | {row['minimum_estimate']} | {row['removable_old_blocks']} | {diag['removed_block_count']} | {diag['estimated_input_tokens_after']} |")
    return "\n".join(lines)+"\n"
