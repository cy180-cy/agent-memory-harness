"""User-turn blocks and conservative, task-local input-window selection."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import ClassVar

from tau2.agent.input_tokens import InputTokenCounter, LocalJsonTokenCounter
from tau2.agent.memory import FullHistory, MemoryMessage, MemoryState, copy_visible_message
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage


class ContextBudgetError(RuntimeError):
    """A pre-generation pause, never an official business reward."""

    def __init__(self, stop_reason: str, diagnostics: dict):
        self.stop_reason = stop_reason
        self.diagnostics = deepcopy(diagnostics)
        self.official_reward = None
        super().__init__(stop_reason)


@dataclass(frozen=True)
class InteractionBlock:
    """Original zero-based message interval [start, end)."""

    start: int
    end: int
    kind: str
    closed_by_next_user: bool


class ToolHistoryError(ValueError):
    def __init__(self, reason: str, *, incomplete: bool = False):
        self.reason = reason
        self.incomplete = incomplete
        super().__init__(reason)


def interaction_blocks(messages: list[MemoryMessage]) -> tuple[list[InteractionBlock], dict]:
    """Validate the entire history before removal; then partition at user turns."""
    pending, seen_calls, seen_results = set(), set(), set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.requestor != "assistant":
                raise ToolHistoryError("wrong_tool_result_requestor")
            if message.id in seen_results:
                raise ToolHistoryError("duplicate_tool_result")
            if message.id not in pending:
                raise ToolHistoryError("orphan_or_wrong_tool_result_id")
            pending.remove(message.id)
            seen_results.add(message.id)
            continue
        if pending:
            raise ToolHistoryError("missing_results_before_next_message")
        if isinstance(message, UserMessage):
            if message.is_tool_call() or not message.has_text_content():
                raise ToolHistoryError("invalid_user_message")
        elif isinstance(message, AssistantMessage):
            if message.is_tool_call():
                if message.has_text_content():
                    raise ToolHistoryError("mixed_text_and_tool_calls")
                for call in message.tool_calls:
                    if not call.id or call.id in seen_calls or call.requestor != "assistant":
                        raise ToolHistoryError("missing_duplicate_or_invalid_call_id")
                    seen_calls.add(call.id)
                    pending.add(call.id)
            elif not message.has_text_content():
                raise ToolHistoryError("empty_assistant_message")
        else:
            raise ToolHistoryError("non_visible_message")
    if pending:
        raise ToolHistoryError("tool_results_not_yet_complete", incomplete=True)
    starts = [i for i, m in enumerate(messages) if isinstance(m, UserMessage)]
    if messages and (not starts or starts[0] != 0):
        starts.insert(0, 0)
    blocks = [InteractionBlock(start, starts[i + 1] if i + 1 < len(starts) else len(messages),
        "user_turn" if isinstance(messages[start], UserMessage) else "preamble", i + 1 < len(starts))
        for i, start in enumerate(starts)]
    return blocks, {"status": "passed", "calls": len(seen_calls), "results": len(seen_results)}


class BudgetWindow(FullHistory):
    """Keep a contiguous suffix of whole turns, protecting the latest two blocks."""

    name: ClassVar[str] = "budget_window"
    version: ClassVar[str] = "1"

    def __init__(
        self,
        *,
        max_input_tokens: int,
        safety_margin_tokens: int = 0,
        counter: InputTokenCounter | None = None,
    ):
        if type(max_input_tokens) is not int or max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be an explicit positive integer")
        if type(safety_margin_tokens) is not int or safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens must be a non-negative integer")
        if counter is not None and not isinstance(counter, InputTokenCounter):
            raise TypeError("counter must implement count_request and describe")
        self.max_input_tokens = max_input_tokens
        self.safety_margin_tokens = safety_margin_tokens
        self.counter = counter if counter is not None else LocalJsonTokenCounter()

    def build_context(
        self,
        state: MemoryState,
        *,
        request_builder: Callable[[list[MemoryMessage]], dict] | None = None,
        record_selection: Callable[[dict], None] | None = None,
    ) -> list[MemoryMessage]:
        """Recount complete candidates; never rewrite fixed inputs or raw state."""
        diagnostic = {
            "strategy": self.name, "version": self.version,
            "max_input_tokens": self.max_input_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
            "estimated_input_tokens_before": None, "estimated_input_tokens_after": None,
            "counter": None, "target_model": None, "candidate_counts": [],
            "removed_blocks": [], "retained_blocks": [],
            "removed_message_count": 0, "retained_message_count": len(state.messages),
            "latest_interaction_retained": bool(state.messages), "official_reward": None,
        }

        def pause(reason, detail):
            diagnostic.update(status=reason, pause_reason=reason, detail=detail)
            if record_selection is not None:
                record_selection(deepcopy(diagnostic))
            raise ContextBudgetError(reason, diagnostic)

        try:
            blocks, pairing = interaction_blocks(state.messages)
        except ToolHistoryError as error:
            diagnostic["tool_pairing"] = {"status": "failed", "reason": error.reason}
            pause("context_incomplete_tools" if error.incomplete else "context_protocol_error", error.reason)
        diagnostic["tool_pairing"] = {"full_history": pairing}
        diagnostic["total_block_count"] = len(blocks)
        diagnostic["minimum_retained_block_count"] = min(2, len(blocks))
        if request_builder is None:
            pause("context_budget_unknown", "fixed_request_view_not_provided")
        try:
            diagnostic["counter"] = deepcopy(self.counter.describe())
        except Exception as error:
            pause("context_budget_unknown", "counter_metadata_failed:" + type(error).__name__)

        removable = max(0, len(blocks) - 2)
        for removed in range(removable + 1):
            start = blocks[removed].start if blocks else 0
            candidate = [copy_visible_message(m) for m in state.messages[start:]]
            diagnostic.update(
                removed_blocks=[asdict(b) for b in blocks[:removed]],
                retained_blocks=[asdict(b) for b in blocks[removed:]],
                removed_block_count=removed,
                retained_block_count=len(blocks) - removed,
                removed_message_count=start,
                retained_message_count=len(candidate),
            )
            try:
                request = request_builder(candidate)
                diagnostic["target_model"] = request["model"]
                count = self.counter.count_request(request)
                if type(count) is not int or count <= 0:
                    raise ValueError("counter_must_return_a_positive_integer")
            except Exception as error:
                pause("context_budget_unknown", "candidate_count_failed:" + type(error).__name__)
            diagnostic["candidate_counts"].append({"removed_block_count": removed, "estimated_input_tokens": count})
            if removed == 0:
                diagnostic["estimated_input_tokens_before"] = count
            diagnostic["estimated_input_tokens_after"] = count
            if count + self.safety_margin_tokens <= self.max_input_tokens:
                _, selected_pairing = interaction_blocks(candidate)
                diagnostic["tool_pairing"]["selected_context"] = selected_pairing
                diagnostic.update(status="trimmed" if removed else "within_budget", pause_reason=None)
                if record_selection is not None:
                    record_selection(deepcopy(diagnostic))
                return candidate
        pause("context_budget_exceeded", "fixed_inputs_and_minimum_interactions_do_not_fit")
