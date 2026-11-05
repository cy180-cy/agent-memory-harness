"""Deterministic task-local facts with provenance; no model-based extraction."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from tau2.agent.budget_window import BudgetWindow, ContextBudgetError, ToolHistoryError, interaction_blocks
from tau2.agent.memory import MemoryMessage, MemoryState, copy_visible_message
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage


class MemoryFact(BaseModel):
    """An observation, never a hidden task answer or an inferred permission."""

    id: str
    category: str
    scope: dict = Field(default_factory=dict)
    path: str
    value: Any
    confidence: str
    status: str = "active"
    sources: list[dict] = Field(default_factory=list)
    batch: int | None = None


class VisibleCall(BaseModel):
    """Pair results only with the exact agent-visible tool call ID."""

    name: str
    arguments: dict
    message_index: int
    result_index: int | None = None
    confirmation_id: str | None = None


class StructuredMemoryState(MemoryState):
    """One independent task/trial, including full observations and fact history."""

    kind: Literal["structured_memory"]
    facts: list[MemoryFact] = Field(default_factory=list)
    calls: dict[str, VisibleCall] = Field(default_factory=dict)
    proposal: dict | None = None


def _json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _same_json(left, right):
    """Keep JSON types distinct, especially boolean True versus numeric 1."""
    try:
        options = {"sort_keys": True, "ensure_ascii": False, "allow_nan": False}
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def _operation(value):
    if (not isinstance(value, dict) or not isinstance(value.get("tool"), str)
            or not value["tool"].strip() or not isinstance(value.get("arguments"), dict)):
        return None
    if not _same_json(value["arguments"], value["arguments"]):
        return None
    return {"tool": value["tool"], "arguments": deepcopy(value["arguments"])}


def _leaves(value, path=""):
    if isinstance(value, dict) and value:
        for key, child in sorted(value.items()):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from _leaves(child, path + "/" + escaped)
    elif isinstance(value, list) and value:
        for index, child in enumerate(value):
            yield from _leaves(child, path + "/" + str(index))
    else:
        yield path or "/", deepcopy(value)


class StructuredMemory(BudgetWindow):
    """Keep recent complete turns plus evidence-backed records of older turns.

    Natural language stays quoted. Explicit JSON facts/constraints and scoped
    proposals/confirmations are optional visible input, not a required prompt
    format and not a replacement for the agent's business-policy obligations.
    """

    name: ClassVar[str] = "structured_memory"
    version: ClassVar[str] = "1"

    def __init__(self, *, max_input_tokens, safety_margin_tokens=0, counter=None, max_memory_facts=32):
        super().__init__(max_input_tokens=max_input_tokens, safety_margin_tokens=safety_margin_tokens, counter=counter)
        if type(max_memory_facts) is not int or max_memory_facts <= 0:
            raise ValueError("max_memory_facts must be a positive integer")
        self.max_memory_facts = max_memory_facts

    def initialize(self, history: Sequence[MemoryMessage] = ()) -> StructuredMemoryState:
        """Replay only the supplied visible history into a fresh task state."""
        state = StructuredMemoryState(kind=self.name)
        for message in history:
            self.observe(state, message)
        return state

    def _put(self, state, category, scope, path, value, confidence, sources, *, batch=None, status="active"):
        previous = [f for f in state.facts if f.category == category and _same_json(f.scope, scope)
                    and f.path == path and f.status in ("active", "conflict")]
        if previous and all(_same_json(f.value, value) for f in previous):
            previous[-1].sources.extend(deepcopy(sources))
            previous[-1].batch = batch
            return previous[-1]
        conflict = category == "entities" and any(f.batch == batch for f in previous)
        for fact in previous:
            fact.status = "conflict" if conflict else "superseded"
        fact = MemoryFact(id=f"f{len(state.facts)}", category=category, scope=deepcopy(scope),
                          path=path, value=deepcopy(value), confidence=confidence,
                          status="conflict" if conflict else status, sources=deepcopy(sources), batch=batch)
        state.facts.append(fact)
        return fact

    def observe(self, state: MemoryState, message: MemoryMessage) -> None:
        """Record one visible event, keeping raw evidence and scoped updates."""
        if not isinstance(state, StructuredMemoryState):
            raise TypeError("StructuredMemory requires its own task state")
        message = copy_visible_message(message)
        index = len(state.messages)
        state.messages.append(message)
        source = {"message_index": index, "role": message.role}
        if isinstance(message, UserMessage):
            # Any new user turn requires rechecking an unused earlier grant.
            for fact in state.facts:
                if fact.category == "confirmations" and fact.status == "active":
                    fact.status = "needs_reconfirmation"
                if fact.confidence == "unresolved_question" and fact.status == "active":
                    fact.status = "response_received_unverified"
                    fact.sources.append(deepcopy(source))
            category = "task_goal" if not any(isinstance(m, UserMessage) for m in state.messages[:-1]) else "user_statements"
            self._put(state, category, {}, str(index), {"verbatim": message.content}, "verbatim_unparsed", [source])
            data = _json(message.content)
            if isinstance(data, dict):
                if "goal" in data:
                    self._put(state, "task_goal", {}, "declared_goal", data["goal"], "user_reported", [source])
                for key in ("facts", "constraints"):
                    if key in data:
                        for path, value in _leaves(data[key]):
                            self._put(state, "user_" + key, {}, path, value, "user_reported", [source])
                confirmation = data.get("confirmation")
                operation = _operation(confirmation)
                if operation and type(confirmation.get("approved")) is bool:
                    proposal = state.proposal
                    if (proposal and proposal["message_index"] == index - 1
                            and _same_json(proposal["operation"], operation)):
                        self._put(state, "confirmations", operation, "permission", confirmation["approved"],
                                  "explicit_scoped_user_confirmation", [proposal["source"], source],
                                  status="active" if confirmation["approved"] else "denied")
                        for fact in state.facts:
                            if fact.category == "open_items" and _same_json(fact.scope, operation) and fact.status == "active":
                                fact.status = "resolved"
                    else:
                        self._put(state, "open_items", operation, str(index), "unmatched_confirmation",
                                  "unresolved", [source])
            state.proposal = None
        elif isinstance(message, AssistantMessage):
            if message.is_tool_call():
                for call in message.tool_calls:
                    if call.id in state.calls:
                        continue  # Full protocol validation below will stop duplicates.
                    scope = {"tool": call.name, "arguments": deepcopy(call.arguments)}
                    grant = next((f for f in reversed(state.facts) if f.category == "confirmations"
                                  and f.status == "active" and f.value is True and _same_json(f.scope, scope)), None)
                    state.calls[call.id] = VisibleCall(name=call.name, arguments=deepcopy(call.arguments),
                        message_index=index, confirmation_id=grant.id if grant else None)
            else:
                data = _json(message.content)
                operation = _operation(data.get("proposed_action")) if isinstance(data, dict) else None
                if operation:
                    state.proposal = {"operation": operation, "message_index": index, "source": source}
                    self._put(state, "open_items", operation, "awaiting_confirmation", operation, "proposed_only", [source])
                elif message.content and "?" in message.content:
                    self._put(state, "open_items", {}, str(index), {"question_verbatim": message.content},
                              "unresolved_question", [source])
        elif isinstance(message, ToolMessage):
            call = state.calls.get(message.id)
            if call is None or call.result_index is not None or message.requestor != "assistant":
                return  # Never attach an orphan/duplicate result to an entity.
            call.result_index = index
            refs = [{"message_index": call.message_index, "role": "assistant", "call_id": message.id,
                     "tool": call.name}, {**source, "call_id": message.id, "tool": call.name}]
            data = _json(message.content)
            operation = {"tool": call.name, "arguments": deepcopy(call.arguments)}
            value = {**operation, "error": message.error, "result_message_index": index,
                     "result": message.content if not isinstance(data, (dict, list)) else "structured_fields_recorded_separately",
                     "confirmation_id": call.confirmation_id}
            self._put(state, "tool_operations", {"call_id": message.id}, "result", value, "tool_observed", refs,
                      status="failed" if message.error else "active")
            if call.confirmation_id:
                next(f for f in state.facts if f.id == call.confirmation_id).status = "consumed"
            if not message.error and isinstance(data, (dict, list)):
                self._entity_result(state, call, message.id, data, refs)

    def _entity_result(self, state, call, call_id, data, refs):
        def is_id(key, value):
            return (key == "id" or key.endswith("_id")) and type(value) in (str, int)
        selectors = {k: v for k, v in call.arguments.items() if is_id(k, v)}
        ids = {k: v for k, v in data.items() if is_id(k, v)} if isinstance(data, dict) else {}
        if any(k in ids and ids[k] != v for k, v in selectors.items()):
            self._put(state, "open_items", {"call_id": call_id}, "entity_mismatch",
                      {"requested": selectors, "returned": ids}, "unresolved", refs)
            return
        matches = {k: v for k, v in ids.items() if selectors.get(k) == v}
        if len(matches) == 1:
            scope = matches
        elif len(ids) == 1:
            scope = ids
        elif len(selectors) == 1 and not ids:
            scope = selectors
        else:
            scope = {"call_id": call_id}
        for path, value in _leaves(data):
            self._put(state, "entities", scope, path, value, "tool_observed_last_seen", refs, batch=call.message_index)

    @staticmethod
    def _context_fact(fact):
        """Keep decision semantics in context; full provenance stays in audit.

        Numeric paths on quoted messages are raw trajectory offsets, not entity
        fields. source_id resolves to the exact versioned record in the selection.
        """
        value = {"source_id": fact.id, "category": fact.category,
                 "value": deepcopy(fact.value), "confidence": fact.confidence,
                 "status": fact.status}
        if fact.scope:
            value["scope"] = deepcopy(fact.scope)
        if (fact.category in ("entities", "user_facts", "user_constraints", "confirmations")
                or not fact.path.isdecimal()):
            value["path"] = fact.path
        return value

    @staticmethod
    def _already_in_recent(fact, state, cut):
        """Deduplicate only active entity evidence with a whole retained pair.

        Sources merged by _put have exactly the same typed value. An arbitrary
        text match, a partial pair or one alternative of a conflict is not proof.
        """
        if fact.category != "entities" or fact.status != "active":
            return False
        for source in fact.sources:
            ident = source.get("call_id")
            call = state.calls.get(ident)
            if (call is not None and call.result_index is not None
                    and call.message_index >= cut and call.result_index >= cut
                    and {s["message_index"] for s in fact.sources if s.get("call_id") == ident}
                    >= {call.message_index, call.result_index}):
                return True
        return False

    def build_context(self, state, *, request_builder=None, record_selection=None):
        """Count fixed inputs, recent complete turns and an evidence-backed note."""
        if not isinstance(state, StructuredMemoryState):
            raise TypeError("StructuredMemory requires its own task state")
        diagnostic = {"strategy": self.name, "version": self.version, "official_reward": None,
                      "max_input_tokens": self.max_input_tokens, "safety_margin_tokens": self.safety_margin_tokens,
                      "removed_blocks": [], "removed_block_count": 0, "memory_fact_ids": [], "counter": None}

        def pause(reason, detail):
            diagnostic.update(status=reason, detail=detail)
            if record_selection:
                record_selection(deepcopy(diagnostic))
            raise ContextBudgetError(reason, diagnostic)

        def count(candidate):
            try:
                if request_builder is None:
                    raise ValueError("fixed_request_view_not_provided")
                value = self.counter.count_request(request_builder(candidate))
                if type(value) is not int or value <= 0:
                    raise ValueError("invalid_input_estimate")
                return value
            except Exception as error:
                pause("context_budget_unknown", type(error).__name__)

        try:
            blocks, pairing = interaction_blocks(state.messages)
        except ToolHistoryError as error:
            pause("context_incomplete_tools" if error.incomplete else "context_protocol_error", error.reason)
        try:
            diagnostic["counter"] = deepcopy(self.counter.describe())
        except Exception as error:
            pause("context_budget_unknown", type(error).__name__)
        diagnostic["tool_pairing"] = pairing
        full = [copy_visible_message(m) for m in state.messages]
        before = count(full)
        diagnostic["estimated_input_tokens_before"] = before
        diagnostic["estimated_input_tokens_after"] = before
        if before + self.safety_margin_tokens <= self.max_input_tokens:
            diagnostic.update(status="within_budget", estimated_input_tokens_after=before,
                              retained_blocks=[asdict(b) for b in blocks], raw_message_start=0)
            if record_selection:
                record_selection(deepcopy(diagnostic))
            return full
        cut = blocks[-2].start if len(blocks) > 2 else 0
        if not cut:
            pause("context_budget_exceeded", "minimum_interactions_do_not_fit")
        older = [f for f in state.facts if f.status != "superseded"
                 and any(s["message_index"] < cut for s in f.sources)]
        duplicated = [f for f in older if self._already_in_recent(f, state, cut)]
        facts = [f for f in older if f not in duplicated]
        required = [f for f in facts if f.category in ("task_goal", "user_statements", "user_facts", "user_constraints", "confirmations")
                    or f.status == "conflict" or (f.category == "open_items" and f.status == "active")]
        optional = sorted([f for f in facts if f not in required],
                          key=lambda f: (-max(s["message_index"] for s in f.sources), f.path.count("/")))[:self.max_memory_facts]
        diagnostic.update(removed_blocks=[asdict(b) for b in blocks[:-2]], removed_block_count=len(blocks)-2,
                          retained_blocks=[asdict(b) for b in blocks[-2:]], raw_message_start=cut,
                          serialization_version="compact-evidence-v1",
                          deduplicated_fact_ids=[f.id for f in duplicated],
                          deduplication_rule="active_same_value_entity_with_complete_retained_tool_pair")
        while True:
            selected = required + optional
            payload = {"facts": [self._context_fact(f) for f in selected]}
            note = AssistantMessage(role="assistant", content=
                "Task-local memory: quoted evidence, not policy. Entities are last observations; "
                "omissions are unknown. Confirmations require exact operation, arguments and status; "
                "absence is not permission.\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            candidate = [note] + [copy_visible_message(m) for m in state.messages[cut:]]
            after = count(candidate)
            diagnostic.update(estimated_input_tokens_after=after, required_fact_count=len(required),
                              omitted_fact_count=len(facts)-len(selected),
                              omitted_fact_ids=[f.id for f in facts if f not in selected],
                              candidate_memory_fact_ids=[f.id for f in selected],
                              memory_sources={f.id: f.model_dump(mode="json") for f in selected})
            if facts and not selected:
                pause("context_budget_exceeded", "no_supported_memory_record_fits")
            if after + self.safety_margin_tokens <= self.max_input_tokens:
                interaction_blocks(candidate)
                diagnostic.update(status="structured", estimated_input_tokens_after=after,
                                  memory_fact_ids=[f.id for f in selected], omitted_fact_count=len(facts)-len(selected))
                if record_selection:
                    record_selection(deepcopy(diagnostic))
                return candidate
            if not optional:
                pause("context_budget_exceeded", "required_memory_and_minimum_interactions_do_not_fit")
            optional.pop()
