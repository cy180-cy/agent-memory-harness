"""Task-local memory contracts, independent of policies, tools and evaluation."""

from collections.abc import Sequence
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from tau2.agent.base_agent import is_valid_agent_history_message
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

MemoryMessage = UserMessage | AssistantMessage | ToolMessage


def copy_visible_message(message: MemoryMessage) -> MemoryMessage:
    """Copy a message allowed in the ordinary agent's conversation history."""
    if not is_valid_agent_history_message(message):
        raise ValueError("Memory accepts agent-visible conversation messages only")
    return message.model_copy(deep=True)


class MemoryState(BaseModel):
    """Memory data owned by one task/trial state, never by a global strategy."""

    messages: list[MemoryMessage] = Field(default_factory=list)


@runtime_checkable
class MemoryStrategy(Protocol):
    """Initialize/reset memory, observe visible events, then build context."""

    name: str
    version: str

    def initialize(self, history: Sequence[MemoryMessage] = ()) -> MemoryState:
        """Return a fresh independent state, including any initial history."""
        ...

    def observe(self, state: MemoryState, message: MemoryMessage) -> None:
        """Receive a newly visible message exactly once."""
        ...

    def build_context(self, state: MemoryState) -> list[MemoryMessage]:
        """Return conversation context; policies and tools stay with the agent."""
        ...


class FullHistory:
    """Baseline strategy: retain all visible messages without rewriting them."""

    name: ClassVar[str] = "full_history"
    version: ClassVar[str] = "1"

    def initialize(self, history: Sequence[MemoryMessage] = ()) -> MemoryState:
        """Create fresh memory; calling again resets it without shared lists."""
        return MemoryState(messages=[copy_visible_message(m) for m in history])

    def observe(self, state: MemoryState, message: MemoryMessage) -> None:
        """Store an independent copy of the next visible event."""
        state.messages.append(copy_visible_message(message))

    def build_context(self, state: MemoryState) -> list[MemoryMessage]:
        """Return independent copies in the original order."""
        return [copy_visible_message(m) for m in state.messages]
