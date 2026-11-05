"""Optional LLMAgent extension with a replaceable task-local memory strategy."""

from copy import deepcopy
from typing import Optional

from pydantic import Field

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.budget_window import BudgetWindow
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.agent.memory import (
    FullHistory,
    MemoryState,
    MemoryStrategy,
    copy_visible_message,
)
from tau2.agent.structured_memory import StructuredMemory, StructuredMemoryState
from tau2.data_model.message import APICompatibleMessage, AssistantMessage, Message
from tau2.environment.tool import Tool


class MemoryAgentState(LLMAgentState):
    """Keep the full raw conversation separate from strategy-owned context data."""

    memory: StructuredMemoryState | MemoryState
    memory_cursor: int
    strategy_name: str
    strategy_version: str
    context_selections: list[dict] = Field(default_factory=list)


class MemoryAgent(LLMAgent[MemoryAgentState]):
    """Reuse ordinary generation, tool handling and output-budget call paths."""

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        memory_strategy: str | MemoryStrategy = "full_history",
        memory_config: Optional[dict] = None,
    ):
        if llm_args is not None and any(k in llm_args for k in ("memory_strategy", "memory_config")):
            raise ValueError("Memory settings are agent configuration, not an LLM argument")
        options = {} if memory_config is None else dict(memory_config)
        if isinstance(memory_strategy, str):
            if memory_strategy == "full_history":
                if options:
                    raise ValueError("FullHistory does not accept budget options")
                strategy = FullHistory()
            elif memory_strategy == "budget_window":
                strategy = BudgetWindow(**options)
            elif memory_strategy == "structured_memory":
                strategy = StructuredMemory(**options)
            else:
                raise ValueError(f"Unknown memory strategy: {memory_strategy}")
        elif isinstance(memory_strategy, MemoryStrategy):
            if options:
                raise ValueError("Configure a strategy object in its own constructor")
            strategy = memory_strategy
        else:
            raise TypeError("memory_strategy must be a known name or MemoryStrategy")
        if not strategy.name or not strategy.version:
            raise ValueError("A memory strategy needs a name and version")
        self.memory_strategy = strategy
        super().__init__(tools, domain_policy, llm, llm_args)

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> MemoryAgentState:
        """Initialize/reset one session without retaining caller-owned objects."""
        state = super().get_init_state(deepcopy(message_history))
        return MemoryAgentState(
            system_messages=state.system_messages,
            messages=state.messages,
            memory=self.memory_strategy.initialize(deepcopy(state.messages)),
            memory_cursor=len(state.messages),
            strategy_name=self.memory_strategy.name,
            strategy_version=self.memory_strategy.version,
        )

    def _observe_pending_messages(self, state: MemoryAgentState) -> None:
        if not isinstance(state, MemoryAgentState):
            raise TypeError("MemoryAgent requires its own session state")
        if (state.strategy_name, state.strategy_version) != (
            self.memory_strategy.name,
            self.memory_strategy.version,
        ):
            raise ValueError("Session state belongs to a different memory strategy")
        if not 0 <= state.memory_cursor <= len(state.messages):
            raise ValueError("Conversation was truncated; reset with get_init_state")
        for message in state.messages[state.memory_cursor :]:
            self.memory_strategy.observe(state.memory, copy_visible_message(message))
            state.memory_cursor += 1

    def _build_model_context(
        self, state: MemoryAgentState
    ) -> list[APICompatibleMessage]:
        # LLMAgent has just appended the incoming user/tool messages.
        self._observe_pending_messages(state)
        if isinstance(self.memory_strategy, BudgetWindow):
            context = self.memory_strategy.build_context(
                state.memory,
                request_builder=lambda history: self._model_input_view(state, history),
                record_selection=state.context_selections.append,
            )
        else:
            context = self.memory_strategy.build_context(state.memory)
        return deepcopy(state.system_messages) + [
            copy_visible_message(message) for message in context
        ]

    def _model_input_view(self, state: MemoryAgentState, history: list) -> dict:
        """Read-only copies of the same fixed inputs used by upstream generate."""
        from tau2.utils.llm_utils import to_litellm_messages

        tools = [deepcopy(tool.openai_schema) for tool in self.tools] if self.tools else None
        choice = self.llm_args.get("tool_choice")
        if tools and choice is None:
            choice = "auto"
        view = {
            "model": self.llm,
            "messages": to_litellm_messages(deepcopy(state.system_messages) + deepcopy(history)),
            "tools": tools,
            "tool_choice": deepcopy(choice),
        }
        if "response_format" in self.llm_args:
            view["response_format"] = deepcopy(self.llm_args["response_format"])
        return view

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: MemoryAgentState
    ) -> tuple[AssistantMessage, MemoryAgentState]:
        """Let the original lifecycle append the reply, then observe that event."""
        reply, state = super().generate_next_message(message, state)
        self._observe_pending_messages(state)
        return reply, state


def create_memory_agent(tools, domain_policy, **kwargs):
    """Registry factory; deliberately do not forward task/hidden runner data."""
    return MemoryAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
        memory_strategy=kwargs.get("memory_strategy", "full_history"),
        memory_config=kwargs.get("memory_config"),
    )
