from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.tools.base import ToolDefinition


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResponse:
    stop_reason: str              # "end_turn" | "tool_use"
    text: str | None              # final answer text
    tool_calls: list[ToolCall] = field(default_factory=list)
    thinking: str | None = None   # reasoning text before tool calls
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = field(default=None, repr=False)


class LLMClient(ABC):
    """Provider-agnostic LLM interface used by the agent loop."""

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> LLMResponse: ...

    @abstractmethod
    def format_tool_result(self, tool_call_id: str, result: str, tool_name: str = "") -> dict:
        """Return the provider-specific message dict for a tool result."""
        ...

    @abstractmethod
    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Return the provider-specific assistant message dict to append to history."""
        ...
