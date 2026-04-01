import anthropic

from app.config import settings
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.tools.base import ToolDefinition


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        tool_specs = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            tools=tool_specs,
            messages=messages,
        )

        stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        text = None
        thinking = None
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                if tool_calls or thinking:
                    # text before tool calls = thinking
                    thinking = (thinking or "") + block.text
                else:
                    text = (text or "") + block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        # If there are tool calls, any earlier text blocks are thinking
        if tool_calls and text:
            thinking = text
            text = None

        return LLMResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            thinking=thinking,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, result: str, tool_name: str = "") -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        """Reconstruct the Anthropic assistant message from the raw response."""
        return {"role": "assistant", "content": response.raw.content}
