import json

from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.tools.base import ToolDefinition


class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4.1-mini") -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)

    def _tool_specs(self, tools: list[ToolDefinition]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in tools
        ]

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        full_messages = [{"role": "system", "content": system}] + messages
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=self._tool_specs(tools),
            tool_choice="auto",
        )

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        msg = choice.message
        text = msg.content  # may be None when tool_calls is set
        tool_calls: list[ToolCall] = []

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=args))

        return LLMResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, result: str, tool_name: str = "") -> dict:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        msg = response.raw.choices[0].message
        d: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        return d
