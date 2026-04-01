import json

import httpx
from openai import AsyncOpenAI

from app.config import settings
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.tools.base import ToolDefinition


class OllamaClient(LLMClient):
    """Ollama local model client using Ollama's OpenAI-compatible /v1 endpoint."""

    def __init__(self, model: str = "llama3.1") -> None:
        self.model = model
        self._base_url = settings.ollama_base_url.rstrip("/") + "/v1"
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key="ollama",  # Ollama ignores the key but the field is required
        )

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
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=full_messages,
                tools=self._tool_specs(tools),
                tool_choice="auto",
            )
        except Exception as exc:
            err_str = str(exc)
            if "Connection refused" in err_str or "connect" in err_str.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama at {settings.ollama_base_url}. "
                    "Make sure the Ollama service is running."
                ) from exc
            raise

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        stop_reason = "tool_use" if finish_reason == "tool_calls" else "end_turn"

        msg = choice.message
        text = msg.content
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
