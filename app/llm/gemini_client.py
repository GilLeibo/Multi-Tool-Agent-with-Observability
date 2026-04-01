import json
from typing import Any

import google.genai as genai
from google.genai import types as genai_types

from app.config import settings
from app.llm.base import LLMClient, LLMResponse, ToolCall
from app.tools.base import ToolDefinition


def _build_gemini_tools(tools: list[ToolDefinition]) -> list[genai_types.Tool]:
    """Convert ToolDefinition list to Gemini Tool objects."""
    declarations = []
    for t in tools:
        # Convert JSON Schema properties to Gemini Schema format
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])

        gemini_props = {}
        for prop_name, prop_def in props.items():
            prop_type = prop_def.get("type", "string").upper()
            # Map JSON Schema types to Gemini types
            type_map = {
                "STRING": genai_types.Type.STRING,
                "NUMBER": genai_types.Type.NUMBER,
                "INTEGER": genai_types.Type.INTEGER,
                "BOOLEAN": genai_types.Type.BOOLEAN,
            }
            gemini_props[prop_name] = genai_types.Schema(
                type=type_map.get(prop_type, genai_types.Type.STRING),
                description=prop_def.get("description", ""),
            )

        func_decl = genai_types.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties=gemini_props,
                required=required,
            ),
        )
        declarations.append(func_decl)

    return [genai_types.Tool(function_declarations=declarations)]


def _messages_to_gemini(system: str, messages: list[dict]) -> tuple[str, list[genai_types.Content]]:
    """Convert OpenAI-style messages to Gemini Content list."""
    contents = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                contents.append(genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=content)],
                ))
            elif isinstance(content, list):
                # Tool results
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_result":
                        parts.append(genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=item.get("name", ""),
                                response={"result": item.get("content", "")},
                            )
                        ))
                    elif isinstance(item, dict) and item.get("role") == "tool":
                        parts.append(genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=item.get("name", "tool"),
                                response={"result": item.get("content", "")},
                            )
                        ))
                if parts:
                    contents.append(genai_types.Content(role="user", parts=parts))
        elif role == "assistant":
            if isinstance(content, str) and content:
                contents.append(genai_types.Content(
                    role="model",
                    parts=[genai_types.Part(text=content)],
                ))
            elif isinstance(content, list):
                # Assistant message with tool calls
                parts = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text" and block.text:
                            parts.append(genai_types.Part(text=block.text))
                        elif block.type == "tool_use":
                            parts.append(genai_types.Part(
                                function_call=genai_types.FunctionCall(
                                    name=block.name,
                                    args=block.input,
                                )
                            ))
                if parts:
                    contents.append(genai_types.Content(role="model", parts=parts))
        elif role == "tool":
            # OpenAI-style tool result
            parts = [genai_types.Part(
                function_response=genai_types.FunctionResponse(
                    name=msg.get("name", "tool"),
                    response={"result": content},
                )
            )]
            contents.append(genai_types.Content(role="user", parts=parts))

    return system, contents


class GeminiClient(LLMClient):
    def __init__(self, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        self._client = genai.Client(api_key=settings.gemini_api_key)

    async def complete(
        self,
        system: str,
        messages: list[dict],
        tools: list[ToolDefinition],
    ) -> LLMResponse:
        system_instruction, contents = _messages_to_gemini(system, messages)
        gemini_tools = _build_gemini_tools(tools)

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=gemini_tools,
                max_output_tokens=4096,
            ),
        )

        candidate = response.candidates[0]
        finish_reason = str(candidate.finish_reason)

        text = None
        tool_calls: list[ToolCall] = []
        thinking = None

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                if tool_calls:
                    thinking = (thinking or "") + part.text
                else:
                    text = (text or "") + part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                # Gemini uses a unique ID based on function name
                call_id = f"gemini-{fc.name}-{len(tool_calls)}"
                args = dict(fc.args) if fc.args else {}
                tool_calls.append(ToolCall(id=call_id, name=fc.name, input=args))

        if tool_calls and text:
            thinking = text
            text = None

        stop_reason = "tool_use" if tool_calls else "end_turn"

        usage = response.usage_metadata
        return LLMResponse(
            stop_reason=stop_reason,
            text=text,
            tool_calls=tool_calls,
            thinking=thinking,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
            raw=response,
        )

    def format_tool_result(self, tool_call_id: str, result: str, tool_name: str = "") -> dict:
        # Store as a special dict that _messages_to_gemini will convert
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result,
        }

    def format_assistant_message(self, response: LLMResponse) -> dict:
        # Store the raw content parts for Gemini
        return {
            "role": "assistant",
            "content": response.raw.candidates[0].content.parts,
        }
