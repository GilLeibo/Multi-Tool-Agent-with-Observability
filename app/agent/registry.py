import logging

from app.tools.base import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get_all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def get_claude_tool_specs(self) -> list[dict]:
        """Return tool specs in Anthropic tool_use format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def get_openai_tool_specs(self) -> list[dict]:
        """Return tool specs in OpenAI function_calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    async def dispatch(self, name: str, input_dict: dict) -> ToolResult:
        """Call a tool by name, catching all exceptions."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {name}")
        try:
            result = await tool.handler(input_dict)
            return result
        except Exception as exc:
            logger.exception("Tool %s raised an error", name)
            return ToolResult(error=str(exc))


# Global registry instance — populated at app startup
registry = ToolRegistry()
