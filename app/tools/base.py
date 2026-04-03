from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class ToolResult:
    result: Any = None
    error: str | None = None


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: dict  # JSON Schema passed to LLM tools= parameter
    handler: Callable[..., Awaitable[ToolResult]] = field(repr=False)
