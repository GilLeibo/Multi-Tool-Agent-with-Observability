import sympy

from app.tools.base import ToolDefinition, ToolResult


async def _handle(input_dict: dict) -> ToolResult:
    expression = input_dict.get("expression", "")
    if not expression:
        return ToolResult(error="expression is required")
    try:
        parsed = sympy.sympify(expression, evaluate=True)
        result = parsed.evalf()
        # Return a clean numeric string if possible
        if result.is_number:
            value = float(result)
            # Use int representation if the result is a whole number
            if value == int(value) and abs(value) < 1e15:
                return ToolResult(result=str(int(value)))
            return ToolResult(result=str(round(value, 10)).rstrip("0").rstrip("."))
        return ToolResult(result=str(result))
    except Exception as exc:
        return ToolResult(error=f"Cannot evaluate expression '{expression}': {exc}")


calculator_tool = ToolDefinition(
    name="calculator",
    description=(
        "Evaluate mathematical expressions safely. Supports arithmetic, algebra, "
        "trigonometry, logarithms, exponentiation, and symbolic math. "
        "Examples: '2**10 + sqrt(144)', 'sin(pi/2)', 'log(100, 10)', '(15/100)*847 + sqrt(144)'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate, e.g. '2**10 + sqrt(144)'",
            }
        },
        "required": ["expression"],
    },
    handler=_handle,
)
