import re

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.session import engine
from app.tools.base import ToolDefinition, ToolResult

# Only SELECT statements are allowed
_SELECT_PATTERN = re.compile(r"^\s*SELECT\b", re.IGNORECASE)


async def _handle(input_dict: dict) -> ToolResult:
    sql = input_dict.get("sql", "").strip()
    if not sql:
        return ToolResult(error="sql is required")

    if not _SELECT_PATTERN.match(sql):
        return ToolResult(error="Only SELECT statements are allowed for security reasons")

    try:
        with Session(engine) as db:
            result = db.execute(text(sql))
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
            return ToolResult(result={
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
            })
    except Exception as exc:
        return ToolResult(error=f"SQL error: {exc}")


database_query_tool = ToolDefinition(
    name="database_query",
    description=(
        "Query the product catalog and orders database using SQL SELECT statements. "
        "Available tables:\n"
        "- products (id, name, category, price, stock_quantity, sku, created_at)\n"
        "  Categories: Electronics, Office, Furniture, Software\n"
        "- orders (id, product_id, quantity, total_price, customer_name, status, ordered_at)\n"
        "  Status values: completed, pending, cancelled\n"
        "Only SELECT queries are permitted. Use JOINs to combine products and orders."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "SQL SELECT query to execute, e.g. 'SELECT name, price FROM products WHERE category = \"Electronics\" ORDER BY price DESC'",
            }
        },
        "required": ["sql"],
    },
    handler=_handle,
)
