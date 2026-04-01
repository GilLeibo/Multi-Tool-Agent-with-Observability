import httpx

from app.config import settings
from app.tools.base import ToolDefinition, ToolResult

_TAVILY_URL = "https://api.tavily.com/search"


async def _handle(input_dict: dict) -> ToolResult:
    query = input_dict.get("query", "").strip()
    if not query:
        return ToolResult(error="query is required")

    max_results = min(int(input_dict.get("max_results", 3)), 5)
    api_key = settings.tavily_api_key
    if not api_key:
        return ToolResult(error="Tavily API key is not configured")

    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.post(
                _TAVILY_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": True,
                },
            )
        except httpx.RequestError as exc:
            return ToolResult(error=f"Network error during web search: {exc}")

    if resp.status_code == 401:
        return ToolResult(error="Invalid Tavily API key")
    if resp.status_code != 200:
        return ToolResult(error=f"Tavily API error: HTTP {resp.status_code}")

    data = resp.json()
    results = []
    for r in data.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", "")[:500],  # trim to 500 chars
            "score": round(r.get("score", 0), 3),
        })

    # Include Tavily's own answer summary if available
    answer = data.get("answer", "")
    return ToolResult(result={"answer_summary": answer, "results": results})


web_search_tool = ToolDefinition(
    name="web_search",
    description=(
        "Search the web for current information and return a list of relevant results "
        "with titles, URLs, and content snippets, plus an answer summary. "
        "Use for recent events, facts, or any information you don't know."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results to return (1–5, default 3)",
                "default": 3,
            },
        },
        "required": ["query"],
    },
    handler=_handle,
)
