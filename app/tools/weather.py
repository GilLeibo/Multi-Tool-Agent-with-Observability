import httpx

from app.config import settings
from app.tools.base import ToolDefinition, ToolResult

_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


async def _handle(input_dict: dict) -> ToolResult:
    city = input_dict.get("city", "").strip()
    if not city:
        return ToolResult(error="city is required")

    api_key = settings.openweathermap_api_key
    if not api_key:
        return ToolResult(error="OpenWeatherMap API key is not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(
                _BASE_URL,
                params={"q": city, "appid": api_key, "units": "metric"},
            )
        except httpx.RequestError as exc:
            return ToolResult(error=f"Network error fetching weather: {exc}")

    if resp.status_code == 404:
        return ToolResult(error=f"City '{city}' not found")
    if resp.status_code == 401:
        return ToolResult(error="Invalid OpenWeatherMap API key")
    if resp.status_code != 200:
        return ToolResult(error=f"Weather API error: HTTP {resp.status_code}")

    data = resp.json()
    temp_c = data["main"]["temp"]
    temp_f = round(temp_c * 9 / 5 + 32, 1)
    result = {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country", ""),
        "temp_c": round(temp_c, 1),
        "temp_f": temp_f,
        "feels_like_c": round(data["main"].get("feels_like", temp_c), 1),
        "description": data["weather"][0]["description"].capitalize(),
        "humidity_pct": data["main"]["humidity"],
        "wind_speed_ms": data.get("wind", {}).get("speed", 0),
    }
    return ToolResult(result=result)


weather_tool = ToolDefinition(
    name="weather",
    description=(
        "Get current weather conditions for a city. Returns temperature in Celsius and "
        "Fahrenheit, feels-like temperature, weather description, humidity percentage, "
        "and wind speed. Use city name like 'London', 'New York', or 'Paris,FR'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, e.g. 'London', 'New York', or 'Paris,FR' for disambiguation",
            }
        },
        "required": ["city"],
    },
    handler=_handle,
)
