import pint

from app.tools.base import ToolDefinition, ToolResult

_ureg = pint.UnitRegistry()

# Map common user-friendly names to pint unit strings
_UNIT_ALIASES: dict[str, str] = {
    "celsius": "degC",
    "centigrade": "degC",
    "fahrenheit": "degF",
    "kelvin": "kelvin",
    "miles": "mile",
    "mile": "mile",
    "kilometers": "kilometer",
    "km": "kilometer",
    "meters": "meter",
    "feet": "foot",
    "foot": "foot",
    "inches": "inch",
    "inch": "inch",
    "yards": "yard",
    "pounds": "pound",
    "lbs": "pound",
    "lb": "pound",
    "kilograms": "kilogram",
    "kg": "kilogram",
    "grams": "gram",
    "g": "gram",
    "ounces": "ounce",
    "oz": "ounce",
    "liters": "liter",
    "litres": "liter",
    "gallons": "gallon",
    "fluid_ounces": "fluid_ounce",
    "mph": "mile / hour",
    "kph": "kilometer / hour",
    "kmh": "kilometer / hour",
    "m/s": "meter / second",
}


def _resolve_unit(unit_str: str) -> str:
    return _UNIT_ALIASES.get(unit_str.lower().strip(), unit_str.strip())


async def _handle(input_dict: dict) -> ToolResult:
    try:
        value = float(input_dict.get("value", 0))
    except (TypeError, ValueError):
        return ToolResult(error="value must be a number")

    from_unit_raw = str(input_dict.get("from_unit", "")).strip()
    to_unit_raw = str(input_dict.get("to_unit", "")).strip()

    if not from_unit_raw or not to_unit_raw:
        return ToolResult(error="from_unit and to_unit are required")

    from_unit = _resolve_unit(from_unit_raw)
    to_unit = _resolve_unit(to_unit_raw)

    try:
        # Handle offset (non-multiplicative) temperature units specially
        is_temp = from_unit in ("degC", "degF") or to_unit in ("degC", "degF")
        if is_temp:
            qty = _ureg.Quantity(value, from_unit)
            converted = qty.to(to_unit)
        else:
            qty = _ureg.Quantity(value, _ureg.parse_units(from_unit))
            converted = qty.to(_ureg.parse_units(to_unit))

        converted_value = round(float(converted.magnitude), 8)
        return ToolResult(result={
            "original_value": value,
            "from_unit": from_unit_raw,
            "converted_value": converted_value,
            "to_unit": to_unit_raw,
        })
    except pint.DimensionalityError:
        return ToolResult(error=f"Cannot convert '{from_unit_raw}' to '{to_unit_raw}': incompatible dimensions")
    except pint.UndefinedUnitError as exc:
        return ToolResult(error=f"Unknown unit: {exc}")
    except Exception as exc:
        return ToolResult(error=str(exc))


unit_converter_tool = ToolDefinition(
    name="unit_converter",
    description=(
        "Convert values between units of measurement. Supports length (miles, km, meters, feet, inches), "
        "mass (kg, pounds, grams, ounces), temperature (celsius, fahrenheit, kelvin), "
        "volume (liters, gallons), speed (mph, kph), and many more. "
        "Examples: 100 miles → km, 32 fahrenheit → celsius, 5 kg → pounds."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "description": "Numeric value to convert",
            },
            "from_unit": {
                "type": "string",
                "description": "Source unit, e.g. 'miles', 'kg', 'fahrenheit', 'liters'",
            },
            "to_unit": {
                "type": "string",
                "description": "Target unit, e.g. 'km', 'pounds', 'celsius', 'gallons'",
            },
        },
        "required": ["value", "from_unit", "to_unit"],
    },
    handler=_handle,
)
