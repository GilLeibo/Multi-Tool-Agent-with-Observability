"""Individual tool unit tests."""
import pytest


class TestDatabaseQueryTool:
    """Test 4: database_query tool returns rows and rejects non-SELECT."""

    @pytest.mark.asyncio
    async def test_select_returns_rows(self):
        from app.tools.database_query import _handle

        result = await _handle({"sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 3"})
        assert result.error is None
        assert result.result is not None
        assert result.result["row_count"] == 3
        assert "name" in result.result["columns"]
        assert "price" in result.result["columns"]
        # Most expensive product should be first
        assert result.result["rows"][0][1] >= result.result["rows"][1][1]

    @pytest.mark.asyncio
    async def test_drop_table_rejected(self):
        from app.tools.database_query import _handle

        result = await _handle({"sql": "DROP TABLE products"})
        assert result.error is not None
        assert "SELECT" in result.error

    @pytest.mark.asyncio
    async def test_insert_rejected(self):
        from app.tools.database_query import _handle

        result = await _handle({"sql": "INSERT INTO products (name) VALUES ('hacked')"})
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_join_query(self):
        from app.tools.database_query import _handle

        result = await _handle({
            "sql": """
                SELECT p.name, SUM(o.quantity) as total_sold
                FROM products p
                JOIN orders o ON p.id = o.product_id
                WHERE o.status = 'completed'
                GROUP BY p.id
                ORDER BY total_sold DESC
                LIMIT 5
            """
        })
        assert result.error is None
        assert result.result["row_count"] > 0


class TestCalculatorTool:
    @pytest.mark.asyncio
    async def test_basic_arithmetic(self):
        from app.tools.calculator import _handle

        result = await _handle({"expression": "2 ** 10"})
        assert result.error is None
        assert result.result == "1024"

    @pytest.mark.asyncio
    async def test_sqrt(self):
        from app.tools.calculator import _handle

        result = await _handle({"expression": "sqrt(144)"})
        assert result.error is None
        assert result.result == "12"

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        from app.tools.calculator import _handle

        result = await _handle({"expression": "import os"})
        assert result.error is not None


class TestUnitConverterTool:
    @pytest.mark.asyncio
    async def test_miles_to_km(self):
        from app.tools.unit_converter import _handle

        result = await _handle({"value": 1, "from_unit": "miles", "to_unit": "km"})
        assert result.error is None
        assert abs(result.result["converted_value"] - 1.60934) < 0.001

    @pytest.mark.asyncio
    async def test_celsius_to_fahrenheit(self):
        from app.tools.unit_converter import _handle

        result = await _handle({"value": 0, "from_unit": "celsius", "to_unit": "fahrenheit"})
        assert result.error is None
        assert result.result["converted_value"] == pytest.approx(32.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_incompatible_units(self):
        from app.tools.unit_converter import _handle

        result = await _handle({"value": 1, "from_unit": "kg", "to_unit": "meters"})
        assert result.error is not None
