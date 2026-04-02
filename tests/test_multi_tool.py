"""Tests that verify agent behaviour when multiple tools are used in a single task."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_llm_response, make_tool_use_response


def _side_effect(responses):
    it = iter(responses)

    async def _fn(*args, **kwargs):
        return next(it)

    return _fn


class TestCalculatorThenUnitConverter:
    def test_two_sequential_tools(self, client: TestClient):
        """Agent calls calculator then unit_converter — trace must have 2 steps."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "100 * 1.2"}, tool_id="tc_001"),
            make_tool_use_response("unit_converter", {"value": 120, "from_unit": "km", "to_unit": "miles"}, tool_id="tc_002"),
            make_llm_response(text="120 km is approximately 74.56 miles."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, tool_name="": {"type": "tool_result", "tool_use_id": id, "content": result}):

            resp = client.post("/task", json={
                "task": "Calculate 100 * 1.2 and convert the result from km to miles.",
                "provider": "anthropic",
            })

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["trace"]) == 2
        tool_names = [s["tool_name"] for s in data["trace"]]
        assert "calculator" in tool_names
        assert "unit_converter" in tool_names
        assert data["final_answer"] is not None


class TestThreeToolsInSequence:
    def test_three_sequential_tools(self, client: TestClient):
        """Agent calls calculator, unit_converter, then database_query — trace must have 3 steps."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "50 * 2"}, tool_id="tc_001"),
            make_tool_use_response("unit_converter", {"value": 100, "from_unit": "kg", "to_unit": "lbs"}, tool_id="tc_002"),
            make_tool_use_response("database_query", {"sql": "SELECT COUNT(*) as total FROM products"}, tool_id="tc_003"),
            make_llm_response(text="Done: 100 kg is 220.46 lbs and there are 10 products in the catalog."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, tool_name="": {"type": "tool_result", "tool_use_id": id, "content": result}):

            resp = client.post("/task", json={
                "task": "Compute 50*2, convert to lbs, and count products in the database.",
                "provider": "anthropic",
            })

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["trace"]) == 3
        tool_names = [s["tool_name"] for s in data["trace"]]
        assert tool_names == ["calculator", "unit_converter", "database_query"]
        assert data["total_input_tokens"] > 0
        assert data["total_output_tokens"] > 0


class TestIterationTracking:
    def test_trace_records_correct_iterations(self, client: TestClient):
        """Each tool call across different iterations is recorded with the correct iteration number."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "7 * 6"}, tool_id="tc_001"),
            make_tool_use_response("calculator", {"expression": "sqrt(1764)"}, tool_id="tc_002"),
            make_llm_response(text="7 * 6 = 42 and sqrt(1764) = 42."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, tool_name="": {"type": "tool_result", "tool_use_id": id, "content": result}):

            resp = client.post("/task", json={
                "task": "What is 7 * 6 and the square root of 1764?",
                "provider": "anthropic",
            })

        assert resp.status_code == 201
        data = resp.json()
        assert len(data["trace"]) == 2
        # Two separate LLM iterations — iteration numbers must differ
        iterations = [s["iteration"] for s in data["trace"]]
        assert iterations[0] != iterations[1]
        # All steps are step_order 0 (first call in each iteration)
        assert all(s["step_order"] == 0 for s in data["trace"])


class TestToolErrorRecovery:
    def test_tool_error_recorded_in_trace(self, client: TestClient):
        """When a tool call fails the error is captured in the trace and the agent continues."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "import os"}, tool_id="tc_001"),
            make_tool_use_response("calculator", {"expression": "6 * 7"}, tool_id="tc_002"),
            make_llm_response(text="6 * 7 = 42."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, tool_name="": {"type": "tool_result", "tool_use_id": id, "content": result}):

            resp = client.post("/task", json={
                "task": "Calculate 6 * 7.",
                "provider": "anthropic",
            })

        assert resp.status_code == 201
        data = resp.json()
        assert len(data["trace"]) == 2
        # First step should have an error recorded
        assert data["trace"][0]["tool_error"] is not None
        # Second step should succeed
        assert data["trace"][1]["tool_error"] is None
        assert data["status"] == "completed"
