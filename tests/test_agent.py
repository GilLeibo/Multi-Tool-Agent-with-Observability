"""Agent reasoning loop tests — structural behaviour and known output validation."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_llm_response, make_tool_use_response


def _side_effect(responses):
    it = iter(responses)

    async def _fn(*args, **kwargs):
        return next(it)

    return _fn


# ── Single-tool agent loop + output correctness ────────────────────────────

class TestAgentOutputs:
    """Full agent loop: task → tool call → correct output → final answer."""

    def test_calculator_power(self, client: TestClient):
        """Agent computes 2**10 via calculator and reports 1024 in the answer."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "2**10"}, tool_id="tc_001"),
            make_llm_response(text="2 to the power of 10 is 1024."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "What is 2 to the power of 10?", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["trace"]) == 1
        assert data["trace"][0]["tool_name"] == "calculator"
        assert data["trace"][0]["tool_output"] == "1024"
        assert data["total_input_tokens"] > 0
        assert data["total_output_tokens"] > 0
        assert "1024" in data["final_answer"]

    def test_calculator_sqrt(self, client: TestClient):
        """Agent computes sqrt(256) via calculator and the result is 16."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "sqrt(256)"}, tool_id="tc_001"),
            make_llm_response(text="The square root of 256 is 16."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "What is the square root of 256?", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        assert data["trace"][0]["tool_output"] == "16"
        assert "16" in data["final_answer"]

    def test_unit_converter_miles_to_km(self, client: TestClient):
        """Agent converts 100 miles to km — tool returns ~160.93."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("unit_converter", {"value": 100, "from_unit": "miles", "to_unit": "km"}, tool_id="tc_001"),
            make_llm_response(text="100 miles is equal to 160.93 kilometers."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "Convert 100 miles to kilometers.", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        converted = data["trace"][0]["tool_output"]["converted_value"]
        assert abs(converted - 160.9344) < 0.001
        assert "160.9" in data["final_answer"]

    def test_unit_converter_celsius_to_fahrenheit(self, client: TestClient):
        """Agent converts 0°C to Fahrenheit — tool returns 32.0."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("unit_converter", {"value": 0, "from_unit": "celsius", "to_unit": "fahrenheit"}, tool_id="tc_001"),
            make_llm_response(text="0 degrees Celsius is equal to 32 degrees Fahrenheit."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "Convert 0 Celsius to Fahrenheit.", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        converted = data["trace"][0]["tool_output"]["converted_value"]
        assert converted == pytest.approx(32.0, abs=0.1)
        assert "32" in data["final_answer"]

    def test_unit_converter_kg_to_lbs(self, client: TestClient):
        """Agent converts 70 kg to lbs — tool returns ~154.32."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("unit_converter", {"value": 70, "from_unit": "kg", "to_unit": "lbs"}, tool_id="tc_001"),
            make_llm_response(text="70 kilograms is approximately 154.32 pounds."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "Convert 70 kg to pounds.", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        converted = data["trace"][0]["tool_output"]["converted_value"]
        assert abs(converted - 154.324) < 0.1
        assert "154" in data["final_answer"]

    def test_database_query_electronics_count(self, client: TestClient):
        """Agent queries DB for Electronics products — tool returns count=6."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("database_query", {"sql": "SELECT COUNT(*) as total FROM products WHERE category = 'Electronics'"}, tool_id="tc_001"),
            make_llm_response(text="There are 6 Electronics products in the catalog."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", side_effect=lambda id, result, *_, **__: {"type": "tool_result", "tool_use_id": id, "content": result}):
            resp = client.post("/task", json={"task": "How many Electronics products are there?", "provider": "anthropic"})

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        rows = data["trace"][0]["tool_output"]["rows"]
        assert rows[0][0] == 6
        assert "6" in data["final_answer"]


# ── Multi-turn conversation ────────────────────────────────────────────────

class TestMultiTurnConversation:
    def test_multi_turn_conversation(self, client: TestClient):
        """Test 3: Turn 2 with the same conversation_id has prior context passed to LLM."""
        from app.llm.anthropic_client import AnthropicClient

        captured_messages = []

        async def _mock_complete(system, messages, tools):
            captured_messages.extend(messages)
            return make_llm_response(text="Got it, nice to meet you!")

        # Turn 1
        with patch.object(AnthropicClient, "complete", side_effect=_mock_complete), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}):
            resp1 = client.post("/task", json={"task": "My name is Alice", "provider": "anthropic"})

        assert resp1.status_code == 201
        conv_id = resp1.json()["conversation_id"]
        assert conv_id

        captured_messages.clear()

        async def _mock_complete2(system, messages, tools):
            captured_messages.extend(messages)
            return make_llm_response(text="Your name is Alice.")

        # Turn 2 — pass conversation_id
        with patch.object(AnthropicClient, "complete", side_effect=_mock_complete2), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}):
            resp2 = client.post("/task", json={
                "task": "What is my name?",
                "provider": "anthropic",
                "conversation_id": conv_id,
            })

        assert resp2.status_code == 201
        # The messages list passed to the LLM should include the prior turn
        roles = [m["role"] for m in captured_messages]
        assert "user" in roles
        # There should be more than 1 message (prior turn + current)
        assert len(captured_messages) >= 2
