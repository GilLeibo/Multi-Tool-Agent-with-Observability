"""Agent reasoning loop tests."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_llm_response, make_tool_use_response


def _side_effect(responses):
    it = iter(responses)

    async def _fn(*args, **kwargs):
        return next(it)

    return _fn


class TestCalculatorTask:
    def test_calculator_task(self, client: TestClient):
        """Test 1: Calculator tool is called, result appears in final_answer, trace has 1 step."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "0.15 * 847 + sqrt(144)"}),
            make_llm_response(text="15% of 847 plus the square root of 144 equals 138.905."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_side_effect(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", return_value={"type": "tool_result", "tool_use_id": "tc_001", "content": "138.905"}):

            resp = client.post("/task", json={
                "task": "What is 15% of 847 plus the square root of 144?",
                "provider": "anthropic",
            })

        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "completed"
        assert data["final_answer"] is not None
        assert len(data["trace"]) == 1
        assert data["trace"][0]["tool_name"] == "calculator"
        assert data["total_input_tokens"] > 0
        assert data["total_output_tokens"] > 0


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
