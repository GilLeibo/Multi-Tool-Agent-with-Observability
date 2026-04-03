"""Endpoint integration tests."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from tests.conftest import make_llm_response, make_tool_use_response


def _mock_anthropic_complete(responses):
    """Return an async mock that yields each response in sequence."""
    call_count = 0

    async def _complete(*args, **kwargs):
        nonlocal call_count
        resp = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return resp

    return _complete


class TestHealthEndpoint:
    def test_health_endpoint(self, client: TestClient):
        """Test 5: Health endpoint returns ok and db_connected=True."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["db_connected"] is True
        assert "version" in data
        assert data["uptime_seconds"] >= 0


class TestGetTask:
    def test_get_task_by_id(self, client: TestClient):
        """Test 2: POST then GET returns the same task."""
        from app.llm.anthropic_client import AnthropicClient

        responses = [
            make_tool_use_response("calculator", {"expression": "25 * 480 / 100"}),
            make_llm_response(text="25% of 480 is 120."),
        ]

        with patch.object(AnthropicClient, "complete", side_effect=_mock_anthropic_complete(responses)), \
             patch.object(AnthropicClient, "format_assistant_message", return_value={"role": "assistant", "content": []}), \
             patch.object(AnthropicClient, "format_tool_result", return_value={"type": "tool_result", "tool_use_id": "tc_001", "content": "120.0"}):
            post_resp = client.post("/task", json={"task": "What is 25% of 480?", "provider": "anthropic"})

        assert post_resp.status_code == 201
        task_id = post_resp.json()["task_id"]
        assert task_id

        get_resp = client.get(f"/tasks/{task_id}")
        assert get_resp.status_code == 200
        data = get_resp.json()
        assert data["task_id"] == task_id
        assert data["total_input_tokens"] > 0
        assert isinstance(data["trace"], list)

    def test_get_nonexistent_task(self, client: TestClient):
        """Test 6: 404 for unknown task_id."""
        resp = client.get("/tasks/00000000-0000-0000-0000-000000000000")
        assert resp.status_code == 404
        assert "detail" in resp.json()


class TestModelsEndpoint:
    def test_models_returns_providers(self, client: TestClient):
        resp = client.get("/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "anthropic" in data
        assert "openai" in data
        assert "gemini" in data
        assert "ollama" in data
        assert isinstance(data["anthropic"]["models"], list)
        assert "configured" in data["anthropic"]
