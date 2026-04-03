from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000, description="Natural language task")
    conversation_id: str | None = Field(None, description="Continue an existing conversation")
    provider: str = Field("anthropic", description="LLM provider: anthropic, openai, gemini, ollama")
    model: str | None = Field(None, description="Model name; uses provider default if omitted")


class TraceStepResponse(BaseModel):
    iteration: int
    step_order: int
    tool_name: str
    tool_input: dict
    tool_output: Any | None
    tool_error: str | None
    thinking: str | None
    latency_ms: float


class TaskResponse(BaseModel):
    task_id: str
    conversation_id: str | None
    status: str
    input_text: str
    final_answer: str | None
    provider: str
    model: str
    trace: list[TraceStepResponse]
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    iterations: int
    error_message: str | None
    created_at: str
    completed_at: str | None


class ProviderInfo(BaseModel):
    models: list[str]
    configured: bool
    default_model: str


class ModelsResponse(BaseModel):
    anthropic: ProviderInfo
    openai: ProviderInfo
    gemini: ProviderInfo
    ollama: ProviderInfo


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    version: str
    uptime_seconds: float
