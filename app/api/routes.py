import json
import time
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.agent.loop import run_agent
from app.agent.registry import ToolRegistry
from app.api.schemas import (
    HealthResponse,
    ModelsResponse,
    ProviderInfo,
    TaskRequest,
    TaskResponse,
    TraceStepResponse,
)
from app.config import settings
from app.db.models import Task, TraceStep
from app.dependencies import get_db, get_registry

router = APIRouter()

# Track server start time for uptime
_START_TIME = time.monotonic()


def _task_to_response(task: Task) -> TaskResponse:
    trace = [
        TraceStepResponse(
            iteration=s.iteration,
            step_order=s.step_order,
            tool_name=s.tool_name,
            tool_input=json.loads(s.tool_input) if s.tool_input else {},
            tool_output=json.loads(s.tool_output) if s.tool_output else None,
            tool_error=s.tool_error,
            thinking=s.thinking,
            latency_ms=s.latency_ms,
        )
        for s in task.trace_steps
    ]
    return TaskResponse(
        task_id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        input_text=task.input_text,
        final_answer=task.final_answer,
        provider=task.provider or "",
        model=task.model or "",
        trace=trace,
        total_input_tokens=task.total_input_tokens or 0,
        total_output_tokens=task.total_output_tokens or 0,
        total_latency_ms=task.total_latency_ms or 0.0,
        iterations=task.iterations or 0,
        error_message=task.error_message,
        created_at=task.created_at.isoformat() if task.created_at else "",
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )


@router.post("/task", response_model=TaskResponse, status_code=201)
async def submit_task(
    request: TaskRequest,
    db: Session = Depends(get_db),
    registry: ToolRegistry = Depends(get_registry),
) -> TaskResponse:
    result = await run_agent(
        task=request.task,
        provider=request.provider,
        model=request.model,
        db=db,
        registry=registry,
        conversation_id=request.conversation_id,
    )

    task = db.get(Task, result.task_id)
    if task is None:
        raise HTTPException(status_code=500, detail="Failed to persist task")

    return _task_to_response(task)


@router.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db)) -> TaskResponse:
    task = db.get(Task, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return _task_to_response(task)


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    return HealthResponse(
        status="ok",
        db_connected=db_connected,
        version=settings.app_version,
        uptime_seconds=round(time.monotonic() - _START_TIME, 2),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    anthropic_models = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]
    openai_models = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
    gemini_models = ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]

    # Fetch Ollama models live
    ollama_models: list[str] = []
    ollama_configured = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
        if resp.status_code == 200:
            data = resp.json()
            ollama_models = [
                m["name"].replace(":latest", "")
                for m in data.get("models", [])
            ]
            ollama_configured = True
    except Exception:
        pass

    return ModelsResponse(
        anthropic=ProviderInfo(
            models=anthropic_models,
            configured=bool(settings.anthropic_api_key),
            default_model="claude-sonnet-4-6",
        ),
        openai=ProviderInfo(
            models=openai_models,
            configured=bool(settings.openai_api_key),
            default_model="gpt-4o-mini",
        ),
        gemini=ProviderInfo(
            models=gemini_models,
            configured=bool(settings.gemini_api_key),
            default_model="gemini-2.0-flash",
        ),
        ollama=ProviderInfo(
            models=ollama_models,
            configured=ollama_configured,
            default_model="llama3.2:3b",
        ),
    )
