"""Core ReAct agent loop — provider-agnostic."""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.registry import ToolRegistry
from app.config import settings
from app.db.models import Conversation, ConversationMessage, Task, TraceStep
from app.llm.base import LLMClient, LLMResponse
from app.llm.factory import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class AgentTraceStep:
    iteration: int
    step_order: int
    tool_name: str
    tool_input: dict
    tool_output: object
    tool_error: str | None
    thinking: str | None
    latency_ms: float


@dataclass
class AgentResult:
    task_id: str
    conversation_id: str | None
    status: str
    final_answer: str | None
    error_message: str | None
    trace: list[AgentTraceStep]
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    iterations: int
    provider: str
    model: str
    created_at: datetime
    completed_at: datetime | None


def _load_conversation_history(conversation_id: str, db: Session) -> list[dict]:
    """Load prior conversation turns as a list of normalized message dicts."""
    messages = (
        db.query(ConversationMessage)
        .filter(ConversationMessage.conversation_id == conversation_id)
        .order_by(ConversationMessage.created_at.asc())
        .all()
    )
    result = []
    for m in messages:
        content = json.loads(m.content)
        result.append({"role": m.role, "content": content})
    return result


def _persist_task(
    db: Session,
    task_id: str,
    conversation_id: str | None,
    input_text: str,
    result: AgentResult,
    trace_steps: list[AgentTraceStep],
) -> None:
    """Write task, trace_steps, and conversation_messages to DB."""
    task = Task(
        id=task_id,
        conversation_id=conversation_id,
        input_text=input_text,
        final_answer=result.final_answer,
        status=result.status,
        provider=result.provider,
        model=result.model,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        total_latency_ms=result.total_latency_ms,
        iterations=result.iterations,
        error_message=result.error_message,
        created_at=result.created_at,
        completed_at=result.completed_at,
    )
    db.add(task)

    for i, step in enumerate(trace_steps):
        db.add(TraceStep(
            task_id=task_id,
            iteration=step.iteration,
            step_order=step.step_order,
            tool_name=step.tool_name,
            tool_input=json.dumps(step.tool_input),
            tool_output=json.dumps(step.tool_output) if step.tool_output is not None else None,
            tool_error=step.tool_error,
            thinking=step.thinking,
            latency_ms=step.latency_ms,
        ))

    # Persist clean conversation messages for multi-turn context
    if conversation_id:
        db.add(ConversationMessage(
            conversation_id=conversation_id,
            task_id=task_id,
            role="user",
            content=json.dumps(input_text),
        ))
        if result.final_answer:
            db.add(ConversationMessage(
                conversation_id=conversation_id,
                task_id=task_id,
                role="assistant",
                content=json.dumps(result.final_answer),
            ))
        # Update conversation metadata
        conv = db.get(Conversation, conversation_id)
        if conv:
            conv.last_activity_at = datetime.utcnow()
            conv.turn_count = (conv.turn_count or 0) + 1

    db.commit()


async def run_agent(
    task: str,
    provider: str,
    model: str | None,
    db: Session,
    registry: ToolRegistry,
    conversation_id: str | None = None,
) -> AgentResult:
    """Run the ReAct agent loop and return a structured result."""
    task_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    start_time = time.monotonic()

    # Resolve / create conversation
    actual_conversation_id = conversation_id
    if actual_conversation_id:
        conv = db.get(Conversation, actual_conversation_id)
        if not conv:
            # Invalid conversation_id — start fresh
            actual_conversation_id = None

    if not actual_conversation_id:
        actual_conversation_id = str(uuid.uuid4())
        conv = Conversation(id=actual_conversation_id)
        db.add(conv)
        db.commit()

    # Determine actual model name
    model_defaults = {
        "anthropic": "claude-sonnet-4-6",
        "openai": "gpt-4o-mini",
        "gemini": "gemini-2.0-flash",
        "ollama": "llama3.2:3b",
    }
    actual_model = model or model_defaults.get(provider.lower(), provider)

    # Build LLM client
    llm: LLMClient = get_llm_client(provider, model)

    # Load conversation history
    messages: list[dict] = _load_conversation_history(actual_conversation_id, db)
    messages.append({"role": "user", "content": task})

    tools = registry.get_all()
    trace_steps: list[AgentTraceStep] = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_answer: str | None = None
    error_message: str | None = None
    status = "completed"
    iteration = 0

    try:
        while iteration < settings.max_agent_iterations:
            iteration += 1
            logger.debug("Agent iteration %d, provider=%s, model=%s", iteration, provider, actual_model)

            response: LLMResponse = await llm.complete(SYSTEM_PROMPT, messages, tools)
            total_input_tokens += response.input_tokens
            total_output_tokens += response.output_tokens

            if response.stop_reason == "end_turn":
                final_answer = response.text or ""
                break

            if response.stop_reason == "tool_use":
                # Append assistant message (with tool_use blocks)
                messages.append(llm.format_assistant_message(response))

                # Process each tool call in this response
                tool_result_messages: list[dict] = []
                for step_order, tool_call in enumerate(response.tool_calls):
                    step_start = time.monotonic()
                    tool_result = await registry.dispatch(tool_call.name, tool_call.input)
                    step_latency_ms = (time.monotonic() - step_start) * 1000

                    output_str = (
                        json.dumps(tool_result.result)
                        if tool_result.result is not None
                        else None
                    )
                    error_str = tool_result.error

                    trace_steps.append(AgentTraceStep(
                        iteration=iteration,
                        step_order=step_order,
                        tool_name=tool_call.name,
                        tool_input=tool_call.input,
                        tool_output=tool_result.result,
                        tool_error=error_str,
                        thinking=response.thinking if step_order == 0 else None,
                        latency_ms=round(step_latency_ms, 2),
                    ))

                    result_content = output_str if not error_str else f"ERROR: {error_str}"
                    tool_result_messages.append(
                        llm.format_tool_result(tool_call.id, result_content or "", tool_call.name)
                    )

                # Append tool results to messages
                # Anthropic expects one user message with a list of tool_result content blocks
                # OpenAI/Ollama expects one message per tool result
                if provider.lower() == "anthropic":
                    messages.append({"role": "user", "content": tool_result_messages})
                else:
                    messages.extend(tool_result_messages)

            else:
                # Unexpected stop_reason
                error_message = f"Unexpected stop_reason: {response.stop_reason}"
                status = "error"
                break

        else:
            # Hit max iterations
            final_answer = f"Reached maximum iterations ({settings.max_agent_iterations}). Partial reasoning may be incomplete."
            status = "error"
            error_message = "Max iterations reached"

    except Exception as exc:
        logger.exception("Agent loop failed for task %s", task_id)
        error_message = str(exc)
        status = "error"

    completed_at = datetime.utcnow()
    total_latency_ms = (time.monotonic() - start_time) * 1000

    result = AgentResult(
        task_id=task_id,
        conversation_id=actual_conversation_id,
        status=status,
        final_answer=final_answer,
        error_message=error_message,
        trace=trace_steps,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_latency_ms=round(total_latency_ms, 2),
        iterations=iteration,
        provider=provider,
        model=actual_model,
        created_at=created_at,
        completed_at=completed_at,
    )

    _persist_task(db, task_id, actual_conversation_id, task, result, trace_steps)
    return result
