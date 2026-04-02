# Multi-Tool Agent with Observability

A general-purpose AI agent REST API that reasons step-by-step using a custom ReAct loop, calls real external tools, and returns a fully structured trace of every reasoning step. Supports multiple LLM providers including local offline inference via Ollama.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI REST API                         │
│  POST /task  │  GET /tasks/{id}  │  GET /tasks  │  GET /health  │
│              │                   │  GET /models │               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Agent Loop    │  app/agent/loop.py
                    │  (ReAct, max    │  Custom hand-rolled loop —
                    │   10 iters)     │  no LangChain/LlamaIndex
                    └────────┬────────┘
                             │ uses
              ┌──────────────▼──────────────┐
              │       LLM Abstraction        │  app/llm/
              │  AnthropicClient             │
              │  OpenAIClient                │
              │  GeminiClient                │
              │  OllamaClient (local)        │
              └──────────────┬──────────────┘
                             │ tool calls
              ┌──────────────▼──────────────┐
              │         Tool Registry        │  app/agent/registry.py
              │  calculator   │  weather     │
              │  web_search   │  unit_conv.  │
              │  database_query (SQLite)     │
              └──────────────┬──────────────┘
                             │ persists
              ┌──────────────▼──────────────┐
              │    SQLite (SQLAlchemy)       │  /app/data/agent.db
              │  tasks  │  trace_steps      │
              │  conversations  │  messages │
              │  products  │  orders        │
              └─────────────────────────────┘
```

### Agent Reasoning Loop

The agent uses a custom **ReAct (Reason + Act)** loop that is entirely visible and inspectable:

1. Send the user task + conversation history + tool definitions to the LLM
2. If the LLM returns `stop_reason == tool_use` → execute all tool calls, record trace steps, feed results back
3. If `stop_reason == end_turn` → the LLM has synthesized a final answer → done
4. Repeat up to `MAX_AGENT_ITERATIONS` times (default: 10)
5. Persist the full trace, token counts, and latency to SQLite

Every tool call is recorded as a `TraceStep` with: tool name, input, output/error, thinking text, and latency.

The loop is **provider-agnostic** — it talks to a `LLMClient` abstract interface, so swapping providers requires no loop changes.

---

## Tools

| Tool | Library / API | What it does |
|---|---|---|
| `calculator` | `sympy` | Evaluates math expressions safely — no `eval()` |
| `weather` | OpenWeatherMap API | Current weather for any city |
| `web_search` | Tavily API | Searches the web, returns snippets + summary |
| `unit_converter` | `pint` | Converts any unit (length, mass, temp, volume…) |
| `database_query` | SQLite / SQLAlchemy | Runs SELECT queries on the product/orders catalog |

Tools are plain Python async functions — not an MCP server. The LLM decides which tools to call based on the task.

---

## LLM Providers

| Provider | Client | Default Model | Requires |
|---|---|---|---|
| `anthropic` | `AnthropicClient` | claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAIClient` | gpt-4o-mini | `OPENAI_API_KEY` |
| `gemini` | `GeminiClient` | gemini-2.0-flash | `GEMINI_API_KEY` |
| `ollama` | `OllamaClient` | llama3.2:3b | Ollama service running |

Ollama runs locally in Docker — no API key required. Models are pre-pulled into a persistent volume on first startup.

---

## Setup & Run

### Prerequisites

- Docker + Docker Compose
- API keys for the providers you want to use (Ollama needs no key)

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in the API keys for your desired providers
```

### 2. Start the service

```bash
docker compose up --build
```

On first run, this will:
- Build the API container
- Pull `llama3.2:3b` into the Ollama volume (~4 GB, once only)
- Initialize the SQLite database and seed the product catalog

Subsequent starts skip the model download — it's cached in the `ollama_data` Docker volume.

### 3. Verify

```bash
curl http://localhost:8000/health
# {"status":"ok","db_connected":true,"version":"1.0.0","uptime_seconds":3.1}
```

### 4. Open the UI

Visit **http://localhost:8000/ui** in your browser.

---

### Running Locally (without Docker)

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
python -m app.db.init_db
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Visit **http://localhost:8000/ui** in your browser.

> **Using Ollama locally:** Ollama must be installed and running on your machine (`ollama serve`), and the desired model must be pulled — e.g. `ollama pull llama3.2:3b`. Download Ollama from [ollama.com](https://ollama.com).

---

## API Reference

### POST /task

Submit a natural language task.

```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"task": "What is 25% of 480?", "provider": "anthropic"}'
```

**Request body:**
```json
{
  "task": "string (required)",
  "provider": "anthropic | openai | gemini | ollama (default: anthropic)",
  "model": "model name (optional, uses provider default)",
  "conversation_id": "UUID (optional, for multi-turn)"
}
```

**Response:** `TaskResponse` with `task_id`, `final_answer`, `trace[]`, token counts, latency.

---

### GET /tasks/{task_id}

Retrieve a past task result.

```bash
curl http://localhost:8000/tasks/abc-123-...
```

---

### GET /tasks

List the most recent tasks (default: last 50), ordered newest first. Used by the UI to restore history across sessions.

```bash
curl http://localhost:8000/tasks
curl http://localhost:8000/tasks?limit=10
```

---

### GET /health

Health check.

---

### GET /models

Returns available models per provider, including which providers are configured and whether Ollama is reachable.

---

## Running Tests

```bash
docker compose run --rm api pytest tests/ -v
```

Or locally (with dependencies installed):

```bash
pip install -r requirements.txt
pytest tests/ -v
```

The test suite covers 25 tests across 4 files:

| File | What it covers |
|---|---|
| `test_agent.py` | Agent output correctness (tool result + final answer content) and multi-turn context |
| `test_multi_tool.py` | Sequential multi-tool chains, iteration tracking, error recovery in trace |
| `test_tools.py` | Tool function unit tests: calculator, unit_converter, database_query (incl. SQL injection rejection) |
| `test_api.py` | API endpoint tests: health, task CRUD, models |

---

## Example Tasks

### 1. Calculator

**Request:**
```json
{"task": "What is 15% of 847 plus the square root of 144?", "provider": "anthropic"}
```

**Trace:**
```
Step 1 · calculator · ~8ms
  Input:  {"expression": "0.15 * 847 + sqrt(144)"}
  Output: "138.905"
```

**Final answer:** "15% of 847 is 127.05, and √144 = 12. Combined: **139.05**"

---

### 2. Weather + Unit Conversion

**Request:**
```json
{"task": "What's the weather in Tokyo and what is that temperature in Fahrenheit?", "provider": "openai"}
```

**Trace:**
```
Step 1 · weather · ~420ms
  Input:  {"city": "Tokyo"}
  Output: {"temp_c": 18.3, "description": "Clear sky", ...}

Step 2 · unit_converter · ~1ms
  Input:  {"value": 18.3, "from_unit": "celsius", "to_unit": "fahrenheit"}
  Output: {"converted_value": 64.94}
```

**Final answer:** "Tokyo is currently 18.3°C (64.9°F) with clear skies."

---

### 3. Database Query (Bonus)

**Request:**
```json
{"task": "Which product category has the highest total revenue from completed orders?", "provider": "anthropic"}
```

**Trace:**
```
Step 1 · database_query · ~3ms
  Input: {"sql": "SELECT p.category, SUM(o.total_price) as revenue FROM products p JOIN orders o ON p.id = o.product_id WHERE o.status = 'completed' GROUP BY p.category ORDER BY revenue DESC"}
  Output: {"columns": ["category", "revenue"], "rows": [["Furniture", 1049.97], ...], "row_count": 4}
```

**Final answer:** "Furniture has the highest revenue from completed orders at $1,049.97."

---

### 4. Web Search

**Request:**
```json
{"task": "What is the latest version of Python and when was it released?", "provider": "gemini"}
```

**Trace:**
```
Step 1 · web_search · ~850ms
  Input:  {"query": "latest Python version release date 2024"}
  Output: {"answer_summary": "Python 3.13...", "results": [...]}
```

**Final answer:** "Python 3.13 was released on October 7, 2024..."

---

### 5. Multi-Step Multi-Turn

**Turn 1:**
```json
{"task": "Convert 100 miles to kilometers", "provider": "anthropic"}
```
Response includes `conversation_id: "abc-123"`

**Turn 2:**
```json
{"task": "Now convert that result to meters", "provider": "anthropic", "conversation_id": "abc-123"}
```

The agent remembers the previous result (160.934 km) from conversation history and converts it to 160,934 meters.

---

## Bonus Features

- **`database_query` tool** — pre-seeded SQLite catalog with 17 products and 30+ orders across 4 categories; SQL injection protected (SELECT-only + semicolon blocking)
- **Multi-turn conversations** — `conversation_id` links task chains; prior turns loaded as context. UI shows "Continue conversation" button on each history item
- **4 LLM providers** — Anthropic, OpenAI, Gemini, Ollama (offline, no API key required)
- **Frontend UI** — provider/model selector with API key validation, task input lock during execution, reasoning trace accordion, persistent session history loaded from DB on startup, token + latency display
- **Test suite** — 25 automated tests across tools, agent loop output correctness, multi-tool trace chains, and API endpoints

## Provider Notes

| Provider | Tool Use Quality | Notes |
|---|---|---|
| Anthropic (claude-sonnet-4-6) | Excellent | Reliably calls tools with correct arguments |
| OpenAI (gpt-4o-mini) | Good | Strong tool call adherence |
| Gemini (gemini-2.0-flash) | Good | Requires `GEMINI_API_KEY` |
| Ollama (llama3.2:3b) | Limited | Small model; may pass schema metadata instead of values for some tools. A larger model (8B+) is recommended for reliable tool use |
