"""Shared pytest fixtures for the test suite."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.agent.registry import ToolRegistry
from app.db.session import Base
from app.db.init_db import _seed_catalog
from app.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# In-memory SQLite for tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="session")
def test_session_factory(test_engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(autouse=True)
def seed_test_db(test_engine, monkeypatch):
    """Seed catalog data into the in-memory test DB."""
    from app.db import session as db_session_module
    from app.tools import database_query as dq_module

    monkeypatch.setattr(db_session_module, "engine", test_engine)
    monkeypatch.setattr(dq_module, "engine", test_engine)
    _seed_catalog()


# ---------------------------------------------------------------------------
# FastAPI TestClient wired to in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_registry():
    reg = ToolRegistry()
    from app.tools.calculator import calculator_tool
    from app.tools.weather import weather_tool
    from app.tools.web_search import web_search_tool
    from app.tools.unit_converter import unit_converter_tool
    from app.tools.database_query import database_query_tool
    reg.register(calculator_tool)
    reg.register(weather_tool)
    reg.register(web_search_tool)
    reg.register(unit_converter_tool)
    reg.register(database_query_tool)
    return reg


@pytest.fixture
def client(test_session_factory, test_registry, test_engine, monkeypatch):
    from app.db import session as db_session_module
    from app.db import init_db as init_db_module
    from app import dependencies

    monkeypatch.setattr(db_session_module, "engine", test_engine)
    monkeypatch.setattr(db_session_module, "SessionLocal", test_session_factory)
    monkeypatch.setattr(init_db_module, "engine", test_engine)

    def override_db():
        db = test_session_factory()
        try:
            yield db
        finally:
            db.close()

    def override_registry():
        return test_registry

    from app.main import app
    from app.dependencies import get_db, get_registry
    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_registry] = override_registry

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------

def make_llm_response(
    stop_reason: str = "end_turn",
    text: str | None = "42",
    tool_calls: list | None = None,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> LLMResponse:
    from app.llm.base import ToolCall
    return LLMResponse(
        stop_reason=stop_reason,
        text=text,
        tool_calls=tool_calls or [],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tc_001") -> LLMResponse:
    from app.llm.base import ToolCall
    return LLMResponse(
        stop_reason="tool_use",
        text=None,
        tool_calls=[ToolCall(id=tool_id, name=tool_name, input=tool_input)],
        input_tokens=80,
        output_tokens=30,
    )
