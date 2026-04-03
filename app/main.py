import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.agent.registry import registry
from app.api.routes import router
from app.config import settings
from app.db.init_db import init_db
from app.tools.calculator import calculator_tool
from app.tools.database_query import database_query_tool
from app.tools.unit_converter import unit_converter_tool
from app.tools.weather import weather_tool
from app.tools.web_search import web_search_tool

logging.basicConfig(level=settings.log_level.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing database...")
    init_db()

    logger.info("Registering tools...")
    registry.register(calculator_tool)
    registry.register(weather_tool)
    registry.register(web_search_tool)
    registry.register(unit_converter_tool)
    registry.register(database_query_tool)
    logger.info("Tools registered: %s", [t.name for t in registry.get_all()])

    yield
    # Shutdown (nothing to clean up for SQLite)


app = FastAPI(
    title="Multi-Tool Agent API",
    description="A general-purpose AI agent with observability",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve frontend if it exists
_frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/ui", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
