from collections.abc import Generator

from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.agent.registry import registry as _registry, ToolRegistry


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_registry() -> ToolRegistry:
    return _registry
