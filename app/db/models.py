from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime, ForeignKey
)
from sqlalchemy import Float as Real
from sqlalchemy.orm import relationship

from app.db.session import Base


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(36), primary_key=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    last_activity_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    turn_count = Column(Integer, nullable=False, default=0)

    tasks = relationship("Task", back_populates="conversation")
    messages = relationship("ConversationMessage", back_populates="conversation")


class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=True)
    input_text = Column(Text, nullable=False)
    final_answer = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    total_input_tokens = Column(Integer, nullable=True, default=0)
    total_output_tokens = Column(Integer, nullable=True, default=0)
    total_latency_ms = Column(Float, nullable=True)
    iterations = Column(Integer, nullable=True, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    conversation = relationship("Conversation", back_populates="tasks")
    trace_steps = relationship("TraceStep", back_populates="task", order_by="TraceStep.iteration, TraceStep.step_order")
    messages = relationship("ConversationMessage", back_populates="task")


class TraceStep(Base):
    __tablename__ = "trace_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    iteration = Column(Integer, nullable=False)
    step_order = Column(Integer, nullable=False)
    tool_name = Column(String(100), nullable=False)
    tool_input = Column(Text, nullable=False)   # JSON string
    tool_output = Column(Text, nullable=True)
    tool_error = Column(Text, nullable=True)
    thinking = Column(Text, nullable=True)
    latency_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    task = relationship("Task", back_populates="trace_steps")


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False)
    task_id = Column(String(36), ForeignKey("tasks.id"), nullable=False)
    role = Column(String(20), nullable=False)   # "user" or "assistant"
    content = Column(Text, nullable=False)       # JSON-serialized message content
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")
    task = relationship("Task", back_populates="messages")


# --- Product catalog (bonus database_query tool) ---

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(Text, nullable=False)
    category = Column(Text, nullable=False)
    price = Column(Real, nullable=False)
    stock_quantity = Column(Integer, nullable=False)
    sku = Column(String(50), unique=True, nullable=False)
    created_at = Column(Text, nullable=False)

    orders = relationship("Order", back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    total_price = Column(Real, nullable=False)
    customer_name = Column(Text, nullable=False)
    status = Column(Text, nullable=False)       # completed / pending / cancelled
    ordered_at = Column(Text, nullable=False)

    product = relationship("Product", back_populates="orders")
