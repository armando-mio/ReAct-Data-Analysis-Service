"""SQLAlchemy 2.0 ORM definitions for SQLite persistence."""

from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all persistence models."""
    pass


class SessionRecord(Base):
    """Database record for a conversation and analysis session."""
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    dataset_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    messages: Mapped[List["MessageRecord"]] = relationship(
        "MessageRecord", back_populates="session", cascade="all, delete-orphan", order_by="MessageRecord.timestamp"
    )
    traces: Mapped[List["TraceStepRecord"]] = relationship(
        "TraceStepRecord", back_populates="session", cascade="all, delete-orphan", order_by="TraceStepRecord.step_index"
    )
    artifacts: Mapped[List["ArtifactRecord"]] = relationship(
        "ArtifactRecord", back_populates="session", cascade="all, delete-orphan", order_by="ArtifactRecord.created_at"
    )


class MessageRecord(Base):
    """Database record for user/assistant conversation messages."""
    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["SessionRecord"] = relationship("SessionRecord", back_populates="messages")


class TraceStepRecord(Base):
    """Database record for structured reasoning trace steps."""
    __tablename__ = "trace_steps"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    step_index: Mapped[int] = mapped_column(Integer, default=0)
    thought: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stdout: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stderr: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, default=0.0)

    session: Mapped["SessionRecord"] = relationship("SessionRecord", back_populates="traces")


class ArtifactRecord(Base):
    """Database record for generated artifact files."""
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), ForeignKey("sessions.id", ondelete="CASCADE"), index=True)
    file_name: Mapped[str] = mapped_column(String(256))
    storage_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    session: Mapped["SessionRecord"] = relationship("SessionRecord", back_populates="artifacts")
