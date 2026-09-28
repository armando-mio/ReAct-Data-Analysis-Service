"""Pure domain entities for the ReAct Data Analysis service.

Adheres strictly to Hexagonal Architecture boundaries: no framework or database dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Artifact:
    """Represents a generated output artifact (e.g. standalone Plotly HTML visualization)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    file_name: str = ""
    storage_path: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStep:
    """Represents an atomic reasoning and execution step in the ReAct loop."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    step_index: int = 0
    thought: str = ""
    code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_seconds: float = 0.0
    execution_time: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Message:
    """Represents a message within a multi-turn analysis session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    role: MessageRole = MessageRole.USER
    content: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Dataset:
    """Domain representation of an uploaded or assigned dataset."""
    path: str
    filename: str
    preview: str
    row_count: int = 0
    column_names: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    summary_stats: Optional[str] = None


@dataclass
class Session:
    """Aggregate root representing an ongoing data analysis session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    dataset_path: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    traces: List[TraceStep] = field(default_factory=list)
    artifacts: List[Artifact] = field(default_factory=list)

    def add_message(self, role: MessageRole, content: str) -> Message:
        """Add a new message to the session."""
        message = Message(session_id=self.id, role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        return message

    def add_trace_step(
        self,
        step_index: int,
        thought: str,
        code: Optional[str] = None,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        duration_seconds: float = 0.0,
    ) -> TraceStep:
        """Add a completed reasoning trace step."""
        trace = TraceStep(
            session_id=self.id,
            step_index=step_index,
            thought=thought,
            code=code,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            execution_time=duration_seconds,
        )
        self.traces.append(trace)
        self.updated_at = datetime.now(timezone.utc)
        return trace

    def add_artifact(self, file_name: str, storage_path: str, metadata: Optional[Dict[str, Any]] = None) -> Artifact:
        """Register a generated artifact."""
        artifact = Artifact(
            session_id=self.id,
            file_name=file_name,
            storage_path=storage_path,
            metadata=metadata or {},
        )
        self.artifacts.append(artifact)
        self.updated_at = datetime.now(timezone.utc)
        return artifact
