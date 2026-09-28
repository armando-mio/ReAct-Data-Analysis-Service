"""Abstract interface for session, message, trace, and artifact persistence."""

from abc import ABC, abstractmethod
from typing import List, Optional
from data_agent.core.entities import Artifact, Message, Session, TraceStep


class ISessionRepository(ABC):
    """Port interface for persisting and querying analysis sessions and associated data."""

    @abstractmethod
    def save_session(self, session: Session) -> Session:
        """Persist or update an entire session entity with its relations."""
        pass

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by its unique ID, including messages, traces, and artifacts."""
        pass

    @abstractmethod
    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """List recent sessions ordered by updated_at descending."""
        pass

    @abstractmethod
    def add_message(self, message: Message) -> Message:
        """Persist an individual message record for a session."""
        pass

    @abstractmethod
    def add_trace_step(self, trace: TraceStep) -> TraceStep:
        """Persist an individual reasoning trace record for a session."""
        pass

    @abstractmethod
    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Persist metadata for a generated artifact."""
        pass

    @abstractmethod
    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Retrieve artifact metadata by its unique ID."""
        pass
