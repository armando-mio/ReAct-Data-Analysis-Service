"""SQLite implementation of ISessionRepository using SQLAlchemy."""

from pathlib import Path
from typing import List, Optional
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as DBSession, sessionmaker

from data_agent.adapters.persistence.models import (
    ArtifactRecord,
    Base,
    MessageRecord,
    SessionRecord,
    TraceStepRecord,
)
from data_agent.core.entities import Artifact, Message, MessageRole, Session, TraceStep
from data_agent.ports.repository_port import ISessionRepository


class SQLiteSessionRepository(ISessionRepository):
    """Persists sessions, messages, traces, and artifacts in a local SQLite database."""

    def __init__(self, db_url: str = "sqlite:///storage/data_agent.db") -> None:
        # Ensure directory for sqlite file exists if using relative path
        if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
            db_path_str = db_url.replace("sqlite:///", "")
            Path(db_path_str).parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
        Base.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def _to_domain_session(self, record: SessionRecord) -> Session:
        """Map ORM SessionRecord to pure domain Session aggregate."""
        session = Session(
            id=record.id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            dataset_path=record.dataset_path,
        )
        session.messages = [
            Message(
                id=msg.id,
                session_id=msg.session_id,
                role=MessageRole(msg.role),
                content=msg.content,
                timestamp=msg.timestamp,
            )
            for msg in record.messages
        ]
        session.traces = [
            TraceStep(
                id=trace.id,
                session_id=trace.session_id,
                step_index=trace.step_index,
                thought=trace.thought,
                code=trace.code,
                stdout=trace.stdout,
                stderr=trace.stderr,
                duration_seconds=trace.duration_seconds,
            )
            for trace in record.traces
        ]
        session.artifacts = [
            Artifact(
                id=art.id,
                session_id=art.session_id,
                file_name=art.file_name,
                storage_path=art.storage_path,
                created_at=art.created_at,
            )
            for art in record.artifacts
        ]
        return session

    def save_session(self, session: Session) -> Session:
        """Persist or update an entire session aggregate."""
        with self.SessionFactory() as db:
            record = db.get(SessionRecord, session.id)
            if not record:
                record = SessionRecord(
                    id=session.id,
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    dataset_path=session.dataset_path,
                )
                db.add(record)
            else:
                record.updated_at = session.updated_at
                record.dataset_path = session.dataset_path

            # Synchronize messages
            existing_msg_ids = {m.id for m in record.messages}
            for msg in session.messages:
                if msg.id not in existing_msg_ids:
                    record.messages.append(
                        MessageRecord(
                            id=msg.id,
                            session_id=session.id,
                            role=msg.role.value,
                            content=msg.content,
                            timestamp=msg.timestamp,
                        )
                    )

            # Synchronize traces
            existing_trace_ids = {t.id for t in record.traces}
            for trace in session.traces:
                if trace.id not in existing_trace_ids:
                    record.traces.append(
                        TraceStepRecord(
                            id=trace.id,
                            session_id=session.id,
                            step_index=trace.step_index,
                            thought=trace.thought,
                            code=trace.code,
                            stdout=trace.stdout,
                            stderr=trace.stderr,
                            duration_seconds=trace.duration_seconds,
                        )
                    )

            # Synchronize artifacts
            existing_art_ids = {a.id for a in record.artifacts}
            for art in session.artifacts:
                if art.id not in existing_art_ids:
                    record.artifacts.append(
                        ArtifactRecord(
                            id=art.id,
                            session_id=session.id,
                            file_name=art.file_name,
                            storage_path=art.storage_path,
                            created_at=art.created_at,
                        )
                    )

            db.commit()
            db.refresh(record)
            return self._to_domain_session(record)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Fetch session by ID with all related records."""
        with self.SessionFactory() as db:
            record = db.get(SessionRecord, session_id)
            if not record:
                return None
            return self._to_domain_session(record)

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """List sessions ordered by update timestamp descending."""
        with self.SessionFactory() as db:
            stmt = select(SessionRecord).order_by(SessionRecord.updated_at.desc()).limit(limit).offset(offset)
            records = db.scalars(stmt).all()
            return [self._to_domain_session(r) for r in records]

    def add_message(self, message: Message) -> Message:
        """Persist a single message record."""
        with self.SessionFactory() as db:
            session_record = db.get(SessionRecord, message.session_id)
            if not session_record:
                session_record = SessionRecord(id=message.session_id)
                db.add(session_record)
            msg_record = MessageRecord(
                id=message.id,
                session_id=message.session_id,
                role=message.role.value,
                content=message.content,
                timestamp=message.timestamp,
            )
            db.add(msg_record)
            db.commit()
            return message

    def add_trace_step(self, trace: TraceStep) -> TraceStep:
        """Persist a single trace step record."""
        with self.SessionFactory() as db:
            session_record = db.get(SessionRecord, trace.session_id)
            if not session_record:
                session_record = SessionRecord(id=trace.session_id)
                db.add(session_record)
            t_record = TraceStepRecord(
                id=trace.id,
                session_id=trace.session_id,
                step_index=trace.step_index,
                thought=trace.thought,
                code=trace.code,
                stdout=trace.stdout,
                stderr=trace.stderr,
                duration_seconds=trace.duration_seconds,
            )
            db.add(t_record)
            db.commit()
            return trace

    def add_artifact(self, artifact: Artifact) -> Artifact:
        """Persist a single artifact record."""
        with self.SessionFactory() as db:
            session_record = db.get(SessionRecord, artifact.session_id)
            if not session_record:
                session_record = SessionRecord(id=artifact.session_id)
                db.add(session_record)
            art_record = ArtifactRecord(
                id=artifact.id,
                session_id=artifact.session_id,
                file_name=artifact.file_name,
                storage_path=artifact.storage_path,
                created_at=artifact.created_at,
            )
            db.add(art_record)
            db.commit()
            return artifact

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """Fetch artifact by ID."""
        with self.SessionFactory() as db:
            record = db.get(ArtifactRecord, artifact_id)
            if not record:
                return None
            return Artifact(
                id=record.id,
                session_id=record.session_id,
                file_name=record.file_name,
                storage_path=record.storage_path,
                created_at=record.created_at,
            )
