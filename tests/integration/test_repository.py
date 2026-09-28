"""Integration tests for SQLiteSessionRepository."""

from pathlib import Path
import pytest

from data_agent.adapters.persistence.sqlite_repository import SQLiteSessionRepository
from data_agent.core.entities import MessageRole, Session


def test_sqlite_session_crud(tmp_path: Path):
    """Test full lifecycle persistence of Session aggregate."""
    db_file = tmp_path / "test_repo.db"
    repo = SQLiteSessionRepository(db_url=f"sqlite:///{db_file}")

    # Create session
    session = Session(dataset_path="/path/to/data.csv")
    session.add_message(MessageRole.USER, "What are the sales trends?")
    session.add_trace_step(
        step_index=1,
        thought="Grouping by month",
        code="df.groupby('month')['sales'].sum()",
        stdout="Jan: 100, Feb: 200",
        duration_seconds=0.35,
    )
    session.add_artifact(
        file_name="monthly_trends.html",
        storage_path="sess_1/monthly_trends.html",
    )

    saved = repo.save_session(session)
    assert saved.id == session.id

    # Retrieve session
    retrieved = repo.get_session(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id
    assert retrieved.dataset_path == "/path/to/data.csv"
    assert len(retrieved.messages) == 1
    assert retrieved.messages[0].content == "What are the sales trends?"
    assert len(retrieved.traces) == 1
    assert retrieved.traces[0].stdout == "Jan: 100, Feb: 200"
    assert len(retrieved.artifacts) == 1
    assert retrieved.artifacts[0].file_name == "monthly_trends.html"

    # List sessions
    sessions_list = repo.list_sessions()
    assert len(sessions_list) == 1
    assert sessions_list[0].id == session.id
