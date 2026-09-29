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


def test_sqlite_clean_retrieval_and_structured_trace(tmp_path: Path):
    """Test get_messages, get_traces, get_artifacts, and get_session_history with structured trace format."""
    db_file = tmp_path / "test_repo_structured.db"
    repo = SQLiteSessionRepository(db_url=f"sqlite:///{db_file}")

    session = Session(dataset_path="/path/to/dataset.csv")
    session.add_message(MessageRole.USER, "Show me top categories")
    session.add_message(MessageRole.ASSISTANT, "Here are the top categories...")
    session.add_trace_step(
        step_index=1,
        thought="Calculate category totals",
        code="print(df.groupby('category')['revenue'].sum())",
        stdout="Category A: 500\nCategory B: 300",
        stderr=None,
        duration_seconds=0.25,
    )
    session.add_artifact(
        file_name="top_categories.html",
        storage_path="sess_2/top_categories.html",
    )

    repo.save_session(session)

    # Clean query methods
    messages = repo.get_messages(session.id)
    assert len(messages) == 2
    assert messages[0].content == "Show me top categories"
    assert messages[1].content == "Here are the top categories..."

    traces = repo.get_traces(session.id)
    assert len(traces) == 1
    assert traces[0].thought == "Calculate category totals"
    assert traces[0].stdout == "Category A: 500\nCategory B: 300"

    artifacts = repo.get_artifacts(session.id)
    assert len(artifacts) == 1
    assert artifacts[0].file_name == "top_categories.html"

    # Full structured history
    history = repo.get_session_history(session.id)
    assert history["session_id"] == session.id
    assert len(history["messages"]) == 2
    assert len(history["trace"]) == 1

    trace_step_json = history["trace"][0]
    assert trace_step_json["step_number"] == 1
    assert trace_step_json["thought"] == "Calculate category totals"
    assert trace_step_json["code_generated"] == "print(df.groupby('category')['revenue'].sum())"
    assert trace_step_json["observation_output"] == "Category A: 500\nCategory B: 300"
    assert trace_step_json["is_error"] is False
    assert trace_step_json["execution_time_ms"] == 250.0

    # Test WAL mode configuration
    with repo.engine.connect() as conn:
        journal_mode = conn.exec_driver_sql("PRAGMA journal_mode;").scalar()
        busy_timeout = conn.exec_driver_sql("PRAGMA busy_timeout;").scalar()
        assert journal_mode.lower() == "wal"
        assert int(busy_timeout) == 5000

    repo.close()
