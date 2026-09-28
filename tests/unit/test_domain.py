"""Unit tests for domain entities, invariants, and exceptions."""

from datetime import datetime, timezone
import pytest

from data_agent.core.entities import (
    Artifact,
    Message,
    MessageRole,
    Session,
    TraceStep,
)
from data_agent.core.exceptions import (
    ArtifactNotFoundError,
    DatasetValidationError,
    DomainError,
    SandboxSecurityError,
    SandboxTimeoutError,
    SessionNotFoundError,
)


def test_session_lifecycle():
    """Verify Session aggregate root methods and invariants."""
    session = Session()
    assert session.id is not None
    assert len(session.messages) == 0
    assert len(session.traces) == 0
    assert len(session.artifacts) == 0

    # Add message
    msg = session.add_message(role=MessageRole.USER, content="Compute average profit")
    assert len(session.messages) == 1
    assert msg.role == MessageRole.USER
    assert msg.content == "Compute average profit"
    assert msg.session_id == session.id

    # Add trace step
    trace = session.add_trace_step(
        step_index=1,
        thought="Calculating mean profit column",
        code="print(df['profit'].mean())",
        stdout="254.28\n",
        duration_seconds=0.45,
    )
    assert len(session.traces) == 1
    assert trace.step_index == 1
    assert trace.stdout == "254.28\n"

    # Add artifact
    artifact = session.add_artifact(
        file_name="profit_distribution.html",
        storage_path="session_123/profit_distribution.html",
        metadata={"chart_type": "histogram"},
    )
    assert len(session.artifacts) == 1
    assert artifact.file_name == "profit_distribution.html"
    assert artifact.metadata["chart_type"] == "histogram"


def test_domain_exceptions():
    """Verify custom domain exceptions and attributes."""
    session_err = SessionNotFoundError("sess-404")
    assert "sess-404" in session_err.message
    assert session_err.session_id == "sess-404"

    artifact_err = ArtifactNotFoundError("art-999")
    assert "art-999" in artifact_err.message
    assert artifact_err.artifact_id == "art-999"

    timeout_err = SandboxTimeoutError(15.0)
    assert timeout_err.timeout_seconds == 15.0
    assert "15.0s" in timeout_err.message

    sec_err = SandboxSecurityError("Socket connect blocked")
    assert sec_err.reason == "Socket connect blocked"
