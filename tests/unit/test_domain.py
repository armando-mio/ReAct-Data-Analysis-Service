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


def test_extract_python_code_standard_fenced():
    """Verify extraction from standard ```python ... ``` markdown blocks."""
    from data_agent.core.utils import extract_python_code
    raw = (
        "Here is the solution:\n"
        "```python\n"
        "import pandas as pd\n"
        "df = pd.read_csv('dataset.csv')\n"
        "print(df.shape)\n"
        "```\n"
        "Hope this helps!"
    )
    code = extract_python_code(raw)
    assert code == "import pandas as pd\ndf = pd.read_csv('dataset.csv')\nprint(df.shape)"


def test_extract_python_code_generic_fenced():
    """Verify extraction from generic ``` ... ``` blocks without language tag."""
    from data_agent.core.utils import extract_python_code
    raw = (
        "```\n"
        "import numpy as np\n"
        "arr = np.array([1, 2, 3])\n"
        "print(arr.sum())\n"
        "```"
    )
    code = extract_python_code(raw)
    assert code == "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.sum())"


def test_extract_python_code_unclosed_backticks():
    """Verify extraction when LLM output is truncated and lacks closing backticks."""
    from data_agent.core.utils import extract_python_code
    raw = (
        "```python\n"
        "import pandas as pd\n"
        "import plotly.express as px\n"
        "df = pd.read_csv('dataset.csv')\n"
        "fig = px.bar(df, x='category', y='sales')\n"
        "fig.write_html('output.html', include_plotlyjs='cdn')"
    )
    code = extract_python_code(raw)
    assert "import pandas as pd" in code
    assert "fig.write_html('output.html', include_plotlyjs='cdn')" in code
    assert "```" not in code


def test_extract_python_code_conversational_no_backticks():
    """Verify extraction when code is presented conversationally without any markdown backticks."""
    from data_agent.core.utils import extract_python_code, clean_code_snippet
    raw = (
        "Sure, here is the analytical script to process your query:\n"
        "import pandas as pd\n"
        "df = pd.read_csv('dataset.csv')\n"
        "summary = df.groupby('category')['revenue'].sum()\n"
        "print(summary)\n"
        "Note: make sure plotly is installed."
    )
    code = extract_python_code(raw)
    assert "import pandas as pd" in code
    assert "print(summary)" in code
    assert "Sure, here is" not in code
    assert "Note: make sure" not in code

    # Also verify backward-compatible clean_code_snippet alias
    alias_code = clean_code_snippet(raw)
    assert alias_code == code


def test_structured_json_logging():
    """Verify StructuredJSONFormatter serializes records and contextual audit fields."""
    import json
    import logging
    from data_agent.core.logging import StructuredJSONFormatter

    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_path.py",
        lineno=42,
        msg="Executing step",
        args=(),
        exc_info=None,
    )
    record.session_id = "test-session-123"
    record.step_index = 2
    record.action_type = "sandbox_execution"
    record.execution_time_ms = 145.8

    output = formatter.format(record)
    parsed = json.loads(output)

    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "test_logger"
    assert parsed["message"] == "Executing step"
    assert parsed["session_id"] == "test-session-123"
    assert parsed["step_index"] == 2
    assert parsed["action_type"] == "sandbox_execution"
    assert parsed["execution_time_ms"] == 145.8
    assert "timestamp" in parsed

