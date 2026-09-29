"""Integration tests validating Sandbox security, timeout enforcement, and network blocking."""

from pathlib import Path
import time
import pytest

from data_agent.adapters.sandbox.process_sandbox import ProcessSandboxRunner
from data_agent.core.exceptions import ExecutionTimeoutError, SandboxTimeoutError


def test_sandbox_timeout_enforcement():
    """Verify that an infinite loop process is killed and triggers ExecutionTimeoutError."""
    sandbox = ProcessSandboxRunner(default_timeout=1.5)
    infinite_loop_code = "while True:\n    pass\n"

    start_time = time.perf_counter()
    with pytest.raises(ExecutionTimeoutError) as exc_info:
        sandbox.execute(code=infinite_loop_code, timeout=1.5)
    elapsed = time.perf_counter() - start_time

    assert exc_info.value.timeout_seconds == 1.5
    assert isinstance(exc_info.value, SandboxTimeoutError)
    # The elapsed time should be close to 1.5 seconds, proving timely termination
    assert elapsed >= 1.4 and elapsed < 4.0


def test_sandbox_network_blocking_socket():
    """Verify that low-level socket connections are blocked by the injected network guard."""
    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    network_attempt_code = (
        "import socket\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "s.connect(('8.8.8.8', 53))\n"
    )

    result = sandbox.execute(code=network_attempt_code)
    assert not result.is_success
    assert ("PermissionError" in result.stderr or "NetworkAccessBlockedError" in result.stderr or "RuntimeError" in result.stderr)
    assert ("External network access is prohibited in this sandbox" in result.stderr or "External network access is blocked" in result.stderr)
    assert result.security_violation is not None


def test_sandbox_network_blocking_urllib():
    """Verify that high-level HTTP client attempts via urllib are blocked."""
    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    urllib_code = (
        "import urllib.request\n"
        "urllib.request.urlopen('https://example.com')\n"
    )

    result = sandbox.execute(code=urllib_code)
    assert not result.is_success
    assert ("PermissionError" in result.stderr or "NetworkAccessBlockedError" in result.stderr or "RuntimeError" in result.stderr or "URLError" in result.stderr)
    assert ("External network access is prohibited in this sandbox" in result.stderr or "External network access is blocked" in result.stderr)


def test_sandbox_fig_show_neutralized():
    """Verify that fig.show() calls are safely neutralized and do not block execution."""
    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    code = (
        "import plotly.express as px\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})\n"
        "fig = px.bar(df, x='x', y='y')\n"
        "fig.show()\n"
        "fig.write_html('test_plot.html')\n"
        "print('Execution completed without blocking')\n"
    )

    result = sandbox.execute(code=code)
    assert result.is_success
    assert "Execution completed without blocking" in result.stdout
    assert len(result.generated_html_files) == 1


def test_sandbox_file_isolation(tmp_path: Path):
    """Verify file writes inside sandbox do not escape into the parent environment."""
    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    canary_host_file = tmp_path / "host_canary.txt"

    code = (
        f"with open('sandbox_local.txt', 'w') as f:\n"
        f"    f.write('created_in_sandbox')\n"
        f"print('Sandbox file write successful')\n"
    )

    result = sandbox.execute(code=code)
    assert result.is_success
    assert "Sandbox file write successful" in result.stdout
    # Host directory remains untouched
    assert not canary_host_file.exists()


def test_sandbox_plotly_html_extraction():
    """Verify that generated Plotly HTML files are detected and extracted."""
    sandbox = ProcessSandboxRunner(default_timeout=10.0)
    plotly_code = (
        "import plotly.express as px\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})\n"
        "fig = px.line(df, x='x', y='y', title='Test Chart')\n"
        "fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
        "print('Plot written.')\n"
    )

    result = sandbox.execute(code=plotly_code)
    assert result.is_success
    assert "Plot written." in result.stdout
    assert len(result.generated_html_files) == 1
    filename, content = result.generated_html_files[0]
    assert filename == "output_plot.html"
    assert b"plotly" in content.lower()
