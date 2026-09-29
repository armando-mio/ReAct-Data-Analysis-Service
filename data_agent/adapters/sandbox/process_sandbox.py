"""Process-based execution sandbox with strict timeout and network isolation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

from data_agent.core.exceptions import (
    ExecutionTimeoutError,
    SandboxSecurityError,
    SandboxTimeoutError,
)
from data_agent.ports.sandbox_port import ISandboxRunner, SandboxExecutionResult


class ProcessSandboxRunner(ISandboxRunner):
    """Executes Python code in an isolated subprocess with security guardrails."""

    def __init__(
        self,
        default_timeout: float = 15.0,
        python_executable: Optional[str] = None,
    ) -> None:
        self.default_timeout = default_timeout
        self.python_executable = python_executable or sys.executable

    def _terminate_process_tree(self, proc: subprocess.Popen) -> None:
        """Forcefully terminate the process and its child processes cross-platform."""
        try:
            if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                import signal
                os.killpg(os.getpgid(proc.pid), getattr(signal, "SIGKILL", signal.SIGTERM))
            else:
                proc.kill()
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def execute(
        self,
        code: str,
        dataset_path: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SandboxExecutionResult:
        """Execute Python code within a dedicated temporary directory with strict isolation."""
        effective_timeout = timeout if timeout is not None else self.default_timeout

        temp_dir = tempfile.mkdtemp(prefix="data_agent_sandbox_")
        temp_path = Path(temp_dir).resolve()

        try:
            # 1. Copy dataset into isolated directory if provided
            if dataset_path and Path(dataset_path).is_file():
                src_path = Path(dataset_path).resolve()
                dest_original = temp_path / src_path.name
                shutil.copy2(src_path, dest_original)
                dest_standard = temp_path / "dataset.csv"
                if not dest_standard.exists():
                    shutil.copy2(src_path, dest_standard)

            # 2. Inject security bootstrap header at the top of the user script:
            # - Activates network guard (blocks sockets & HTTP connections)
            # - Neutralizes interactive fig.show() and plt.show() without regex/AST parsing
            bootstrap_header = (
                "import sys\n"
                "from data_agent.adapters.sandbox.network_guard import block_network\n"
                "block_network()\n\n"
                "# Neutralize interactive display methods at runtime without altering user code\n"
                "try:\n"
                "    import plotly.graph_objects as go\n"
                "    go.Figure.show = lambda *args, **kwargs: None\n"
                "except ImportError:\n"
                "    pass\n\n"
                "try:\n"
                "    import matplotlib.pyplot as plt\n"
                "    plt.show = lambda *args, **kwargs: None\n"
                "except ImportError:\n"
                "    pass\n\n"
            )

            script_path = temp_path / "sandbox_script.py"
            full_script_content = bootstrap_header + code
            script_path.write_text(full_script_content, encoding="utf-8")

            # 3. Configure environment: ensure repo root is in PYTHONPATH
            repo_root = Path(__file__).resolve().parents[3]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{repo_root}{os.pathsep}{current_pythonpath}"
                if current_pythonpath
                else str(repo_root)
            )

            start_time = time.perf_counter()
            proc = None
            try:
                proc = subprocess.Popen(
                    [self.python_executable, "sandbox_script.py"],
                    cwd=str(temp_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    start_new_session=True,
                )

                stdout, stderr = proc.communicate(timeout=effective_timeout)
                duration = time.perf_counter() - start_time
                return_code = proc.returncode

            except subprocess.TimeoutExpired:
                if proc:
                    self._terminate_process_tree(proc)
                raise ExecutionTimeoutError(timeout_seconds=effective_timeout)
            except Exception as exc:
                if proc:
                    self._terminate_process_tree(proc)
                raise exc

            # 4. Detect network security violations in stderr
            security_violation: Optional[str] = None
            if (
                "NetworkAccessBlockedError" in stderr
                or "External network access is prohibited in this sandbox" in stderr
                or "External network access is blocked" in stderr
            ):
                security_violation = "External network access is prohibited in this sandbox."

            # 5. Collect generated HTML artifacts (Plotly visualizations) if any
            # Optional: returns empty list if no HTML artifact was generated
            generated_html: List[Tuple[str, bytes]] = []
            for file_path in temp_path.glob("*.html"):
                try:
                    generated_html.append((file_path.name, file_path.read_bytes()))
                except Exception:
                    pass

            return SandboxExecutionResult(
                stdout=stdout or "",
                stderr=stderr or "",
                return_code=return_code,
                duration_seconds=duration,
                generated_html_files=generated_html,
                timed_out=False,
                security_violation=security_violation,
            )

        finally:
            # Guarantee cleanup of temporary sandbox directory without orphan artifacts on disk
            shutil.rmtree(temp_dir, ignore_errors=True)
