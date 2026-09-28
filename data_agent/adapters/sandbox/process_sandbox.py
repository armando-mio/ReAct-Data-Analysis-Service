"""Process-based execution sandbox with strict timeout and network isolation."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

from data_agent.adapters.sandbox.network_guard import BOOTSTRAP_NETWORK_GUARD
from data_agent.core.exceptions import SandboxSecurityError, SandboxTimeoutError
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
        """Forcefully terminate a process and any child processes it spawned."""
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    import psutil
                    parent = psutil.Process(proc.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except Exception:
                    proc.kill()
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
        """Execute Python code within a dedicated temporary directory."""
        effective_timeout = timeout if timeout is not None else self.default_timeout

        with tempfile.TemporaryDirectory(prefix="data_agent_sandbox_") as temp_dir:
            temp_path = Path(temp_dir).resolve()

            # Copy dataset into isolated directory if provided
            if dataset_path and Path(dataset_path).is_file():
                src_path = Path(dataset_path).resolve()
                dest_original = temp_path / src_path.name
                shutil.copy2(src_path, dest_original)
                # Also provide standardized alias 'dataset.csv' for convenience
                dest_standard = temp_path / "dataset.csv"
                if not dest_standard.exists():
                    shutil.copy2(src_path, dest_standard)

            # Build bootstrap script with network isolation prepended
            script_content = f"{BOOTSTRAP_NETWORK_GUARD}\n# --- User Generated Code ---\n{code}\n"
            script_path = temp_path / "sandbox_script.py"
            script_path.write_text(script_content, encoding="utf-8")

            # Environment variables for sandbox execution
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            start_time = time.perf_counter()
            proc = None
            try:
                proc = subprocess.Popen(
                    [self.python_executable, str(script_path)],
                    cwd=str(temp_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )

                stdout, stderr = proc.communicate(timeout=effective_timeout)
                duration = time.perf_counter() - start_time
                return_code = proc.returncode

            except subprocess.TimeoutExpired:
                if proc:
                    self._terminate_process_tree(proc)
                raise SandboxTimeoutError(timeout_seconds=effective_timeout)
            except Exception as exc:
                if proc:
                    self._terminate_process_tree(proc)
                raise exc

            # Detect any network permission errors raised by security guard
            security_violation: Optional[str] = None
            if "External network access is blocked by sandbox security policy" in stderr:
                security_violation = "External network access blocked by sandbox security policy."

            # Collect generated HTML artifacts (Plotly visualizations)
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
