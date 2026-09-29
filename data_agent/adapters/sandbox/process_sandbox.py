"""Process-based execution sandbox with strict timeout and network isolation."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

from data_agent.adapters.sandbox.network_guard import BOOTSTRAP_NETWORK_GUARD, write_network_guard_init
from data_agent.core.exceptions import (
    ExecutionTimeoutError,
    NetworkAccessBlockedError,
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

    def _sanitize_code(self, code: str) -> str:
        """Scan code and neutralize calls to fig.show(), figure.show(), and plt.show() with a no-op."""
        pattern = r"(\b(?:fig|figure|plt|plot)\.show\s*\([^)]*\))"
        return re.sub(pattern, "pass  # fig.show() neutralized by sandbox", code)

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
                import signal
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGKILL)
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

            # 2. Write network guard init scripts into sandbox environment
            guard_path = write_network_guard_init(temp_path)

            # 3. Sanitize user code to neutralize blocking fig.show() calls
            sanitized_code = self._sanitize_code(code)

            # 4. Write script to execute
            script_path = temp_path / "sandbox_script.py"
            # Prepend guard snippet directly as additional safeguard
            full_script_content = f"{BOOTSTRAP_NETWORK_GUARD}\n# --- User Generated Code ---\n{sanitized_code}\n"
            script_path.write_text(full_script_content, encoding="utf-8")

            # 5. Environment configuration enforcing network isolation
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONSTARTUP"] = str(guard_path)
            # Ensure sandbox directory is at the head of PYTHONPATH so sitecustomize is loaded immediately
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{temp_path}{os.pathsep}{current_pythonpath}" if current_pythonpath else str(temp_path)

            # Process group creation flags
            extra_popen_kwargs = {}
            if sys.platform != "win32":
                extra_popen_kwargs["start_new_session"] = True
            else:
                extra_popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

            start_time = time.perf_counter()
            proc = None
            try:
                proc = subprocess.Popen(
                    [
                        self.python_executable,
                        "-c",
                        "import network_guard_init; import runpy, sys; sys.argv = ['sandbox_script.py']; runpy.run_path('sandbox_script.py', run_name='__main__')",
                    ],
                    cwd=str(temp_path),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                    **extra_popen_kwargs,
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

            # Detect network security violations
            security_violation: Optional[str] = None
            if (
                "External network access is blocked by sandbox security policy" in stderr
                or "NetworkAccessBlockedError" in stderr
            ):
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

        finally:
            # Guarantee cleanup of temporary sandbox directory without orphan artifacts on disk
            shutil.rmtree(temp_dir, ignore_errors=True)
