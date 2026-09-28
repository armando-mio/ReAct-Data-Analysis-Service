"""Abstract interface and result models for secure code execution sandbox."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class SandboxExecutionResult:
    """Encapsulates the result of a subprocess sandbox execution run."""
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    generated_html_files: List[Tuple[str, bytes]] = field(default_factory=list)
    timed_out: bool = False
    security_violation: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """True if the code executed cleanly without errors or timeouts."""
        return self.return_code == 0 and not self.timed_out and not self.security_violation


class ISandboxRunner(ABC):
    """Port interface for executing untrusted Python code inside an isolated environment."""

    @abstractmethod
    def execute(
        self,
        code: str,
        dataset_path: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> SandboxExecutionResult:
        """Execute the provided Python code in an isolated subprocess.

        Args:
            code: Python script code to execute.
            dataset_path: Optional path to CSV file to mount/copy into the sandbox.
            timeout: Maximum allowed execution time in seconds.

        Returns:
            SandboxExecutionResult with stdout, stderr, execution duration, and extracted HTML artifacts.

        Raises:
            SandboxTimeoutError: If execution exceeds the allocated timeout.
            SandboxSecurityError: If code attempts forbidden actions like socket network calls.
        """
        pass
