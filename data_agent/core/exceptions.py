"""Custom domain exceptions for the ReAct Data Analysis service.

Adheres strictly to Hexagonal Architecture boundaries: no framework or database dependencies.
"""

from typing import Optional


class DomainError(Exception):
    """Base exception for all domain-specific errors."""

    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class EntityNotFoundError(DomainError):
    """Raised when an expected domain entity does not exist."""
    pass


class SessionNotFoundError(EntityNotFoundError):
    """Raised when an analysis session cannot be located."""

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session with ID '{session_id}' not found.", {"session_id": session_id})
        self.session_id = session_id


class ArtifactNotFoundError(EntityNotFoundError):
    """Raised when a requested artifact does not exist."""

    def __init__(self, artifact_id: str) -> None:
        super().__init__(f"Artifact with ID '{artifact_id}' not found.", {"artifact_id": artifact_id})
        self.artifact_id = artifact_id


class DatasetValidationError(DomainError):
    """Raised when an uploaded dataset fails validation (unsupported format, corrupt, empty)."""
    pass


class SandboxExecutionError(DomainError):
    """Raised when the execution of sandbox code fails critically."""
    pass


class SandboxTimeoutError(SandboxExecutionError):
    """Raised when sandbox execution exceeds the configured timeout threshold."""

    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            f"Execution timed out after exceeding safety threshold of {timeout_seconds}s.",
            {"timeout_seconds": timeout_seconds},
        )
        self.timeout_seconds = timeout_seconds


class SandboxSecurityError(SandboxExecutionError):
    """Raised when code violates sandbox security policy (e.g. unauthorized network access)."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Sandbox security violation: {reason}", {"reason": reason})
        self.reason = reason


class LLMExecutionError(DomainError):
    """Raised when communication or parsing with the LLM provider fails."""
    pass


class MaxIterationsReachedError(DomainError):
    """Raised when the ReAct agent reaches maximum reflection iterations without convergence."""

    def __init__(self, max_iterations: int) -> None:
        super().__init__(
            f"Agent reached maximum limit of {max_iterations} iterations without resolving the query.",
            {"max_iterations": max_iterations},
        )
        self.max_iterations = max_iterations
