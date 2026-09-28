"""Abstract interface for LLM client providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from data_agent.core.entities import Artifact, TraceStep


class ILLMClient(ABC):
    """Port interface for Large Language Model communication and generation."""

    @abstractmethod
    def plan(
        self,
        question: str,
        dataset_preview: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Formulate a step-by-step reasoning plan given the user query and dataset schema."""
        pass

    @abstractmethod
    def generate_code(
        self,
        question: str,
        dataset_preview: str,
        plan: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Generate executable Python code using pandas, numpy, and plotly based on the plan."""
        pass

    @abstractmethod
    def reflect_and_evaluate(
        self,
        question: str,
        plan: str,
        code: str,
        stdout: Optional[str],
        stderr: Optional[str],
        has_error: bool,
    ) -> Dict[str, Any]:
        """Evaluate execution output, determining whether the answer is resolved or requires recovery.

        Returns a dictionary containing:
        - "is_resolved": bool
        - "needs_code_fix": bool
        - "feedback": str
        """
        pass

    @abstractmethod
    def summarize(
        self,
        question: str,
        dataset_preview: str,
        traces: List[TraceStep],
        artifacts: List[Artifact],
    ) -> str:
        """Formulate a comprehensive natural language conclusion synthesizing all findings and plots."""
        pass
