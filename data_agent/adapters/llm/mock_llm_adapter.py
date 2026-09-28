"""Deterministic Mock LLM adapter for offline testing and self-healing verification."""

from typing import Any, Dict, List, Optional
from data_agent.core.entities import Artifact, TraceStep
from data_agent.ports.llm_port import ILLMClient


class MockLLMAdapter(ILLMClient):
    """Deterministic mock implementation of ILLMClient for unit and integration tests."""

    def __init__(
        self,
        scripted_plans: Optional[List[str]] = None,
        scripted_codes: Optional[List[str]] = None,
        scripted_evaluations: Optional[List[Dict[str, Any]]] = None,
        scripted_summaries: Optional[List[str]] = None,
    ) -> None:
        self.scripted_plans = list(scripted_plans or [])
        self.scripted_codes = list(scripted_codes or [])
        self.scripted_evaluations = list(scripted_evaluations or [])
        self.scripted_summaries = list(scripted_summaries or [])

        self.call_history: List[Dict[str, Any]] = []

    def plan(
        self,
        question: str,
        dataset_preview: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Return scripted or default plan."""
        self.call_history.append({"method": "plan", "question": question})
        if self.scripted_plans:
            return self.scripted_plans.pop(0)
        return (
            "1. Inspect the dataset columns.\n"
            "2. Perform data transformation and aggregation.\n"
            "3. Generate an interactive Plotly HTML visualization if appropriate."
        )

    def generate_code(
        self,
        question: str,
        dataset_preview: str,
        plan: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Return scripted code or default code."""
        self.call_history.append({
            "method": "generate_code",
            "question": question,
            "previous_error": previous_error,
            "previous_code": previous_code,
        })
        if self.scripted_codes:
            return self.scripted_codes.pop(0)

        # Default self-healing demonstration code
        if previous_error:
            # Self-healed version
            return (
                "import pandas as pd\n"
                "import plotly.express as px\n"
                "df = pd.read_csv('dataset.csv')\n"
                "print(f'Clean rows count: {len(df)}')\n"
                "fig = px.bar(df.head(5), title='Recovered Analysis')\n"
                "fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
            )

        # Standard default code
        return (
            "import pandas as pd\n"
            "import plotly.express as px\n"
            "df = pd.read_csv('dataset.csv')\n"
            "print(f'Total rows: {len(df)}')\n"
            "fig = px.bar(df.head(5), title='Data Summary')\n"
            "fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
        )

    def reflect_and_evaluate(
        self,
        question: str,
        plan: str,
        code: str,
        stdout: Optional[str],
        stderr: Optional[str],
        has_error: bool,
    ) -> Dict[str, Any]:
        """Return scripted or deterministic reflection evaluation."""
        self.call_history.append({
            "method": "reflect_and_evaluate",
            "has_error": has_error,
            "stderr": stderr,
        })
        if self.scripted_evaluations:
            return self.scripted_evaluations.pop(0)

        if has_error:
            return {
                "is_resolved": False,
                "needs_code_fix": True,
                "feedback": f"Execution failed with error: {stderr}. Fix the issue and retry.",
            }

        return {
            "is_resolved": True,
            "needs_code_fix": False,
            "feedback": "Code executed successfully and answered the user query.",
        }

    def summarize(
        self,
        question: str,
        dataset_preview: str,
        traces: List[TraceStep],
        artifacts: List[Artifact],
    ) -> str:
        """Return scripted or standard summary."""
        self.call_history.append({"method": "summarize", "question": question})
        if self.scripted_summaries:
            return self.scripted_summaries.pop(0)

        artifact_mentions = [a.file_name for a in artifacts]
        return (
            f"Analysis complete for question: '{question}'. "
            f"Executed {len(traces)} reasoning step(s). "
            f"Generated artifacts: {', '.join(artifact_mentions) if artifact_mentions else 'None'}."
        )
