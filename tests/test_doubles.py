"""Test doubles adhering to domain ports for deterministic testing."""

from typing import Any, Dict, List, Optional
from data_agent.core.entities import Artifact, TraceStep
from data_agent.ports.llm_port import ILLMClient


class MockLLMClient(ILLMClient):
    """Deterministic test double for ILLMClient strictly isolated inside the test suite."""

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
        """Return scripted plan or default analytical plan."""
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
        """Return scripted code or deterministic analytical code."""
        self.call_history.append({
            "method": "generate_code",
            "question": question,
            "previous_error": previous_error,
            "previous_code": previous_code,
        })
        if self.scripted_codes:
            return self.scripted_codes.pop(0)

        # Self-healed version upon recovery
        if previous_error:
            return (
                "import pandas as pd\n"
                "import plotly.express as px\n"
                "df = pd.read_csv('dataset.csv')\n"
                "print(f'Clean rows count: {len(df)}')\n"
                "num_cols = df.select_dtypes(include='number').columns\n"
                "val_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]\n"
                "fig = px.bar(df.head(5), y=val_col, title='Recovered Analysis')\n"
                "fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
            )

        cat_col = "Category" if "Category" in dataset_preview else ("category" if "category" in dataset_preview else None)
        val_col = "Revenue" if "Revenue" in dataset_preview else ("sales" if "sales" in dataset_preview else None)

        if cat_col and val_col:
            return (
                "import pandas as pd\n"
                "import plotly.express as px\n"
                "df = pd.read_csv('dataset.csv')\n"
                f"print('=== Total {val_col} by {cat_col} ===')\n"
                f"grouped = df.groupby('{cat_col}')['{val_col}'].sum().reset_index()\n"
                f"grouped = grouped.sort_values(by='{val_col}', ascending=False)\n"
                "print(grouped.to_string(index=False))\n"
                f"top = grouped.iloc[0]\n"
                f"print(f\"\\nTop Performing: {{top['{cat_col}']}} with {{top['{val_col}']:.2f}}\")\n"
                f"fig = px.bar(grouped, x='{cat_col}', y='{val_col}', color='{cat_col}', title='Total {val_col} by {cat_col}')\n"
                "fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
                "print('Successfully generated Plotly chart: output_plot.html')\n"
            )

        return (
            "import pandas as pd\n"
            "import plotly.express as px\n"
            "df = pd.read_csv('dataset.csv')\n"
            "print(f'Total rows: {len(df)}')\n"
            "num_cols = df.select_dtypes(include='number').columns\n"
            "val_col = num_cols[0] if len(num_cols) > 0 else df.columns[0]\n"
            "fig = px.bar(df.head(5), y=val_col, title='Data Summary')\n"
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
        """Return scripted evaluation or deterministic outcome."""
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
        """Return scripted summary or natural language synthesis."""
        self.call_history.append({"method": "summarize", "question": question})
        if self.scripted_summaries:
            return self.scripted_summaries.pop(0)

        artifact_mentions = [a.file_name for a in artifacts]
        return (
            f"Analysis complete for question: '{question}'. "
            f"Executed {len(traces)} reasoning step(s). "
            f"Generated artifacts: {', '.join(artifact_mentions) if artifact_mentions else 'None'}."
        )


# Backward-compatible alias for test files
MockLLMAdapter = MockLLMClient
