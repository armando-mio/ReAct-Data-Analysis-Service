"""Unit tests for the ReAct agent state machine and self-healing recovery loop."""

from pathlib import Path
import pytest

from tests.test_doubles import MockLLMAdapter
from data_agent.adapters.sandbox.process_sandbox import ProcessSandboxRunner
from data_agent.use_cases.react_graph import ReActGraphBuilder


def test_react_loop_successful_execution(sample_csv_file: Path):
    """Test standard single-turn plan -> act -> observe -> finalize loop."""
    mock_llm = MockLLMAdapter(
        scripted_plans=["Plan: Calculate total sales and output to stdout."],
        scripted_codes=[
            "import pandas as pd\n"
            "df = pd.read_csv('dataset.csv')\n"
            "print(f'Total Sales: {df[\"sales\"].sum()}')\n"
        ],
        scripted_evaluations=[
            {"is_resolved": True, "needs_code_fix": False, "feedback": "Total sales calculated."}
        ],
        scripted_summaries=["Total sales across all regions amounts to $6,540.70."],
    )
    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    builder = ReActGraphBuilder(llm_client=mock_llm, sandbox_runner=sandbox)
    graph = builder.build()

    state = {
        "session_id": "test-session-1",
        "question": "What is the total sales amount?",
        "dataset_path": str(sample_csv_file),
        "dataset_preview": "region,category,sales...",
        "iteration": 0,
        "max_iterations": 4,
        "trace": [],
        "artifact_ids": [],
        "artifacts": [],
        "current_plan": None,
        "current_code": None,
        "execution_output": None,
        "execution_error": None,
        "final_answer": None,
        "generated_html_files": [],
        "is_resolved": False,
        "needs_code_fix": False,
        "feedback": None,
    }

    final_state = graph.invoke(state)

    assert final_state["is_resolved"] is True
    assert final_state["iteration"] == 1
    assert len(final_state["trace"]) == 1
    assert "Total Sales: 6540.7" in final_state["trace"][0].stdout
    assert "6,540.70" in final_state["final_answer"]


def test_react_loop_self_healing_recovery(sample_csv_file: Path):
    """Test self-healing: Step 1 fails with KeyError, Step 2 corrects code and succeeds."""
    mock_llm = MockLLMAdapter(
        scripted_plans=[
            "Step 1 Plan: Compute revenue by reading non-existent column 'revenue'.",
            "Step 2 Plan: Fix column name to 'sales' and recompute.",
        ],
        scripted_codes=[
            # Step 1: Intentionally buggy code that causes KeyError
            (
                "import pandas as pd\n"
                "df = pd.read_csv('dataset.csv')\n"
                "print(df['non_existent_revenue'].sum())\n"
            ),
            # Step 2: Self-healed code targeting correct column 'sales'
            (
                "import pandas as pd\n"
                "df = pd.read_csv('dataset.csv')\n"
                "print(f'Recovered Sales: {df[\"sales\"].sum()}')\n"
            ),
        ],
        scripted_evaluations=[
            # Step 1 evaluation: Error detected, needs code fix
            {
                "is_resolved": False,
                "needs_code_fix": True,
                "feedback": "KeyError: 'non_existent_revenue' column does not exist. Use 'sales'.",
            },
            # Step 2 evaluation: Successfully recovered
            {
                "is_resolved": True,
                "needs_code_fix": False,
                "feedback": "Execution succeeded on retry.",
            },
        ],
        scripted_summaries=[
            "After correcting the column name, total sales were calculated as $6,540.70."
        ],
    )

    sandbox = ProcessSandboxRunner(default_timeout=5.0)
    builder = ReActGraphBuilder(llm_client=mock_llm, sandbox_runner=sandbox)
    graph = builder.build()

    state = {
        "session_id": "test-self-heal-session",
        "question": "What is the total revenue?",
        "dataset_path": str(sample_csv_file),
        "dataset_preview": "region,category,sales,units,profit...",
        "iteration": 0,
        "max_iterations": 4,
        "trace": [],
        "artifact_ids": [],
        "artifacts": [],
        "current_plan": None,
        "current_code": None,
        "execution_output": None,
        "execution_error": None,
        "final_answer": None,
        "generated_html_files": [],
        "is_resolved": False,
        "needs_code_fix": False,
        "feedback": None,
    }

    final_state = graph.invoke(state)

    # Assertions on self-healing behavior
    assert final_state["is_resolved"] is True
    assert final_state["iteration"] == 2
    assert len(final_state["trace"]) == 2

    # Step 1 had KeyError in stderr
    assert "KeyError" in final_state["trace"][0].stderr

    # Step 2 successfully executed
    assert "Recovered Sales: 6540.7" in final_state["trace"][1].stdout
    assert "After correcting the column name" in final_state["final_answer"]
