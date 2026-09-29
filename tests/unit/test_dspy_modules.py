"""Unit tests for DSPy reasoning signatures, ReAct module, and adapter."""

import dspy
import pytest

from data_agent.adapters.llm.dspy_modules import (
    CodeGenerationSignature,
    DataAnalysisReActModule,
    DSPyLLMAdapter,
    PlanSignature,
    ReflectionSignature,
    SummarizeSignature,
    clean_code_snippet,
)
from data_agent.core.entities import Artifact, TraceStep
from data_agent.ports.llm_port import ILLMClient


def test_clean_code_snippet():
    """Verify robust code block extraction and sanitation."""
    # 1. Standard markdown python fence
    snippet1 = "```python\nimport pandas as pd\nprint(123)\n```"
    assert clean_code_snippet(snippet1) == "import pandas as pd\nprint(123)"

    # 2. Generic fence without language tag
    snippet2 = "```\nimport numpy as np\n```"
    assert clean_code_snippet(snippet2) == "import numpy as np"

    # 3. Raw Python code without fences
    snippet3 = "import plotly.express as px\nfig = px.bar()"
    assert clean_code_snippet(snippet3) == snippet3

    # 4. Text with explanations followed by code block
    snippet4 = "Here is the code you asked for:\n```python\nx = 1\n```\nHope it helps!"
    assert clean_code_snippet(snippet4) == "x = 1"


def test_dspy_signatures_structure():
    """Verify DSPy signature input and output field definitions."""
    # PlanSignature
    assert "question" in PlanSignature.input_fields
    assert "dataset_preview" in PlanSignature.input_fields
    assert "plan" in PlanSignature.output_fields

    # CodeGenerationSignature
    assert "question" in CodeGenerationSignature.input_fields
    assert "dataset_preview" in CodeGenerationSignature.input_fields
    assert "plan" in CodeGenerationSignature.input_fields
    assert "previous_error" in CodeGenerationSignature.input_fields
    assert "previous_code" in CodeGenerationSignature.input_fields
    assert "code" in CodeGenerationSignature.output_fields

    # ReflectionSignature
    assert "stdout" in ReflectionSignature.input_fields
    assert "stderr" in ReflectionSignature.input_fields
    assert "is_resolved" in ReflectionSignature.output_fields
    assert "needs_code_fix" in ReflectionSignature.output_fields
    assert "feedback" in ReflectionSignature.output_fields

    # SummarizeSignature
    assert "execution_summary" in SummarizeSignature.input_fields
    assert "summary" in SummarizeSignature.output_fields


def test_dspy_react_module_predictors():
    """Verify that DataAnalysisReActModule registers predictors discoverable by GEPA."""
    module = DataAnalysisReActModule()
    predictors = dict(module.named_predictors())

    assert "planner.predict" in predictors
    assert "coder.predict" in predictors
    assert "reflector" in predictors
    assert "summarizer" in predictors


def test_dspy_llm_adapter_implements_port():
    """Verify DSPyLLMAdapter implements ILLMClient and coordinates reasoning steps."""
    class DummyPredictor:
        def __init__(self, output_dict):
            self.output_dict = output_dict

        def __call__(self, **kwargs):
            return dspy.Prediction(**self.output_dict)

    module = DataAnalysisReActModule()
    module.planner = DummyPredictor({"plan": "1. Group by category. 2. Plotly bar chart."})
    module.coder = DummyPredictor({"code": "```python\nimport pandas as pd\nprint('ok')\n```"})
    module.reflector = DummyPredictor({
        "is_resolved": True,
        "needs_code_fix": False,
        "feedback": "Execution succeeded cleanly.",
    })
    module.summarizer = DummyPredictor({"summary": "Final synthesized analysis answer."})

    adapter = DSPyLLMAdapter(module=module)
    assert isinstance(adapter, ILLMClient)

    # 1. Plan
    plan_out = adapter.plan(question="What is total revenue?", dataset_preview="col1,col2")
    assert "Plotly bar chart" in plan_out

    # 2. Generate code
    code_out = adapter.generate_code(
        question="What is total revenue?",
        dataset_preview="col1,col2",
        plan=plan_out,
    )
    assert code_out == "import pandas as pd\nprint('ok')"

    # 3. Reflect and evaluate
    eval_out = adapter.reflect_and_evaluate(
        question="What is total revenue?",
        plan=plan_out,
        code=code_out,
        stdout="Total: 100",
        stderr=None,
        has_error=False,
    )
    assert eval_out["is_resolved"] is True
    assert eval_out["needs_code_fix"] is False

    # 4. Summarize
    trace = TraceStep(
        session_id="s1",
        step_index=1,
        thought=plan_out,
        code=code_out,
        stdout="Total: 100",
        stderr=None,
        duration_seconds=0.5,
    )
    summary_out = adapter.summarize(
        question="What is total revenue?",
        dataset_preview="col1,col2",
        traces=[trace],
        artifacts=[Artifact(session_id="s1", file_name="plot.html", storage_path="/path")],
    )
    assert summary_out == "Final synthesized analysis answer."


def test_dspy_react_agent_instantiation():
    """Verify DSPyReActAgent properly configures built-in dspy.ReAct and its execution tools."""
    from data_agent.adapters.llm.dspy_modules import DSPyReActAgent

    agent = DSPyReActAgent(dataset_path="data/sample_sales.csv")
    assert hasattr(agent, "react")
    assert isinstance(agent.react, dspy.ReAct)
    assert "execute_analysis_code" in agent.react.tools


def test_container_requires_api_key(monkeypatch):
    """Verify Container raises ValueError when GEMINI_API_KEY is missing or empty."""
    from data_agent.adapters.api.dependencies import Container

    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        Container(gemini_api_key="", gemini_model="test-model")


def test_container_requires_model_name(monkeypatch):
    """Verify Container raises ValueError when GEMINI_MODEL is missing or empty."""
    from data_agent.adapters.api.dependencies import Container

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "")
    with pytest.raises(ValueError, match="GEMINI_MODEL environment variable is required"):
        Container(gemini_api_key="test-key", gemini_model="")

