"""Integration tests for DSPy + GEPA prompt optimization, dev set, and metrics."""

import os
from pathlib import Path
import pytest
import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from data_agent.optimization.dev_set import DEV_QUESTIONS, get_dev_set
from data_agent.optimization.metrics import (
    code_and_plot_execution_metric,
    evaluate_execution_score,
)
from data_agent.optimization.optimizer import PromptOptimizationRunner


def test_dev_set_construction():
    """Verify the curated development set properties and schema."""
    dev_set = get_dev_set(dataset_path="data/sample_sales.csv")
    assert len(dev_set) == len(DEV_QUESTIONS)
    assert len(dev_set) == 5

    for ex in dev_set:
        assert hasattr(ex, "question")
        assert hasattr(ex, "dataset_preview")
        assert hasattr(ex, "expected_keywords")
        assert "Date" in ex.dataset_preview or "Category" in ex.dataset_preview
        assert ex.dataset_path == "data/sample_sales.csv"


def test_evaluate_execution_score_success():
    """Verify that valid code producing Plotly HTML receives full credit (1.0)."""
    valid_code = """
import pandas as pd
import plotly.express as px

df = pd.read_csv('dataset.csv')
summary = df.groupby('Category')['Revenue'].sum().reset_index()
print('Total Revenue:')
print(summary)

fig = px.bar(summary, x='Category', y='Revenue', title='Category Revenue')
fig.write_html('plot.html')
"""
    score, feedback, diag = evaluate_execution_score(
        code=valid_code,
        dataset_path="data/sample_sales.csv",
        expected_keywords=["Revenue", "Category"],
    )
    assert score == 1.0
    assert diag["syntax_valid"] is True
    assert diag["executed"] is True
    assert diag["html_generated"] is True
    assert "plot.html" in diag["plot_name"]
    assert "Success" in feedback


def test_evaluate_execution_score_missing_plotly_html():
    """Verify that code executing without errors but missing Plotly HTML receives a penalty (0.65)."""
    code_no_html = """
import pandas as pd
df = pd.read_csv('dataset.csv')
print(df.groupby('Category')['Revenue'].sum())
"""
    score, feedback, diag = evaluate_execution_score(
        code=code_no_html,
        dataset_path="data/sample_sales.csv",
    )
    assert score == 0.65
    assert diag["syntax_valid"] is True
    assert diag["executed"] is True
    assert diag["html_generated"] is False
    assert "FAILED to generate an interactive Plotly HTML" in feedback


def test_evaluate_execution_score_syntax_error():
    """Verify syntax error handling and score penalty."""
    syntax_error_code = "def invalid_syntax(:"
    score, feedback, diag = evaluate_execution_score(
        code=syntax_error_code,
        dataset_path="data/sample_sales.csv",
    )
    assert score == 0.15
    assert diag["syntax_valid"] is False
    assert "SyntaxError" in feedback


def test_evaluate_execution_score_runtime_error():
    """Verify runtime exception capture and diagnostic feedback."""
    runtime_error_code = """
import pandas as pd
df = pd.read_csv('dataset.csv')
print(df['NonExistentColumn'])
"""
    score, feedback, diag = evaluate_execution_score(
        code=runtime_error_code,
        dataset_path="data/sample_sales.csv",
    )
    assert score == 0.35
    assert diag["syntax_valid"] is True
    assert diag["executed"] is False
    assert "KeyError" in feedback or "NonExistentColumn" in feedback


def test_code_and_plot_execution_metric_gepa_protocol():
    """Verify metric returns ScoreWithFeedback compliant with GEPA optimizer."""
    valid_code = """
import pandas as pd
import plotly.express as px
df = pd.read_csv('dataset.csv')
fig = px.bar(df, x='Category', y='Revenue')
fig.write_html('plot.html')
print('Done')
"""
    gold = dspy.Example(
        question="Plot revenue",
        dataset_path="data/sample_sales.csv",
        expected_keywords=["Done"],
    )
    pred = dspy.Prediction(code=valid_code)

    result = code_and_plot_execution_metric(gold=gold, pred=pred)
    assert isinstance(result, ScoreWithFeedback)
    assert result.score == 1.0
    assert "Success" in result.feedback


def test_prompt_optimization_runner_end_to_end_mock(tmp_path):
    """Verify that PromptOptimizationRunner completes the full before/after GEPA pipeline."""
    runner = PromptOptimizationRunner(use_mock=True)
    report = runner.run_optimization(
        max_metric_calls=4,
        dataset_path="data/sample_sales.csv",
    )

    assert report.dev_set_size == 5
    assert report.baseline_avg_score == 0.65
    assert report.optimized_avg_score == 1.0
    assert report.relative_improvement_pct > 50.0
    assert len(report.mutations) >= 2

    # Verify mutations contain prompt changes
    changed = [m for m in report.mutations if m["has_changed"]]
    assert len(changed) >= 2
    coder_mutation = next(m for m in changed if m["predictor_name"] == "coder.predict")
    assert "Plotly visualization" in coder_mutation["optimized_instructions"]


def test_llm_judge_answer_quality():
    """Verify optional LLM-as-a-judge answer quality evaluation."""
    from data_agent.optimization.metrics import llm_judge_answer_quality

    # 1. Non-empty relevant answer
    score, critique = llm_judge_answer_quality(
        question="Which category has highest revenue?",
        stdout_data="Electronics: 14508.62, Clothing: 7767.26",
        answer="Electronics generated the highest total revenue of $14,508.62.",
    )
    assert score > 0.0
    assert isinstance(critique, str)

    # 2. Empty answer receives 0.0
    score_empty, critique_empty = llm_judge_answer_quality(
        question="Which category has highest revenue?",
        stdout_data="Electronics: 14508.62",
        answer="",
    )
    assert score_empty == 0.0
    assert "missing" in critique_empty.lower()


def test_prompt_optimization_runner_optimizer_choice():
    """Verify PromptOptimizationRunner supports optimizer selection (gepa, miprov2, bootstrap)."""
    runner = PromptOptimizationRunner(use_mock=True)
    report_bootstrap = runner.run_optimization(
        max_metric_calls=2,
        dataset_path="data/sample_sales.csv",
        optimizer_type="bootstrap",
    )
    assert "BOOTSTRAP" in report_bootstrap.optimizer_name
    assert report_bootstrap.optimized_avg_score == 1.0
