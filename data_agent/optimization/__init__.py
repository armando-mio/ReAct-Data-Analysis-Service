"""DSPy + GEPA prompt optimization package."""

from data_agent.optimization.dev_set import SAMPLE_SALES_PREVIEW, get_dev_set
from data_agent.optimization.metrics import (
    code_and_plot_execution_metric,
    evaluate_execution_score,
)

def get_optimizer_runner():
    from data_agent.optimization.optimizer import PromptOptimizationRunner
    return PromptOptimizationRunner

__all__ = [
    "SAMPLE_SALES_PREVIEW",
    "get_dev_set",
    "code_and_plot_execution_metric",
    "evaluate_execution_score",
    "get_optimizer_runner",
]
