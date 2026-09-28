"""DSPy + GEPA prompt optimization package."""

from data_agent.optimization.dev_set import SAMPLE_SALES_PREVIEW, get_dev_set
from data_agent.optimization.metrics import (
    code_and_plot_execution_metric,
    evaluate_execution_score,
)

__all__ = [
    "SAMPLE_SALES_PREVIEW",
    "get_dev_set",
    "code_and_plot_execution_metric",
    "evaluate_execution_score",
]
