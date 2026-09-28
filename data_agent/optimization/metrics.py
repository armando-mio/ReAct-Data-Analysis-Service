"""Evaluation metric for DSPy + GEPA prompt optimization.

Measures whether generated code:
1. Is syntactically valid Python.
2. Executes cleanly within the sandbox (exit code 0, no errors/timeouts).
3. Produces a valid, non-empty Plotly HTML visualization artifact.
4. Outputs meaningful analytical insights to stdout.

Returns ScoreWithFeedback for reflective evolutionary prompt mutation in GEPA.
"""

import ast
import os
from typing import Any, Dict, Optional, Tuple, Union
import dspy
from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

from data_agent.adapters.llm.dspy_modules import clean_code_snippet
from data_agent.adapters.sandbox.process_sandbox import ProcessSandboxRunner
from data_agent.ports.sandbox_port import ISandboxRunner


def evaluate_execution_score(
    code: str,
    dataset_path: str = "data/sample_sales.csv",
    sandbox_runner: Optional[ISandboxRunner] = None,
    expected_keywords: Optional[list] = None,
    timeout: float = 15.0,
) -> Tuple[float, str, Dict[str, Any]]:
    """Execute code in isolated sandbox and compute fine-grained score and diagnostics."""
    cleaned_code = clean_code_snippet(code)

    if not cleaned_code or not cleaned_code.strip():
        return 0.0, "Code generation produced empty or unparseable code.", {
            "syntax_valid": False,
            "executed": False,
            "html_generated": False,
            "stdout_valid": False,
        }

    # 1. Syntax check
    try:
        ast.parse(cleaned_code)
    except SyntaxError as err:
        msg = f"Python SyntaxError: {err.msg} at line {err.lineno}. Model must produce valid Python syntax."
        return 0.15, msg, {
            "syntax_valid": False,
            "executed": False,
            "html_generated": False,
            "stdout_valid": False,
        }

    # 2. Sandbox execution check
    runner = sandbox_runner or ProcessSandboxRunner(default_timeout=timeout)
    # Ensure dataset path exists; fallback to default if relative path needs resolve
    resolved_dataset = dataset_path
    if not os.path.exists(resolved_dataset) and os.path.exists(os.path.join("data", "sample_sales.csv")):
        resolved_dataset = os.path.join("data", "sample_sales.csv")

    try:
        exec_res = runner.execute(code=cleaned_code, dataset_path=resolved_dataset, timeout=timeout)
    except Exception as exc:
        msg = f"Execution invocation failure: {exc}"
        return 0.2, msg, {
            "syntax_valid": True,
            "executed": False,
            "html_generated": False,
            "stdout_valid": False,
        }

    if not exec_res.is_success:
        err_msg = exec_res.stderr or ("Execution timed out" if exec_res.timed_out else "Execution failed")
        msg = (
            f"Code failed at runtime in sandbox with error: {err_msg[:300]}. "
            f"Check column names, imports (import plotly.express as px, pandas as pd), and data types."
        )
        return 0.35, msg, {
            "syntax_valid": True,
            "executed": False,
            "html_generated": False,
            "stdout_valid": False,
            "stderr": exec_res.stderr,
        }

    # 3. Plotly HTML file artifact check
    html_files = exec_res.generated_html_files or []
    valid_plot = False
    plot_name = "none"

    for fname, content in html_files:
        if fname.endswith(".html") and len(content) > 100:
            str_content = content[:2000].decode("utf-8", errors="ignore").lower()
            if "plotly" in str_content or "div" in str_content or "script" in str_content:
                valid_plot = True
                plot_name = fname
                break

    if not valid_plot:
        msg = (
            "Code executed cleanly (exit code 0), but FAILED to generate an interactive Plotly HTML file! "
            "The model must include `fig.write_html('plot.html')` or `fig.write_html('visualization.html')`."
        )
        return 0.65, msg, {
            "syntax_valid": True,
            "executed": True,
            "html_generated": False,
            "stdout_valid": bool(exec_res.stdout),
            "stdout": exec_res.stdout,
        }

    # 4. Analytical relevance & stdout check
    stdout_score = 0.1
    stdout_text = exec_res.stdout or ""
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in stdout_text.lower())
        if matches == 0 and len(stdout_text.strip()) < 10:
            stdout_score = 0.05

    total_score = min(1.0, 0.20 + 0.40 + 0.30 + stdout_score)
    feedback = (
        f"Success (score: {total_score:.2f})! Code executed cleanly with exit code 0, "
        f"generated valid Plotly HTML visualization '{plot_name}', and printed analysis to stdout: "
        f"'{stdout_text.strip()[:100]}...'"
    )

    return total_score, feedback, {
        "syntax_valid": True,
        "executed": True,
        "html_generated": True,
        "stdout_valid": True,
        "plot_name": plot_name,
        "stdout": stdout_text,
    }


def code_and_plot_execution_metric(
    gold: dspy.Example,
    pred: dspy.Prediction,
    trace: Optional[Any] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[Any] = None,
    program_trace: Optional[Any] = None,
    sandbox_runner: Optional[ISandboxRunner] = None,
) -> Union[float, ScoreWithFeedback]:
    """DSPy & GEPA compatible evaluation metric.

    Evaluates execution of generated Python code in the sandbox, verifying Plotly HTML creation,
    runtime success, and output relevance.
    """
    raw_code = getattr(pred, "code", "")
    dataset_path = getattr(gold, "dataset_path", "data/sample_sales.csv")
    expected_keywords = getattr(gold, "expected_keywords", None)

    score, feedback, _ = evaluate_execution_score(
        code=raw_code,
        dataset_path=dataset_path,
        sandbox_runner=sandbox_runner,
        expected_keywords=expected_keywords,
    )

    return ScoreWithFeedback(score=score, feedback=feedback)
