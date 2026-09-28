"""DSPy reasoning signatures, modules, and adapter for prompt optimization."""

import os
import re
from typing import Any, Dict, List, Optional

import dspy
from data_agent.core.entities import Artifact, TraceStep
from data_agent.core.exceptions import LLMExecutionError
from data_agent.ports.llm_port import ILLMClient


def clean_code_snippet(raw_text: str) -> str:
    """Extract and sanitize raw Python code from markdown blocks or backticks."""
    text = (raw_text or "").strip()
    pattern = r"```(?:python)?\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


class PlanSignature(dspy.Signature):
    """Analyze the tabular dataset preview and user query, formulating a step-by-step analytical plan including metrics and Plotly visualization design."""

    question: str = dspy.InputField(desc="User analytical question")
    dataset_preview: str = dspy.InputField(desc="Preview, shape, and column data types of the tabular dataset")
    plan: str = dspy.OutputField(desc="Strategic step-by-step plan for calculations and Plotly visualization")


class CodeGenerationSignature(dspy.Signature):
    """Generate standalone, runnable Python code using pandas, numpy, and plotly.
    The script must load the dataset from 'dataset.csv' (or path provided in scope),
    compute all requested aggregations and metrics, print clear insights to stdout,
    and create an interactive Plotly visualization saved to an HTML file via fig.write_html('plot.html').
    Never use plt.show() or input(). Avoid syntax errors and handle missing data gracefully.
    """

    question: str = dspy.InputField(desc="Analytical question to answer")
    dataset_preview: str = dspy.InputField(desc="Dataset column names, types, and sample records")
    plan: str = dspy.InputField(desc="Execution plan detailing data processing and chart selection")
    previous_error: str = dspy.InputField(desc="Error message or traceback from previous execution attempt, or 'None'")
    previous_code: str = dspy.InputField(desc="Code from previous execution attempt that failed, or 'None'")
    code: str = dspy.OutputField(desc="Executable Python code without markdown wrappers")


class ReflectionSignature(dspy.Signature):
    """Observe execution results (stdout and stderr) and evaluate whether the analysis succeeded and generated a Plotly HTML artifact."""

    question: str = dspy.InputField(desc="Original analytical question")
    plan: str = dspy.InputField(desc="Execution plan")
    code: str = dspy.InputField(desc="Executed Python code")
    stdout: str = dspy.InputField(desc="Standard output produced during sandbox execution")
    stderr: str = dspy.InputField(desc="Standard error or exception traceback, if any")
    is_resolved: bool = dspy.OutputField(desc="True if the execution completed with exit code 0 and answered the query")
    needs_code_fix: bool = dspy.OutputField(desc="True if an error occurred or output/plot is missing")
    feedback: str = dspy.OutputField(desc="Diagnostic feedback on what succeeded or how to correct the code")


class SummarizeSignature(dspy.Signature):
    """Synthesize execution outputs, metrics, and generated Plotly artifacts into an analytical executive summary answering the user query."""

    question: str = dspy.InputField(desc="User analytical query")
    dataset_preview: str = dspy.InputField(desc="Dataset preview and schema")
    execution_summary: str = dspy.InputField(desc="Stdout output and computed findings from code execution")
    artifacts: str = dspy.InputField(desc="List of generated HTML visualization artifacts")
    summary: str = dspy.OutputField(desc="Executive summary interpreting data findings and referencing the visualization")


class DataAnalysisReActModule(dspy.Module):
    """DSPy module implementing the ReAct reasoning pipeline for data analysis and visualization."""

    def __init__(self) -> None:
        super().__init__()
        self.planner = dspy.ChainOfThought(PlanSignature)
        self.coder = dspy.ChainOfThought(CodeGenerationSignature)
        self.reflector = dspy.Predict(ReflectionSignature)
        self.summarizer = dspy.Predict(SummarizeSignature)

    def forward(
        self,
        question: str,
        dataset_preview: str,
        sandbox_runner: Any = None,
        dataset_path: Optional[str] = None,
        max_iterations: int = 2,
    ) -> dspy.Prediction:
        """Execute reasoning and code generation, optionally executing in sandbox."""
        plan_pred = self.planner(question=question, dataset_preview=dataset_preview)
        plan = getattr(plan_pred, "plan", "")

        prev_error = "None"
        prev_code = "None"
        last_stdout = ""
        last_stderr = ""
        generated_html_files = []
        is_success = False

        for _ in range(max_iterations):
            code_pred = self.coder(
                question=question,
                dataset_preview=dataset_preview,
                plan=plan,
                previous_error=prev_error,
                previous_code=prev_code,
            )
            raw_code = getattr(code_pred, "code", "")
            code = clean_code_snippet(raw_code)

            if sandbox_runner is not None and dataset_path is not None:
                res = sandbox_runner.execute(code=code, dataset_path=dataset_path)
                last_stdout = res.stdout
                last_stderr = res.stderr
                generated_html_files = res.generated_html_files
                is_success = res.is_success and bool(generated_html_files)

                if is_success:
                    break
                else:
                    prev_error = res.stderr or "Plotly HTML file was not generated."
                    prev_code = code
            else:
                is_success = True
                break

        art_names = [f for f, _ in generated_html_files]
        sum_pred = self.summarizer(
            question=question,
            dataset_preview=dataset_preview,
            execution_summary=last_stdout or "Analysis generated.",
            artifacts=", ".join(art_names) if art_names else "None",
        )

        return dspy.Prediction(
            plan=plan,
            code=code,
            stdout=last_stdout,
            stderr=last_stderr,
            generated_html_files=generated_html_files,
            is_success=is_success,
            final_answer=getattr(sum_pred, "summary", ""),
        )


class DSPyLLMAdapter(ILLMClient):
    """Adapter bridging DSPy modules into the Hexagonal Architecture ILLMClient port."""

    def __init__(
        self,
        module: Optional[DataAnalysisReActModule] = None,
        lm: Optional[dspy.LM] = None,
    ) -> None:
        self.module = module or DataAnalysisReActModule()
        if lm is not None:
            dspy.settings.configure(lm=lm)
        elif not getattr(dspy.settings, "lm", None):
            # Attempt default Gemini LM if configured in environment
            api_key = os.getenv("GEMINI_API_KEY")
            model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
            if api_key:
                try:
                    configured_lm = dspy.LM(f"gemini/{model_name}", api_key=api_key)
                    dspy.settings.configure(lm=configured_lm)
                except Exception:
                    pass

    def plan(
        self,
        question: str,
        dataset_preview: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Formulate execution plan via DSPy planner."""
        try:
            res = self.module.planner(question=question, dataset_preview=dataset_preview)
            return getattr(res, "plan", "")
        except Exception as exc:
            raise LLMExecutionError(f"DSPy planning error: {exc}") from exc

    def generate_code(
        self,
        question: str,
        dataset_preview: str,
        plan: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Generate code via DSPy coder predictor."""
        try:
            res = self.module.coder(
                question=question,
                dataset_preview=dataset_preview,
                plan=plan,
                previous_error=previous_error or "None",
                previous_code=previous_code or "None",
            )
            raw = getattr(res, "code", "")
            return clean_code_snippet(raw)
        except Exception as exc:
            raise LLMExecutionError(f"DSPy code generation error: {exc}") from exc

    def reflect_and_evaluate(
        self,
        question: str,
        plan: str,
        code: str,
        stdout: Optional[str],
        stderr: Optional[str],
        has_error: bool,
    ) -> Dict[str, Any]:
        """Evaluate execution output via DSPy reflector."""
        try:
            res = self.module.reflector(
                question=question,
                plan=plan,
                code=code,
                stdout=stdout or "",
                stderr=stderr or "",
            )
            is_resolved = bool(getattr(res, "is_resolved", not has_error))
            needs_code_fix = bool(getattr(res, "needs_code_fix", has_error))
            feedback = getattr(res, "feedback", "Execution evaluated.")
            return {
                "is_resolved": is_resolved and not has_error,
                "needs_code_fix": needs_code_fix or has_error,
                "feedback": feedback,
            }
        except Exception:
            return {
                "is_resolved": not has_error,
                "needs_code_fix": has_error,
                "feedback": stderr if has_error else "Code executed successfully.",
            }

    def summarize(
        self,
        question: str,
        dataset_preview: str,
        traces: List[TraceStep],
        artifacts: List[Artifact],
    ) -> str:
        """Summarize findings via DSPy summarizer."""
        try:
            findings = []
            for t in traces:
                if t.stdout:
                    findings.append(t.stdout)
            exec_summary = "\n".join(findings) if findings else "Execution complete."
            art_str = ", ".join([a.file_name for a in artifacts]) if artifacts else "None"

            res = self.module.summarizer(
                question=question,
                dataset_preview=dataset_preview,
                execution_summary=exec_summary,
                artifacts=art_str,
            )
            return getattr(res, "summary", "Analysis completed.")
        except Exception as exc:
            raise LLMExecutionError(f"DSPy summary error: {exc}") from exc
