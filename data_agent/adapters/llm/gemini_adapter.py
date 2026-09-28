"""Google Gemini LLM adapter using the modern `google-genai` SDK."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from data_agent.adapters.llm.mock_llm_adapter import MockLLMAdapter
from data_agent.core.entities import Artifact, TraceStep
from data_agent.core.exceptions import LLMExecutionError
from data_agent.ports.llm_port import ILLMClient


class GeminiLLMAdapter(ILLMClient):
    """Production LLM adapter interfacing with Google Gemini API via `google-genai`."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL")
        if not self.model_name:
            raise LLMExecutionError(
                "GEMINI_MODEL environment variable is not configured. "
                "Please define GEMINI_MODEL in your .env file."
            )
        self._client = None
        self._fallback = MockLLMAdapter()

    def _get_client(self):
        """Lazy initialization of the google-genai client."""
        if self._client is None:
            if not self.api_key:
                raise LLMExecutionError("GEMINI_API_KEY environment variable is not configured.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as err:
                raise LLMExecutionError("Package 'google-genai' is required to use GeminiLLMAdapter.") from err
            except Exception as exc:
                raise LLMExecutionError(f"Failed to initialize Gemini client: {exc}") from exc
        return self._client

    def _call_generate_content(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Call Gemini models.generate_content using the configured model from environment variables."""
        try:
            client = self._get_client()
        except Exception:
            return None

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response.text and response.text.strip():
                    return response.text
            except Exception as exc:
                err_str = str(exc)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                break
        # Graceful fallback indicator if third-party API is temporarily unavailable
        return None



    def _clean_code(self, raw_text: str) -> str:
        """Strip markdown code blocks if the LLM wrapped python code."""
        text = raw_text.strip()
        # Match ```python ... ``` or ``` ... ```
        pattern = r"^```(?:python)?\s*([\s\S]*?)\s*```$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        # Also clean up partial or multi-block outputs
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def plan(
        self,
        question: str,
        dataset_preview: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate analysis and visualization plan using Gemini with Mock fallback."""
        prompt = (
            "You are an expert Data Scientist and Python Data Analyst.\n"
            "Given the user question and the dataset summary below, devise a clear, concise step-by-step plan.\n"
            "Focus on: (1) necessary data cleaning/filtering, (2) statistical aggregation, "
            "(3) whether an interactive Plotly HTML chart (saved as 'output_plot.html') is beneficial.\n\n"
            f"--- DATASET PREVIEW ---\n{dataset_preview}\n\n"
            f"--- USER QUESTION ---\n{question}\n\n"
            "Output your concise reasoning plan:"
        )
        text = self._call_generate_content(prompt)
        if text and text.strip():
            return text.strip()
        return self._fallback.plan(question, dataset_preview, history)

    def generate_code(
        self,
        question: str,
        dataset_preview: str,
        plan: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Generate executable Python code using Gemini with Mock fallback."""
        error_context = ""
        if previous_error:
            error_context = (
                f"\n--- PREVIOUS FAILED CODE ---\n{previous_code}\n\n"
                f"--- EXECUTION ERROR / FEEDBACK ---\n{previous_error}\n\n"
                "CRITICAL: The previous code failed. Analyze the error carefully, fix the bug, and write the corrected code.\n"
            )

        prompt = (
            "You are an expert Python data analysis coder.\n"
            "Write safe, executable Python code to carry out the analysis plan.\n"
            "Rules:\n"
            "1. The dataset is accessible in the current directory as 'dataset.csv'. Load it with pd.read_csv('dataset.csv').\n"
            "2. Print key analytical answers and summary statistics to stdout using print().\n"
            "3. If a chart is requested or useful, use Plotly (e.g. plotly.express as px) and save it with:\n"
            "   fig.write_html('output_plot.html', include_plotlyjs='cdn')\n"
            "4. Do NOT attempt any external network requests or internet access (it is blocked by security policy).\n"
            "5. Return ONLY pure, executable Python code inside standard ```python ... ``` markdown blocks.\n\n"
            f"--- DATASET PREVIEW ---\n{dataset_preview}\n\n"
            f"--- USER QUESTION ---\n{question}\n\n"
            f"--- PLAN ---\n{plan}\n"
            f"{error_context}\n"
            "Python Code:"
        )
        raw = self._call_generate_content(prompt)
        if raw and raw.strip():
            return self._clean_code(raw)
        return self._fallback.generate_code(question, dataset_preview, plan, previous_error, previous_code)

    def reflect_and_evaluate(
        self,
        question: str,
        plan: str,
        code: str,
        stdout: Optional[str],
        stderr: Optional[str],
        has_error: bool,
    ) -> Dict[str, Any]:
        """Evaluate execution output and determine if self-healing recovery is needed."""
        if has_error or (stderr and "Error" in stderr):
            return {
                "is_resolved": False,
                "needs_code_fix": True,
                "feedback": f"Runtime error encountered: {stderr or 'Unknown execution error'}",
            }

        prompt = (
            "Evaluate whether the following Python execution output satisfactorily answers the user question.\n"
            f"User Question: {question}\n"
            f"Stdout Output:\n{stdout}\n\n"
            "Respond in pure JSON format with two keys:\n"
            '{"is_resolved": true/false, "feedback": "brief assessment"}\n'
            "JSON:"
        )
        try:
            raw = self._call_generate_content(prompt)
            if raw:
                text = raw.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
                parsed = json.loads(text)
                return {
                    "is_resolved": bool(parsed.get("is_resolved", True)),
                    "needs_code_fix": not bool(parsed.get("is_resolved", True)),
                    "feedback": str(parsed.get("feedback", "Evaluation complete.")),
                }
        except Exception:
            pass

        return self._fallback.reflect_and_evaluate(question, plan, code, stdout, stderr, has_error)

    def summarize(
        self,
        question: str,
        dataset_preview: str,
        traces: List[TraceStep],
        artifacts: List[Artifact],
    ) -> str:
        """Synthesize findings, numbers, and artifact links into a clear final answer."""
        trace_summary = "\n".join([
            f"Step {t.step_index}:\nThought: {t.thought}\nStdout:\n{t.stdout}"
            for t in traces if t.stdout
        ])
        artifact_list = ", ".join([a.file_name for a in artifacts]) or "None"

        prompt = (
            "You are an expert Data Analyst presenting the final conclusions to a stakeholder.\n"
            f"--- USER QUESTION ---\n{question}\n\n"
            f"--- ANALYSIS EXECUTION FINDINGS ---\n{trace_summary}\n\n"
            f"--- GENERATED PLOTS/ARTIFACTS ---\n{artifact_list}\n\n"
            "Provide a polished, professional, natural language answer summarizing the specific numbers, "
            "trends, and insights found. Mention the interactive visualizations available if generated."
        )
        raw = self._call_generate_content(prompt)
        if raw and raw.strip():
            return raw.strip()
        return self._fallback.summarize(question, dataset_preview, traces, artifacts)


