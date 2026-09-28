"""Google Gemini LLM adapter using the modern `google-genai` SDK."""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from data_agent.core.entities import Artifact, TraceStep
from data_agent.core.exceptions import LLMExecutionError
from data_agent.ports.llm_port import ILLMClient


class GeminiLLMAdapter(ILLMClient):
    """Production LLM adapter interfacing strictly with Google Gemini API via `google-genai`."""

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

    def _call_generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini models.generate_content using configured and candidate models to survive free-tier per-model quotas."""
        client = self._get_client()
        last_error = None

        config = None
        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        except Exception:
            pass

        # Build list of real Gemini models to try if the primary hits free-tier per-model quotas (429 RESOURCE_EXHAUSTED)
        candidate_models = [self.model_name]
        for fallback_m in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]:
            if fallback_m not in candidate_models:
                candidate_models.append(fallback_m)

        for model_to_use in candidate_models:
            for attempt in range(max_retries):
                try:
                    kwargs = {"model": model_to_use, "contents": prompt}
                    if config is not None:
                        kwargs["config"] = config
                    response = client.models.generate_content(**kwargs)
                    if response.text and response.text.strip():
                        return response.text.strip()
                except Exception as exc:
                    last_error = exc
                    err_str = str(exc)
                    # If this specific model hit a hard per-model daily quota (429 RESOURCE_EXHAUSTED), switch to next candidate model
                    if "RESOURCE_EXHAUSTED" in err_str and "limit: 20" in err_str:
                        break
                    if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        sleep_time = 1.5 * (attempt + 1)
                        time.sleep(sleep_time)
                        continue
                    raise LLMExecutionError(f"Gemini API invocation error: {exc}") from exc

        raise LLMExecutionError(
            f"Gemini API request failed across candidate models: {last_error}"
        )

    def _clean_code(self, raw_text: str) -> str:
        """Strip markdown code blocks if the LLM wrapped python code."""
        text = raw_text.strip()
        pattern = r"^```(?:python)?\s*([\s\S]*?)\s*```$"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
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
        """Generate analysis and visualization plan using Gemini."""
        prompt = (
            "You are an expert Data Scientist and Python Data Analyst.\n"
            "Given the user question and the dataset summary below, devise a clear, concise step-by-step plan.\n\n"
            "CRITICAL RELEVANCE & SANITY CHECK:\n"
            "- First inspect the user question. If the user question is nonsensical gibberish (e.g. random letters like 'spidhf...', keyboard mashing, or completely unintelligible), DO NOT invent an analysis. State clearly in your plan that the input is meaningless gibberish and that no analysis should be performed.\n"
            "- If the question is a valid analytical query, create an autonomous plan focusing on:\n"
            "  1. Necessary data cleaning and filtering\n"
            "  2. Statistical aggregation and metrics to compute\n"
            "  3. Autonomous Plotly chart design (e.g. bar chart, line chart, scatter plot, or distribution) to be saved as 'output_plot.html'\n\n"
            f"--- DATASET PREVIEW ---\n{dataset_preview}\n\n"
            f"--- USER QUESTION ---\n{question}\n\n"
            "Output your concise reasoning plan:"
        )
        return self._call_generate_content(prompt)

    def generate_code(
        self,
        question: str,
        dataset_preview: str,
        plan: str,
        previous_error: Optional[str] = None,
        previous_code: Optional[str] = None,
    ) -> str:
        """Generate executable Python code using Gemini."""
        error_context = ""
        if previous_error:
            error_context = (
                f"\n--- PREVIOUS FAILED CODE ---\n{previous_code}\n\n"
                f"--- EXECUTION ERROR / FEEDBACK ---\n{previous_error}\n\n"
                "CRITICAL: The previous code failed. Analyze the error carefully, fix the bug, and write the corrected code.\n"
            )

        prompt = (
            "You are an expert Python data analysis coder.\n"
            "Write safe, executable Python code to carry out the analysis plan.\n\n"
            "SPECIAL RULE FOR GIBBERISH / INVALID QUESTIONS:\n"
            "If the plan or question indicates that the user query is nonsensical gibberish or invalid, write ONLY the following code:\n"
            "print('Input query is not recognizable or intelligible. Please submit a valid data analysis question.')\n"
            "Do NOT calculate arbitrary metrics or create misleading charts.\n\n"
            "Rules for valid queries:\n"
            "1. The dataset is accessible in the current directory as 'dataset.csv'. Load it with pd.read_csv('dataset.csv').\n"
            "2. Print key analytical answers and summary statistics to stdout using print().\n"
            "3. MANDATORY AUTONOMOUS PLOTLY VISUALIZATION: You MUST ALWAYS create an insightful, interactive Plotly visualization (e.g. using plotly.express as px or plotly.graph_objects as go) that visually answers or enriches the user's question. Save it to the current directory with:\n"
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
        return self._clean_code(raw)

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
            "You are an evaluation agent verifying if code execution solved the question.\n"
            f"Question: {question}\n"
            f"Code: {code}\n"
            f"Stdout: {stdout}\n"
            f"Stderr: {stderr}\n"
            "Return a strictly valid JSON object with:\n"
            '{"is_resolved": true/false, "feedback": "brief assessment"}\n'
            "JSON:"
        )
        try:
            raw = self._call_generate_content(prompt)
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
            return {
                "is_resolved": not has_error,
                "needs_code_fix": has_error,
                "feedback": "Evaluation complete.",
            }

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
            "Guidelines:\n"
            "- If the user question was unintelligible gibberish, clearly and politely inform the user that their request could not be parsed, and invite them to submit a valid analytical question.\n"
            "- If the question was valid, provide a polished, professional, natural language answer summarizing the specific numbers, trends, and insights found. Mention the interactive visualizations available if generated."
        )
        return self._call_generate_content(prompt)
