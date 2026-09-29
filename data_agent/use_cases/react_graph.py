"""LangGraph ReAct agent state machine implementing Plan -> Act -> Observe -> Recover."""

import time
from typing import Any, Callable, Dict, Literal, Optional
from langgraph.graph import END, START, StateGraph

from data_agent.core.entities import Artifact, TraceStep
from data_agent.core.exceptions import SandboxSecurityError, SandboxTimeoutError
from data_agent.core.logging import get_logger
from data_agent.ports.llm_port import ILLMClient
from data_agent.ports.sandbox_port import ISandboxRunner
from data_agent.use_cases.agent_state import ReActAgentState

logger = get_logger("data_agent.react_graph")


class ReActGraphBuilder:
    """Constructs and compiles the ReAct StateGraph parameterized with domain ports."""

    def __init__(
        self,
        llm_client: ILLMClient,
        sandbox_runner: ISandboxRunner,
    ) -> None:
        self.llm_client = llm_client
        self.sandbox_runner = sandbox_runner

    def _planner_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """Formulate execution plan based on question and dataset preview."""
        start_time = time.perf_counter()
        session_id = state.get("session_id", "")
        step_index = state.get("iteration", 0)

        plan = self.llm_client.plan(
            question=state["question"],
            dataset_preview=state.get("dataset_preview", ""),
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Planner node formulated execution plan",
            extra={
                "session_id": session_id,
                "step_index": step_index,
                "action_type": "plan",
                "execution_time_ms": elapsed_ms,
            },
        )
        return {"current_plan": plan}

    def _code_generator_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """Generate executable Python code, incorporating previous errors during recovery."""
        start_time = time.perf_counter()
        session_id = state.get("session_id", "")
        step_index = state.get("iteration", 0)

        code = self.llm_client.generate_code(
            question=state["question"],
            dataset_preview=state.get("dataset_preview", ""),
            plan=state.get("current_plan", ""),
            previous_error=state.get("execution_error"),
            previous_code=state.get("current_code"),
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Code generator produced executable script",
            extra={
                "session_id": session_id,
                "step_index": step_index,
                "action_type": "code_generation",
                "execution_time_ms": elapsed_ms,
            },
        )
        return {"current_code": code}

    def _sandbox_runner_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """Execute generated code inside isolated sandbox with timeout and network guards."""
        start_time = time.perf_counter()
        session_id = state.get("session_id", "")
        step_index = state.get("iteration", 0)
        code = state.get("current_code", "")
        dataset_path = state.get("dataset_path")

        try:
            result = self.sandbox_runner.execute(code=code, dataset_path=dataset_path)
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.info(
                f"Sandbox runner completed execution (success={result.is_success})",
                extra={
                    "session_id": session_id,
                    "step_index": step_index,
                    "action_type": "sandbox_execution",
                    "execution_time_ms": elapsed_ms,
                },
            )
            return {
                "execution_output": result.stdout,
                "execution_error": result.stderr if not result.is_success else None,
                "generated_html_files": result.generated_html_files,
                "execution_duration": result.duration_seconds,
            }
        except SandboxTimeoutError as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                f"Sandbox execution timed out after {exc.timeout_seconds}s",
                extra={
                    "session_id": session_id,
                    "step_index": step_index,
                    "action_type": "sandbox_execution",
                    "execution_time_ms": elapsed_ms,
                },
            )
            return {
                "execution_output": "",
                "execution_error": f"Execution timed out after {exc.timeout_seconds} seconds.",
                "generated_html_files": [],
                "execution_duration": exc.timeout_seconds,
            }
        except SandboxSecurityError as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                f"Sandbox security violation: {exc.reason}",
                extra={
                    "session_id": session_id,
                    "step_index": step_index,
                    "action_type": "sandbox_execution",
                    "execution_time_ms": elapsed_ms,
                },
            )
            return {
                "execution_output": "",
                "execution_error": f"Security violation: {exc.reason}",
                "generated_html_files": [],
                "execution_duration": 0.0,
            }
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                f"Sandbox execution failed unexpectedly: {exc}",
                extra={
                    "session_id": session_id,
                    "step_index": step_index,
                    "action_type": "sandbox_execution",
                    "execution_time_ms": elapsed_ms,
                },
            )
            return {
                "execution_output": "",
                "execution_error": f"Execution failed: {exc}",
                "generated_html_files": [],
                "execution_duration": 0.0,
            }

    def _reflection_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """Observe results, append trace step, and determine whether self-healing is required."""
        start_time = time.perf_counter()
        session_id = state.get("session_id", "")
        iteration = state.get("iteration", 0) + 1
        output = state.get("execution_output", "") or ""
        error = state.get("execution_error")
        code = state.get("current_code", "")
        plan = state.get("current_plan", "")
        duration = state.get("execution_duration", 0.0)

        # Create trace step
        step = TraceStep(
            session_id=session_id,
            step_index=iteration,
            thought=plan,
            code=code,
            stdout=output,
            stderr=error,
            duration_seconds=duration,
        )

        has_error = bool(error and error.strip())

        # Evaluate progress
        eval_result = self.llm_client.reflect_and_evaluate(
            question=state["question"],
            plan=plan,
            code=code,
            stdout=output,
            stderr=error,
            has_error=has_error,
        )

        needs_code_fix = has_error or eval_result.get("needs_code_fix", False)
        is_resolved = not needs_code_fix
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"Reflection evaluated step {iteration} (needs_fix={needs_code_fix}, is_resolved={is_resolved})",
            extra={
                "session_id": session_id,
                "step_index": iteration,
                "action_type": "reflection",
                "execution_time_ms": elapsed_ms,
            },
        )

        return {
            "iteration": iteration,
            "trace": [step],
            "needs_code_fix": needs_code_fix,
            "is_resolved": is_resolved,
            "feedback": eval_result.get("feedback", ""),
        }

    def _route_after_reflection(self, state: ReActAgentState) -> Literal["code_generator", "finalizer"]:
        """Determine whether to re-enter code generation for self-healing or finish."""
        needs_fix = state.get("needs_code_fix", False)
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 4)

        if needs_fix and iteration < max_iterations:
            return "code_generator"
        return "finalizer"

    def _finalizer_node(self, state: ReActAgentState) -> Dict[str, Any]:
        """Formulate natural language final answer integrating traces and artifacts."""
        start_time = time.perf_counter()
        session_id = state.get("session_id", "")
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 4)
        traces = state.get("trace", [])
        artifacts = list(state.get("artifacts", []))
        if not artifacts and state.get("generated_html_files"):
            for fname, _ in state.get("generated_html_files", []):
                artifacts.append(
                    Artifact(session_id=session_id, file_name=fname, storage_path="")
                )

        if not state.get("is_resolved", False) and iteration >= max_iterations:
            last_err = state.get("execution_error", "Unknown error")
            fallback_answer = (
                f"I attempted to analyze the dataset across {iteration} iterations, but encountered persistent errors: "
                f"{last_err}. The partial output gathered is: {state.get('execution_output', 'None')}."
            )
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning(
                f"Finalizer reached max iterations ({iteration}) without full resolution",
                extra={
                    "session_id": session_id,
                    "step_index": iteration,
                    "action_type": "finalizer",
                    "execution_time_ms": elapsed_ms,
                },
            )
            return {"final_answer": fallback_answer}

        answer = self.llm_client.summarize(
            question=state["question"],
            dataset_preview=state.get("dataset_preview", ""),
            traces=traces,
            artifacts=artifacts,
        )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(
            "Finalizer synthesized user answer",
            extra={
                "session_id": session_id,
                "step_index": iteration,
                "action_type": "finalizer",
                "execution_time_ms": elapsed_ms,
            },
        )
        return {"final_answer": answer}

    def build(self) -> Any:
        """Compile and return the executable LangGraph StateGraph."""
        workflow = StateGraph(ReActAgentState)

        # Register nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("code_generator", self._code_generator_node)
        workflow.add_node("sandbox_runner", self._sandbox_runner_node)
        workflow.add_node("reflection", self._reflection_node)
        workflow.add_node("finalizer", self._finalizer_node)

        # Configure standard linear and conditional edges
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "code_generator")
        workflow.add_edge("code_generator", "sandbox_runner")
        workflow.add_edge("sandbox_runner", "reflection")

        # Conditional self-healing edge
        workflow.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {
                "code_generator": "code_generator",
                "finalizer": "finalizer",
            },
        )
        workflow.add_edge("finalizer", END)

        return workflow.compile()
