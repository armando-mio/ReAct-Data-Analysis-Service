"""Orchestration use case for executing data analysis with the ReAct agent."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd

from data_agent.core.entities import Artifact, MessageRole, Session, TraceStep
from data_agent.core.exceptions import DatasetValidationError, SessionNotFoundError
from data_agent.ports.llm_port import ILLMClient
from data_agent.ports.repository_port import ISessionRepository
from data_agent.ports.sandbox_port import ISandboxRunner
from data_agent.ports.storage_port import IArtifactStorage
from data_agent.use_cases.react_graph import ReActGraphBuilder


@dataclass
class AnalysisResult:
    """Structured response container for the analysis use case."""
    session_id: str
    answer: str
    artifacts: List[Dict[str, str]] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class AnalyzeDataUseCase:
    """Main orchestration service executing analysis workflows over tabular datasets."""

    def __init__(
        self,
        llm_client: ILLMClient,
        sandbox_runner: ISandboxRunner,
        repository: ISessionRepository,
        storage: IArtifactStorage,
        uploads_dir: str = "storage/uploads",
    ) -> None:
        self.llm_client = llm_client
        self.sandbox_runner = sandbox_runner
        self.repository = repository
        self.storage = storage
        self.uploads_dir = Path(uploads_dir).resolve()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def _extract_dataset_preview(self, csv_path: str) -> str:
        """Read CSV, computing top 5 rows, data types, and summary statistics."""
        try:
            df = pd.read_csv(csv_path)
            shape_info = f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
            dtypes_info = f"Column Types:\n{df.dtypes.to_string()}\n\n"
            head_info = f"First 5 Rows:\n{df.head(5).to_string()}\n\n"
            stats_info = f"Descriptive Statistics:\n{df.describe(include='all').to_string()}\n"
            return shape_info + dtypes_info + head_info + stats_info
        except Exception as exc:
            raise DatasetValidationError(f"Failed to parse uploaded dataset CSV: {exc}") from exc

    def execute(
        self,
        question: str,
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> AnalysisResult:
        """Execute the end-to-end ReAct data analysis pipeline."""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        # Resolve or create Session
        session: Optional[Session] = None
        if session_id:
            session = self.repository.get_session(session_id)
            if not session:
                # If session_id provided but not found, initialize new session with that ID
                session = Session(id=session_id)
        else:
            session = Session(id=str(uuid.uuid4()))

        # Handle dataset upload or retrieval
        if file_bytes and filename:
            session_upload_dir = self.uploads_dir / session.id
            session_upload_dir.mkdir(parents=True, exist_ok=True)
            saved_csv_path = session_upload_dir / filename
            saved_csv_path.write_bytes(file_bytes)
            session.dataset_path = str(saved_csv_path)

        if not session.dataset_path or not Path(session.dataset_path).is_file():
            raise DatasetValidationError(
                "No dataset available. Please upload a CSV dataset to analyze or specify an existing session."
            )

        # Generate dataset preview
        dataset_preview = self._extract_dataset_preview(session.dataset_path)

        # Log User Message
        session.add_message(role=MessageRole.USER, content=question)

        # Build and invoke ReAct StateGraph
        graph_builder = ReActGraphBuilder(
            llm_client=self.llm_client,
            sandbox_runner=self.sandbox_runner,
        )
        graph = graph_builder.build()

        initial_state = {
            "session_id": session.id,
            "question": question,
            "dataset_path": session.dataset_path,
            "dataset_preview": dataset_preview,
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

        final_state = graph.invoke(initial_state)

        # Ingest and persist generated HTML artifacts
        generated_html_files = final_state.get("generated_html_files", [])
        artifact_responses: List[Dict[str, str]] = []

        for html_name, html_bytes in generated_html_files:
            storage_path = self.storage.save_artifact(
                session_id=session.id,
                file_name=html_name,
                content=html_bytes,
            )
            artifact = session.add_artifact(file_name=html_name, storage_path=storage_path)
            artifact_responses.append({
                "id": artifact.id,
                "file_name": artifact.file_name,
                "url": f"/artifacts/{artifact.id}",
            })

        # Persist traces
        new_traces: List[TraceStep] = final_state.get("trace", [])
        trace_responses: List[Dict[str, Any]] = []
        for step in new_traces:
            session.traces.append(step)
            trace_responses.append({
                "step_index": step.step_index,
                "thought": step.thought,
                "code": step.code,
                "stdout": step.stdout,
                "stderr": step.stderr,
                "duration_seconds": step.duration_seconds,
            })

        # Save assistant answer message
        final_answer = final_state.get("final_answer") or "Analysis completed."
        session.add_message(role=MessageRole.ASSISTANT, content=final_answer)

        # Update session in repository
        self.repository.save_session(session)

        return AnalysisResult(
            session_id=session.id,
            answer=final_answer,
            artifacts=artifact_responses,
            trace=trace_responses,
        )
