"""State definition for the ReAct Data Analysis LangGraph workflow."""

from typing import Annotated, Any, Dict, List, Optional, Tuple, TypedDict
from data_agent.core.entities import Artifact, TraceStep


def append_traces(existing: List[TraceStep], new_traces: List[TraceStep]) -> List[TraceStep]:
    """Reducer for accumulating reasoning trace steps."""
    return existing + new_traces


def append_artifacts(existing: List[Artifact], new_artifacts: List[Artifact]) -> List[Artifact]:
    """Reducer for accumulating generated domain artifacts."""
    return existing + new_artifacts


def append_artifact_ids(existing: List[str], new_ids: List[str]) -> List[str]:
    """Reducer for accumulating artifact IDs."""
    return existing + new_ids


class ReActAgentState(TypedDict, total=False):
    """LangGraph state representation for the ReAct loop."""

    session_id: str
    question: str
    dataset_path: str
    dataset_preview: str
    iteration: int
    max_iterations: int
    trace: Annotated[List[TraceStep], append_traces]
    current_plan: Optional[str]
    current_code: Optional[str]
    execution_output: Optional[str]
    execution_error: Optional[str]
    final_answer: Optional[str]
    artifact_ids: Annotated[List[str], append_artifact_ids]
    artifacts: Annotated[List[Artifact], append_artifacts]
    generated_html_files: List[Tuple[str, bytes]]
    is_resolved: bool
    needs_code_fix: bool
    feedback: Optional[str]
