"""Router for GET /sessions and GET /sessions/{id} endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query, status

from data_agent.adapters.api.dependencies import get_manage_session_use_case
from data_agent.adapters.api.schemas import (
    ArtifactSummary,
    MessageSummary,
    SessionDetailResponse,
    TraceStepSummary,
)
from data_agent.core.exceptions import SessionNotFoundError
from data_agent.use_cases.manage_session import ManageSessionUseCase

router = APIRouter(tags=["Sessions"])


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve session metadata, historical messages, traces, and generated artifacts",
)
def get_session(
    session_id: str,
    use_case: ManageSessionUseCase = Depends(get_manage_session_use_case),
) -> SessionDetailResponse:
    """Fetch complete analysis session by ID."""
    try:
        session = use_case.get_session(session_id)
    except SessionNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=err.message,
        ) from err

    return SessionDetailResponse(
        session_id=session.id,
        created_at=session.created_at,
        updated_at=session.updated_at,
        dataset_path=session.dataset_path,
        messages=[
            MessageSummary(
                id=m.id,
                role=m.role.value,
                content=m.content,
                timestamp=m.timestamp,
            )
            for m in session.messages
        ],
        traces=[
            TraceStepSummary(
                step_index=t.step_index,
                thought=t.thought,
                code=t.code,
                stdout=t.stdout,
                stderr=t.stderr,
                duration_seconds=t.duration_seconds,
            )
            for t in session.traces
        ],
        artifacts=[
            ArtifactSummary(
                id=a.id,
                file_name=a.file_name,
                url=f"/artifacts/{a.id}",
            )
            for a in session.artifacts
        ],
    )


@router.get(
    "/sessions",
    response_model=List[SessionDetailResponse],
    status_code=status.HTTP_200_OK,
    summary="List recent analysis sessions",
)
def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    use_case: ManageSessionUseCase = Depends(get_manage_session_use_case),
) -> List[SessionDetailResponse]:
    """Retrieve list of recent sessions."""
    sessions = use_case.list_sessions(limit=limit, offset=offset)
    return [
        SessionDetailResponse(
            session_id=s.id,
            created_at=s.created_at,
            updated_at=s.updated_at,
            dataset_path=s.dataset_path,
            messages=[
                MessageSummary(
                    id=m.id,
                    role=m.role.value,
                    content=m.content,
                    timestamp=m.timestamp,
                )
                for m in s.messages
            ],
            traces=[
                TraceStepSummary(
                    step_index=t.step_index,
                    thought=t.thought,
                    code=t.code,
                    stdout=t.stdout,
                    stderr=t.stderr,
                    duration_seconds=t.duration_seconds,
                )
                for t in s.traces
            ],
            artifacts=[
                ArtifactSummary(
                    id=a.id,
                    file_name=a.file_name,
                    url=f"/artifacts/{a.id}",
                )
                for a in s.artifacts
            ],
        )
        for s in sessions
    ]
