"""Router for POST /analyze endpoint with graceful error handling and self-healing recovery."""

from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from data_agent.adapters.api.dependencies import get_analyze_use_case
from data_agent.adapters.api.schemas import AnalyzeResponse, ArtifactSummary, TraceStepSummary
from data_agent.core.exceptions import DatasetValidationError, DomainError, SessionNotFoundError
from data_agent.use_cases.analyze_data import AnalyzeDataUseCase

router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute ReAct data analysis over an uploaded CSV or existing session dataset",
)
async def analyze_data(
    question: str = Form(..., description="Analytical question or task for the ReAct agent"),
    file: Optional[UploadFile] = File(None, description="Optional CSV dataset file to analyze"),
    session_id: Optional[str] = Form(None, description="Optional existing session ID for multi-turn conversations"),
    use_case: AnalyzeDataUseCase = Depends(get_analyze_use_case),
) -> AnalyzeResponse:
    """Run ReAct loop (Plan -> Act -> Observe -> Recover) on tabular data."""
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None

    if file and file.filename:
        filename = file.filename
        file_bytes = await file.read()

    try:
        result = use_case.execute(
            question=question,
            file_bytes=file_bytes,
            filename=filename,
            session_id=session_id,
        )
    except DatasetValidationError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err.message,
        ) from err
    except SessionNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=err.message,
        ) from err
    except DomainError as err:
        # Graceful failure returning structured payload rather than HTTP 500
        return AnalyzeResponse(
            session_id=session_id or "error",
            answer=f"Analysis could not be completed due to a domain error: {err.message}",
            status="failed",
            error=str(err),
            artifacts=[],
            trace=[],
        )
    except Exception as exc:
        # Graceful failure for unexpected runtime exceptions
        return AnalyzeResponse(
            session_id=session_id or "error",
            answer=f"An unexpected error occurred during analysis: {exc}",
            status="failed",
            error=str(exc),
            artifacts=[],
            trace=[],
        )

    return AnalyzeResponse(
        session_id=result.session_id,
        answer=result.answer,
        status=result.status,
        error=result.error,
        artifacts=[ArtifactSummary(**art) for art in result.artifacts],
        trace=[TraceStepSummary(**t) for t in result.trace],
    )
