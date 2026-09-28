"""Router for GET /artifacts/{id} endpoint."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from data_agent.adapters.api.dependencies import get_manage_session_use_case
from data_agent.core.exceptions import ArtifactNotFoundError
from data_agent.use_cases.manage_session import ManageSessionUseCase

router = APIRouter(tags=["Artifacts"])


@router.get(
    "/artifacts/{artifact_id}",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    summary="Retrieve generated standalone Plotly HTML visualization",
)
def get_artifact(
    artifact_id: str,
    use_case: ManageSessionUseCase = Depends(get_manage_session_use_case),
) -> Response:
    """Stream or return the raw Plotly HTML artifact file."""
    try:
        filename, content = use_case.get_artifact_content(artifact_id)
    except ArtifactNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=err.message,
        ) from err

    # Return pure HTML with text/html media type
    return Response(
        content=content,
        media_type="text/html",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
