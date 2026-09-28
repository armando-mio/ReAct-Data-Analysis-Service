"""Query and management use cases for sessions and artifacts."""

from typing import List, Optional, Tuple
from data_agent.core.entities import Artifact, Session
from data_agent.core.exceptions import ArtifactNotFoundError, SessionNotFoundError
from data_agent.ports.repository_port import ISessionRepository
from data_agent.ports.storage_port import IArtifactStorage


class ManageSessionUseCase:
    """Provides querying capabilities for conversation sessions and generated artifacts."""

    def __init__(
        self,
        repository: ISessionRepository,
        storage: IArtifactStorage,
    ) -> None:
        self.repository = repository
        self.storage = storage

    def get_session(self, session_id: str) -> Session:
        """Retrieve full session aggregate including history, traces, and artifact records."""
        session = self.repository.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session

    def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Session]:
        """List historical sessions ordered by recent activity."""
        return self.repository.list_sessions(limit=limit, offset=offset)

    def get_artifact_content(self, artifact_id: str) -> Tuple[str, bytes]:
        """Retrieve the binary content and filename of a stored visualization artifact."""
        artifact = self.repository.get_artifact(artifact_id)
        if not artifact:
            raise ArtifactNotFoundError(artifact_id)
        content = self.storage.load_artifact(artifact.storage_path)
        return artifact.file_name, content
