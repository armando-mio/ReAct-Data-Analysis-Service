"""Abstract interface for artifact file storage."""

from abc import ABC, abstractmethod


class IArtifactStorage(ABC):
    """Port interface for storing and retrieving raw binary artifacts (e.g. HTML plots)."""

    @abstractmethod
    def save_artifact(self, session_id: str, file_name: str, content: bytes) -> str:
        """Save artifact binary content to persistent storage and return its storage path.

        Args:
            session_id: The ID of the session the artifact belongs to.
            file_name: The original file name (e.g. 'scatter_plot.html').
            content: Raw byte contents of the file.

        Returns:
            The unique storage path or reference key where the file can be retrieved.
        """
        pass

    @abstractmethod
    def load_artifact(self, storage_path: str) -> bytes:
        """Read and return binary content of an artifact from storage.

        Args:
            storage_path: Path or key returned by `save_artifact`.

        Returns:
            Raw byte content of the file.

        Raises:
            ArtifactNotFoundError: If the storage path does not exist.
        """
        pass

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Check if an artifact file exists in storage."""
        pass

    @abstractmethod
    def delete_artifact(self, storage_path: str) -> bool:
        """Delete an artifact file from storage. Returns True if deleted, False otherwise."""
        pass
