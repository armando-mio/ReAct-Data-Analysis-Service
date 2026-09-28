"""Local filesystem implementation of IArtifactStorage."""

import os
from pathlib import Path
from typing import Union
from data_agent.core.exceptions import ArtifactNotFoundError
from data_agent.ports.storage_port import IArtifactStorage


class LocalArtifactStorage(IArtifactStorage):
    """Stores artifact files directly in a specified local directory."""

    def __init__(self, base_directory: Union[str, Path] = "storage/artifacts") -> None:
        self.base_dir = Path(base_directory).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, session_id: str, file_name: str, content: bytes) -> str:
        """Save artifact binary content inside session subfolder and return relative path."""
        session_folder = self.base_dir / session_id
        session_folder.mkdir(parents=True, exist_ok=True)

        target_file = session_folder / file_name
        # If file already exists, avoid collision by preserving uniqueness
        counter = 1
        stem = Path(file_name).stem
        suffix = Path(file_name).suffix
        while target_file.exists():
            target_file = session_folder / f"{stem}_{counter}{suffix}"
            counter += 1

        target_file.write_bytes(content)
        # Store relative to base directory for portability
        return str(target_file.relative_to(self.base_dir).as_posix())

    def load_artifact(self, storage_path: str) -> bytes:
        """Load artifact binary content from storage."""
        target_file = self.base_dir / storage_path
        if not target_file.is_file():
            raise ArtifactNotFoundError(f"Artifact at path '{storage_path}' does not exist on disk.")
        return target_file.read_bytes()

    def exists(self, storage_path: str) -> bool:
        """Check whether the artifact file exists on disk."""
        target_file = self.base_dir / storage_path
        return target_file.is_file()

    def delete_artifact(self, storage_path: str) -> bool:
        """Delete artifact file from disk."""
        target_file = self.base_dir / storage_path
        if target_file.is_file():
            try:
                target_file.unlink()
                return True
            except OSError:
                return False
        return False
