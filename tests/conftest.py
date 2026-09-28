"""Pytest fixtures and test environment configuration."""

import os
from pathlib import Path
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from data_agent.adapters.api.app import create_app
from data_agent.adapters.api.dependencies import Container, get_container
from data_agent.adapters.llm.mock_llm_adapter import MockLLMAdapter
from data_agent.adapters.persistence.sqlite_repository import SQLiteSessionRepository
from data_agent.adapters.sandbox.process_sandbox import ProcessSandboxRunner
from data_agent.adapters.storage.local_storage import LocalArtifactStorage


@pytest.fixture
def sample_csv_content() -> bytes:
    """Sample sales and performance CSV dataset."""
    return (
        b"region,category,sales,units,profit\n"
        b"North,Electronics,1200.5,12,300.2\n"
        b"South,Furniture,850.0,5,150.0\n"
        b"East,Office Supplies,340.2,25,85.5\n"
        b"West,Electronics,2100.0,18,520.0\n"
        b"North,Furniture,450.0,4,60.0\n"
        b"South,Electronics,1600.0,14,410.0\n"
    )


@pytest.fixture
def sample_csv_file(sample_csv_content: bytes, tmp_path: Path) -> Path:
    """Write sample CSV content to a temporary file."""
    csv_file = tmp_path / "sales_data.csv"
    csv_file.write_bytes(sample_csv_content)
    return csv_file


@pytest.fixture
def test_container(tmp_path: Path) -> Container:
    """Isolated dependency injection container for tests."""
    container = Container()
    container.db_url = f"sqlite:///{tmp_path / 'test_data_agent.db'}"
    container.artifacts_dir = str(tmp_path / "artifacts")
    container.uploads_dir = str(tmp_path / "uploads")
    container.sandbox_timeout = 5.0

    container.storage = LocalArtifactStorage(base_directory=container.artifacts_dir)
    container.repository = SQLiteSessionRepository(db_url=container.db_url)
    container.sandbox_runner = ProcessSandboxRunner(default_timeout=container.sandbox_timeout)
    container.llm_client = MockLLMAdapter()

    return container


@pytest.fixture
def test_app(test_container: Container) -> TestClient:
    """TestClient configured with overridden test container dependencies."""
    app = create_app()

    # Override dependency container
    from data_agent.adapters.api import dependencies
    dependencies._container_instance = test_container

    client = TestClient(app)
    return client
