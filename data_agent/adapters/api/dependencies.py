"""Dependency injection container and service factories for FastAPI."""

import os
from functools import lru_cache
from typing import Optional
from dotenv import load_dotenv

# Automatically load .env file if present, overriding existing cached env vars
load_dotenv(override=True)


from data_agent.adapters.llm.gemini_adapter import GeminiLLMAdapter
from data_agent.adapters.llm.mock_llm_adapter import MockLLMAdapter
from data_agent.adapters.persistence.sqlite_repository import SQLiteSessionRepository
from data_agent.adapters.sandbox.process_sandbox import ProcessSandboxRunner
from data_agent.adapters.storage.local_storage import LocalArtifactStorage
from data_agent.ports.llm_port import ILLMClient
from data_agent.ports.repository_port import ISessionRepository
from data_agent.ports.sandbox_port import ISandboxRunner
from data_agent.ports.storage_port import IArtifactStorage
from data_agent.use_cases.analyze_data import AnalyzeDataUseCase
from data_agent.use_cases.manage_session import ManageSessionUseCase


class Container:
    """Singleton container maintaining shared adapter instances."""

    def __init__(self) -> None:
        load_dotenv(override=True)
        self.db_url = os.getenv("DATABASE_URL", "sqlite:///storage/data_agent.db")
        self.artifacts_dir = os.getenv("ARTIFACTS_DIR", "storage/artifacts")
        self.uploads_dir = os.getenv("UPLOADS_DIR", "storage/uploads")
        self.sandbox_timeout = float(os.getenv("SANDBOX_TIMEOUT", "15.0"))
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

        # Port adapters
        self.storage: IArtifactStorage = LocalArtifactStorage(base_directory=self.artifacts_dir)
        self.repository: ISessionRepository = SQLiteSessionRepository(db_url=self.db_url)
        self.sandbox_runner: ISandboxRunner = ProcessSandboxRunner(default_timeout=self.sandbox_timeout)

        # Configurable LLM client (defaults to Gemini if API key is present, otherwise MockLLMAdapter for tests/offline)
        if self.gemini_api_key:
            self.llm_client: ILLMClient = GeminiLLMAdapter(
                api_key=self.gemini_api_key,
                model_name=self.gemini_model,
            )
        else:
            self.llm_client: ILLMClient = MockLLMAdapter()

    def set_llm_client(self, client: ILLMClient) -> None:
        """Override LLM client (e.g. for testing)."""
        self.llm_client = client

    def get_analyze_use_case(self) -> AnalyzeDataUseCase:
        """Create AnalyzeDataUseCase with current configured adapters."""
        return AnalyzeDataUseCase(
            llm_client=self.llm_client,
            sandbox_runner=self.sandbox_runner,
            repository=self.repository,
            storage=self.storage,
            uploads_dir=self.uploads_dir,
        )

    def get_manage_session_use_case(self) -> ManageSessionUseCase:
        """Create ManageSessionUseCase with current configured adapters."""
        return ManageSessionUseCase(
            repository=self.repository,
            storage=self.storage,
        )


_container_instance: Optional[Container] = None


def get_container() -> Container:
    """Retrieve or initialize the global DI container."""
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
    return _container_instance


def get_analyze_use_case() -> AnalyzeDataUseCase:
    """FastAPI dependency provider for AnalyzeDataUseCase."""
    return get_container().get_analyze_use_case()


def get_manage_session_use_case() -> ManageSessionUseCase:
    """FastAPI dependency provider for ManageSessionUseCase."""
    return get_container().get_manage_session_use_case()
