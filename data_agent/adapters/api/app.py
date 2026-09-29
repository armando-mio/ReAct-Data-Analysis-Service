"""FastAPI application factory, middleware, and exception handlers."""

import json
import logging
import sys
import time
from typing import Callable
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from data_agent.adapters.api.routers.analyze import router as analyze_router
from data_agent.adapters.api.routers.artifacts import router as artifacts_router
from data_agent.adapters.api.routers.sessions import router as sessions_router
from data_agent.core.exceptions import (
    ArtifactNotFoundError,
    DatasetValidationError,
    DomainError,
    SessionNotFoundError,
)


from data_agent.core.logging import configure_structured_logging, get_logger

logger = get_logger("data_agent.api")


def create_app() -> FastAPI:
    """Factory creating and configuring the production FastAPI instance."""
    configure_structured_logging()

    app = FastAPI(
        title="ReAct Data Analysis Service",
        description="Hexagonal Architecture ReAct Agent Service powered by LangGraph, SQLite, and Gemini",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Lifecycle & Latency Logging Middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            f"HTTP {request.method} {request.url.path} responded {response.status_code} in {duration_ms}ms",
            extra={
                "action_type": "http_request",
                "execution_time_ms": duration_ms,
            },
        )
        return response

    # Global Domain Exception Handlers
    @app.exception_handler(SessionNotFoundError)
    async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": exc.message, "error_type": "SessionNotFoundError"},
        )

    @app.exception_handler(ArtifactNotFoundError)
    async def artifact_not_found_handler(request: Request, exc: ArtifactNotFoundError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": exc.message, "error_type": "ArtifactNotFoundError"},
        )

    @app.exception_handler(DatasetValidationError)
    async def dataset_validation_handler(request: Request, exc: DatasetValidationError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": exc.message, "error_type": "DatasetValidationError"},
        )

    @app.exception_handler(DomainError)
    async def generic_domain_error_handler(request: Request, exc: DomainError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": exc.message, "error_type": exc.__class__.__name__},
        )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    def health_check():
        return {"status": "healthy", "service": "react-data-agent", "version": "1.0.0"}

    # Include Routers
    app.include_router(analyze_router)
    app.include_router(sessions_router)
    app.include_router(artifacts_router)

    return app


app = create_app()
