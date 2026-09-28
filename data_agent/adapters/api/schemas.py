"""Pydantic schemas for HTTP API request and response models."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ArtifactSummary(BaseModel):
    """Metadata and retrieval URL for a generated plot or file."""
    id: str
    file_name: str
    url: str


class TraceStepSummary(BaseModel):
    """Single reasoning or execution step inside the ReAct trace."""
    step_index: int
    thought: str
    code: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    duration_seconds: float = Field(default=0.0)


class AnalyzeResponse(BaseModel):
    """Response model for POST /analyze."""
    session_id: str
    answer: str
    artifacts: List[ArtifactSummary] = Field(default_factory=list)
    trace: List[TraceStepSummary] = Field(default_factory=list)


class MessageSummary(BaseModel):
    """Individual message in session history."""
    id: str
    role: str
    content: str
    timestamp: datetime


class SessionDetailResponse(BaseModel):
    """Full session details returned by GET /sessions/{id}."""
    session_id: str
    created_at: datetime
    updated_at: datetime
    dataset_path: Optional[str] = None
    messages: List[MessageSummary] = Field(default_factory=list)
    traces: List[TraceStepSummary] = Field(default_factory=list)
    artifacts: List[ArtifactSummary] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standardized error response payload."""
    detail: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
