"""Pydantic schemas for Run resources."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from backend.app.schemas.artifact import ArtifactResponse
from backend.app.schemas.metric import MetricResponse


class RunCreate(BaseModel):
    """Schema for creating a new simulation run."""

    geometry_version_id: UUID
    parameters_override: Optional[dict] = None  # Optional solver param overrides


class RunResponse(BaseModel):
    """Schema for returning a simulation run."""

    id: UUID
    simulation_case_id: UUID
    geometry_version_id: UUID
    optimization_study_id: Optional[UUID]
    iteration_number: Optional[int]
    status: str  # queued, running, completed, failed, cancelled
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    compute_time_s: Optional[float]
    error_message: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RunDetailResponse(RunResponse):
    """Schema for returning a simulation run with full details."""

    solver_log: Optional[str] = None
    input_snapshot: Optional[dict] = None
    metrics: List[MetricResponse] = []
    artifacts: List[ArtifactResponse] = []
