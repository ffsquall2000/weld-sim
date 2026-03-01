"""Pydantic schemas for Comparison resources."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ComparisonCreate(BaseModel):
    """Schema for creating a new comparison."""

    name: str
    description: Optional[str] = None
    run_ids: List[UUID]
    metric_names: Optional[List[str]] = None
    baseline_run_id: Optional[UUID] = None


class ComparisonResultItem(BaseModel):
    """Schema for a single comparison result row."""

    run_id: UUID
    metric_name: str
    value: float
    delta_from_baseline: Optional[float]
    delta_percent: Optional[float]


class ComparisonResponse(BaseModel):
    """Schema for returning a comparison with results."""

    id: UUID
    project_id: UUID
    name: str
    description: Optional[str]
    run_ids: List[UUID]
    metric_names: Optional[List[str]]
    baseline_run_id: Optional[UUID]
    results: List[ComparisonResultItem] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
