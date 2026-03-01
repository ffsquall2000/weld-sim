"""Pydantic schemas for Metric resources."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MetricResponse(BaseModel):
    """Schema for returning a simulation metric."""

    id: UUID
    run_id: UUID
    metric_name: str
    value: float
    unit: Optional[str]
    metadata_json: Optional[dict]

    model_config = ConfigDict(from_attributes=True)


class MetricSummary(BaseModel):
    """Schema for a lightweight metric summary."""

    metric_name: str
    value: float
    unit: Optional[str]
