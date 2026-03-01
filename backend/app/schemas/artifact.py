"""Pydantic schemas for Artifact resources."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ArtifactResponse(BaseModel):
    """Schema for returning a simulation artifact."""

    id: UUID
    run_id: UUID
    artifact_type: str
    file_path: str
    file_size_bytes: Optional[int]
    mime_type: Optional[str]
    description: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
