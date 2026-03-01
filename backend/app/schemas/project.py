"""Pydantic schemas for Project resources."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ProjectCreate(BaseModel):
    """Schema for creating a new project."""

    name: str
    description: Optional[str] = None
    application_type: str  # li_battery_tab, li_battery_busbar, general_metal, etc.
    settings: Optional[dict] = None
    tags: Optional[List[str]] = None


class ProjectUpdate(BaseModel):
    """Schema for partially updating a project."""

    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[dict] = None
    tags: Optional[List[str]] = None


class ProjectResponse(BaseModel):
    """Schema for returning a single project."""

    id: UUID
    name: str
    description: Optional[str]
    application_type: str
    settings: Optional[dict]
    tags: Optional[List[str]]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectListResponse(BaseModel):
    """Schema for returning a paginated list of projects."""

    items: List[ProjectResponse]
    total: int
