"""CRUD endpoints for Project resources."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("/", response_model=ProjectResponse, status_code=201)
async def create_project(body: ProjectCreate) -> ProjectResponse:
    """Create a new simulation project."""
    # TODO: call project service to persist in DB
    now = datetime.now(tz=timezone.utc)
    return ProjectResponse(
        id=uuid.uuid4(),
        name=body.name,
        description=body.description,
        application_type=body.application_type,
        settings=body.settings,
        tags=body.tags,
        created_at=now,
        updated_at=now,
    )


@router.get("/", response_model=ProjectListResponse)
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    application_type: Optional[str] = None,
    search: Optional[str] = None,
) -> ProjectListResponse:
    """List projects with optional filtering and pagination."""
    # TODO: call project service to query DB
    return ProjectListResponse(items=[], total=0)


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: uuid.UUID) -> ProjectResponse:
    """Get a single project by ID."""
    # TODO: call project service to fetch from DB
    raise HTTPException(status_code=404, detail="Project not found")


@router.patch("/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: uuid.UUID, body: ProjectUpdate
) -> ProjectResponse:
    """Partially update a project."""
    # TODO: call project service to update in DB
    raise HTTPException(status_code=404, detail="Project not found")


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: uuid.UUID) -> None:
    """Delete a project and all associated resources."""
    # TODO: call project service to delete from DB
    raise HTTPException(status_code=404, detail="Project not found")
