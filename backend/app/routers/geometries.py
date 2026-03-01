"""Endpoints for Geometry (GeometryVersion) resources."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile

from backend.app.schemas.geometry import (
    GeometryCreate,
    GeometryResponse,
    MeshPreviewResponse,
)

router = APIRouter(tags=["geometries"])


@router.post(
    "/projects/{project_id}/geometries",
    response_model=GeometryResponse,
    status_code=201,
)
async def create_geometry(
    project_id: uuid.UUID, body: GeometryCreate
) -> GeometryResponse:
    """Create a new geometry version within a project."""
    # TODO: call geometry service to persist
    now = datetime.now(tz=timezone.utc)
    return GeometryResponse(
        id=uuid.uuid4(),
        project_id=project_id,
        version_number=1,
        label=body.label,
        source_type=body.source_type,
        parametric_params=(
            body.parametric_params.model_dump() if body.parametric_params else None
        ),
        file_path=None,
        mesh_file_path=None,
        metadata_json=None,
        created_at=now,
    )


@router.get(
    "/projects/{project_id}/geometries",
    response_model=List[GeometryResponse],
)
async def list_geometries(project_id: uuid.UUID) -> List[GeometryResponse]:
    """List all geometry versions for a project."""
    # TODO: call geometry service to query DB
    return []


@router.get("/geometries/{geometry_id}", response_model=GeometryResponse)
async def get_geometry(geometry_id: uuid.UUID) -> GeometryResponse:
    """Get a single geometry version by ID."""
    # TODO: call geometry service to fetch from DB
    raise HTTPException(status_code=404, detail="Geometry not found")


@router.post("/geometries/{geometry_id}/generate", response_model=GeometryResponse)
async def generate_geometry(geometry_id: uuid.UUID) -> GeometryResponse:
    """Generate parametric geometry (CAD model) from stored parameters."""
    # TODO: call geometry generation service
    raise HTTPException(status_code=404, detail="Geometry not found")


@router.post("/geometries/{geometry_id}/mesh", response_model=GeometryResponse)
async def mesh_geometry(geometry_id: uuid.UUID) -> GeometryResponse:
    """Generate mesh for a geometry version."""
    # TODO: call meshing service
    raise HTTPException(status_code=404, detail="Geometry not found")


@router.post("/geometries/upload", response_model=GeometryResponse, status_code=201)
async def upload_geometry(
    project_id: uuid.UUID,
    file: UploadFile,
    label: Optional[str] = None,
) -> GeometryResponse:
    """Upload a STEP or STL file as a new geometry version."""
    # TODO: save uploaded file and create geometry version
    if file.content_type not in (
        "model/step",
        "model/stl",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    now = datetime.now(tz=timezone.utc)
    source_type = (
        "imported_step"
        if file.filename and file.filename.endswith(".step")
        else "imported_stl"
    )
    return GeometryResponse(
        id=uuid.uuid4(),
        project_id=project_id,
        version_number=1,
        label=label,
        source_type=source_type,
        parametric_params=None,
        file_path=f"storage/geometries/{file.filename}",
        mesh_file_path=None,
        metadata_json=None,
        created_at=now,
    )


@router.get(
    "/geometries/{geometry_id}/preview", response_model=MeshPreviewResponse
)
async def preview_geometry(geometry_id: uuid.UUID) -> MeshPreviewResponse:
    """Get a mesh preview for 3D visualization of a geometry."""
    # TODO: call geometry service to generate or load preview mesh
    raise HTTPException(status_code=404, detail="Geometry not found")
