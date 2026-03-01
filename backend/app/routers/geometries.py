"""Endpoints for Geometry (GeometryVersion) resources."""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.geometry import GeometryCreate, GeometryResponse, MeshPreviewResponse
from backend.app.services.geometry_service import GeometryService

router = APIRouter(tags=["geometries"])


@router.post("/projects/{project_id}/geometries", response_model=GeometryResponse, status_code=201)
async def create_geometry(project_id: uuid.UUID, body: GeometryCreate, db: AsyncSession = Depends(get_db)) -> GeometryResponse:
    svc = GeometryService(db)
    geom = await svc.create(project_id, body)
    return GeometryResponse.model_validate(geom)


@router.get("/projects/{project_id}/geometries", response_model=List[GeometryResponse])
async def list_geometries(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> List[GeometryResponse]:
    svc = GeometryService(db)
    items = await svc.list_by_project(project_id)
    return [GeometryResponse.model_validate(g) for g in items]


@router.get("/geometries/{geometry_id}", response_model=GeometryResponse)
async def get_geometry(geometry_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> GeometryResponse:
    svc = GeometryService(db)
    geom = await svc.get(geometry_id)
    if not geom:
        raise HTTPException(status_code=404, detail="Geometry not found")
    return GeometryResponse.model_validate(geom)


@router.post("/geometries/{geometry_id}/generate", response_model=GeometryResponse)
async def generate_geometry(geometry_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> GeometryResponse:
    svc = GeometryService(db)
    geom = await svc.generate_parametric(geometry_id)
    if not geom:
        raise HTTPException(status_code=404, detail="Geometry not found or no parametric params")
    return GeometryResponse.model_validate(geom)


@router.post("/geometries/{geometry_id}/mesh", response_model=GeometryResponse)
async def mesh_geometry(geometry_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> GeometryResponse:
    svc = GeometryService(db)
    geom = await svc.generate_mesh(geometry_id)
    if not geom:
        raise HTTPException(status_code=404, detail="Geometry not found")
    return GeometryResponse.model_validate(geom)


@router.post("/geometries/upload", response_model=GeometryResponse, status_code=201)
async def upload_geometry(project_id: uuid.UUID, file: UploadFile, label: Optional[str] = None, db: AsyncSession = Depends(get_db)) -> GeometryResponse:
    if file.content_type not in ("model/step", "model/stl", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    content = await file.read()
    svc = GeometryService(db)
    geom = await svc.upload(project_id, content, file.filename or "uploaded.step", label)
    return GeometryResponse.model_validate(geom)


@router.get("/geometries/{geometry_id}/preview", response_model=MeshPreviewResponse)
async def preview_geometry(geometry_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> MeshPreviewResponse:
    svc = GeometryService(db)
    preview = await svc.get_preview(geometry_id)
    if not preview:
        raise HTTPException(status_code=404, detail="Geometry not found")
    return MeshPreviewResponse(**preview)
