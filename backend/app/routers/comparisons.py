"""Endpoints for Comparison resources."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.comparison import ComparisonCreate, ComparisonResponse
from backend.app.services.comparison_service import ComparisonService

router = APIRouter(tags=["comparisons"])


@router.post("/projects/{project_id}/comparisons", response_model=ComparisonResponse, status_code=201)
async def create_comparison(project_id: uuid.UUID, body: ComparisonCreate, db: AsyncSession = Depends(get_db)) -> ComparisonResponse:
    svc = ComparisonService(db)
    comp = await svc.create(project_id, body)
    return ComparisonResponse.model_validate(comp)


@router.get("/comparisons/{comparison_id}", response_model=ComparisonResponse)
async def get_comparison(comparison_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> ComparisonResponse:
    svc = ComparisonService(db)
    comp = await svc.get(comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return ComparisonResponse.model_validate(comp)


@router.post("/comparisons/{comparison_id}/refresh", response_model=ComparisonResponse)
async def refresh_comparison(comparison_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> ComparisonResponse:
    svc = ComparisonService(db)
    comp = await svc.refresh(comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")
    return ComparisonResponse.model_validate(comp)
