"""Endpoints for Material resources."""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.material import MaterialCreate, MaterialResponse
from backend.app.services.material_service import BUILTIN_FEA_MATERIALS, MaterialService

router = APIRouter(prefix="/materials", tags=["materials"])


@router.get("/", response_model=List[MaterialResponse])
async def list_materials(
    category: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> List[MaterialResponse]:
    svc = MaterialService(db)
    items = await svc.list_all(category, search)
    return [MaterialResponse(**m) for m in items]


@router.get("/fea", response_model=List[MaterialResponse])
async def list_fea_materials() -> List[MaterialResponse]:
    return [MaterialResponse(**m) for m in BUILTIN_FEA_MATERIALS]


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(material_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> MaterialResponse:
    svc = MaterialService(db)
    mat = await svc.get(material_id)
    if not mat:
        raise HTTPException(status_code=404, detail="Material not found")
    return MaterialResponse.model_validate(mat)


@router.post("/", response_model=MaterialResponse, status_code=201)
async def create_material(body: MaterialCreate, db: AsyncSession = Depends(get_db)) -> MaterialResponse:
    svc = MaterialService(db)
    mat = await svc.create(body)
    return MaterialResponse.model_validate(mat)
