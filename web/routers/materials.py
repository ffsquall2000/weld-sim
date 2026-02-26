"""Material query endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from web.dependencies import get_engine_service
from web.services.engine_service import EngineService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["materials"])


@router.get("/materials")
async def list_materials(
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Return all available material type names."""
    materials = svc.get_materials()
    return {"materials": materials}


@router.get("/materials/combination/{mat_a}/{mat_b}")
async def get_combination(
    mat_a: str,
    mat_b: str,
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Return combination properties for two material types."""
    result = svc.get_material_combination(mat_a, mat_b)
    return result


@router.get("/materials/{material_type}")
async def get_material(
    material_type: str,
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Return properties for a single material type."""
    result = svc.get_material(material_type)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Material '{material_type}' not found",
        )
    return {"material_type": material_type, "properties": result}
