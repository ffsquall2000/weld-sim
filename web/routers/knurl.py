"""Knurl optimization endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knurl", tags=["knurl"])


class KnurlOptimizeRequest(BaseModel):
    upper_material: str = "Cu"
    lower_material: str = "Al"
    upper_hardness_hv: float = Field(default=50.0, gt=0)
    lower_hardness_hv: float = Field(default=23.0, gt=0)
    mu_base: float = Field(default=0.3, gt=0, le=1.0)
    weld_area_mm2: float = Field(default=75.0, gt=0)
    pressure_mpa: float = Field(default=2.0, gt=0)
    amplitude_um: float = Field(default=30.0, gt=0)
    frequency_khz: float = Field(default=20.0, gt=0)
    n_layers: int = Field(default=1, ge=1)
    max_results: int = Field(default=10, ge=1, le=50)


class KnurlOptimizeResponse(BaseModel):
    recommendations: list[dict]
    pareto_front: list[dict]
    analysis_summary: dict


@router.post("/optimize", response_model=KnurlOptimizeResponse)
async def optimize_knurl(request: KnurlOptimizeRequest):
    """Find optimal knurl parameters for given materials and requirements."""
    try:
        from web.services.knurl_service import KnurlOptimizer
        optimizer = KnurlOptimizer()
        result = optimizer.optimize(
            upper_material=request.upper_material,
            lower_material=request.lower_material,
            upper_hardness_hv=request.upper_hardness_hv,
            lower_hardness_hv=request.lower_hardness_hv,
            mu_base=request.mu_base,
            weld_area_mm2=request.weld_area_mm2,
            pressure_mpa=request.pressure_mpa,
            amplitude_um=request.amplitude_um,
            frequency_khz=request.frequency_khz,
            n_layers=request.n_layers,
            max_results=request.max_results,
        )
        return KnurlOptimizeResponse(**result)
    except Exception as exc:
        logger.error("Knurl optimization error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Knurl optimization failed: {exc}") from exc
