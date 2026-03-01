"""V2 knurl optimization API endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.knurl import (
    KnurlOptimizeRequest,
    KnurlOptimizeResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knurl", tags=["knurl"])

# Lazy-initialise the service from the V1 layer.  If the legacy module is not
# installed the endpoint will return 501 (Not Implemented).
try:
    from web.services.knurl_service import KnurlOptimizer

    _knurl_optimizer = KnurlOptimizer()
except ImportError:
    _knurl_optimizer = None


@router.post("/optimize", response_model=KnurlOptimizeResponse)
async def optimize_knurl(request: KnurlOptimizeRequest):
    """Find optimal knurl parameters for given materials and welding conditions.

    Performs a multi-objective parameter sweep across knurl types, pitches,
    tooth widths, and depths.  Returns ranked recommendations, the Pareto
    front (energy coupling vs. material damage), and sweep statistics.
    """
    if _knurl_optimizer is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Knurl optimization service is not available. "
                "The legacy web.services.knurl_service module could not be imported."
            ),
        )

    try:
        result = _knurl_optimizer.optimize(
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
        raise HTTPException(
            status_code=500,
            detail=f"Knurl optimization failed: {exc}",
        ) from exc
