"""V2 weld simulation / calculation endpoints.

Wraps the V1 EngineService with Pydantic v2 schemas and the /api/v2 prefix
convention used by the new backend.
"""
from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, HTTPException

from backend.app.schemas.calculation import (
    BatchSimulateRequest,
    BatchSimulateResponse,
    SimulateRequest,
    SimulateResponse,
    ValidationInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/calculation", tags=["calculation"])

# ---------------------------------------------------------------------------
# Lazy-load the V1 engine service singleton.  The import may fail when the
# legacy ``web`` package is not installed, so we degrade gracefully.
# ---------------------------------------------------------------------------
try:
    from web.dependencies import get_engine_service

    _engine_service = get_engine_service()
except (ImportError, Exception):
    _engine_service = None
    logger.warning(
        "V1 EngineService not available; /calculation endpoints will return 503"
    )


def _get_engine_service():
    """Return the cached engine service or raise 503."""
    if _engine_service is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation engine is not available",
        )
    return _engine_service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_response(recipe, validation) -> SimulateResponse:
    """Convert a core WeldRecipe + ValidationResult into a SimulateResponse."""
    return SimulateResponse(
        recipe_id=recipe.recipe_id,
        application=recipe.application,
        parameters=recipe.parameters,
        safety_window=recipe.safety_window,
        risk_assessment=recipe.risk_assessment,
        quality_estimate=recipe.quality_estimate,
        recommendations=recipe.recommendations,
        validation=ValidationInfo(
            status=validation.status.value,
            messages=validation.messages,
        ),
        created_at=recipe.created_at,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/simulate", response_model=SimulateResponse)
async def simulate(request: SimulateRequest) -> SimulateResponse:
    """Run a single weld simulation.

    Accepts material stack, geometry, and machine parameters and returns the
    full recipe with safety window, risk assessment, and quality estimate.
    """
    svc = _get_engine_service()
    try:
        inputs = request.model_dump(exclude_none=True)
        recipe, validation = svc.calculate(request.application, inputs)
        return _build_response(recipe, validation)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Simulation error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/simulate/batch", response_model=BatchSimulateResponse)
async def simulate_batch(request: BatchSimulateRequest) -> BatchSimulateResponse:
    """Run multiple simulations in a single request.

    Each item is processed independently; failures are collected in the
    ``errors`` list without aborting the remaining items.
    """
    svc = _get_engine_service()
    results: list[SimulateResponse] = []
    errors: list[dict] = []

    for idx, item in enumerate(request.items):
        try:
            inputs = item.model_dump(exclude_none=True)
            recipe, validation = svc.calculate(item.application, inputs)
            results.append(_build_response(recipe, validation))
        except Exception as exc:
            logger.warning("Batch item %d failed: %s", idx, exc)
            errors.append({"index": idx, "error": str(exc)})

    return BatchSimulateResponse(results=results, errors=errors)


@router.get("/simulate/schema/{application}")
async def get_schema(application: str) -> dict:
    """Return the input JSON Schema for a given application type.

    This is useful for dynamically building UI forms based on the
    application's required and optional parameters.
    """
    svc = _get_engine_service()
    try:
        schema = svc.get_input_schema(application)
        return schema
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Schema error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc
