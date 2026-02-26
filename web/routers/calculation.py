"""Calculation / simulation endpoints."""
from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException

from web.dependencies import get_engine_service
from web.schemas.calculation import (
    BatchSimulateRequest,
    BatchSimulateResponse,
    SimulateRequest,
    SimulateResponse,
    ValidationResponse,
)
from web.services.engine_service import EngineService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["calculation"])


def _build_response(recipe, validation) -> SimulateResponse:
    """Convert core WeldRecipe + ValidationResult into a SimulateResponse."""
    return SimulateResponse(
        recipe_id=recipe.recipe_id,
        application=recipe.application,
        parameters=recipe.parameters,
        safety_window=recipe.safety_window,
        risk_assessment=recipe.risk_assessment,
        quality_estimate=recipe.quality_estimate,
        recommendations=recipe.recommendations,
        validation=ValidationResponse(
            status=validation.status.value,
            messages=validation.messages,
        ),
        created_at=recipe.created_at,
    )


@router.post("/simulate", response_model=SimulateResponse)
async def simulate(
    request: SimulateRequest,
    svc: EngineService = Depends(get_engine_service),
) -> SimulateResponse:
    """Run a single weld simulation."""
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
async def simulate_batch(
    request: BatchSimulateRequest,
    svc: EngineService = Depends(get_engine_service),
) -> BatchSimulateResponse:
    """Run multiple simulations in one request."""
    results: list[SimulateResponse] = []
    errors: list[dict] = []

    for idx, item in enumerate(request.items):
        try:
            inputs = item.model_dump(exclude_none=True)
            recipe, validation = svc.calculate(item.application, inputs)
            results.append(_build_response(recipe, validation))
        except Exception as exc:
            errors.append({"index": idx, "error": str(exc)})

    return BatchSimulateResponse(results=results, errors=errors)


@router.get("/simulate/schema/{application}")
async def get_schema(
    application: str,
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Return the input JSON Schema for an application."""
    try:
        schema = svc.get_input_schema(application)
        return schema
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Schema error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc
