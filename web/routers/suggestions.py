"""Real-time welding suggestion endpoints."""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])


class SuggestionRequest(BaseModel):
    """Request for parameter suggestions."""
    experiment_params: dict  # Current welding parameters
    simulation_recipe: dict  # Simulation recipe (from /simulate endpoint)
    trial_history: Optional[list[dict]] = None  # Previous trial results


class SuggestionResponse(BaseModel):
    """Response with prioritized suggestions."""
    suggestions: list[dict]
    deviations: dict
    safety_status: dict
    quality_trend: dict


@router.post("/analyze", response_model=SuggestionResponse)
async def analyze_suggestions(request: SuggestionRequest):
    """Analyze experiment parameters and generate adjustment suggestions."""
    try:
        from web.services.suggestion_service import SuggestionService
        svc = SuggestionService()
        result = svc.generate_suggestions(
            experiment_params=request.experiment_params,
            simulation_recipe=request.simulation_recipe,
            trial_history=request.trial_history,
        )
        return SuggestionResponse(**result)
    except Exception as exc:
        logger.error("Suggestion analysis error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Suggestion analysis failed: {exc}") from exc
