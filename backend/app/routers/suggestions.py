"""V2 real-time welding suggestion analysis API endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.suggestions import (
    SuggestionAnalysisRequest,
    SuggestionAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/suggestions", tags=["suggestions"])

# Lazy-initialise the service from the V1 layer.  If the legacy module is not
# installed the endpoint will return 501 (Not Implemented).
try:
    from web.services.suggestion_service import SuggestionService

    _suggestion_service = SuggestionService()
except ImportError:
    _suggestion_service = None


@router.post("/analyze", response_model=SuggestionAnalysisResponse)
async def analyze_suggestions(request: SuggestionAnalysisRequest):
    """Analyze experiment parameters against simulation and generate suggestions.

    Compares actual experiment parameters with simulation recommendations,
    checks safety windows, analyzes quality trends from trial history, and
    returns prioritized adjustment suggestions.
    """
    if _suggestion_service is None:
        raise HTTPException(
            status_code=501,
            detail=(
                "Suggestion analysis service is not available. "
                "The legacy web.services.suggestion_service module could not be imported."
            ),
        )

    try:
        result = _suggestion_service.generate_suggestions(
            experiment_params=request.experiment_params,
            simulation_recipe=request.simulation_recipe,
            trial_history=request.trial_history,
        )
        return SuggestionAnalysisResponse(**result)
    except Exception as exc:
        logger.error("Suggestion analysis error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Suggestion analysis failed: {exc}",
        ) from exc
