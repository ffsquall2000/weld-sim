"""Schemas for real-time welding suggestion analysis endpoints."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SuggestionAnalysisRequest(BaseModel):
    """Request body for welding parameter suggestion analysis."""

    experiment_params: dict[str, Any] = Field(
        ...,
        description=(
            "Current welding parameters from the experiment "
            "(e.g. amplitude_um, pressure_mpa, energy_j, time_ms)"
        ),
    )
    simulation_recipe: dict[str, Any] = Field(
        ...,
        description=(
            "Recommended parameters from simulation. "
            "Contains 'parameters' dict and optional 'safety_window' / 'risk_assessment' dicts."
        ),
    )
    trial_history: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description=(
            "Optional list of previous trial results. "
            "Each dict may include trial_number, parameters, and quality metrics."
        ),
    )


class SuggestionItem(BaseModel):
    """A single parameter adjustment suggestion."""

    parameter: str
    current_value: float
    suggested_value: float
    unit: str = ""
    reason: str = ""
    priority: str = "medium"
    confidence: float = 0.7


class DeviationDetail(BaseModel):
    """Deviation information for a single parameter."""

    actual: float
    recommended: float
    difference: float
    percent: float


class SafetyWindowStatus(BaseModel):
    """Safety window status for a single parameter."""

    within_window: bool
    current: float
    window_low: float
    window_high: float
    deviation_from_center: float


class QualityTrendDetail(BaseModel):
    """Quality trend analysis result."""

    declining: bool = False
    improving: bool = False
    stable: bool = True
    trend_percent: float = 0.0
    trial_count: int = 0
    quality_scores: list[float] = Field(default_factory=list)
    reason: str = ""


class SuggestionAnalysisResponse(BaseModel):
    """Response body for welding parameter suggestion analysis."""

    suggestions: list[dict[str, Any]] = Field(
        description="Prioritized parameter adjustment suggestions"
    )
    deviations: dict[str, Any] = Field(
        description="Parameter deviations from simulation recommendations"
    )
    safety_status: dict[str, Any] = Field(
        description="Safety window status for each checked parameter"
    )
    quality_trend: dict[str, Any] = Field(
        description="Quality trend analysis over trial history"
    )
