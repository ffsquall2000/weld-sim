"""Schemas for manual optimization endpoints."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ManualSessionCreate(BaseModel):
    """Request body for creating a manual optimization session."""

    project_id: str
    baseline_params: dict[str, float] = Field(
        ..., description="Starting parameter values from simulation"
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Material/application context for knowledge rules",
    )


class ManualSessionResponse(BaseModel):
    """Response body for a manual optimization session."""

    session_id: str
    project_id: str
    strategy: str = "manual_guided"
    baseline_params: dict[str, float]
    current_params: dict[str, float]
    iteration_count: int
    status: str
    created_at: str


class SuggestionRequest(BaseModel):
    """Request body for generating parameter suggestions."""

    current_params: dict[str, float]
    baseline_params: dict[str, float]
    context: dict[str, Any] = Field(default_factory=dict)
    trial_history: list[dict[str, Any]] | None = None


class SuggestionItem(BaseModel):
    """A single parameter adjustment suggestion."""

    parameter: str
    current_value: float
    suggested_value: float
    unit: str = ""
    reason: str = ""
    priority: str = "medium"
    confidence: float = 0.7
    direction: str = ""
    source: str = "deviation_analysis"
    rule_id: str | None = None


class DeviationInfo(BaseModel):
    """Deviation information for a single parameter."""

    current: float
    baseline: float
    deviation_pct: float


class SafetyInfo(BaseModel):
    """Safety window status for a single parameter."""

    in_window: bool
    safe_min: float
    safe_max: float


class KnowledgeRecommendation(BaseModel):
    """A matched knowledge rule recommendation."""

    rule_id: str
    description: str
    adjustments: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    priority: int = 0


class QualityTrend(BaseModel):
    """Quality trend analysis result."""

    trend: str
    message: str


class SuggestionResponse(BaseModel):
    """Response body for parameter suggestions."""

    suggestions: list[SuggestionItem]
    deviations: dict[str, DeviationInfo]
    safety_status: dict[str, SafetyInfo]
    knowledge_recommendations: list[KnowledgeRecommendation]
    quality_trend: QualityTrend


class RecordIterationRequest(BaseModel):
    """Request body for recording an optimization iteration."""

    params: dict[str, float]
    results: dict[str, Any]
    notes: str = ""


class IterationRecord(BaseModel):
    """A single recorded iteration."""

    iteration: int
    params: dict[str, float]
    results: dict[str, Any]
    notes: str
    timestamp: str


class TrendValue(BaseModel):
    """Trend data for a single parameter or quality metric."""

    values: list[float]
    current: float
    min: float | None = None
    max: float | None = None
    best: float | None = None
    trend: str


class TrendAnalysisResponse(BaseModel):
    """Response body for trend analysis."""

    parameter_trends: dict[str, TrendValue]
    quality_trends: dict[str, TrendValue]
    best_iteration: int | None
    recommendation: str
