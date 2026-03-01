"""Pydantic v2 schemas for recipe management endpoints."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RecipeListResponse(BaseModel):
    """Response for listing recipes."""

    recipes: list[dict[str, Any]] = Field(
        default_factory=list, description="List of recipe summaries"
    )
    count: int = Field(..., description="Total number of recipes returned")


class RecipeDetailResponse(BaseModel):
    """Full detail response for a single recipe."""

    id: Optional[str] = Field(default=None, description="Database row ID or recipe ID")
    recipe_id: Optional[str] = Field(default=None, description="Unique recipe identifier")
    application: Optional[str] = Field(default=None, description="Application type")
    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Original input parameters"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Calculated welding parameters"
    )
    safety_window: dict[str, Any] = Field(
        default_factory=dict, description="Safe operating ranges"
    )
    risk_assessment: dict[str, Any] = Field(
        default_factory=dict, description="Risk assessment results"
    )
    quality_estimate: dict[str, Any] = Field(
        default_factory=dict, description="Estimated quality metrics"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Process recommendations"
    )
    created_at: Optional[str] = Field(
        default=None, description="ISO-8601 creation timestamp"
    )
