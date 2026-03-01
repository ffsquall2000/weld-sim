"""Schemas for knurl optimization endpoints."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnurlOptimizeRequest(BaseModel):
    """Request body for knurl pattern optimization."""

    upper_material: str = Field(
        default="Cu", description="Upper workpiece material code"
    )
    lower_material: str = Field(
        default="Al", description="Lower workpiece material code"
    )
    upper_hardness_hv: float = Field(
        default=50.0, gt=0, description="Upper material Vickers hardness (HV)"
    )
    lower_hardness_hv: float = Field(
        default=23.0, gt=0, description="Lower material Vickers hardness (HV)"
    )
    mu_base: float = Field(
        default=0.3,
        gt=0,
        le=1.0,
        description="Base friction coefficient between workpieces",
    )
    weld_area_mm2: float = Field(
        default=75.0, gt=0, description="Nominal weld area in mm^2"
    )
    pressure_mpa: float = Field(
        default=2.0, gt=0, description="Applied clamping pressure in MPa"
    )
    amplitude_um: float = Field(
        default=30.0, gt=0, description="Vibration amplitude in micrometers"
    )
    frequency_khz: float = Field(
        default=20.0, gt=0, description="Vibration frequency in kHz"
    )
    n_layers: int = Field(
        default=1, ge=1, description="Number of foil layers in the stack"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of ranked results to return",
    )


class KnurlRecommendation(BaseModel):
    """A single knurl configuration recommendation."""

    knurl_type: str
    pitch_mm: float
    tooth_width_mm: float
    depth_mm: float
    direction: str
    effective_friction: float
    contact_pressure_mpa: float
    energy_coupling_efficiency: float
    material_damage_index: float
    overall_score: float
    rank: int


class KnurlAnalysisSummary(BaseModel):
    """Summary statistics for a knurl optimization sweep."""

    total_configs_evaluated: int
    pareto_front_size: int
    best_coupling_efficiency: float
    best_damage_index: float
    upper_material: str
    lower_material: str


class KnurlOptimizeResponse(BaseModel):
    """Response body for knurl pattern optimization."""

    recommendations: list[dict[str, Any]] = Field(
        description="Top-N ranked knurl configurations"
    )
    pareto_front: list[dict[str, Any]] = Field(
        description="Pareto-optimal configurations (energy coupling vs damage)"
    )
    analysis_summary: dict[str, Any] = Field(
        description="Statistics about the parameter sweep"
    )
