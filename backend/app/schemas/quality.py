"""Pydantic schemas for the welding quality assessment endpoint."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class QualityAssessmentRequest(BaseModel):
    """Input: simulation results to evaluate against industry standards."""

    application_type: str = Field(
        default="li_battery_tab",
        description=(
            "Application type for the weld. "
            "One of: li_battery_tab, busbar, collector, general_metal"
        ),
    )
    parameters: Dict[str, float] = Field(
        ...,
        description=(
            "Welding parameters to evaluate. Common keys: "
            "amplitude_um, pressure_mpa, energy_j, time_ms, "
            "frequency_deviation_pct, temperature_rise_c, "
            "amplitude_uniformity_pct, stress_safety_factor"
        ),
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Standardised simulation metrics if already available.",
    )
    material_upper: Optional[str] = Field(
        default=None, description="Upper workpiece material, e.g. 'Cu', 'Al', 'Ni'."
    )
    material_lower: Optional[str] = Field(
        default=None, description="Lower workpiece material, e.g. 'Al', 'Cu'."
    )
    total_thickness_mm: Optional[float] = Field(
        default=None, description="Total material stack thickness in mm."
    )


class CriterionResult(BaseModel):
    """Result of evaluating a single quality criterion."""

    name: str = Field(..., description="Machine-readable criterion identifier.")
    description: str = Field(..., description="Human-readable criterion description.")
    value: Optional[float] = Field(
        default=None, description="Measured / simulated value."
    )
    unit: str = Field(default="", description="Physical unit of the value.")
    min_threshold: Optional[float] = Field(
        default=None, description="Lower acceptable limit."
    )
    max_threshold: Optional[float] = Field(
        default=None, description="Upper acceptable limit."
    )
    status: str = Field(
        ...,
        description='Evaluation result: "pass", "warning", "fail", or "not_tested".',
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of the evaluation outcome."
    )


class AnalysisSection(BaseModel):
    """A logical grouping of criteria (e.g. 'Process Parameters')."""

    title: str
    items: List[CriterionResult]
    summary: str


class QualityAssessmentResponse(BaseModel):
    """Full quality assessment report returned to the client."""

    overall_score: float = Field(
        ..., ge=0, le=100, description="Aggregate quality score (0-100)."
    )
    overall_verdict: str = Field(
        ..., description='Overall verdict: "pass", "warning", or "fail".'
    )
    verdict_description: str = Field(
        ..., description="Short narrative describing the overall quality."
    )
    sections: List[AnalysisSection] = Field(
        ..., description="Detailed per-section analysis results."
    )
    recommendations: List[str] = Field(
        ..., description="Actionable recommendations for improvement."
    )
    application_notes: str = Field(
        ..., description="Application-specific guidance and warnings."
    )
