"""Pydantic v2 schemas for weld simulation / calculation endpoints."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Input parameters for a single weld simulation."""

    application: str = Field(
        ...,
        description=(
            "Application type: li_battery_tab, li_battery_busbar, "
            "li_battery_collector, general_metal"
        ),
    )

    # --- Material stack ---
    upper_material_type: str = Field(..., description="Upper workpiece material")
    upper_thickness_mm: float = Field(..., gt=0, description="Upper workpiece thickness (mm)")
    upper_layers: int = Field(default=1, ge=1, le=200, description="Number of upper layers")
    lower_material_type: str = Field(..., description="Lower workpiece material")
    lower_thickness_mm: float = Field(..., gt=0, description="Lower workpiece thickness (mm)")

    # --- Weld geometry ---
    weld_width_mm: float = Field(default=3.0, gt=0, description="Weld width (mm)")
    weld_length_mm: float = Field(default=25.0, gt=0, description="Weld length (mm)")

    # --- Machine settings ---
    frequency_khz: float = Field(default=20.0, description="Ultrasonic frequency (kHz)")
    max_power_w: float = Field(default=3500.0, description="Maximum power (W)")

    # --- Optional horn / sonotrode ---
    horn_type: Optional[str] = Field(default=None, description="Horn type identifier")
    horn_gain: Optional[float] = Field(default=None, description="Horn gain ratio")
    knurl_type: Optional[str] = Field(default=None, description="Knurl pattern type")
    knurl_pitch_mm: Optional[float] = Field(default=None, description="Knurl pitch (mm)")
    knurl_tooth_width_mm: Optional[float] = Field(
        default=None, description="Knurl tooth width (mm)"
    )
    knurl_depth_mm: Optional[float] = Field(default=None, description="Knurl depth (mm)")

    # --- Optional chamfer / edge treatment ---
    chamfer_radius_mm: Optional[float] = Field(default=None, description="Chamfer radius (mm)")
    chamfer_angle_deg: Optional[float] = Field(default=None, description="Chamfer angle (deg)")
    edge_treatment: Optional[Literal["none", "chamfer", "fillet", "compound"]] = Field(
        default=None, description="Edge treatment method"
    )

    # --- Optional anvil ---
    anvil_type: Optional[str] = Field(default=None, description="Anvil type identifier")
    anvil_resonant_freq_khz: Optional[float] = Field(
        default=None, description="Anvil resonant frequency (kHz)"
    )

    # --- Optional booster / cylinder ---
    booster_gain: Optional[float] = Field(default=None, description="Booster gain ratio")
    cylinder_bore_mm: Optional[float] = Field(
        default=None, description="Pneumatic cylinder bore (mm)"
    )
    cylinder_min_air_bar: Optional[float] = Field(
        default=None, description="Minimum air pressure (bar)"
    )
    cylinder_max_air_bar: Optional[float] = Field(
        default=None, description="Maximum air pressure (bar)"
    )


class ValidationInfo(BaseModel):
    """Validation status for a simulation result."""

    status: str = Field(..., description="Validation status (pass, warn, fail)")
    messages: list[str] = Field(default_factory=list, description="Validation messages")


class SimulateResponse(BaseModel):
    """Full response from a single weld simulation."""

    recipe_id: str = Field(..., description="Unique recipe identifier")
    application: str = Field(..., description="Application type used")
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
    validation: ValidationInfo = Field(..., description="Validation information")
    created_at: str = Field(..., description="ISO-8601 creation timestamp")


class BatchSimulateRequest(BaseModel):
    """Batch of simulation requests."""

    items: list[SimulateRequest] = Field(
        ..., min_length=1, description="List of simulation requests"
    )


class BatchSimulateResponse(BaseModel):
    """Results from a batch simulation run."""

    results: list[SimulateResponse] = Field(
        default_factory=list, description="Successful simulation results"
    )
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Errors indexed by request position",
    )
