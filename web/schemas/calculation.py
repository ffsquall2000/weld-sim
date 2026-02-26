"""Pydantic request/response models for calculation endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Input parameters for a weld simulation."""

    application: str  # li_battery_tab, li_battery_busbar, li_battery_collector, general_metal
    upper_material_type: str
    upper_thickness_mm: float = Field(gt=0)
    upper_layers: int = Field(default=1, ge=1, le=200)
    lower_material_type: str
    lower_thickness_mm: float = Field(gt=0)
    weld_width_mm: float = Field(default=3.0, gt=0)
    weld_length_mm: float = Field(default=25.0, gt=0)
    frequency_khz: float = 20.0
    max_power_w: float = 3500.0
    # Optional sonotrode/horn
    horn_type: Optional[str] = None
    horn_gain: Optional[float] = None
    knurl_type: Optional[str] = None
    knurl_pitch_mm: Optional[float] = None
    knurl_tooth_width_mm: Optional[float] = None
    knurl_depth_mm: Optional[float] = None
    # Optional anvil
    anvil_type: Optional[str] = None
    anvil_resonant_freq_khz: Optional[float] = None
    # Optional booster/cylinder
    booster_gain: Optional[float] = None
    cylinder_bore_mm: Optional[float] = None
    cylinder_min_air_bar: Optional[float] = None
    cylinder_max_air_bar: Optional[float] = None


class ValidationResponse(BaseModel):
    """Validation status returned alongside a simulation result."""

    status: str
    messages: list[str] = []


class SimulateResponse(BaseModel):
    """Full response from a simulation run."""

    recipe_id: str
    application: str
    parameters: dict
    safety_window: dict
    risk_assessment: dict
    quality_estimate: dict
    recommendations: list[str]
    validation: ValidationResponse
    created_at: str


class BatchSimulateRequest(BaseModel):
    """Batch of simulation requests."""

    items: list[SimulateRequest]


class BatchSimulateResponse(BaseModel):
    """Results from a batch simulation."""

    results: list[SimulateResponse]
    errors: list[dict] = []
