"""Pydantic models for horn generation and chamfer analysis endpoints."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class HornGenerateRequest(BaseModel):
    """Request to generate a parametric horn."""

    horn_type: str = "flat"  # flat | cylindrical | exponential | blade | stepped
    width_mm: float = Field(default=25.0, gt=0)
    height_mm: float = Field(default=80.0, gt=0)
    length_mm: float = Field(default=25.0, gt=0)
    material: str = "Titanium Ti-6Al-4V"
    # Knurl parameters
    knurl_type: str = "none"  # none | linear | cross_hatch | diamond | conical | spherical
    knurl_pitch_mm: float = Field(default=1.0, gt=0)
    knurl_tooth_width_mm: float = Field(default=0.5, gt=0)
    knurl_depth_mm: float = Field(default=0.3, ge=0)
    knurl_direction: str = "perpendicular"
    # Chamfer parameters
    chamfer_radius_mm: float = Field(default=0.0, ge=0)
    chamfer_angle_deg: float = Field(default=45.0, gt=0, le=90)
    edge_treatment: Optional[Literal["none", "chamfer", "fillet", "compound"]] = "none"


class HornGenerateResponse(BaseModel):
    """Response from horn generation."""

    horn_type: str
    dimensions: dict
    knurl_info: dict
    chamfer_info: dict
    volume_mm3: float
    surface_area_mm2: float
    has_cad_export: bool
    mesh: dict  # {vertices: [[x,y,z],...], faces: [[a,b,c],...]}
    download_id: Optional[str] = None  # ID for downloading STEP/STL files


class ChamferAnalysisRequest(BaseModel):
    """Request for chamfer impact analysis."""

    horn_type: str = "flat"
    contact_width_mm: float = Field(default=25.0, gt=0)
    contact_length_mm: float = Field(default=25.0, gt=0)
    chamfer_radius_mm: float = Field(default=0.0, ge=0)
    chamfer_angle_deg: float = Field(default=45.0, gt=0, le=90)
    edge_treatment: Optional[Literal["none", "chamfer", "fillet", "compound"]] = "none"
    # Welding context for damage analysis
    pressure_mpa: float = Field(default=2.0, gt=0)
    amplitude_um: float = Field(default=30.0, gt=0)
    material_yield_mpa: float = Field(default=70.0, gt=0)
    weld_time_s: float = Field(default=0.2, gt=0)


class ChamferAnalysisResponse(BaseModel):
    """Response from chamfer analysis."""

    stress_concentration_factor: float
    risk_level: str
    peak_stress_mpa: float
    damage_ratio: float
    contact_area_nominal_mm2: float
    contact_area_corrected_mm2: float
    area_reduction_percent: float
    energy_redistribution_factor: float
    recommendations: list[str]
