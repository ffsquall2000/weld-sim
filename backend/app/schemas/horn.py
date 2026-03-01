"""Pydantic v2 schemas for horn generation and chamfer analysis endpoints."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class HornGenerateRequest(BaseModel):
    """Request body for parametric horn generation."""

    horn_type: str = Field(
        default="flat",
        description="Horn shape type: flat, cylindrical, exponential, blade, stepped",
    )
    width_mm: float = Field(default=25.0, gt=0, description="Horn width in mm")
    height_mm: float = Field(default=80.0, gt=0, description="Horn height in mm")
    length_mm: float = Field(default=25.0, gt=0, description="Horn length in mm")
    material: str = Field(
        default="Titanium Ti-6Al-4V", description="Horn material name"
    )
    # Knurl parameters
    knurl_type: str = Field(
        default="none",
        description="Knurl pattern: none, linear, cross_hatch, diamond, conical, spherical",
    )
    knurl_pitch_mm: float = Field(
        default=1.0, gt=0, description="Knurl pitch in mm"
    )
    knurl_tooth_width_mm: float = Field(
        default=0.5, gt=0, description="Knurl tooth width in mm"
    )
    knurl_depth_mm: float = Field(
        default=0.3, ge=0, description="Knurl depth in mm"
    )
    knurl_direction: str = Field(
        default="perpendicular", description="Knurl direction"
    )
    # Chamfer parameters
    chamfer_radius_mm: float = Field(
        default=0.0, ge=0, description="Chamfer radius in mm"
    )
    chamfer_angle_deg: float = Field(
        default=45.0, gt=0, le=90, description="Chamfer angle in degrees"
    )
    edge_treatment: Optional[Literal["none", "chamfer", "fillet", "compound"]] = Field(
        default="none", description="Edge treatment type"
    )


class HornGenerateResponse(BaseModel):
    """Response body from horn generation."""

    horn_type: str
    dimensions: dict[str, Any]
    knurl_info: dict[str, Any]
    chamfer_info: dict[str, Any]
    volume_mm3: float
    surface_area_mm2: float
    has_cad_export: bool
    mesh: dict[str, Any] = Field(
        description="Mesh data: {vertices: [[x,y,z],...], faces: [[a,b,c],...]}"
    )
    download_id: Optional[str] = Field(
        default=None, description="ID for downloading STEP/STL export files"
    )


class ChamferAnalysisRequest(BaseModel):
    """Request body for chamfer impact analysis on welding."""

    contact_width_mm: float = Field(
        default=25.0, gt=0, description="Contact surface width in mm"
    )
    contact_length_mm: float = Field(
        default=25.0, gt=0, description="Contact surface length in mm"
    )
    chamfer_radius_mm: float = Field(
        default=0.0, ge=0, description="Chamfer radius in mm"
    )
    chamfer_angle_deg: float = Field(
        default=45.0, gt=0, le=90, description="Chamfer angle in degrees"
    )
    edge_treatment: Optional[Literal["none", "chamfer", "fillet", "compound"]] = Field(
        default="none", description="Edge treatment type"
    )
    # Welding context for damage analysis
    pressure_mpa: float = Field(
        default=2.0, gt=0, description="Welding pressure in MPa"
    )
    amplitude_um: float = Field(
        default=30.0, gt=0, description="Vibration amplitude in micrometers"
    )
    material_yield_mpa: float = Field(
        default=70.0, gt=0, description="Material yield strength in MPa"
    )
    weld_time_s: float = Field(
        default=0.2, gt=0, description="Weld time in seconds"
    )


class ChamferAnalysisResponse(BaseModel):
    """Response body from chamfer analysis."""

    stress_concentration_factor: float = Field(
        description="Stress concentration factor Kt"
    )
    risk_level: str = Field(description="Risk level: low, medium, high, critical")
    peak_stress_mpa: float = Field(description="Peak stress at chamfer edge in MPa")
    damage_ratio: float = Field(description="Material damage ratio")
    contact_area_nominal_mm2: float = Field(
        description="Nominal contact area in mm^2"
    )
    contact_area_corrected_mm2: float = Field(
        description="Corrected contact area after chamfer in mm^2"
    )
    area_reduction_percent: float = Field(
        description="Contact area reduction percentage"
    )
    energy_redistribution_factor: float = Field(
        description="Energy redistribution factor due to chamfer"
    )
    recommendations: list[str] = Field(
        description="List of recommendations for improving the chamfer design"
    )
