"""Schemas for assembly analysis endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ComponentRequest(BaseModel):
    """A single component in the ultrasonic welding assembly stack."""

    name: str = Field(
        ..., description="Component name: horn, booster, or transducer"
    )
    horn_type: str = Field(
        default="cylindrical",
        description="Geometry type for GmshMesher",
    )
    dimensions: dict = Field(
        ..., description="Horn dimensions in mm (keys vary by horn_type)"
    )
    material_name: str = Field(
        default="Titanium Ti-6Al-4V",
        description="Material name for FEA properties lookup",
    )
    mesh_size: float = Field(
        default=2.0, gt=0, description="Target mesh element size in mm"
    )


class AssemblyAnalysisRequest(BaseModel):
    """Request body for full assembly analysis of a multi-component stack."""

    components: list[ComponentRequest] = Field(
        ..., description="List of assembly components to analyze"
    )
    coupling_method: str = Field(
        default="bonded",
        description="Component coupling method",
    )
    penalty_factor: float = Field(
        default=1e3, gt=0, description="Penalty factor for coupling constraints"
    )
    analyses: list[str] = Field(
        default=["modal", "harmonic"],
        description="Analyses to run: modal, harmonic",
    )
    frequency_hz: float = Field(
        default=20000.0, gt=0, description="Target operating frequency in Hz"
    )
    n_modes: int = Field(
        default=20, ge=1, description="Number of modes to compute"
    )
    damping_ratio: float = Field(
        default=0.005, ge=0, description="Damping ratio for harmonic analysis"
    )
    use_gmsh: bool = Field(
        default=True,
        description="Use Gmsh TET10 pipeline (True) or legacy solver (False)",
    )


class AssemblyAnalysisResponse(BaseModel):
    """Response body for assembly analysis results."""

    success: bool = Field(..., description="Whether analysis completed successfully")
    message: str = Field(..., description="Status or error message")
    n_total_dof: int = Field(
        default=0, description="Total degrees of freedom in the assembly"
    )
    n_components: int = Field(
        default=0, description="Number of components in the assembly"
    )
    # Modal results
    frequencies_hz: list[float] = Field(
        default_factory=list,
        description="Natural frequencies in Hz",
    )
    mode_types: list[str] = Field(
        default_factory=list,
        description="Classification of each mode (longitudinal, flexural, etc.)",
    )
    # Harmonic results
    resonance_frequency_hz: float = Field(
        default=0.0, description="Primary resonance frequency in Hz"
    )
    gain: float = Field(default=0.0, description="Assembly amplitude gain")
    q_factor: float = Field(default=0.0, description="Quality factor")
    uniformity: float = Field(
        default=0.0, description="Output amplitude uniformity (0-1)"
    )
    # Gain chain
    gain_chain: dict = Field(
        default_factory=dict,
        description="Per-component amplitude gain values",
    )
    # Impedance
    impedance: dict = Field(
        default_factory=dict,
        description="Impedance data per component",
    )
    transmission_coefficients: dict = Field(
        default_factory=dict,
        description="Transmission coefficients between components",
    )
    # Timing
    solve_time_s: float = Field(
        default=0.0, description="Total solver wall-clock time in seconds"
    )


class MaterialListItem(BaseModel):
    """Material summary for listing available FEA materials."""

    name: str = Field(..., description="Canonical material name")
    E_gpa: float = Field(..., description="Young's modulus in GPa")
    density_kg_m3: float = Field(..., description="Density in kg/m^3")
    poisson_ratio: float = Field(..., description="Poisson's ratio")
    acoustic_velocity_m_s: Optional[float] = Field(
        default=None, description="Acoustic velocity in m/s (if available)"
    )


class BoosterProfileItem(BaseModel):
    """Booster profile type descriptor."""

    profile: str = Field(..., description="Profile name identifier")
    description: str = Field(
        ..., description="Human-readable description of the profile"
    )
