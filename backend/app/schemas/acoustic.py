"""Schemas for acoustic analysis endpoints."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class StressHotspot(BaseModel):
    """A stress concentration point in the horn geometry."""

    location: list[float] = Field(
        ..., description="[x, y, z] coordinates in mm"
    )
    von_mises_mpa: float = Field(
        ..., description="Von Mises stress at this point in MPa"
    )
    node_index: int = Field(
        ..., description="FEA mesh node index"
    )


class AcousticAnalysisRequest(BaseModel):
    """Request body for acoustic analysis of an ultrasonic horn."""

    horn_type: str = Field(
        default="cylindrical",
        description="Horn geometry type (cylindrical, block, etc.)",
    )
    width_mm: float = Field(
        default=25.0, gt=0, description="Horn width/diameter in mm"
    )
    height_mm: float = Field(
        default=80.0, gt=0, description="Horn height/length in mm"
    )
    length_mm: float = Field(
        default=25.0, gt=0, description="Horn depth in mm"
    )
    material: str = Field(
        default="Titanium Ti-6Al-4V",
        description="Material name for FEA properties lookup",
    )
    frequency_khz: float = Field(
        default=20.0, gt=0, description="Target operating frequency in kHz"
    )
    mesh_density: str = Field(
        default="medium",
        description="Mesh density preset: coarse, medium, or fine",
    )
    use_gmsh: bool = Field(
        default=True,
        description=(
            "Use Gmsh TET10 + SolverA pipeline (True) or legacy HEX8 solver (False)"
        ),
    )


class AcousticAnalysisResponse(BaseModel):
    """Response body for acoustic analysis results."""

    # Modal analysis
    modes: list[dict] = Field(
        ..., description="List of computed vibration modes"
    )
    closest_mode_hz: float = Field(
        ..., description="Frequency of mode closest to target in Hz"
    )
    target_frequency_hz: float = Field(
        ..., description="Requested target frequency in Hz"
    )
    frequency_deviation_percent: float = Field(
        ...,
        description="Deviation of closest mode from target as percentage",
    )
    # Harmonic response
    harmonic_response: dict = Field(
        ...,
        description="Harmonic response data: {frequencies_hz, amplitudes}",
    )
    # Amplitude distribution
    amplitude_distribution: dict = Field(
        ...,
        description=(
            "Amplitude distribution: {node_positions: [[x,y,z],...], amplitudes: [...]}"
        ),
    )
    amplitude_uniformity: float = Field(
        ..., description="Amplitude uniformity index (0-1, higher is better)"
    )
    # Stress
    stress_hotspots: list[StressHotspot] = Field(
        ..., description="Top stress concentration points"
    )
    stress_max_mpa: float = Field(
        ..., description="Maximum von Mises stress in MPa"
    )
    # Mesh info
    node_count: int = Field(..., description="Total number of mesh nodes")
    element_count: int = Field(
        ..., description="Total number of mesh elements"
    )
    solve_time_s: float = Field(
        ..., description="Total solver wall-clock time in seconds"
    )
    mesh: Optional[dict] = Field(
        default=None,
        description="Optional mesh geometry data for visualization",
    )
