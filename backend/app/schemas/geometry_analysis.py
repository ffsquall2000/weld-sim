"""Pydantic v2 schemas for geometry analysis (CAD/PDF upload, FEA run) and mesh data."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# FEA Run
# ---------------------------------------------------------------------------


class FEARunRequest(BaseModel):
    """Request body for running FEA modal analysis."""

    geometry_id: Optional[str] = Field(
        default=None,
        description=(
            "UUID of an uploaded geometry to analyze. When provided, "
            "the stored mesh/geometry file is used instead of parametric horn params."
        ),
    )
    horn_type: str = Field(
        default="cylindrical",
        description="Horn geometry type (cylindrical, blade, exponential, block, flat)",
    )
    width_mm: float = Field(gt=0, default=25.0, description="Horn width / diameter in mm")
    height_mm: float = Field(gt=0, default=80.0, description="Horn height / length in mm")
    length_mm: float = Field(gt=0, default=25.0, description="Horn depth in mm")
    material: str = Field(
        default="Titanium Ti-6Al-4V",
        description="Material name from the FEA material library",
    )
    frequency_khz: float = Field(
        gt=0, default=20.0, description="Target resonance frequency in kHz"
    )
    mesh_density: str = Field(
        default="medium",
        description="Mesh density preset: coarse, medium, or fine",
    )
    use_gmsh: bool = Field(
        default=True,
        description=(
            "Use Gmsh TET10 + SolverA pipeline (True, default) "
            "or legacy HEX8 internal solver (False)"
        ),
    )


class ModeShapeResponse(BaseModel):
    """A single eigenmode result from FEA analysis."""

    frequency_hz: float = Field(description="Natural frequency of this mode in Hz")
    mode_type: str = Field(description="Classification of mode shape (axial, bending, torsional, ...)")
    participation_factor: float = Field(description="Modal participation factor")
    effective_mass_ratio: float = Field(description="Effective mass ratio for this mode")
    displacement_max: float = Field(description="Maximum displacement magnitude")


class FEARunResponse(BaseModel):
    """Response from FEA modal analysis."""

    mode_shapes: list[ModeShapeResponse] = Field(description="Extracted eigenmodes")
    closest_mode_hz: float = Field(description="Closest mode frequency to target in Hz")
    target_frequency_hz: float = Field(description="Requested target frequency in Hz")
    frequency_deviation_percent: float = Field(
        description="Deviation between closest mode and target in percent"
    )
    node_count: int = Field(description="Total number of mesh nodes")
    element_count: int = Field(description="Total number of finite elements")
    solve_time_s: float = Field(description="Wall-clock solve time in seconds")
    mesh: Optional[dict[str, Any]] = Field(
        default=None, description="3D mesh for visualization (vertices + faces)"
    )
    stress_max_mpa: Optional[float] = Field(
        default=None, description="Maximum von-Mises stress in MPa"
    )
    temperature_max_c: Optional[float] = Field(
        default=None, description="Maximum estimated temperature in Celsius"
    )


# ---------------------------------------------------------------------------
# CAD Analysis
# ---------------------------------------------------------------------------


class CADAnalysisResponse(BaseModel):
    """Response from uploading and analyzing a STEP / CAD file."""

    horn_type: str = Field(description="Classified horn type")
    dimensions: dict[str, float] = Field(
        description="Extracted dimensions (width_mm, height_mm, length_mm)"
    )
    gain_estimate: float = Field(description="Estimated acoustic gain")
    confidence: float = Field(description="Classification confidence 0-1")
    knurl: Optional[dict[str, Any]] = Field(
        default=None, description="Detected knurl pattern data"
    )
    bounding_box: list[float] = Field(
        description="Axis-aligned bounding box [xmin, ymin, zmin, xmax, ymax, zmax]"
    )
    volume_mm3: float = Field(description="Part volume in cubic millimeters")
    surface_area_mm2: float = Field(description="Part surface area in square millimeters")
    mesh: Optional[dict[str, Any]] = Field(
        default=None,
        description="Simplified visualization mesh {vertices: [[x,y,z],...], faces: [[a,b,c],...]}",
    )


# ---------------------------------------------------------------------------
# PDF Analysis
# ---------------------------------------------------------------------------


class PDFAnalysisResponse(BaseModel):
    """Response from uploading and analyzing a PDF engineering drawing."""

    detected_dimensions: list[dict[str, Any]] = Field(
        description="Extracted dimension annotations"
    )
    tolerances: list[dict[str, Any]] = Field(
        description="Extracted tolerance specifications"
    )
    notes: list[str] = Field(description="Extracted engineering notes")
    confidence: float = Field(description="Overall extraction confidence 0-1")
    page_count: int = Field(description="Number of pages in the PDF")


# ---------------------------------------------------------------------------
# FEA Material
# ---------------------------------------------------------------------------


class FEAMaterialResponse(BaseModel):
    """A single material available for FEA analysis."""

    name: str = Field(description="Material display name")
    E_gpa: float = Field(description="Young's modulus in GPa")
    density_kg_m3: float = Field(description="Density in kg/m^3")
    poisson_ratio: float = Field(description="Poisson's ratio")


# ---------------------------------------------------------------------------
# Mesh Info (JSON metadata for the binary mesh endpoints)
# ---------------------------------------------------------------------------


class MeshModeInfo(BaseModel):
    """Summary of a single mode in the mesh cache."""

    index: int
    frequency_hz: Optional[float] = None
    mode_type: Optional[str] = None


class MeshInfoResponse(BaseModel):
    """JSON metadata for a cached mesh."""

    task_id: str = Field(description="Unique mesh/task identifier")
    node_count: int = Field(description="Number of mesh nodes")
    face_count: int = Field(description="Number of triangular faces")
    has_normals: bool = Field(description="Whether vertex normals are present")
    available_scalars: list[str] = Field(
        description="Names of available scalar fields (e.g. von_mises, displacement_mag)"
    )
    mode_count: int = Field(description="Number of stored mode shapes")
    modes: list[MeshModeInfo] = Field(
        default_factory=list, description="Per-mode summary"
    )
