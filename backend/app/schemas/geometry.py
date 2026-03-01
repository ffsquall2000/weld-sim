"""Pydantic schemas for Geometry resources."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class HornParams(BaseModel):
    """Parametric horn geometry parameters."""

    horn_type: str = "flat"  # flat, cylindrical, exponential, blade, stepped
    width_mm: float = 25.0
    height_mm: float = 80.0
    length_mm: float = 25.0
    material: str = "Titanium Ti-6Al-4V"
    knurl_type: Optional[str] = None
    knurl_pitch_mm: Optional[float] = None
    knurl_depth_mm: Optional[float] = None
    chamfer_radius_mm: Optional[float] = None
    edge_treatment: Optional[str] = None


class GeometryCreate(BaseModel):
    """Schema for creating a new geometry version."""

    label: Optional[str] = None
    source_type: str = "parametric"  # parametric, imported_step, imported_stl
    parametric_params: Optional[HornParams] = None
    mesh_config: Optional[dict] = None


class GeometryResponse(BaseModel):
    """Schema for returning a geometry version."""

    id: UUID
    project_id: UUID
    version_number: int
    label: Optional[str]
    source_type: str
    parametric_params: Optional[dict]
    file_path: Optional[str]
    mesh_file_path: Optional[str]
    metadata_json: Optional[dict]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MeshPreviewResponse(BaseModel):
    """Schema for returning a mesh preview for 3D visualization."""

    vertices: List[List[float]]
    faces: List[List[int]]
    scalar_field: Optional[List[float]] = None
    node_count: int
    element_count: int
