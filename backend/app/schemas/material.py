"""Pydantic schemas for Material resources."""

from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class MaterialResponse(BaseModel):
    """Schema for returning a material."""

    id: Optional[UUID] = None
    name: str
    category: Optional[str]
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    yield_strength_mpa: Optional[float]
    thermal_conductivity: Optional[float]
    specific_heat: Optional[float]
    acoustic_impedance: Optional[float]
    properties_json: Optional[dict] = None

    model_config = ConfigDict(from_attributes=True)


class MaterialCreate(BaseModel):
    """Schema for creating a custom material."""

    name: str
    category: Optional[str] = None
    density_kg_m3: float
    youngs_modulus_pa: float
    poisson_ratio: float
    yield_strength_mpa: Optional[float] = None
    thermal_conductivity: Optional[float] = None
    specific_heat: Optional[float] = None
    acoustic_impedance: Optional[float] = None
    properties_json: Optional[dict] = None
