"""Endpoints for Material resources."""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.material import MaterialCreate, MaterialResponse

router = APIRouter(prefix="/materials", tags=["materials"])


# Built-in material database for FEA solvers
_BUILTIN_FEA_MATERIALS: List[dict] = [
    {
        "name": "Titanium Ti-6Al-4V",
        "category": "titanium",
        "density_kg_m3": 4430.0,
        "youngs_modulus_pa": 113.8e9,
        "poisson_ratio": 0.342,
        "yield_strength_mpa": 880.0,
        "thermal_conductivity": 6.7,
        "specific_heat": 526.3,
        "acoustic_impedance": 27.3e6,
    },
    {
        "name": "Aluminum 6061-T6",
        "category": "aluminum",
        "density_kg_m3": 2700.0,
        "youngs_modulus_pa": 68.9e9,
        "poisson_ratio": 0.33,
        "yield_strength_mpa": 276.0,
        "thermal_conductivity": 167.0,
        "specific_heat": 896.0,
        "acoustic_impedance": 17.1e6,
    },
    {
        "name": "Copper C11000",
        "category": "copper",
        "density_kg_m3": 8960.0,
        "youngs_modulus_pa": 117.0e9,
        "poisson_ratio": 0.34,
        "yield_strength_mpa": 69.0,
        "thermal_conductivity": 388.0,
        "specific_heat": 385.0,
        "acoustic_impedance": 44.6e6,
    },
    {
        "name": "Nickel 200",
        "category": "nickel",
        "density_kg_m3": 8890.0,
        "youngs_modulus_pa": 204.0e9,
        "poisson_ratio": 0.31,
        "yield_strength_mpa": 148.0,
        "thermal_conductivity": 70.2,
        "specific_heat": 456.0,
        "acoustic_impedance": 49.5e6,
    },
    {
        "name": "Steel AISI 1020",
        "category": "steel",
        "density_kg_m3": 7870.0,
        "youngs_modulus_pa": 200.0e9,
        "poisson_ratio": 0.29,
        "yield_strength_mpa": 350.0,
        "thermal_conductivity": 51.9,
        "specific_heat": 486.0,
        "acoustic_impedance": 46.0e6,
    },
]


@router.get("/", response_model=List[MaterialResponse])
async def list_materials(
    category: Optional[str] = Query(None, description="Filter by material category"),
    search: Optional[str] = Query(None, description="Search by material name"),
) -> List[MaterialResponse]:
    """List all materials (built-in and custom)."""
    # TODO: merge built-in materials with custom materials from DB
    results: List[MaterialResponse] = []
    for mat_data in _BUILTIN_FEA_MATERIALS:
        if category and mat_data.get("category") != category:
            continue
        if search and search.lower() not in mat_data["name"].lower():
            continue
        results.append(MaterialResponse(**mat_data))
    return results


@router.get("/fea", response_model=List[MaterialResponse])
async def list_fea_materials() -> List[MaterialResponse]:
    """List built-in materials suitable for FEA analysis."""
    return [MaterialResponse(**m) for m in _BUILTIN_FEA_MATERIALS]


@router.get("/{material_id}", response_model=MaterialResponse)
async def get_material(material_id: uuid.UUID) -> MaterialResponse:
    """Get a single material by ID."""
    # TODO: call material service to fetch from DB
    raise HTTPException(status_code=404, detail="Material not found")


@router.post("/", response_model=MaterialResponse, status_code=201)
async def create_material(body: MaterialCreate) -> MaterialResponse:
    """Create a custom material definition."""
    # TODO: call material service to persist in DB
    return MaterialResponse(
        id=uuid.uuid4(),
        name=body.name,
        category=body.category,
        density_kg_m3=body.density_kg_m3,
        youngs_modulus_pa=body.youngs_modulus_pa,
        poisson_ratio=body.poisson_ratio,
        yield_strength_mpa=body.yield_strength_mpa,
        thermal_conductivity=body.thermal_conductivity,
        specific_heat=body.specific_heat,
        acoustic_impedance=body.acoustic_impedance,
        properties_json=body.properties_json,
    )
