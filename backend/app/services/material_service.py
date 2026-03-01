"""Service layer for Material operations."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.material import Material
from backend.app.schemas.material import MaterialCreate


# Built-in FEA materials
BUILTIN_FEA_MATERIALS = [
    {"name": "Titanium Ti-6Al-4V", "category": "titanium", "density_kg_m3": 4430.0, "youngs_modulus_pa": 113.8e9, "poisson_ratio": 0.342, "yield_strength_mpa": 880.0, "thermal_conductivity": 6.7, "specific_heat": 526.3, "acoustic_impedance": 27.3e6},
    {"name": "Aluminum 6061-T6", "category": "aluminum", "density_kg_m3": 2700.0, "youngs_modulus_pa": 68.9e9, "poisson_ratio": 0.33, "yield_strength_mpa": 276.0, "thermal_conductivity": 167.0, "specific_heat": 896.0, "acoustic_impedance": 17.1e6},
    {"name": "Copper C11000", "category": "copper", "density_kg_m3": 8960.0, "youngs_modulus_pa": 117.0e9, "poisson_ratio": 0.34, "yield_strength_mpa": 69.0, "thermal_conductivity": 388.0, "specific_heat": 385.0, "acoustic_impedance": 44.6e6},
    {"name": "Nickel 200", "category": "nickel", "density_kg_m3": 8890.0, "youngs_modulus_pa": 204.0e9, "poisson_ratio": 0.31, "yield_strength_mpa": 148.0, "thermal_conductivity": 70.2, "specific_heat": 456.0, "acoustic_impedance": 49.5e6},
    {"name": "Steel AISI 1020", "category": "steel", "density_kg_m3": 7870.0, "youngs_modulus_pa": 200.0e9, "poisson_ratio": 0.29, "yield_strength_mpa": 350.0, "thermal_conductivity": 51.9, "specific_heat": 486.0, "acoustic_impedance": 46.0e6},
]


class MaterialService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def list_all(self, category: str | None = None, search: str | None = None) -> list[dict]:
        """List built-in + custom materials."""
        results = []
        for mat in BUILTIN_FEA_MATERIALS:
            if category and mat.get("category") != category:
                continue
            if search and search.lower() not in mat["name"].lower():
                continue
            results.append(mat)
        # Add custom materials from DB
        query = select(Material)
        if category:
            query = query.where(Material.category == category)
        if search:
            query = query.where(Material.name.ilike(f"%{search}%"))
        db_result = await self.session.execute(query)
        for mat in db_result.scalars().all():
            results.append({
                "id": mat.id,
                "name": mat.name,
                "category": mat.category,
                "density_kg_m3": mat.density_kg_m3,
                "youngs_modulus_pa": mat.youngs_modulus_pa,
                "poisson_ratio": mat.poisson_ratio,
                "yield_strength_mpa": mat.yield_strength_mpa,
                "thermal_conductivity": mat.thermal_conductivity,
                "specific_heat": mat.specific_heat,
                "acoustic_impedance": mat.acoustic_impedance,
                "properties_json": mat.properties_json,
            })
        return results

    async def get(self, material_id: UUID) -> Material | None:
        return await self.session.get(Material, material_id)

    async def create(self, data: MaterialCreate) -> Material:
        mat = Material(
            name=data.name,
            category=data.category,
            density_kg_m3=data.density_kg_m3,
            youngs_modulus_pa=data.youngs_modulus_pa,
            poisson_ratio=data.poisson_ratio,
            yield_strength_mpa=data.yield_strength_mpa,
            thermal_conductivity=data.thermal_conductivity,
            specific_heat=data.specific_heat,
            acoustic_impedance=data.acoustic_impedance,
            properties_json=data.properties_json,
        )
        self.session.add(mat)
        await self.session.flush()
        await self.session.refresh(mat)
        return mat
