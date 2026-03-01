"""Service for assembly simulation -- multi-body contact analysis."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.geometry_version import GeometryVersion
from backend.app.models.simulation_case import SimulationCase
from backend.app.solvers.base import BoundaryCondition, MaterialAssignment, SolverConfig


class AssemblyService:
    """Manage assembly simulations with multiple bodies."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def build_assembly_config(
        self, simulation_id: UUID
    ) -> SolverConfig:
        """Build a solver config that includes all assembly bodies.

        Reads assembly_components from the simulation case,
        loads each referenced GeometryVersion, and merges their
        mesh/material/BC data into a single SolverConfig.
        """
        sim = await self.session.get(SimulationCase, simulation_id)
        if not sim or not sim.assembly_components:
            raise ValueError("Simulation has no assembly components")

        # Load all geometry versions referenced in assembly
        components = sim.assembly_components  # list of {geometry_id, role, material, contact_pairs}
        all_materials: list[MaterialAssignment] = []
        all_bcs: list[BoundaryCondition] = []
        mesh_paths: list[str] = []

        for comp in (components if isinstance(components, list) else [components]):
            geom_id = comp.get("geometry_id")
            role = comp.get("role", "body")  # horn, anvil, workpiece
            material_name = comp.get("material", "Titanium Ti-6Al-4V")

            if geom_id:
                geom = await self.session.get(GeometryVersion, UUID(geom_id))
                if geom and geom.mesh_file_path:
                    mesh_paths.append(geom.mesh_file_path)

            # Material assignment for this body
            from backend.app.domain.material_properties import get_material
            mat_props = get_material(material_name) or {}
            all_materials.append(MaterialAssignment(
                region_id=role,
                material_name=material_name,
                properties=mat_props,
            ))

            # Contact conditions between bodies
            contact_pairs = comp.get("contact_pairs", [])
            for cp in contact_pairs:
                all_bcs.append(BoundaryCondition(
                    bc_type="contact",
                    region=f"{cp.get('master', role)}-{cp.get('slave', 'workpiece')}",
                    values={
                        "friction_coefficient": cp.get("friction", 0.3),
                        "contact_stiffness": cp.get("stiffness", 1e6),
                    },
                ))

        # Add default BCs from simulation config
        if sim.boundary_conditions:
            for bc_data in (sim.boundary_conditions if isinstance(sim.boundary_conditions, list) else []):
                all_bcs.append(BoundaryCondition(**bc_data))

        return SolverConfig(
            analysis_type=sim.analysis_type if hasattr(sim.analysis_type, 'value') else sim.analysis_type,
            mesh_path=mesh_paths[0] if mesh_paths else "",
            material_assignments=all_materials,
            boundary_conditions=all_bcs,
            parameters={
                **(sim.configuration or {}),
                "assembly_mesh_paths": mesh_paths,
                "n_bodies": len(components if isinstance(components, list) else [components]),
            },
        )
