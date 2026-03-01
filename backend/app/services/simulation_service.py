"""Service layer for SimulationCase operations."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.simulation_case import SimulationCase
from backend.app.schemas.simulation import SimulationCaseCreate, SimulationCaseUpdate


class SimulationService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, project_id: UUID, data: SimulationCaseCreate) -> SimulationCase:
        sim = SimulationCase(
            project_id=project_id,
            name=data.name,
            description=data.description,
            analysis_type=data.analysis_type,
            solver_backend=data.solver_backend or "preview",
            configuration=data.configuration,
            boundary_conditions=data.boundary_conditions,
            material_assignments=data.material_assignments,
            assembly_components=data.assembly_components,
            workflow_dag=data.workflow_dag,
        )
        self.session.add(sim)
        await self.session.flush()
        await self.session.refresh(sim)
        return sim

    async def get(self, simulation_id: UUID) -> SimulationCase | None:
        return await self.session.get(SimulationCase, simulation_id)

    async def list_by_project(self, project_id: UUID) -> list[SimulationCase]:
        query = (
            select(SimulationCase)
            .where(SimulationCase.project_id == project_id)
            .order_by(SimulationCase.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update(self, simulation_id: UUID, data: SimulationCaseUpdate) -> SimulationCase | None:
        sim = await self.get(simulation_id)
        if not sim:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(sim, field, value)
        await self.session.flush()
        await self.session.refresh(sim)
        return sim

    async def validate(self, simulation_id: UUID) -> dict:
        """Validate a simulation case configuration."""
        sim = await self.get(simulation_id)
        if not sim:
            return {"valid": False, "errors": ["Simulation case not found"], "warnings": []}
        errors = []
        warnings = []
        # Check solver backend is registered
        from backend.app.solvers.registry import is_registered
        if not is_registered(sim.solver_backend):
            errors.append(f"Solver backend '{sim.solver_backend}' is not registered")
        # Check analysis type is supported
        if not sim.analysis_type:
            errors.append("Analysis type is required")
        # Check material assignments
        if not sim.material_assignments:
            warnings.append("No material assignments specified")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}
