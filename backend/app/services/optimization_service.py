"""Service layer for OptimizationStudy operations."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.metric import Metric
from backend.app.models.optimization_study import OptimizationStudy
from backend.app.models.run import Run
from backend.app.schemas.optimization import OptimizationCreate


class OptimizationService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, simulation_id: UUID, data: OptimizationCreate) -> OptimizationStudy:
        study = OptimizationStudy(
            simulation_case_id=simulation_id,
            name=data.name,
            strategy=data.strategy,
            design_variables=[dv.model_dump() for dv in data.design_variables],
            constraints=[c.model_dump() for c in data.constraints] if data.constraints else None,
            objectives=[obj.model_dump() for obj in data.objectives],
            status="pending",
            total_iterations=data.max_iterations,
            completed_iterations=0,
        )
        self.session.add(study)
        await self.session.flush()
        await self.session.refresh(study)
        return study

    async def get(self, study_id: UUID) -> OptimizationStudy | None:
        return await self.session.get(OptimizationStudy, study_id)

    async def get_iterations(self, study_id: UUID) -> list[dict]:
        """Get all iterations (runs) for an optimization study."""
        query = (
            select(Run)
            .where(Run.optimization_study_id == study_id)
            .options(selectinload(Run.metrics))
            .order_by(Run.iteration_number)
        )
        result = await self.session.execute(query)
        runs = result.scalars().all()

        iterations = []
        for run in runs:
            metrics_dict = {m.metric_name: m.value for m in run.metrics}
            iterations.append({
                "iteration": run.iteration_number or 0,
                "run_id": run.id,
                "parameters": run.input_snapshot or {},
                "metrics": metrics_dict,
                "feasible": True,  # TODO: evaluate constraints
            })
        return iterations

    async def get_pareto(self, study_id: UUID) -> list[dict]:
        """Get Pareto front iterations for a multi-objective optimization."""
        study = await self.get(study_id)
        if not study or not study.pareto_front_run_ids:
            return []
        query = (
            select(Run)
            .where(Run.id.in_(study.pareto_front_run_ids))
            .options(selectinload(Run.metrics))
        )
        result = await self.session.execute(query)
        runs = result.scalars().all()

        pareto = []
        for run in runs:
            metrics_dict = {m.metric_name: m.value for m in run.metrics}
            pareto.append({
                "iteration": run.iteration_number or 0,
                "run_id": run.id,
                "parameters": run.input_snapshot or {},
                "metrics": metrics_dict,
                "feasible": True,
            })
        return pareto

    async def pause(self, study_id: UUID) -> OptimizationStudy | None:
        study = await self.get(study_id)
        if not study:
            return None
        if study.status == "running":
            study.status = "paused"
            await self.session.flush()
            await self.session.refresh(study)
        return study

    async def resume(self, study_id: UUID) -> OptimizationStudy | None:
        study = await self.get(study_id)
        if not study:
            return None
        if study.status == "paused":
            study.status = "running"
            await self.session.flush()
            await self.session.refresh(study)
        return study
