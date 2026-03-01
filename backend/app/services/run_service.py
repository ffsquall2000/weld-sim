"""Service layer for Run operations."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.artifact import Artifact
from backend.app.models.metric import Metric
from backend.app.models.run import Run
from backend.app.schemas.run import RunCreate


class RunService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def submit(self, simulation_id: UUID, data: RunCreate) -> Run:
        """Create a new run and dispatch it for execution."""
        run = Run(
            simulation_case_id=simulation_id,
            geometry_version_id=data.geometry_version_id,
            status="queued",
            input_snapshot=data.parameters_override,
        )
        self.session.add(run)
        await self.session.flush()
        await self.session.refresh(run)
        # Dispatch Celery task
        try:
            from backend.app.tasks.solver_tasks import run_solver_task
            run_solver_task.delay(str(run.id))
        except Exception:
            # If Celery/Redis not available, run inline (dev mode)
            pass
        return run

    async def get(self, run_id: UUID) -> Run | None:
        return await self.session.get(Run, run_id)

    async def get_detail(self, run_id: UUID) -> Run | None:
        """Get run with eager-loaded metrics and artifacts."""
        query = (
            select(Run)
            .where(Run.id == run_id)
            .options(selectinload(Run.metrics), selectinload(Run.artifacts))
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def list_by_simulation(self, simulation_id: UUID) -> list[Run]:
        query = (
            select(Run)
            .where(Run.simulation_case_id == simulation_id)
            .order_by(Run.created_at.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def cancel(self, run_id: UUID) -> Run | None:
        run = await self.get(run_id)
        if not run:
            return None
        if run.status in ("queued", "running"):
            run.status = "cancelled"
            run.completed_at = datetime.now(tz=timezone.utc)
            await self.session.flush()
            await self.session.refresh(run)
        return run

    async def get_metrics(self, run_id: UUID) -> list[Metric]:
        query = select(Metric).where(Metric.run_id == run_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_artifacts(self, run_id: UUID) -> list[Artifact]:
        query = select(Artifact).where(Artifact.run_id == run_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_artifact(self, run_id: UUID, artifact_id: UUID) -> Artifact | None:
        query = select(Artifact).where(Artifact.id == artifact_id, Artifact.run_id == run_id)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def save_metrics(self, run_id: UUID, metrics: dict[str, float]) -> list[Metric]:
        """Save metrics for a run."""
        saved = []
        for name, value in metrics.items():
            metric = Metric(
                run_id=run_id,
                metric_name=name,
                value=value,
                unit=self._guess_unit(name),
            )
            self.session.add(metric)
            saved.append(metric)
        await self.session.flush()
        return saved

    @staticmethod
    def _guess_unit(metric_name: str) -> str:
        units = {
            "natural_frequency_hz": "Hz",
            "frequency_deviation_pct": "%",
            "amplitude_uniformity": "",
            "max_von_mises_stress_mpa": "MPa",
            "stress_safety_factor": "",
            "max_temperature_rise_c": "\u00b0C",
            "contact_pressure_uniformity": "",
            "effective_contact_area_mm2": "mm\u00b2",
            "energy_coupling_efficiency": "",
            "horn_gain": "",
            "modal_separation_hz": "Hz",
            "fatigue_cycle_estimate": "cycles",
        }
        return units.get(metric_name, "")
