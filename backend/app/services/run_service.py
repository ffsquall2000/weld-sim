"""Service layer for Run operations."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.artifact import Artifact
from backend.app.models.metric import Metric
from backend.app.models.run import Run
from backend.app.schemas.run import RunCreate

logger = logging.getLogger(__name__)


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
        # Dispatch execution
        dispatched = False
        try:
            from backend.app.tasks.solver_tasks import run_solver_task
            run_solver_task.delay(str(run.id))
            dispatched = True
            logger.info("Run %s dispatched to Celery worker", run.id)
        except Exception as exc:
            logger.warning("Celery dispatch failed for run %s: %s", run.id, exc)

        if not dispatched:
            # Fallback: run inline as async background task
            import asyncio
            asyncio.create_task(_execute_run_inline(str(run.id)))
            logger.info("Run %s scheduled for inline execution (no Celery)", run.id)

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


async def _execute_run_inline(run_id_str: str) -> None:
    """Execute a solver run inline (no Celery) using async DB session.

    This fallback is used when Celery/Redis are unavailable.
    It runs the core solver logic in a background coroutine.
    """
    import uuid as _uuid

    from backend.app.dependencies import async_session_factory
    from backend.app.models.run import Run

    try:
        async with async_session_factory() as session:
            run = await session.get(Run, _uuid.UUID(run_id_str))
            if not run:
                logger.error("Run %s not found for inline execution", run_id_str)
                return

            run.status = "running"
            await session.commit()

            try:
                # Import the actual solver execution logic
                from backend.app.tasks.solver_tasks import _execute_solver_sync
                import asyncio

                # Run the synchronous solver in a thread pool to not block the event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, _execute_solver_sync, run_id_str
                )
                run.status = "completed"
                run.compute_time_s = result.get("elapsed_s", 0) if isinstance(result, dict) else 0
                logger.info("Inline run %s completed successfully", run_id_str)
            except Exception as exc:
                run.status = "failed"
                run.error_message = str(exc)[:2000]
                logger.error("Inline run %s failed: %s", run_id_str, exc)

            await session.commit()
    except Exception as exc:
        logger.error("Inline execution wrapper failed for run %s: %s", run_id_str, exc)
