"""Service layer for Run operations."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.artifact import Artifact
from backend.app.models.metric import Metric
from backend.app.models.run import Run
from backend.app.schemas.run import RunCreate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Strong references to background tasks to prevent garbage collection
_background_tasks: set[asyncio.Task] = set()  # type: ignore[type-arg]


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

        # BUG-4 fix: Dispatch Celery task with async fallback
        # Check Redis + active Celery workers + task registration before dispatch
        celery_dispatched = False
        try:
            from backend.app.dependencies import celery_app
            inspector = celery_app.control.inspect(timeout=1.0)
            active = inspector.active()
            if not active:
                raise RuntimeError("No active Celery workers")
            # Verify our task is actually registered in the worker
            registered = inspector.registered()
            if registered:
                all_tasks = {t for tasks in registered.values() for t in tasks}
                if "weldsim.run_solver" not in all_tasks:
                    raise RuntimeError(
                        f"Task 'weldsim.run_solver' not registered in workers "
                        f"(registered: {all_tasks})"
                    )
            else:
                raise RuntimeError("Cannot query registered tasks")
            from backend.app.tasks.solver_tasks import run_solver_task
            run_solver_task.delay(str(run.id))
            celery_dispatched = True
            logger.info("Run %s dispatched via Celery to %d worker(s)", run.id, len(active))
        except Exception as exc:
            logger.info("Celery not available (%s), using inline execution for run %s", exc, run.id)

        if not celery_dispatched:
            # Run inline using asyncio background task
            # Keep strong reference to prevent GC from collecting the task
            task = asyncio.create_task(self._execute_inline(str(run.id)))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
            logger.info("Inline execution task created for run %s", run.id)

        return run

    @staticmethod
    async def _execute_inline(run_id: str) -> None:
        """Execute a solver run inline when Celery is not available."""
        from backend.app.dependencies import get_db_context

        try:
            async with get_db_context() as session:
                run = await session.get(Run, UUID(run_id))
                if not run:
                    logger.error("Run %s not found for inline execution", run_id)
                    return

                run.status = "running"
                run.started_at = datetime.utcnow()
                await session.commit()

                t0 = time.time()

                try:
                    from backend.app.models.simulation_case import SimulationCase
                    from backend.app.models.geometry_version import GeometryVersion

                    sim = await session.get(SimulationCase, run.simulation_case_id)
                    geom = await session.get(GeometryVersion, run.geometry_version_id) if run.geometry_version_id else None

                    if not sim:
                        raise ValueError("Simulation case not found")

                    # Get solver from registry
                    from backend.app.solvers.registry import get_solver, init_solvers
                    try:
                        solver = get_solver(sim.solver_backend)
                    except KeyError:
                        init_solvers()
                        solver = get_solver(sim.solver_backend)

                    from backend.app.solvers.base import (
                        AnalysisType,
                        BoundaryCondition,
                        MaterialAssignment,
                        SolverConfig,
                    )

                    material_assignments = []
                    if sim.material_assignments:
                        if isinstance(sim.material_assignments, list):
                            for ma in sim.material_assignments:
                                material_assignments.append(MaterialAssignment(**ma))
                        elif isinstance(sim.material_assignments, dict):
                            material_assignments.append(
                                MaterialAssignment(
                                    region_id="default",
                                    material_name=sim.material_assignments.get(
                                        "material", "Titanium Ti-6Al-4V"
                                    ),
                                    properties=sim.material_assignments,
                                )
                            )

                    boundary_conditions = []
                    if sim.boundary_conditions:
                        if isinstance(sim.boundary_conditions, list):
                            for bc in sim.boundary_conditions:
                                boundary_conditions.append(BoundaryCondition(**bc))

                    parameters = dict(sim.configuration or {})
                    if geom and geom.parametric_params:
                        parameters.update(geom.parametric_params)
                    if run.input_snapshot:
                        parameters.update(run.input_snapshot)

                    config = SolverConfig(
                        analysis_type=(
                            AnalysisType(sim.analysis_type)
                            if sim.analysis_type
                            else AnalysisType.MODAL
                        ),
                        mesh_path=geom.mesh_file_path if geom else None,
                        material_assignments=material_assignments,
                        boundary_conditions=boundary_conditions,
                        parameters=parameters,
                    )

                    job = await solver.prepare(config)
                    result = await solver.run(job)

                    # Save metrics
                    for name, value in result.metrics.items():
                        metric = Metric(
                            run_id=run.id,
                            metric_name=name,
                            value=float(value),
                            unit=RunService._guess_unit(name),
                        )
                        session.add(metric)

                    compute_time = time.time() - t0
                    run.status = "completed" if result.success else "failed"
                    run.compute_time_s = compute_time
                    run.completed_at = datetime.utcnow()
                    if not result.success:
                        run.error_message = result.error_message

                    await session.commit()
                    logger.info("Inline run %s completed: %s", run_id, run.status)

                except Exception as e:
                    logger.exception("Inline solver run failed for %s", run_id)
                    run.status = "failed"
                    run.error_message = str(e)
                    run.completed_at = datetime.utcnow()
                    run.compute_time_s = time.time() - t0
                    await session.commit()

        except Exception as e:
            logger.exception("Failed to execute inline run %s", run_id)

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
            run.completed_at = datetime.utcnow()
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
