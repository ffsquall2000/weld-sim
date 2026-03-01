"""Celery tasks for running solver jobs asynchronously."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.config import settings

logger = logging.getLogger(__name__)


def _get_sync_session() -> Session:
    """Create a synchronous DB session for Celery tasks."""
    sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace(
        "+aiosqlite", ""
    )
    engine = create_engine(sync_url, echo=False)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory()


def _get_redis():
    """Get Redis client for pub/sub."""
    try:
        import redis

        return redis.Redis.from_url(settings.REDIS_URL)
    except Exception:
        return None


def _publish_progress(redis_client, run_id: str, data: dict):
    """Publish progress message to Redis pub/sub."""
    if redis_client:
        try:
            channel = f"run:{run_id}:progress"
            redis_client.publish(channel, json.dumps(data))
        except Exception as e:
            logger.warning("Failed to publish progress: %s", e)


def _guess_unit(metric_name: str) -> str:
    """Guess the unit for a metric based on its name."""
    units = {
        "natural_frequency_hz": "Hz",
        "frequency_deviation_pct": "%",
        "frequency_deviation_percent": "%",
        "amplitude_uniformity": "",
        "max_von_mises_stress_mpa": "MPa",
        "max_stress_mpa": "MPa",
        "stress_safety_factor": "",
        "max_temperature_rise_c": "\u00b0C",
        "thermal_penetration_mm": "mm",
        "contact_pressure_uniformity": "",
        "effective_contact_area_mm2": "mm\u00b2",
        "energy_coupling_efficiency": "",
        "horn_gain": "",
        "modal_separation_hz": "Hz",
        "fatigue_cycle_estimate": "cycles",
        "node_count": "",
        "element_count": "",
        "target_frequency_hz": "Hz",
        "max_deflection_mm": "mm",
    }
    return units.get(metric_name, "")


def _execute_solver_sync(run_id: str) -> dict:
    """Execute the solver synchronously for a given run ID.

    This is the core solver execution logic, used both by the Celery task
    and by the inline fallback in run_service.py.

    Returns a dict with at least ``elapsed_s`` on success.
    Raises on failure.
    """
    session = _get_sync_session()
    redis_client = _get_redis()
    start_time = time.time()

    try:
        from backend.app.models.artifact import Artifact
        from backend.app.models.geometry_version import GeometryVersion
        from backend.app.models.metric import Metric
        from backend.app.models.run import Run
        from backend.app.models.simulation_case import SimulationCase

        run = session.get(Run, UUID(run_id))
        if not run:
            logger.error("Run %s not found", run_id)
            return {"elapsed_s": 0}

        # Update status to running
        run.status = "running"
        run.started_at = datetime.now(tz=timezone.utc)
        session.commit()

        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "progress",
                "run_id": run_id,
                "percent": 0,
                "phase": "preparing",
                "message": "Preparing solver...",
                "elapsed_s": 0,
            },
        )

        # Load simulation case and geometry
        sim = session.get(SimulationCase, run.simulation_case_id)
        geom = session.get(GeometryVersion, run.geometry_version_id)

        if not sim:
            raise ValueError("Simulation case not found")

        # Get solver from registry
        from backend.app.solvers.registry import get_solver, init_solvers

        try:
            solver = get_solver(sim.solver_backend)
        except KeyError:
            init_solvers()
            solver = get_solver(sim.solver_backend)

        # Build solver config
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

        # Build parameters from geometry + simulation config
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

        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "progress",
                "run_id": run_id,
                "percent": 10,
                "phase": "meshing",
                "message": "Preparing mesh...",
                "elapsed_s": time.time() - start_time,
            },
        )

        # Prepare solver job (async -> sync via asyncio.run)
        job = asyncio.run(solver.prepare(config))

        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "progress",
                "run_id": run_id,
                "percent": 30,
                "phase": "solving",
                "message": f"Running {solver.name} solver...",
                "elapsed_s": time.time() - start_time,
            },
        )

        # Define progress callback matching ProgressCallback(float, str)
        def progress_callback(percent: float, message: str) -> None:
            actual_percent = 30 + int(percent * 0.6)  # Map 0-100 to 30-90
            _publish_progress(
                redis_client,
                run_id,
                {
                    "type": "progress",
                    "run_id": run_id,
                    "percent": actual_percent,
                    "phase": "solving",
                    "message": message,
                    "elapsed_s": time.time() - start_time,
                },
            )

        # Run solver (async -> sync via asyncio.run)
        result = asyncio.run(solver.run(job, progress_callback))

        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "progress",
                "run_id": run_id,
                "percent": 90,
                "phase": "postprocessing",
                "message": "Saving results...",
                "elapsed_s": time.time() - start_time,
            },
        )

        # Save metrics
        for name, value in result.metrics.items():
            metric = Metric(
                run_id=run.id,
                metric_name=name,
                value=float(value),
                unit=_guess_unit(name),
            )
            session.add(metric)

        # Save artifacts (field data as VTK JSON)
        if result.field_data:
            storage_dir = Path(settings.STORAGE_PATH) / "runs" / str(run.id)
            storage_dir.mkdir(parents=True, exist_ok=True)

            from backend.app.solvers.result_reader import ResultReader

            reader = ResultReader()
            vtk_json = reader.field_to_vtk_json(result.field_data, "displacement")
            result_path = storage_dir / "result.json"
            result_path.write_text(json.dumps(vtk_json))
            session.add(
                Artifact(
                    run_id=run.id,
                    artifact_type="result_vtu",
                    file_path=str(result_path),
                    file_size_bytes=result_path.stat().st_size,
                    mime_type="application/json",
                )
            )

        # Save solver log
        if result.solver_log:
            storage_dir = Path(settings.STORAGE_PATH) / "runs" / str(run.id)
            storage_dir.mkdir(parents=True, exist_ok=True)
            log_path = storage_dir / "solver.log"
            log_path.write_text(result.solver_log)
            session.add(
                Artifact(
                    run_id=run.id,
                    artifact_type="solver_log",
                    file_path=str(log_path),
                    file_size_bytes=log_path.stat().st_size,
                    mime_type="text/plain",
                )
            )

        # Update run
        compute_time = time.time() - start_time
        run.status = "completed" if result.success else "failed"
        run.compute_time_s = compute_time
        run.completed_at = datetime.now(tz=timezone.utc)
        run.solver_log = result.solver_log
        if not result.success:
            run.error_message = result.error_message

        session.commit()

        # Publish completion
        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "completed",
                "run_id": run_id,
                "status": run.status,
                "compute_time_s": compute_time,
                "metrics_summary": result.metrics,
            },
        )

        return {"elapsed_s": compute_time, "status": run.status}

    except Exception as e:
        logger.exception("Solver execution failed for run %s", run_id)
        try:
            from backend.app.models.run import Run

            run = session.get(Run, UUID(run_id))
            if run:
                run.status = "failed"
                run.error_message = str(e)
                run.completed_at = datetime.now(tz=timezone.utc)
                run.compute_time_s = time.time() - start_time
                session.commit()
        except Exception:
            session.rollback()

        _publish_progress(
            redis_client,
            run_id,
            {
                "type": "error",
                "run_id": run_id,
                "error": str(e),
            },
        )
        raise
    finally:
        session.close()
        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass


@shared_task(bind=True, max_retries=0, name="weldsim.run_solver")
def run_solver_task(self, run_id: str):
    """Execute solver for a simulation run (Celery task wrapper).

    This task:
    1. Loads the Run from the database using a synchronous session
    2. Gets the solver from the registry
    3. Executes solver.prepare() and solver.run()
    4. Saves metrics and artifacts to the database
    5. Publishes progress via Redis pub/sub
    6. Handles errors and updates run status
    """
    _execute_solver_sync(run_id)
