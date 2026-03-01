"""Celery tasks for running optimization studies iteratively."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

from celery import shared_task

from backend.app.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from solver_tasks)
# ---------------------------------------------------------------------------


def _get_sync_session():
    """Create a synchronous DB session for Celery tasks."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

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


def _publish_optimization_progress(redis_client, study_id: str, data: dict):
    """Publish optimisation progress to Redis pub/sub."""
    if redis_client:
        try:
            channel = f"optimization:{study_id}:progress"
            redis_client.publish(channel, json.dumps(data, default=str))
        except Exception as e:
            logger.warning("Failed to publish optimization progress: %s", e)


def _run_solver_for_params(
    session,
    simulation_case,
    geometry_version,
    study,
    iteration_number: int,
    params: dict,
) -> tuple[dict[str, float], str]:
    """Create a Run, execute the solver synchronously, and return metrics.

    Returns
    -------
    (metrics_dict, run_id)
        metrics_dict maps metric name -> float value; run_id is the Run UUID.
    """
    import asyncio

    from backend.app.models.metric import Metric
    from backend.app.models.run import Run

    # Create Run record
    run = Run(
        simulation_case_id=simulation_case.id,
        geometry_version_id=geometry_version.id,
        optimization_study_id=study.id,
        iteration_number=iteration_number,
        status="queued",
        input_snapshot=params,
    )
    session.add(run)
    session.flush()
    run_id = str(run.id)

    try:
        run.status = "running"
        run.started_at = datetime.now(tz=timezone.utc)
        session.commit()

        # ---------- Build solver config (mirrors solver_tasks.py) ----------
        from backend.app.solvers.base import (
            AnalysisType,
            BoundaryCondition,
            MaterialAssignment,
            SolverConfig,
        )
        from backend.app.solvers.registry import get_solver, init_solvers

        try:
            solver = get_solver(simulation_case.solver_backend)
        except KeyError:
            init_solvers()
            solver = get_solver(simulation_case.solver_backend)

        material_assignments = []
        if simulation_case.material_assignments:
            if isinstance(simulation_case.material_assignments, list):
                for ma in simulation_case.material_assignments:
                    material_assignments.append(MaterialAssignment(**ma))
            elif isinstance(simulation_case.material_assignments, dict):
                material_assignments.append(
                    MaterialAssignment(
                        region_id="default",
                        material_name=simulation_case.material_assignments.get(
                            "material", "Titanium Ti-6Al-4V"
                        ),
                        properties=simulation_case.material_assignments,
                    )
                )

        boundary_conditions = []
        if simulation_case.boundary_conditions:
            if isinstance(simulation_case.boundary_conditions, list):
                for bc in simulation_case.boundary_conditions:
                    boundary_conditions.append(BoundaryCondition(**bc))

        # Merge parameters: sim config + geometry params + optimization params
        parameters = dict(simulation_case.configuration or {})
        if geometry_version and geometry_version.parametric_params:
            parameters.update(geometry_version.parametric_params)
        parameters.update(params)

        config = SolverConfig(
            analysis_type=(
                AnalysisType(simulation_case.analysis_type)
                if simulation_case.analysis_type
                else AnalysisType.MODAL
            ),
            mesh_path=geometry_version.mesh_file_path if geometry_version else None,
            material_assignments=material_assignments,
            boundary_conditions=boundary_conditions,
            parameters=parameters,
        )

        # Run solver synchronously
        start = time.time()
        job = asyncio.run(solver.prepare(config))

        def _noop_progress(pct: float, msg: str) -> None:
            pass

        result = asyncio.run(solver.run(job, _noop_progress))
        elapsed = time.time() - start

        # Save metrics
        metrics_dict: dict[str, float] = {}
        for name, value in result.metrics.items():
            metric = Metric(
                run_id=run.id,
                metric_name=name,
                value=float(value),
                unit=_guess_unit(name),
            )
            session.add(metric)
            metrics_dict[name] = float(value)

        run.status = "completed" if result.success else "failed"
        run.compute_time_s = elapsed
        run.completed_at = datetime.now(tz=timezone.utc)
        if not result.success:
            run.error_message = result.error_message
        session.commit()

        return metrics_dict, run_id

    except Exception as exc:
        logger.exception("Solver failed for optimization iter %d", iteration_number)
        run.status = "failed"
        run.error_message = str(exc)
        run.completed_at = datetime.now(tz=timezone.utc)
        session.commit()
        return {}, run_id


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


# ---------------------------------------------------------------------------
# Main optimisation Celery task
# ---------------------------------------------------------------------------


@shared_task(bind=True, max_retries=0, name="weldsim.run_optimization")
def run_optimization_task(self, study_id: str):
    """Run optimisation iterations for a study.

    Workflow
    --------
    1. Load the OptimizationStudy from the database.
    2. Create an ``OptimizationEngine`` with the study's strategy.
    3. Generate initial samples (or resume from history).
    4. For each parameter set:
       a. Create a Run with the new parameters.
       b. Execute the solver synchronously and collect metrics.
       c. Record iteration result in history.
       d. Update study progress.
       e. Publish progress via Redis.
    5. After initial samples are evaluated, enter an iterative loop (for
       Bayesian / GA) calling ``engine.suggest_next`` until convergence or
       ``max_iterations`` is reached.
    6. Compute Pareto front and best run.
    7. Mark study as completed.
    """
    session = _get_sync_session()
    redis_client = _get_redis()

    try:
        from backend.app.models.geometry_version import GeometryVersion
        from backend.app.models.metric import Metric
        from backend.app.models.optimization_study import OptimizationStudy
        from backend.app.models.run import Run
        from backend.app.models.simulation_case import SimulationCase
        from backend.app.services.optimization_engine import OptimizationEngine

        # 1. Load study
        study = session.get(OptimizationStudy, UUID(study_id))
        if not study:
            logger.error("OptimizationStudy %s not found", study_id)
            return

        # Check if study was cancelled before we start
        if study.status in ("completed", "cancelled"):
            return

        study.status = "running"
        session.commit()

        _publish_optimization_progress(
            redis_client,
            study_id,
            {
                "type": "started",
                "study_id": study_id,
                "strategy": study.strategy,
                "total_iterations": study.total_iterations,
            },
        )

        # Load simulation case and most recent geometry version
        sim_case = session.get(SimulationCase, study.simulation_case_id)
        if not sim_case:
            raise ValueError("SimulationCase not found")

        # Get the most recent geometry version for this project
        from sqlalchemy import select

        geom_query = (
            select(GeometryVersion)
            .where(GeometryVersion.project_id == sim_case.project_id)
            .order_by(GeometryVersion.version_number.desc())
            .limit(1)
        )
        geom_result = session.execute(geom_query)
        geometry_version = geom_result.scalars().first()
        if not geometry_version:
            raise ValueError("No GeometryVersion found for project")

        # 2. Create engine
        engine = OptimizationEngine(study.strategy)

        # Decode schema objects from JSON stored in DB
        design_variables = study.design_variables  # list[dict]
        objectives = study.objectives  # list[dict]
        constraints = study.constraints  # list[dict] or None
        max_iterations = study.total_iterations

        # Reconstruct history from existing runs (for resume support)
        history: list[dict] = []
        existing_runs_query = (
            select(Run)
            .where(Run.optimization_study_id == study.id)
            .order_by(Run.iteration_number)
        )
        existing_runs_result = session.execute(existing_runs_query)
        existing_runs = existing_runs_result.scalars().all()
        for existing_run in existing_runs:
            metrics_query = (
                select(Metric)
                .where(Metric.run_id == existing_run.id)
            )
            metrics_result = session.execute(metrics_query)
            run_metrics = metrics_result.scalars().all()
            metrics_dict = {m.metric_name: m.value for m in run_metrics}
            history.append({
                "iteration": existing_run.iteration_number or 0,
                "run_id": str(existing_run.id),
                "parameters": existing_run.input_snapshot or {},
                "metrics": metrics_dict,
                "feasible": engine.evaluate_constraints(metrics_dict, constraints),
            })

        current_iteration = len(history)

        # 3. Generate initial samples if no history yet
        if current_iteration == 0:
            # Number of initial samples depends on strategy
            if study.strategy == "parametric_sweep":
                n_initial = max_iterations  # sweep evaluates all at once
            elif study.strategy == "genetic":
                # Population-based: initial population size
                pop_size = min(20, max_iterations)
                n_initial = pop_size
            else:
                # Bayesian: start with a small initial DoE
                n_initial = min(max(5, max_iterations // 5), max_iterations)

            initial_samples = engine.generate_initial_samples(
                design_variables, n_initial
            )

            # Evaluate initial samples
            for i, params in enumerate(initial_samples):
                # Check if study has been paused/cancelled
                session.refresh(study)
                if study.status in ("paused", "cancelled"):
                    logger.info(
                        "Optimization %s %s at iteration %d",
                        study_id, study.status, current_iteration,
                    )
                    return

                iteration_num = current_iteration + 1
                metrics, run_id = _run_solver_for_params(
                    session, sim_case, geometry_version, study,
                    iteration_num, params,
                )

                feasible = engine.evaluate_constraints(metrics, constraints)
                history.append({
                    "iteration": iteration_num,
                    "run_id": run_id,
                    "parameters": params,
                    "metrics": metrics,
                    "feasible": feasible,
                })

                current_iteration = iteration_num
                study.completed_iterations = current_iteration
                session.commit()

                _publish_optimization_progress(
                    redis_client,
                    study_id,
                    {
                        "type": "iteration_complete",
                        "study_id": study_id,
                        "iteration": iteration_num,
                        "total_iterations": max_iterations,
                        "parameters": params,
                        "metrics": metrics,
                        "feasible": feasible,
                    },
                )

        # 4. Iterative loop (only for Bayesian and GA)
        if study.strategy != "parametric_sweep":
            while current_iteration < max_iterations:
                # Check pause/cancel
                session.refresh(study)
                if study.status in ("paused", "cancelled"):
                    logger.info(
                        "Optimization %s %s at iteration %d",
                        study_id, study.status, current_iteration,
                    )
                    return

                # For GA: evolve a whole generation at once
                if study.strategy == "genetic" and isinstance(engine.strategy, type(engine.strategy)):
                    # Get suggestions for the full population
                    suggested = engine.suggest_next(
                        design_variables, objectives, constraints, history,
                    )
                    if suggested is None:
                        logger.info("Optimization %s converged at iteration %d", study_id, current_iteration)
                        break
                    params = suggested
                else:
                    # Bayesian: one point at a time
                    params = engine.suggest_next(
                        design_variables, objectives, constraints, history,
                    )
                    if params is None:
                        logger.info("Optimization %s converged at iteration %d", study_id, current_iteration)
                        break

                iteration_num = current_iteration + 1
                metrics, run_id = _run_solver_for_params(
                    session, sim_case, geometry_version, study,
                    iteration_num, params,
                )

                feasible = engine.evaluate_constraints(metrics, constraints)
                history.append({
                    "iteration": iteration_num,
                    "run_id": run_id,
                    "parameters": params,
                    "metrics": metrics,
                    "feasible": feasible,
                })

                current_iteration = iteration_num
                study.completed_iterations = current_iteration
                session.commit()

                _publish_optimization_progress(
                    redis_client,
                    study_id,
                    {
                        "type": "iteration_complete",
                        "study_id": study_id,
                        "iteration": iteration_num,
                        "total_iterations": max_iterations,
                        "parameters": params,
                        "metrics": metrics,
                        "feasible": feasible,
                    },
                )

        # 5. Compute Pareto front
        pareto_indices = engine.compute_pareto_front(history, objectives)
        pareto_run_ids = []
        for idx in pareto_indices:
            entry = history[idx]
            rid = entry.get("run_id")
            if rid:
                pareto_run_ids.append(UUID(rid) if isinstance(rid, str) else rid)

        study.pareto_front_run_ids = pareto_run_ids if pareto_run_ids else None

        # 6. Find best run (single-objective: best feasible; multi-objective: first Pareto)
        best_run_id = _find_best_run(history, objectives, constraints, engine)
        if best_run_id:
            study.best_run_id = UUID(best_run_id) if isinstance(best_run_id, str) else best_run_id

        # 7. Mark complete
        study.status = "completed"
        study.completed_iterations = current_iteration
        session.commit()

        _publish_optimization_progress(
            redis_client,
            study_id,
            {
                "type": "completed",
                "study_id": study_id,
                "total_evaluated": current_iteration,
                "pareto_front_size": len(pareto_indices),
                "best_run_id": str(best_run_id) if best_run_id else None,
            },
        )

        logger.info(
            "Optimization %s completed: %d iterations, %d Pareto-optimal",
            study_id, current_iteration, len(pareto_indices),
        )

    except Exception as e:
        logger.exception("Optimization task failed for study %s", study_id)
        try:
            from backend.app.models.optimization_study import OptimizationStudy as OS

            study = session.get(OS, UUID(study_id))
            if study:
                study.status = "failed"
                session.commit()
        except Exception:
            session.rollback()

        _publish_optimization_progress(
            redis_client,
            study_id,
            {
                "type": "error",
                "study_id": study_id,
                "error": str(e),
            },
        )
    finally:
        session.close()
        if redis_client:
            try:
                redis_client.close()
            except Exception:
                pass


def _find_best_run(
    history: list[dict],
    objectives: list[dict] | None,
    constraints: list[dict] | None,
    engine,
) -> str | None:
    """Find the best run from the history.

    For single-objective: the feasible run with the best objective value.
    For multi-objective: the first Pareto-optimal feasible run.
    """
    from backend.app.schemas.optimization import Objective as ObjSchema

    if not history or not objectives:
        return None

    objs = [ObjSchema(**o) if isinstance(o, dict) else o for o in objectives]

    # Filter feasible entries
    feasible_entries = [
        (i, entry) for i, entry in enumerate(history)
        if entry.get("feasible", True) and entry.get("metrics")
    ]

    if not feasible_entries:
        # Fall back to all entries
        feasible_entries = [
            (i, entry) for i, entry in enumerate(history)
            if entry.get("metrics")
        ]

    if not feasible_entries:
        return None

    if len(objs) == 1:
        # Single-objective: find best
        obj = objs[0]
        best_val = None
        best_run_id = None
        for _, entry in feasible_entries:
            val = entry["metrics"].get(obj.metric)
            if val is None:
                continue
            val = float(val)
            if obj.direction == "maximize":
                val = -val
            if best_val is None or val < best_val:
                best_val = val
                best_run_id = entry.get("run_id")
        return best_run_id
    else:
        # Multi-objective: first Pareto member
        pareto = engine.compute_pareto_front(history, objectives)
        if pareto:
            return history[pareto[0]].get("run_id")
        return feasible_entries[0][1].get("run_id") if feasible_entries else None
