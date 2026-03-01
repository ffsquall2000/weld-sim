"""Endpoints for Optimization resources."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException

from backend.app.schemas.optimization import (
    IterationResult,
    OptimizationCreate,
    OptimizationResponse,
)

router = APIRouter(tags=["optimizations"])


@router.post(
    "/simulations/{simulation_id}/optimize",
    response_model=OptimizationResponse,
    status_code=201,
)
async def create_optimization(
    simulation_id: uuid.UUID, body: OptimizationCreate
) -> OptimizationResponse:
    """Create and start an optimization study for a simulation case."""
    # TODO: call optimization service to persist and enqueue
    now = datetime.now(tz=timezone.utc)
    return OptimizationResponse(
        id=uuid.uuid4(),
        simulation_case_id=simulation_id,
        name=body.name,
        strategy=body.strategy,
        design_variables=[dv.model_dump() for dv in body.design_variables],
        constraints=(
            [c.model_dump() for c in body.constraints]
            if body.constraints
            else None
        ),
        objectives=[obj.model_dump() for obj in body.objectives],
        status="pending",
        total_iterations=body.max_iterations,
        completed_iterations=0,
        best_run_id=None,
        pareto_front_run_ids=None,
        created_at=now,
        updated_at=now,
    )


@router.get(
    "/optimizations/{optimization_id}",
    response_model=OptimizationResponse,
)
async def get_optimization(
    optimization_id: uuid.UUID,
) -> OptimizationResponse:
    """Get optimization study details and status."""
    # TODO: call optimization service to fetch from DB
    raise HTTPException(status_code=404, detail="Optimization study not found")


@router.get(
    "/optimizations/{optimization_id}/iterations",
    response_model=List[IterationResult],
)
async def get_optimization_iterations(
    optimization_id: uuid.UUID,
) -> List[IterationResult]:
    """Get all iteration results for an optimization study."""
    # TODO: call optimization service to fetch iteration data
    return []


@router.get(
    "/optimizations/{optimization_id}/pareto",
    response_model=List[IterationResult],
)
async def get_pareto_front(
    optimization_id: uuid.UUID,
) -> List[IterationResult]:
    """Get the Pareto front for a multi-objective optimization study."""
    # TODO: call optimization service to compute Pareto front
    return []


@router.post(
    "/optimizations/{optimization_id}/pause",
    response_model=OptimizationResponse,
)
async def pause_optimization(
    optimization_id: uuid.UUID,
) -> OptimizationResponse:
    """Pause a running optimization study."""
    # TODO: call optimization service to pause
    raise HTTPException(status_code=404, detail="Optimization study not found")


@router.post(
    "/optimizations/{optimization_id}/resume",
    response_model=OptimizationResponse,
)
async def resume_optimization(
    optimization_id: uuid.UUID,
) -> OptimizationResponse:
    """Resume a paused optimization study."""
    # TODO: call optimization service to resume
    raise HTTPException(status_code=404, detail="Optimization study not found")
