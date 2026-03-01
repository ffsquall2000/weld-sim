"""Endpoints for Optimization resources."""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.optimization import (
    IterationResult,
    OptimizationCreate,
    OptimizationResponse,
)
from backend.app.services.optimization_service import OptimizationService

router = APIRouter(tags=["optimizations"])


@router.post(
    "/simulations/{simulation_id}/optimize",
    response_model=OptimizationResponse,
    status_code=201,
)
async def create_optimization(
    simulation_id: uuid.UUID,
    body: OptimizationCreate,
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Create and start an optimization study for a simulation case."""
    svc = OptimizationService(db)
    study = await svc.create(simulation_id, body)
    return OptimizationResponse.model_validate(study)


@router.get(
    "/optimizations/{optimization_id}",
    response_model=OptimizationResponse,
)
async def get_optimization(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Get optimization study details and status."""
    svc = OptimizationService(db)
    study = await svc.get(optimization_id)
    if not study:
        raise HTTPException(status_code=404, detail="Optimization study not found")
    return OptimizationResponse.model_validate(study)


@router.get(
    "/optimizations/{optimization_id}/iterations",
    response_model=List[IterationResult],
)
async def get_optimization_iterations(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> List[IterationResult]:
    """Get all iteration results for an optimization study."""
    svc = OptimizationService(db)
    iterations = await svc.get_iterations(optimization_id)
    return [IterationResult(**it) for it in iterations]


@router.get(
    "/optimizations/{optimization_id}/pareto",
    response_model=List[IterationResult],
)
async def get_pareto_front(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> List[IterationResult]:
    """Get the Pareto front for a multi-objective optimization study."""
    svc = OptimizationService(db)
    pareto = await svc.get_pareto(optimization_id)
    return [IterationResult(**it) for it in pareto]


@router.post(
    "/optimizations/{optimization_id}/pause",
    response_model=OptimizationResponse,
)
async def pause_optimization(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Pause a running optimization study."""
    svc = OptimizationService(db)
    study = await svc.pause(optimization_id)
    if not study:
        raise HTTPException(status_code=404, detail="Optimization study not found")
    return OptimizationResponse.model_validate(study)


@router.post(
    "/optimizations/{optimization_id}/resume",
    response_model=OptimizationResponse,
)
async def resume_optimization(
    optimization_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> OptimizationResponse:
    """Resume a paused optimization study."""
    svc = OptimizationService(db)
    study = await svc.resume(optimization_id)
    if not study:
        raise HTTPException(status_code=404, detail="Optimization study not found")
    return OptimizationResponse.model_validate(study)
