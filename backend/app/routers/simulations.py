"""Endpoints for SimulationCase resources."""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.simulation import SimulationCaseCreate, SimulationCaseResponse, SimulationCaseUpdate
from backend.app.services.simulation_service import SimulationService

router = APIRouter(tags=["simulations"])


@router.post("/projects/{project_id}/simulations", response_model=SimulationCaseResponse, status_code=201)
async def create_simulation(project_id: uuid.UUID, body: SimulationCaseCreate, db: AsyncSession = Depends(get_db)) -> SimulationCaseResponse:
    svc = SimulationService(db)
    sim = await svc.create(project_id, body)
    return SimulationCaseResponse.model_validate(sim)


@router.get("/projects/{project_id}/simulations", response_model=List[SimulationCaseResponse])
async def list_simulations(project_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> List[SimulationCaseResponse]:
    svc = SimulationService(db)
    items = await svc.list_by_project(project_id)
    return [SimulationCaseResponse.model_validate(s) for s in items]


@router.get("/simulations/{simulation_id}", response_model=SimulationCaseResponse)
async def get_simulation(simulation_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> SimulationCaseResponse:
    svc = SimulationService(db)
    sim = await svc.get(simulation_id)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation case not found")
    return SimulationCaseResponse.model_validate(sim)


@router.patch("/simulations/{simulation_id}", response_model=SimulationCaseResponse)
async def update_simulation(simulation_id: uuid.UUID, body: SimulationCaseUpdate, db: AsyncSession = Depends(get_db)) -> SimulationCaseResponse:
    svc = SimulationService(db)
    sim = await svc.update(simulation_id, body)
    if not sim:
        raise HTTPException(status_code=404, detail="Simulation case not found")
    return SimulationCaseResponse.model_validate(sim)


@router.post("/simulations/{simulation_id}/validate", response_model=dict)
async def validate_simulation(simulation_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    svc = SimulationService(db)
    return await svc.validate(simulation_id)
