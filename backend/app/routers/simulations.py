"""Endpoints for SimulationCase resources."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException

from backend.app.schemas.simulation import (
    SimulationCaseCreate,
    SimulationCaseResponse,
    SimulationCaseUpdate,
)

router = APIRouter(tags=["simulations"])


@router.post(
    "/projects/{project_id}/simulations",
    response_model=SimulationCaseResponse,
    status_code=201,
)
async def create_simulation(
    project_id: uuid.UUID, body: SimulationCaseCreate
) -> SimulationCaseResponse:
    """Create a new simulation case within a project."""
    # TODO: call simulation service to persist
    now = datetime.now(tz=timezone.utc)
    return SimulationCaseResponse(
        id=uuid.uuid4(),
        project_id=project_id,
        name=body.name,
        description=body.description,
        analysis_type=body.analysis_type,
        solver_backend=body.solver_backend,
        configuration=body.configuration,
        boundary_conditions=body.boundary_conditions,
        material_assignments=body.material_assignments,
        assembly_components=(
            {str(i): c for i, c in enumerate(body.assembly_components)}
            if body.assembly_components
            else None
        ),
        workflow_dag=body.workflow_dag,
        created_at=now,
        updated_at=now,
    )


@router.get(
    "/projects/{project_id}/simulations",
    response_model=List[SimulationCaseResponse],
)
async def list_simulations(
    project_id: uuid.UUID,
) -> List[SimulationCaseResponse]:
    """List all simulation cases for a project."""
    # TODO: call simulation service to query DB
    return []


@router.get(
    "/simulations/{simulation_id}",
    response_model=SimulationCaseResponse,
)
async def get_simulation(
    simulation_id: uuid.UUID,
) -> SimulationCaseResponse:
    """Get a single simulation case by ID."""
    # TODO: call simulation service to fetch from DB
    raise HTTPException(status_code=404, detail="Simulation case not found")


@router.patch(
    "/simulations/{simulation_id}",
    response_model=SimulationCaseResponse,
)
async def update_simulation(
    simulation_id: uuid.UUID, body: SimulationCaseUpdate
) -> SimulationCaseResponse:
    """Partially update a simulation case."""
    # TODO: call simulation service to update in DB
    raise HTTPException(status_code=404, detail="Simulation case not found")


@router.post(
    "/simulations/{simulation_id}/validate",
    response_model=dict,
)
async def validate_simulation(
    simulation_id: uuid.UUID,
) -> dict:
    """Validate a simulation case configuration before running."""
    # TODO: call simulation validation service
    return {
        "valid": True,
        "errors": [],
        "warnings": [],
    }
