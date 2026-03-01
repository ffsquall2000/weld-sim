"""Endpoints for Run resources."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.schemas.artifact import ArtifactResponse
from backend.app.schemas.metric import MetricResponse
from backend.app.schemas.run import RunCreate, RunDetailResponse, RunResponse

router = APIRouter(tags=["runs"])


@router.post(
    "/simulations/{simulation_id}/runs",
    response_model=RunResponse,
    status_code=201,
)
async def create_run(
    simulation_id: uuid.UUID, body: RunCreate
) -> RunResponse:
    """Submit a new simulation run for execution."""
    # TODO: call run service to create and enqueue
    from datetime import datetime, timezone

    now = datetime.now(tz=timezone.utc)
    return RunResponse(
        id=uuid.uuid4(),
        simulation_case_id=simulation_id,
        geometry_version_id=body.geometry_version_id,
        optimization_study_id=None,
        iteration_number=None,
        status="queued",
        started_at=None,
        completed_at=None,
        compute_time_s=None,
        error_message=None,
        created_at=now,
    )


@router.get(
    "/simulations/{simulation_id}/runs",
    response_model=List[RunResponse],
)
async def list_runs(simulation_id: uuid.UUID) -> List[RunResponse]:
    """List all runs for a simulation case."""
    # TODO: call run service to query DB
    return []


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: uuid.UUID) -> RunDetailResponse:
    """Get a single run with full details including metrics and artifacts."""
    # TODO: call run service to fetch from DB
    raise HTTPException(status_code=404, detail="Run not found")


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(run_id: uuid.UUID) -> RunResponse:
    """Cancel a running or queued simulation run."""
    # TODO: call run service to cancel
    raise HTTPException(status_code=404, detail="Run not found")


@router.get(
    "/runs/{run_id}/metrics", response_model=List[MetricResponse]
)
async def get_run_metrics(run_id: uuid.UUID) -> List[MetricResponse]:
    """Get all metrics for a specific run."""
    # TODO: call metric service to query DB
    return []


@router.get(
    "/runs/{run_id}/artifacts", response_model=List[ArtifactResponse]
)
async def get_run_artifacts(run_id: uuid.UUID) -> List[ArtifactResponse]:
    """Get all artifacts for a specific run."""
    # TODO: call artifact service to query DB
    return []


@router.get("/runs/{run_id}/artifacts/{artifact_id}/download")
async def download_artifact(
    run_id: uuid.UUID, artifact_id: uuid.UUID
) -> FileResponse:
    """Download a specific artifact file."""
    # TODO: call artifact service to locate file
    raise HTTPException(status_code=404, detail="Artifact not found")


@router.get("/runs/{run_id}/results/field/{field_name}")
async def get_field_results(
    run_id: uuid.UUID, field_name: str
) -> dict:
    """Get field results (displacement, stress, temperature, etc.) for visualization."""
    # TODO: call post-processing service to load field data
    raise HTTPException(status_code=404, detail="Run not found")
