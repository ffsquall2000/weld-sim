"""Endpoints for Run resources."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.artifact import ArtifactResponse
from backend.app.schemas.metric import MetricResponse
from backend.app.schemas.run import RunCreate, RunDetailResponse, RunResponse
from backend.app.services.run_service import RunService

router = APIRouter(tags=["runs"])


@router.post("/simulations/{simulation_id}/runs", response_model=RunResponse, status_code=201)
async def create_run(simulation_id: uuid.UUID, body: RunCreate, db: AsyncSession = Depends(get_db)) -> RunResponse:
    svc = RunService(db)
    run = await svc.submit(simulation_id, body)
    return RunResponse.model_validate(run)


@router.get("/simulations/{simulation_id}/runs", response_model=List[RunResponse])
async def list_runs(simulation_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> List[RunResponse]:
    svc = RunService(db)
    items = await svc.list_by_simulation(simulation_id)
    return [RunResponse.model_validate(r) for r in items]


@router.get("/runs/{run_id}", response_model=RunDetailResponse)
async def get_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> RunDetailResponse:
    svc = RunService(db)
    run = await svc.get_detail(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunDetailResponse.model_validate(run)


@router.post("/runs/{run_id}/cancel", response_model=RunResponse)
async def cancel_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> RunResponse:
    svc = RunService(db)
    run = await svc.cancel(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse.model_validate(run)


@router.get("/runs/{run_id}/metrics", response_model=List[MetricResponse])
async def get_run_metrics(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> List[MetricResponse]:
    svc = RunService(db)
    metrics = await svc.get_metrics(run_id)
    return [MetricResponse.model_validate(m) for m in metrics]


@router.get("/runs/{run_id}/artifacts", response_model=List[ArtifactResponse])
async def get_run_artifacts(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> List[ArtifactResponse]:
    svc = RunService(db)
    artifacts = await svc.get_artifacts(run_id)
    return [ArtifactResponse.model_validate(a) for a in artifacts]


@router.get("/runs/{run_id}/artifacts/{artifact_id}/download")
async def download_artifact(run_id: uuid.UUID, artifact_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> FileResponse:
    svc = RunService(db)
    artifact = await svc.get_artifact(run_id, artifact_id)
    if not artifact or not Path(artifact.file_path).exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(artifact.file_path, media_type=artifact.mime_type, filename=Path(artifact.file_path).name)


@router.get("/runs/{run_id}/metrics/standard")
async def get_standard_metrics(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> dict:
    """Get standardized metrics with quality score for a run."""
    svc = RunService(db)
    metrics_list = await svc.get_metrics(run_id)
    raw_metrics = {m.metric_name: m.value for m in metrics_list}

    # Load geometry params for derived metrics
    run = await svc.get(run_id)
    geo_params: dict = {}
    mat_props: dict = {}
    if run and run.geometry_version_id:
        from backend.app.services.geometry_service import GeometryService
        geo_svc = GeometryService(db)
        geom = await geo_svc.get(run.geometry_version_id)
        if geom:
            geo_params = geom.parametric_params or {}

    from backend.app.services.metrics_service import MetricsService
    standard = MetricsService.compute_standard_metrics(raw_metrics, mat_props, geo_params)
    quality_score = MetricsService.compute_quality_score(standard)

    return {
        "metrics": standard,
        "quality_score": quality_score,
        "metric_info": MetricsService.get_metric_info(),
    }


@router.get("/metrics/catalog")
async def get_metrics_catalog() -> dict:
    """Get the standardized metrics catalog."""
    from backend.app.services.metrics_service import MetricsService
    return {"metrics": MetricsService.get_metric_info()}


@router.get("/runs/{run_id}/results/field/{field_name}")
async def get_field_results(run_id: uuid.UUID, field_name: str, db: AsyncSession = Depends(get_db)) -> dict:
    svc = RunService(db)
    artifacts = await svc.get_artifacts(run_id)
    vtu_artifact = next((a for a in artifacts if a.artifact_type == "result_vtu"), None)
    if not vtu_artifact or not Path(vtu_artifact.file_path).exists():
        raise HTTPException(status_code=404, detail="No result data found for this run")
    from backend.app.solvers.result_reader import ResultReader
    reader = ResultReader()
    field_data = reader.read_vtu(vtu_artifact.file_path)
    return reader.field_to_vtk_json(field_data, field_name)
