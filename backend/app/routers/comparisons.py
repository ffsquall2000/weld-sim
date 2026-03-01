"""Endpoints for Comparison resources."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.app.schemas.comparison import (
    ComparisonCreate,
    ComparisonResponse,
)

router = APIRouter(tags=["comparisons"])


@router.post(
    "/projects/{project_id}/comparisons",
    response_model=ComparisonResponse,
    status_code=201,
)
async def create_comparison(
    project_id: uuid.UUID, body: ComparisonCreate
) -> ComparisonResponse:
    """Create a new cross-run comparison."""
    # TODO: call comparison service to persist and compute deltas
    now = datetime.now(tz=timezone.utc)
    return ComparisonResponse(
        id=uuid.uuid4(),
        project_id=project_id,
        name=body.name,
        description=body.description,
        run_ids=body.run_ids,
        metric_names=body.metric_names,
        baseline_run_id=body.baseline_run_id,
        results=[],
        created_at=now,
    )


@router.get(
    "/comparisons/{comparison_id}",
    response_model=ComparisonResponse,
)
async def get_comparison(
    comparison_id: uuid.UUID,
) -> ComparisonResponse:
    """Get comparison details and results."""
    # TODO: call comparison service to fetch from DB
    raise HTTPException(status_code=404, detail="Comparison not found")


@router.post(
    "/comparisons/{comparison_id}/refresh",
    response_model=ComparisonResponse,
)
async def refresh_comparison(
    comparison_id: uuid.UUID,
) -> ComparisonResponse:
    """Refresh comparison results (re-compute deltas from latest metrics)."""
    # TODO: call comparison service to recompute
    raise HTTPException(status_code=404, detail="Comparison not found")
