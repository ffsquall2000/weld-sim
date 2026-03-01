"""Pydantic schemas for WebSocket messages."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import BaseModel


class WSMessage(BaseModel):
    """Base WebSocket message schema."""

    type: str  # progress, metric_update, completed, error, cancelled
    run_id: str
    timestamp: Optional[str] = None


class WSProgress(WSMessage):
    """WebSocket progress update message."""

    type: str = "progress"
    percent: float
    phase: str  # meshing, solving, postprocessing
    message: str
    elapsed_s: float


class WSMetricUpdate(WSMessage):
    """WebSocket metric update message."""

    type: str = "metric_update"
    metric_name: str
    value: float
    unit: Optional[str] = None


class WSCompleted(WSMessage):
    """WebSocket run-completed message."""

    type: str = "completed"
    status: str = "completed"
    compute_time_s: float
    metrics_summary: Dict[str, float] = {}


class WSError(WSMessage):
    """WebSocket error message."""

    type: str = "error"
    error: str
    solver_log_tail: Optional[str] = None
