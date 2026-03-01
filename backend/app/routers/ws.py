"""WebSocket endpoints for real-time simulation progress."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/runs/{run_id}")
async def ws_run_progress(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for streaming run progress updates.

    Sends periodic heartbeat messages while the connection is open.
    In production, this will stream real progress from the solver task.
    """
    await websocket.accept()
    try:
        while True:
            # TODO: replace with actual progress from task queue / Redis pub-sub
            heartbeat = {
                "type": "heartbeat",
                "run_id": run_id,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(heartbeat))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/optimizations/{optimization_id}")
async def ws_optimization_progress(
    websocket: WebSocket, optimization_id: str
) -> None:
    """WebSocket endpoint for streaming optimization study progress.

    Sends periodic heartbeat messages while the connection is open.
    In production, this will stream iteration results from the optimizer.
    """
    await websocket.accept()
    try:
        while True:
            # TODO: replace with actual progress from optimization service
            heartbeat = {
                "type": "heartbeat",
                "optimization_id": optimization_id,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            }
            await websocket.send_text(json.dumps(heartbeat))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close(code=1011, reason="Internal server error")
