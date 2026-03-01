"""WebSocket endpoints for real-time simulation progress.

When Redis is available, progress messages are received via Redis pub/sub
channels published by the Celery solver task.  When Redis is unavailable
the endpoint falls back to a simple heartbeat loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.config import settings

router = APIRouter(prefix="/ws", tags=["websocket"])
logger = logging.getLogger(__name__)


async def _get_async_redis():
    """Get an async Redis client, or *None* if Redis is not reachable."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.Redis.from_url(settings.REDIS_URL)
        # Quick connectivity check
        await client.ping()
        return client
    except Exception:
        return None


@router.websocket("/runs/{run_id}")
async def ws_run_progress(websocket: WebSocket, run_id: str) -> None:
    """WebSocket endpoint for streaming run progress updates via Redis pub/sub.

    Messages forwarded to the client include:
      - ``{"type": "progress", ...}`` -- ongoing progress updates
      - ``{"type": "completed", ...}`` -- solver finished successfully
      - ``{"type": "error", ...}`` -- solver encountered an error

    The connection is closed automatically when a terminal message
    (``completed`` or ``error``) is received from the solver task.
    """
    await websocket.accept()
    redis = await _get_async_redis()

    if redis:
        pubsub = redis.pubsub()
        channel = f"run:{run_id}:progress"
        await pubsub.subscribe(channel)
        try:
            while True:
                # Check for Redis messages
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.5
                )
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    await websocket.send_text(data)
                    # Check if this was a terminal message
                    try:
                        parsed = json.loads(data)
                        if parsed.get("type") in (
                            "completed",
                            "error",
                            "cancelled",
                        ):
                            break
                    except json.JSONDecodeError:
                        pass

                # Handle client messages (ping/pong keepalive)
                try:
                    client_data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.1
                    )
                    try:
                        msg = json.loads(client_data)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning("WebSocket error for run %s: %s", run_id, e)
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await redis.close()
    else:
        # Fallback: simple heartbeat when Redis is not available
        try:
            while True:
                heartbeat = {
                    "type": "heartbeat",
                    "run_id": run_id,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                await websocket.send_text(json.dumps(heartbeat))
                # Handle client messages
                try:
                    client_data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=5.0
                    )
                    try:
                        msg = json.loads(client_data)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception:
            await websocket.close(code=1011, reason="Internal server error")


@router.websocket("/optimizations/{optimization_id}")
async def ws_optimization_progress(
    websocket: WebSocket, optimization_id: str
) -> None:
    """WebSocket endpoint for streaming optimization study progress via Redis pub/sub.

    Messages are published by the optimization service under the channel
    ``optimization:{optimization_id}:progress``.
    """
    await websocket.accept()
    redis = await _get_async_redis()

    if redis:
        pubsub = redis.pubsub()
        channel = f"optimization:{optimization_id}:progress"
        await pubsub.subscribe(channel)
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.5
                )
                if message and message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")
                    await websocket.send_text(data)
                    # Check for terminal message
                    try:
                        parsed = json.loads(data)
                        if parsed.get("type") in (
                            "completed",
                            "error",
                            "cancelled",
                        ):
                            break
                    except json.JSONDecodeError:
                        pass

                # Handle client messages
                try:
                    client_data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=0.1
                    )
                    try:
                        msg = json.loads(client_data)
                        if msg.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                    except json.JSONDecodeError:
                        pass
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.warning(
                "Optimization WebSocket error for %s: %s", optimization_id, e
            )
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await redis.close()
    else:
        # Fallback: simple heartbeat
        try:
            while True:
                heartbeat = {
                    "type": "heartbeat",
                    "optimization_id": optimization_id,
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
                await websocket.send_text(json.dumps(heartbeat))
                try:
                    await asyncio.wait_for(
                        websocket.receive_text(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass
        except Exception:
            await websocket.close(code=1011, reason="Internal server error")
