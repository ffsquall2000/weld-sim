"""WebSocket router for real-time analysis progress."""
import asyncio
import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.analysis_manager import analysis_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/analysis/{task_id}")
async def analysis_progress_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time analysis progress updates.

    Clients connect here immediately after receiving a task_id from the
    HTTP response header or body.  The endpoint waits up to 10 seconds
    for the task to appear (in case the WebSocket connects before the
    HTTP handler has created the task).
    """
    await websocket.accept()
    logger.info("WS connected for task %s, waiting for task to appear…", task_id)

    # Wait for the task to appear (race between WS connect and HTTP handler)
    task = analysis_manager.get_task(task_id)
    if not task:
        for i in range(20):  # 10 seconds max
            await asyncio.sleep(0.5)
            task = analysis_manager.get_task(task_id)
            if task:
                logger.info("WS task %s found after %.1fs", task_id, (i + 1) * 0.5)
                break

    if not task:
        logger.warning("WS task %s NOT found after 10s – closing", task_id)
        await websocket.send_json({"type": "error", "message": f"Task {task_id} not found"})
        await websocket.close()
        return

    # Send initial status
    logger.info("WS task %s: sending initial status (status=%s, progress=%.2f)",
                task_id, task.status.value, task.progress)
    await websocket.send_json({
        "type": "connected",
        "task_id": task_id,
        "task_type": task.task_type,
        "status": task.status.value,
        "current_step": task.current_step,
        "total_steps": len(task.steps),
        "progress": task.progress,
    })

    # Subscribe to updates
    queue = analysis_manager.subscribe(task_id)
    logger.info("WS task %s: subscribed, waiting for progress messages…", task_id)

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                msg_type = message.get("type", "?")
                msg_progress = message.get("progress", "?")
                logger.info("WS task %s: forwarding %s (progress=%s)",
                            task_id, msg_type, msg_progress)
                await websocket.send_json(message)

                if message.get("type") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                logger.debug("WS task %s: ping (no message for 30s)", task_id)
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        logger.info("WS task %s: client disconnected", task_id)
    finally:
        analysis_manager.unsubscribe(task_id, queue)


@router.websocket("/analysis")
async def analysis_list_ws(websocket: WebSocket):
    """WebSocket endpoint for monitoring all analysis tasks."""
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)

            if cmd.get("action") == "list":
                tasks = []
                for task in analysis_manager._tasks.values():
                    tasks.append({
                        "task_id": task.task_id,
                        "task_type": task.task_type,
                        "status": task.status.value,
                        "progress": task.progress,
                    })
                await websocket.send_json({"type": "task_list", "tasks": tasks})

            elif cmd.get("action") == "cancel":
                task_id_to_cancel = cmd.get("task_id")
                if task_id_to_cancel:
                    success = await analysis_manager.cancel_task(task_id_to_cancel)
                    await websocket.send_json({
                        "type": "cancel_result",
                        "task_id": task_id_to_cancel,
                        "success": success,
                    })

    except WebSocketDisconnect:
        pass
