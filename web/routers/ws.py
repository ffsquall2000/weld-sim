"""WebSocket router for real-time analysis progress."""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.analysis_manager import analysis_manager

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

    # Wait for the task to appear (race between WS connect and HTTP handler)
    task = analysis_manager.get_task(task_id)
    if not task:
        for _ in range(20):  # 10 seconds max
            await asyncio.sleep(0.5)
            task = analysis_manager.get_task(task_id)
            if task:
                break

    if not task:
        await websocket.send_json({"type": "error", "message": f"Task {task_id} not found"})
        await websocket.close()
        return

    # Send initial status
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

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(message)

                if message.get("type") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
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
