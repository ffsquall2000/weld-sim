"""WebSocket router for real-time analysis progress."""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.analysis_manager import analysis_manager

router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/analysis/{task_id}")
async def analysis_progress_ws(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time analysis progress updates."""
    await websocket.accept()

    # Check if task exists
    task = analysis_manager.get_task(task_id)
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
            # Wait for updates with timeout (for keepalive)
            try:
                message = await asyncio.wait_for(queue.get(), timeout=30.0)
                await websocket.send_json(message)

                # Close if task is done
                if message.get("type") in ("completed", "failed", "cancelled"):
                    break
            except asyncio.TimeoutError:
                # Send keepalive ping
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
            # Receive commands from client
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
                task_id = cmd.get("task_id")
                if task_id:
                    success = await analysis_manager.cancel_task(task_id)
                    await websocket.send_json({"type": "cancel_result", "task_id": task_id, "success": success})

    except WebSocketDisconnect:
        pass
