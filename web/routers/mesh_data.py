"""Binary mesh data endpoints for efficient frontend transfer."""
import struct
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
import numpy as np

router = APIRouter(prefix="/mesh", tags=["mesh"])

# In-memory cache for demo (in production, use Redis or file storage)
_mesh_cache: dict = {}


def register_mesh(task_id: str, mesh_data: dict):
    """Register mesh data for a task (called by FEA service after analysis)."""
    _mesh_cache[task_id] = mesh_data


def get_mesh(task_id: str) -> Optional[dict]:
    """Get mesh data for a task."""
    return _mesh_cache.get(task_id)


@router.get("/{task_id}/geometry")
async def get_mesh_geometry(task_id: str):
    """
    Get mesh geometry as binary data.

    Binary format:
    - Header (12 bytes): node_count (uint32), face_count (uint32), has_normals (uint32)
    - Positions: float32[node_count * 3] - flattened XYZ coordinates
    - Indices: uint32[face_count * 3] - flattened triangle indices
    - Normals (optional): float32[node_count * 3] - vertex normals
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(status_code=404, detail=f"Mesh not found for task {task_id}")

    positions = mesh.get("positions")  # numpy array (N, 3) or (N*3,)
    indices = mesh.get("indices")      # numpy array (F, 3) or (F*3,)
    normals = mesh.get("normals")      # optional numpy array

    if positions is None or indices is None:
        raise HTTPException(status_code=404, detail="Mesh geometry data not available")

    # Ensure correct types
    positions = np.asarray(positions, dtype=np.float32).flatten()
    indices = np.asarray(indices, dtype=np.uint32).flatten()

    node_count = len(positions) // 3
    face_count = len(indices) // 3
    has_normals = 1 if normals is not None else 0

    # Build binary response
    header = struct.pack("<III", node_count, face_count, has_normals)
    data = header + positions.tobytes() + indices.tobytes()

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).flatten()
        data += normals.tobytes()

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename=mesh_{task_id}.bin",
            "X-Node-Count": str(node_count),
            "X-Face-Count": str(face_count),
        }
    )


@router.get("/{task_id}/scalars")
async def get_mesh_scalars(
    task_id: str,
    field: str = Query(..., description="Scalar field name (e.g., 'von_mises', 'displacement_mag', 'temperature')"),
):
    """
    Get scalar field data as binary Float32Array.

    Binary format: float32[node_count] - one value per node
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(status_code=404, detail=f"Mesh not found for task {task_id}")

    scalars = mesh.get("scalars", {}).get(field)
    if scalars is None:
        available = list(mesh.get("scalars", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Scalar field '{field}' not found. Available: {available}"
        )

    scalars = np.asarray(scalars, dtype=np.float32).flatten()

    # Include min/max in headers for frontend colormap setup
    min_val = float(np.min(scalars))
    max_val = float(np.max(scalars))

    return Response(
        content=scalars.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Scalar-Min": str(min_val),
            "X-Scalar-Max": str(max_val),
            "X-Node-Count": str(len(scalars)),
        }
    )


@router.get("/{task_id}/modes/{mode_index}")
async def get_mode_shape(task_id: str, mode_index: int):
    """
    Get mode shape displacement as binary Float32Array.

    Binary format:
    - Header (12 bytes): frequency (float32), mode_type_len (uint32), reserved (uint32)
    - Mode type string: utf-8 bytes (mode_type_len)
    - Shape: float32[node_count * 3] - displacement vector per node
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(status_code=404, detail=f"Mesh not found for task {task_id}")

    modes = mesh.get("modes", [])
    if mode_index < 0 or mode_index >= len(modes):
        raise HTTPException(
            status_code=404,
            detail=f"Mode {mode_index} not found. Available modes: 0-{len(modes)-1}"
        )

    mode = modes[mode_index]
    frequency = float(mode.get("frequency_hz", 0))
    mode_type = mode.get("mode_type", "unknown").encode("utf-8")
    shape = np.asarray(mode.get("shape"), dtype=np.float32).flatten()

    # Build binary response
    header = struct.pack("<fII", frequency, len(mode_type), 0)
    data = header + mode_type + shape.tobytes()

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Frequency-Hz": str(frequency),
            "X-Mode-Type": mode_type.decode("utf-8"),
            "X-Node-Count": str(len(shape) // 3),
        }
    )


@router.get("/{task_id}/info")
async def get_mesh_info(task_id: str):
    """Get mesh metadata as JSON."""
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(status_code=404, detail=f"Mesh not found for task {task_id}")

    positions = mesh.get("positions")
    indices = mesh.get("indices")

    return {
        "task_id": task_id,
        "node_count": len(positions) // 3 if positions is not None else 0,
        "face_count": len(indices) // 3 if indices is not None else 0,
        "has_normals": mesh.get("normals") is not None,
        "available_scalars": list(mesh.get("scalars", {}).keys()),
        "mode_count": len(mesh.get("modes", [])),
        "modes": [
            {
                "index": i,
                "frequency_hz": m.get("frequency_hz"),
                "mode_type": m.get("mode_type"),
            }
            for i, m in enumerate(mesh.get("modes", []))
        ],
    }
