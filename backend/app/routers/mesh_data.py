"""V2 API endpoints for binary mesh data delivery.

These endpoints stream mesh geometry, scalar fields, and mode-shape data
as compact binary buffers so the WebGL frontend can consume them directly
without JSON parsing overhead.
"""
from __future__ import annotations

import struct
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.app.schemas.geometry_analysis import MeshInfoResponse, MeshModeInfo

router = APIRouter(prefix="/mesh", tags=["mesh-data"])

# ---------------------------------------------------------------------------
# Shared in-memory mesh cache
# ---------------------------------------------------------------------------
# The FEA endpoints (or any producer) call ``register_mesh(task_id, data)``
# after analysis completes.  The binary endpoints below read from this cache.
#
# In production this would be backed by Redis or an object store; the
# in-memory dict is sufficient for demo / single-process usage.
# ---------------------------------------------------------------------------

_mesh_cache: dict[str, dict] = {}


def register_mesh(task_id: str, mesh_data: dict) -> None:
    """Store mesh data for later retrieval by the binary endpoints.

    Parameters
    ----------
    task_id:
        Unique identifier for this mesh (typically the FEA run id).
    mesh_data:
        Dictionary with keys:
        - ``positions``: np.ndarray (N,3) or (N*3,) float32 vertex positions
        - ``indices``: np.ndarray (F,3) or (F*3,) uint32 triangle indices
        - ``normals``: Optional np.ndarray (N,3) vertex normals
        - ``scalars``: Optional dict[str, np.ndarray] scalar fields per node
        - ``modes``: Optional list[dict] with keys frequency_hz, mode_type, shape
    """
    _mesh_cache[task_id] = mesh_data


def get_mesh(task_id: str) -> Optional[dict]:
    """Retrieve cached mesh data for *task_id*, or ``None``."""
    return _mesh_cache.get(task_id)


# ---------------------------------------------------------------------------
# GET /mesh/{task_id}/geometry -- binary octet-stream
# ---------------------------------------------------------------------------


@router.get("/{task_id}/geometry")
async def get_mesh_geometry(task_id: str):
    """Stream mesh geometry as a compact binary buffer.

    **Binary format** (little-endian):

    | Section   | Type              | Count             |
    |-----------|-------------------|-------------------|
    | Header    | 3x uint32         | node_count, face_count, has_normals |
    | Positions | float32           | node_count * 3    |
    | Indices   | uint32            | face_count * 3    |
    | Normals   | float32 (opt.)    | node_count * 3    |
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(
            status_code=404, detail=f"Mesh not found for task {task_id}"
        )

    positions = mesh.get("positions")
    indices = mesh.get("indices")
    normals = mesh.get("normals")

    if positions is None or indices is None:
        raise HTTPException(
            status_code=404, detail="Mesh geometry data not available"
        )

    # Ensure correct dtypes and flatten
    positions = np.asarray(positions, dtype=np.float32).flatten()
    indices = np.asarray(indices, dtype=np.uint32).flatten()

    node_count = len(positions) // 3
    face_count = len(indices) // 3
    has_normals = 1 if normals is not None else 0

    # Build binary payload
    header = struct.pack("<III", node_count, face_count, has_normals)
    data = header + positions.tobytes() + indices.tobytes()

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32).flatten()
        data += normals.tobytes()

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="mesh_{task_id}.bin"',
            "X-Node-Count": str(node_count),
            "X-Face-Count": str(face_count),
        },
    )


# ---------------------------------------------------------------------------
# GET /mesh/{task_id}/scalars -- binary float32 array
# ---------------------------------------------------------------------------


@router.get("/{task_id}/scalars")
async def get_mesh_scalars(
    task_id: str,
    field: str = Query(
        ...,
        description=(
            "Scalar field name, e.g. 'von_mises', 'displacement_mag', 'temperature'"
        ),
    ),
):
    """Stream a per-node scalar field as a raw Float32Array.

    **Binary format**: ``float32[node_count]`` -- one value per mesh node.

    The ``X-Scalar-Min`` and ``X-Scalar-Max`` response headers carry the
    data range so the frontend can set up a colour-map without a second pass.
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(
            status_code=404, detail=f"Mesh not found for task {task_id}"
        )

    scalars = mesh.get("scalars", {}).get(field)
    if scalars is None:
        available = list(mesh.get("scalars", {}).keys())
        raise HTTPException(
            status_code=404,
            detail=f"Scalar field '{field}' not found. Available: {available}",
        )

    scalars = np.asarray(scalars, dtype=np.float32).flatten()
    min_val = float(np.min(scalars))
    max_val = float(np.max(scalars))

    return Response(
        content=scalars.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Scalar-Min": str(min_val),
            "X-Scalar-Max": str(max_val),
            "X-Node-Count": str(len(scalars)),
        },
    )


# ---------------------------------------------------------------------------
# GET /mesh/{task_id}/modes/{mode_index} -- binary mode shape data
# ---------------------------------------------------------------------------


@router.get("/{task_id}/modes/{mode_index}")
async def get_mode_shape(task_id: str, mode_index: int):
    """Stream a single mode-shape displacement field as binary data.

    **Binary format** (little-endian):

    | Section         | Type            | Count           |
    |-----------------|-----------------|-----------------|
    | Header          | float32 + 2x uint32 | frequency, mode_type_len, reserved |
    | Mode type       | utf-8 bytes     | mode_type_len   |
    | Displacements   | float32         | node_count * 3  |
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(
            status_code=404, detail=f"Mesh not found for task {task_id}"
        )

    modes = mesh.get("modes", [])
    if mode_index < 0 or mode_index >= len(modes):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Mode {mode_index} not found. "
                f"Available modes: 0-{len(modes) - 1}" if modes else "No modes available"
            ),
        )

    mode = modes[mode_index]
    frequency = float(mode.get("frequency_hz", 0))
    mode_type_bytes = mode.get("mode_type", "unknown").encode("utf-8")
    shape = np.asarray(mode.get("shape"), dtype=np.float32).flatten()

    # Build binary payload
    header = struct.pack("<fII", frequency, len(mode_type_bytes), 0)
    data = header + mode_type_bytes + shape.tobytes()

    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={
            "X-Frequency-Hz": str(frequency),
            "X-Mode-Type": mode_type_bytes.decode("utf-8"),
            "X-Node-Count": str(len(shape) // 3),
        },
    )


# ---------------------------------------------------------------------------
# GET /mesh/{task_id}/info -- JSON metadata
# ---------------------------------------------------------------------------


@router.get("/{task_id}/info", response_model=MeshInfoResponse)
async def get_mesh_info(task_id: str):
    """Return JSON metadata for a cached mesh.

    Useful for the frontend to discover available scalar fields and mode
    counts before fetching the (potentially large) binary payloads.
    """
    mesh = get_mesh(task_id)
    if not mesh:
        raise HTTPException(
            status_code=404, detail=f"Mesh not found for task {task_id}"
        )

    positions = mesh.get("positions")
    indices = mesh.get("indices")

    positions_flat = (
        np.asarray(positions).flatten() if positions is not None else np.array([])
    )
    indices_flat = (
        np.asarray(indices).flatten() if indices is not None else np.array([])
    )

    modes_raw = mesh.get("modes", [])

    return MeshInfoResponse(
        task_id=task_id,
        node_count=len(positions_flat) // 3,
        face_count=len(indices_flat) // 3,
        has_normals=mesh.get("normals") is not None,
        available_scalars=list(mesh.get("scalars", {}).keys()),
        mode_count=len(modes_raw),
        modes=[
            MeshModeInfo(
                index=i,
                frequency_hz=m.get("frequency_hz"),
                mode_type=m.get("mode_type"),
            )
            for i, m in enumerate(modes_raw)
        ],
    )
