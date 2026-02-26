"""Health check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "service": "UltrasonicWeldMaster Simulation Service",
    }
