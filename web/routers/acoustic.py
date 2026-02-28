"""Acoustic analysis endpoints."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/acoustic", tags=["acoustic"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AcousticAnalysisRequest(BaseModel):
    horn_type: str = "cylindrical"
    width_mm: float = Field(default=25.0, gt=0)
    height_mm: float = Field(default=80.0, gt=0)
    length_mm: float = Field(default=25.0, gt=0)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(default=20.0, gt=0)
    mesh_density: str = "medium"
    use_gmsh: bool = True  # Default: Gmsh TET10 + SolverA pipeline (set False for legacy HEX8)


class StressHotspot(BaseModel):
    location: list[float]  # [x, y, z] in mm
    von_mises_mpa: float
    node_index: int


class AcousticAnalysisResponse(BaseModel):
    # Modal analysis
    modes: list[dict]
    closest_mode_hz: float
    target_frequency_hz: float
    frequency_deviation_percent: float
    # Harmonic response
    harmonic_response: dict  # {frequencies_hz: [...], amplitudes: [...]}
    # Amplitude distribution
    amplitude_distribution: dict  # {node_positions: [[x,y,z],...], amplitudes: [...]}
    amplitude_uniformity: float  # 0-1, higher is better
    # Stress
    stress_hotspots: list[StressHotspot]
    stress_max_mpa: float
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    mesh: Optional[dict] = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AcousticAnalysisResponse)
async def run_acoustic_analysis(request: AcousticAnalysisRequest):
    """Run comprehensive acoustic analysis on a horn geometry.

    Uses subprocess isolation with timeout and real-time progress.
    """
    if not request.use_gmsh:
        # Legacy path
        try:
            from web.services.fea_service import FEAService
            svc = FEAService()
            result = await asyncio.to_thread(
                svc.run_acoustic_analysis,
                horn_type=request.horn_type, width_mm=request.width_mm,
                height_mm=request.height_mm, length_mm=request.length_mm,
                material=request.material, frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
            return AcousticAnalysisResponse(**result)
        except Exception as exc:
            logger.error("Acoustic legacy error: %s", exc, exc_info=True)
            raise HTTPException(500, f"Acoustic analysis failed: {exc}") from exc

    from web.services.fea_process_runner import FEAProcessRunner
    from web.services.analysis_manager import analysis_manager

    steps = ["init", "meshing", "assembly", "solving", "classifying", "packaging"]
    task_id = analysis_manager.create_task("acoustic", steps)
    runner = FEAProcessRunner()
    analysis_manager.set_cancel_hook(task_id, runner)

    async def _on_progress(phase, progress, message):
        step_map = {s: i for i, s in enumerate(steps)}
        await analysis_manager.update_progress(
            task_id, step_map.get(phase, 0), progress, message)

    params = {
        "horn_type": request.horn_type,
        "diameter_mm": request.width_mm,
        "length_mm": request.height_mm,
        "material": request.material,
        "frequency_khz": request.frequency_khz,
        "mesh_density": request.mesh_density,
    }
    try:
        result = await runner.run("acoustic", params, on_progress=_on_progress)
        await analysis_manager.complete_task(task_id, result)
        result["task_id"] = task_id
        return AcousticAnalysisResponse(**result)
    except TimeoutError as exc:
        await analysis_manager.fail_task(task_id, str(exc))
        raise HTTPException(504, str(exc)) from exc
    except asyncio.CancelledError:
        await analysis_manager.fail_task(task_id, "Cancelled")
        raise HTTPException(499, "Analysis cancelled")
    except ValueError as exc:
        await analysis_manager.fail_task(task_id, str(exc))
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        await analysis_manager.fail_task(task_id, str(exc))
        logger.error("Acoustic analysis error: %s", exc, exc_info=True)
        raise HTTPException(500, f"Acoustic analysis failed: {exc}") from exc
