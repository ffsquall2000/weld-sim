"""Assembly analysis endpoints for multi-body ultrasonic welding stacks."""
from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assembly", tags=["assembly"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ComponentRequest(BaseModel):
    """A single component in the assembly."""

    name: str  # "horn", "booster", "transducer"
    horn_type: str = "cylindrical"  # For GmshMesher
    dimensions: dict  # Horn dimensions in mm
    material_name: str = "Titanium Ti-6Al-4V"
    mesh_size: float = Field(default=2.0, gt=0)


class AssemblyAnalysisRequest(BaseModel):
    """Request for full assembly analysis."""

    components: list[ComponentRequest]
    coupling_method: str = "bonded"
    penalty_factor: float = Field(default=1e3, gt=0)
    analyses: list[str] = ["modal", "harmonic"]
    frequency_hz: float = Field(default=20000.0, gt=0)
    n_modes: int = Field(default=20, ge=1)
    damping_ratio: float = Field(default=0.005, ge=0)
    use_gmsh: bool = True
    task_id: Optional[str] = None  # Client-generated UUID for early WebSocket connection


class AssemblyAnalysisResponse(BaseModel):
    """Response from assembly analysis."""

    success: bool
    message: str
    n_total_dof: int = 0
    n_components: int = 0
    # Modal results
    frequencies_hz: list[float] = []
    mode_types: list[str] = []
    # Harmonic results
    resonance_frequency_hz: float = 0.0
    gain: float = 0.0
    q_factor: float = 0.0
    uniformity: float = 0.0
    # Gain chain
    gain_chain: dict = {}
    # Impedance
    impedance: dict = {}
    transmission_coefficients: dict = {}
    # Timing
    solve_time_s: float = 0.0


class MaterialListItem(BaseModel):
    """Material summary for listing."""

    name: str
    E_gpa: float
    density_kg_m3: float
    poisson_ratio: float
    acoustic_velocity_m_s: Optional[float] = None


class BoosterProfileItem(BaseModel):
    """Booster profile type for listing."""

    profile: str
    description: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


async def _run_assembly_subprocess(request: AssemblyAnalysisRequest, analyses: list[str]):
    """Shared helper for assembly analysis with subprocess isolation."""
    from web.services.fea_process_runner import FEAProcessRunner
    from web.services.analysis_manager import analysis_manager

    steps = ["init", "component_analysis", "aggregation", "packaging"]
    task_id = analysis_manager.create_task("assembly", steps, task_id=request.task_id)
    runner = FEAProcessRunner()
    analysis_manager.set_cancel_hook(task_id, runner)

    async def _on_progress(phase, progress, message):
        step_map = {s: i for i, s in enumerate(steps)}
        await analysis_manager.update_progress(
            task_id, step_map.get(phase, 0), progress, message)

    components = [c.model_dump() for c in request.components]
    params = {
        "components": components,
        "coupling_method": request.coupling_method,
        "penalty_factor": request.penalty_factor,
        "analyses": analyses,
        "frequency_hz": request.frequency_hz,
        "n_modes": request.n_modes,
        "damping_ratio": request.damping_ratio,
        "use_gmsh": request.use_gmsh,
    }
    result = await runner.run("assembly", params, on_progress=_on_progress)
    await analysis_manager.complete_task(task_id, result)
    result["task_id"] = task_id
    return result


async def _run_assembly_legacy(request: AssemblyAnalysisRequest, analyses: list[str]):
    """Legacy path using FEAService (use_gmsh=False)."""
    from web.services.fea_service import FEAService
    svc = FEAService()
    components = [c.model_dump() for c in request.components]
    result = await asyncio.to_thread(
        svc.run_assembly_analysis,
        components=components,
        coupling_method=request.coupling_method,
        penalty_factor=request.penalty_factor,
        analyses=analyses,
        frequency_hz=request.frequency_hz,
        n_modes=request.n_modes,
        damping_ratio=request.damping_ratio,
    )
    return result


@router.post("/analyze", response_model=AssemblyAnalysisResponse)
async def analyze_assembly(request: AssemblyAnalysisRequest):
    """Run full assembly analysis pipeline.

    Uses subprocess isolation with timeout and real-time progress.
    """
    if not request.use_gmsh:
        try:
            result = await _run_assembly_legacy(request, request.analyses)
            return AssemblyAnalysisResponse(**result)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.error("Assembly legacy error: %s", exc, exc_info=True)
            raise HTTPException(500, f"Assembly analysis failed: {exc}") from exc

    try:
        result = await _run_assembly_subprocess(request, request.analyses)
        return AssemblyAnalysisResponse(**result)
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except asyncio.CancelledError:
        raise HTTPException(499, "Assembly analysis cancelled")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Assembly analysis error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(500, f"Assembly analysis failed: {exc}") from exc


@router.post("/modal", response_model=AssemblyAnalysisResponse)
async def assembly_modal(request: AssemblyAnalysisRequest):
    """Run modal analysis only on assembly.

    Uses subprocess isolation with timeout and real-time progress.
    """
    if not request.use_gmsh:
        try:
            result = await _run_assembly_legacy(request, ["modal"])
            return AssemblyAnalysisResponse(**result)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            logger.error("Assembly modal legacy error: %s", exc, exc_info=True)
            raise HTTPException(500, f"Assembly modal failed: {exc}") from exc

    try:
        result = await _run_assembly_subprocess(request, ["modal"])
        return AssemblyAnalysisResponse(**result)
    except TimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except asyncio.CancelledError:
        raise HTTPException(499, "Assembly analysis cancelled")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Assembly modal error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(500, f"Assembly modal analysis failed: {exc}") from exc


@router.get("/materials", response_model=list[MaterialListItem])
async def list_materials():
    """List available materials for assembly components."""
    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
        FEA_MATERIALS,
    )

    result = []
    for name, props in FEA_MATERIALS.items():
        result.append(
            MaterialListItem(
                name=name,
                E_gpa=round(props["E_pa"] / 1e9, 1),
                density_kg_m3=props["rho_kg_m3"],
                poisson_ratio=props["nu"],
                acoustic_velocity_m_s=props.get("acoustic_velocity_m_s"),
            )
        )
    return result


@router.get("/profiles", response_model=list[BoosterProfileItem])
async def list_booster_profiles():
    """List available booster profile types."""
    profiles = [
        BoosterProfileItem(
            profile="uniform",
            description="Constant diameter cylinder, gain = 1.0",
        ),
        BoosterProfileItem(
            profile="stepped",
            description="Two cylinders joined at a step, gain = (D_in/D_out)^2",
        ),
        BoosterProfileItem(
            profile="exponential",
            description="Smooth exponential taper, gain = D_in/D_out",
        ),
        BoosterProfileItem(
            profile="catenoidal",
            description="Smooth catenoidal taper (minimum stress), gain = D_in/D_out",
        ),
    ]
    return profiles
