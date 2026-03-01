"""Assembly analysis endpoints for multi-body ultrasonic welding stacks."""
from __future__ import annotations

import asyncio
import logging
import signal
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _run_in_thread(func, *args, **kwargs):
    """Wrapper that patches signal.signal for Gmsh thread-safety."""
    orig = signal.signal
    signal.signal = lambda *a, **kw: signal.SIG_DFL
    try:
        return func(*args, **kwargs)
    finally:
        signal.signal = orig

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


@router.post("/analyze", response_model=AssemblyAnalysisResponse)
async def analyze_assembly(request: AssemblyAnalysisRequest):
    """Run full assembly analysis pipeline.

    Generates meshes for each component, couples them into a global
    assembly, and runs the requested analyses (modal, harmonic).
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.gpu_backend import get_analysis_semaphore
        from web.services.fea_service import FEAService

        svc = FEAService()
        async with get_analysis_semaphore():
            result = await asyncio.to_thread(
                _run_in_thread,
                svc.run_assembly_analysis,
                components=[c.model_dump() for c in request.components],
                coupling_method=request.coupling_method,
                penalty_factor=request.penalty_factor,
                analyses=request.analyses,
                frequency_hz=request.frequency_hz,
                n_modes=request.n_modes,
                damping_ratio=request.damping_ratio,
                use_gmsh=request.use_gmsh,
            )
        return AssemblyAnalysisResponse(**result)
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

    Convenience endpoint that forces analyses to ``["modal"]`` only.
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.gpu_backend import get_analysis_semaphore
        from web.services.fea_service import FEAService

        svc = FEAService()
        async with get_analysis_semaphore():
            result = await asyncio.to_thread(
                _run_in_thread,
                svc.run_assembly_analysis,
                components=[c.model_dump() for c in request.components],
                coupling_method=request.coupling_method,
                penalty_factor=request.penalty_factor,
                analyses=["modal"],
                frequency_hz=request.frequency_hz,
                n_modes=request.n_modes,
                damping_ratio=request.damping_ratio,
                use_gmsh=request.use_gmsh,
            )
        return AssemblyAnalysisResponse(**result)
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
