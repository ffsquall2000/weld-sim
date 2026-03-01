"""V2 assembly analysis API endpoints.

Wraps the V1 FEAService assembly analysis methods with Pydantic v2 schemas
and the V2 router pattern.  Also exposes material and booster-profile
reference data used by the frontend assembly builder.
"""
from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, HTTPException

from backend.app.schemas.assembly import (
    AssemblyAnalysisRequest,
    AssemblyAnalysisResponse,
    BoosterProfileItem,
    MaterialListItem,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assembly", tags=["assembly"])

# ---------------------------------------------------------------------------
# Lazy-init FEA service (tolerate missing V1 dependencies)
# ---------------------------------------------------------------------------

try:
    from web.services.fea_service import FEAService

    _fea_service = FEAService()
except ImportError:
    _fea_service = None

# ---------------------------------------------------------------------------
# Material database import (domain layer)
# ---------------------------------------------------------------------------

try:
    from backend.app.domain.material_properties import FEA_MATERIALS
except ImportError:
    FEA_MATERIALS = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=AssemblyAnalysisResponse)
async def analyze_assembly(request: AssemblyAnalysisRequest):
    """Run full assembly analysis pipeline.

    Generates meshes for each component, couples them into a global
    assembly, and runs the requested analyses (modal, harmonic).
    """
    if _fea_service is None:
        raise HTTPException(
            status_code=503,
            detail="FEA service unavailable (missing dependencies)",
        )

    try:
        result = _fea_service.run_assembly_analysis(
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Assembly analysis error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            status_code=500,
            detail=f"Assembly analysis failed: {exc}",
        ) from exc


@router.post("/modal", response_model=AssemblyAnalysisResponse)
async def assembly_modal(request: AssemblyAnalysisRequest):
    """Run modal analysis only on assembly.

    Convenience endpoint that forces analyses to ``["modal"]`` only,
    regardless of the ``analyses`` field in the request body.
    """
    if _fea_service is None:
        raise HTTPException(
            status_code=503,
            detail="FEA service unavailable (missing dependencies)",
        )

    try:
        result = _fea_service.run_assembly_analysis(
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
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Assembly modal error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            status_code=500,
            detail=f"Assembly modal analysis failed: {exc}",
        ) from exc


@router.get("/materials", response_model=list[MaterialListItem])
async def list_materials():
    """List available materials for assembly components.

    Returns material properties from the FEA material database,
    formatted for UI display.
    """
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
    """List available booster profile types.

    Returns the standard set of booster taper profiles supported
    by the assembly builder.
    """
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
