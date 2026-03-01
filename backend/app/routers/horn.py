"""V2 API endpoints for horn generation, chamfer analysis, and file download."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.schemas.horn import (
    ChamferAnalysisRequest,
    ChamferAnalysisResponse,
    HornGenerateRequest,
    HornGenerateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/horn", tags=["horn"])

# Try to import V1 horn service; degrade gracefully if unavailable.
try:
    from web.services.horn_service import HornService

    _horn_service = HornService()
except ImportError:
    _horn_service = None  # type: ignore[assignment]
    logger.info("web.services.horn_service not available; horn endpoints will return 503")


def _require_service() -> "HornService":
    """Return the horn service or raise 503 if not available."""
    if _horn_service is None:
        raise HTTPException(
            status_code=503,
            detail="Horn service is not available. Required dependencies may be missing.",
        )
    return _horn_service


@router.post("/generate", response_model=HornGenerateResponse)
async def generate_horn(request: HornGenerateRequest):
    """Generate a parametric horn with optional knurl pattern and chamfer.

    Returns horn geometry data including mesh for 3D preview and an optional
    download ID for STEP/STL export.
    """
    svc = _require_service()

    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
            HornParams,
        )

        params = HornParams(
            horn_type=request.horn_type,
            width_mm=request.width_mm,
            height_mm=request.height_mm,
            length_mm=request.length_mm,
            material=request.material,
            knurl_type=request.knurl_type,
            knurl_pitch_mm=request.knurl_pitch_mm,
            knurl_tooth_width_mm=request.knurl_tooth_width_mm,
            knurl_depth_mm=request.knurl_depth_mm,
            knurl_direction=request.knurl_direction,
            chamfer_radius_mm=request.chamfer_radius_mm,
            chamfer_angle_deg=request.chamfer_angle_deg,
            edge_treatment=request.edge_treatment or "none",
        )

        result, download_id = svc.generate_horn(params)

        return HornGenerateResponse(
            horn_type=result.horn_type,
            dimensions=result.dimensions,
            knurl_info=result.knurl_info,
            chamfer_info=result.chamfer_info,
            volume_mm3=round(result.volume_mm3, 2),
            surface_area_mm2=round(result.surface_area_mm2, 2),
            has_cad_export=result.has_cad_export,
            mesh=result.mesh,
            download_id=download_id,
        )
    except ImportError as exc:
        logger.error("Missing horn generator dependency: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Horn generator plugin is not installed.",
        ) from exc
    except Exception as exc:
        logger.error("Horn generation error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Horn generation failed: {exc}"
        ) from exc


@router.post("/chamfer-analysis", response_model=ChamferAnalysisResponse)
async def analyze_chamfer(request: ChamferAnalysisRequest):
    """Analyze the impact of chamfer/edge treatment on welding performance.

    Returns stress concentration, damage risk, area corrections, and
    recommendations for improving the edge treatment design.
    """
    svc = _require_service()

    try:
        result = svc.analyze_chamfer(
            contact_width_mm=request.contact_width_mm,
            contact_length_mm=request.contact_length_mm,
            chamfer_radius_mm=request.chamfer_radius_mm,
            chamfer_angle_deg=request.chamfer_angle_deg,
            edge_treatment=request.edge_treatment or "none",
            pressure_mpa=request.pressure_mpa,
            amplitude_um=request.amplitude_um,
            material_yield_mpa=request.material_yield_mpa,
            weld_time_s=request.weld_time_s,
        )
        return ChamferAnalysisResponse(**result)
    except Exception as exc:
        logger.error("Chamfer analysis error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Chamfer analysis failed: {exc}"
        ) from exc


@router.get("/download/{file_id}")
async def download_horn_file(file_id: str, fmt: str = "step"):
    """Download a generated horn CAD file in STEP or STL format.

    The file_id is returned by the ``POST /horn/generate`` endpoint when
    CAD export is available.
    """
    if fmt not in ("step", "stl"):
        raise HTTPException(
            status_code=400, detail="Format must be 'step' or 'stl'"
        )

    svc = _require_service()
    path = svc.get_download_path(file_id, fmt)

    if path is None:
        raise HTTPException(
            status_code=404, detail="File not found or expired"
        )

    media_type = "application/step" if fmt == "step" else "application/sla"
    filename = f"horn_{file_id}.{fmt}"
    return FileResponse(path, media_type=media_type, filename=filename)
