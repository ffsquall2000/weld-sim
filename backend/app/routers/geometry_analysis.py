"""V2 API endpoints for geometry analysis -- CAD/PDF upload and FEA modal analysis."""
from __future__ import annotations

import logging
import os
import tempfile
import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.geometry_analysis import (
    CADAnalysisResponse,
    FEAMaterialResponse,
    FEARunRequest,
    FEARunResponse,
    PDFAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geometry", tags=["geometry-analysis"])

# ---------------------------------------------------------------------------
# Lazy-loaded service singletons
# ---------------------------------------------------------------------------

try:
    from web.services.geometry_service import GeometryService

    _geometry_service: GeometryService | None = GeometryService()
except ImportError:
    _geometry_service = None

try:
    from web.services.fea_service import FEAService

    _fea_service: FEAService | None = FEAService()
except ImportError:
    _fea_service = None


# ---------------------------------------------------------------------------
# CAD Upload
# ---------------------------------------------------------------------------


@router.post("/upload/cad", response_model=CADAnalysisResponse)
async def upload_and_analyze_cad(
    file: UploadFile = File(..., description="STEP/STP CAD file to analyze"),
):
    """Upload a STEP file and analyze horn geometry.

    Accepts `.step`, `.stp`, `.x_t`, `.x_b` files.
    Returns classified horn type, dimensions, gain estimate, and a
    simplified visualization mesh.
    """
    if _geometry_service is None:
        raise HTTPException(
            status_code=503,
            detail="Geometry analysis service is not available (missing dependencies)",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".step", ".stp", ".x_t", ".x_b"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: .step, .stp, .x_t, .x_b",
        )

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _geometry_service.analyze_step_file(tmp_path)
            return CADAnalysisResponse(**result)
        finally:
            os.unlink(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("CAD analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"CAD analysis failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# PDF Upload
# ---------------------------------------------------------------------------


@router.post("/upload/pdf", response_model=PDFAnalysisResponse)
async def upload_and_analyze_pdf(
    file: UploadFile = File(..., description="PDF engineering drawing to analyze"),
):
    """Upload a PDF drawing and extract dimensions, tolerances, and notes.

    Returns detected dimensions (linear, diameter, radius), tolerance
    specifications, and engineering notes extracted from the document.
    """
    if _geometry_service is None:
        raise HTTPException(
            status_code=503,
            detail="Geometry analysis service is not available (missing dependencies)",
        )

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(
            status_code=400, detail=f"Only PDF files supported, got: {ext}"
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = _geometry_service.analyze_pdf(tmp_path)
            return PDFAnalysisResponse(**result)
        finally:
            os.unlink(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("PDF analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"PDF analysis failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# FEA Run
# ---------------------------------------------------------------------------


@router.post("/fea/run", response_model=FEARunResponse)
async def run_fea_analysis(request: FEARunRequest):
    """Run FEA modal analysis on the specified horn geometry.

    When ``geometry_id`` is provided, the endpoint loads the stored geometry
    dimensions from the database (metadata_json or parametric_params) and uses
    them instead of the default parametric parameters.

    When ``use_gmsh`` is True (default), the high-accuracy Gmsh TET10 +
    SolverA pipeline is used.  Set to False for the legacy HEX8 solver.
    """
    if _fea_service is None:
        raise HTTPException(
            status_code=503,
            detail="FEA service is not available (missing dependencies)",
        )

    try:
        # Determine horn parameters -- from geometry_id or from request
        horn_type = request.horn_type
        width_mm = request.width_mm
        height_mm = request.height_mm
        length_mm = request.length_mm

        if request.geometry_id:
            # Try to load dimensions from the uploaded geometry record
            from backend.app.dependencies import async_session_factory
            from backend.app.services.geometry_service import GeometryService
            import uuid as _uuid

            async with async_session_factory() as session:
                svc = GeometryService(session)
                geom = await svc.get(_uuid.UUID(request.geometry_id))
                if not geom:
                    raise HTTPException(
                        status_code=404, detail="Geometry not found"
                    )

                # 1) Prefer dimensions stored in metadata_json (set by CAD analysis)
                meta = geom.metadata_json or {}
                dims = meta.get("dimensions", {})
                if dims:
                    horn_type = dims.get("horn_type", horn_type)
                    width_mm = dims.get("width_mm", width_mm)
                    height_mm = dims.get("height_mm", height_mm)
                    length_mm = dims.get("length_mm", length_mm)
                    logger.info(
                        "FEA using geometry %s metadata dimensions: %s %s x %s x %s mm",
                        request.geometry_id,
                        horn_type,
                        width_mm,
                        height_mm,
                        length_mm,
                    )

                # 2) Fallback: check parametric_params on the geometry record
                if not dims and geom.parametric_params:
                    pp = geom.parametric_params
                    horn_type = pp.get("horn_type", horn_type)
                    width_mm = pp.get("width_mm", width_mm)
                    height_mm = pp.get("height_mm", height_mm)
                    length_mm = pp.get("length_mm", length_mm)
                    logger.info(
                        "FEA using geometry %s parametric_params: %s %s x %s x %s mm",
                        request.geometry_id,
                        horn_type,
                        width_mm,
                        height_mm,
                        length_mm,
                    )

        if request.use_gmsh:
            result = _fea_service.run_modal_analysis_gmsh(
                horn_type=horn_type,
                diameter_mm=width_mm,
                length_mm=height_mm,
                material=request.material,
                frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
        else:
            result = _fea_service.run_modal_analysis(
                horn_type=horn_type,
                width_mm=width_mm,
                height_mm=height_mm,
                length_mm=length_mm,
                material=request.material,
                frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
        return FEARunResponse(**result)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("FEA analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"FEA analysis failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# FEA Materials
# ---------------------------------------------------------------------------


@router.get("/fea/materials", response_model=list[FEAMaterialResponse])
async def list_fea_materials():
    """List available FEA horn materials with their mechanical properties."""
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
            FEA_MATERIALS,
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="FEA material library is not available (missing dependencies)",
        )

    result: list[FEAMaterialResponse] = []
    for name, props in FEA_MATERIALS.items():
        result.append(
            FEAMaterialResponse(
                name=name,
                E_gpa=round(props["E_pa"] / 1e9, 1),
                density_kg_m3=props["rho_kg_m3"],
                poisson_ratio=props["nu"],
            )
        )
    return result
