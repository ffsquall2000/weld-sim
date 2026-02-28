"""Geometry analysis and FEA simulation endpoints."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import traceback
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/geometry", tags=["geometry"])


# --- Request/Response models ---


class GeometryAnalysisResponse(BaseModel):
    """Response from CAD geometry analysis."""

    horn_type: str
    dimensions: dict
    contact_dimensions: Optional[dict] = None  # {width_mm, length_mm} of welding tip
    gain_estimate: float
    confidence: float
    knurl: Optional[dict] = None
    bounding_box: list[float]
    volume_mm3: float
    surface_area_mm2: float
    mesh: Optional[dict] = None  # {vertices: [[x,y,z],...], faces: [[a,b,c],...]}


class FEARequest(BaseModel):
    """Request for FEA modal analysis."""

    horn_type: str = "cylindrical"
    width_mm: float = Field(gt=0, default=25.0)
    height_mm: float = Field(gt=0, default=80.0)
    length_mm: float = Field(gt=0, default=25.0)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(gt=0, default=20.0)
    mesh_density: str = "medium"  # coarse, medium, fine
    use_gmsh: bool = True  # Default: Gmsh TET10 + SolverA pipeline (set False for legacy HEX8)


class ModeShapeResponse(BaseModel):
    frequency_hz: float
    mode_type: str
    participation_factor: float
    effective_mass_ratio: float
    displacement_max: float


class FEAResponse(BaseModel):
    """Response from FEA analysis."""

    mode_shapes: list[ModeShapeResponse]
    closest_mode_hz: float
    target_frequency_hz: float
    frequency_deviation_percent: float
    node_count: int
    element_count: int
    solve_time_s: float
    mesh: Optional[dict] = None  # 3D mesh for visualization
    stress_max_mpa: Optional[float] = None
    temperature_max_c: Optional[float] = None


class PDFAnalysisResponse(BaseModel):
    """Response from PDF drawing analysis."""

    detected_dimensions: list[dict]
    tolerances: list[dict]
    notes: list[str]
    confidence: float
    page_count: int


class FEAMaterialResponse(BaseModel):
    name: str
    E_gpa: float
    density_kg_m3: float
    poisson_ratio: float


# --- Endpoints ---


@router.post("/upload/cad", response_model=GeometryAnalysisResponse)
async def upload_and_analyze_cad(
    file: UploadFile = File(...),
):
    """Upload a CAD file and analyze the horn geometry."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    supported_step = (".step", ".stp")
    supported_parasolid = (".x_t", ".x_b")
    all_supported = supported_step + supported_parasolid

    if ext not in all_supported:
        raise HTTPException(
            400,
            f"Unsupported file format: {ext}. "
            f"Supported: {', '.join(all_supported)}",
        )

    # Parasolid format: currently not supported for parsing
    if ext in supported_parasolid:
        raise HTTPException(
            400,
            f"Parasolid format ({ext}) is not yet supported for direct parsing. "
            f"Please convert to STEP format (.step / .stp) using your CAD software "
            f"(e.g. SolidWorks: File > Save As > .step, "
            f"NX/UG: File > Export > STEP).",
        )

    try:
        from web.services.geometry_service import GeometryService

        geo_svc = GeometryService()

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await asyncio.to_thread(geo_svc.analyze_step_file, tmp_path)
            return result
        finally:
            os.unlink(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("CAD analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Analysis failed: {exc}") from exc


@router.post("/upload/pdf", response_model=PDFAnalysisResponse)
async def upload_and_analyze_pdf(
    file: UploadFile = File(...),
):
    """Upload a PDF drawing and extract dimensions."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(400, f"Only PDF files supported, got: {ext}")

    try:
        from web.services.geometry_service import GeometryService

        geo_svc = GeometryService()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = await asyncio.to_thread(geo_svc.analyze_pdf, tmp_path)
            return result
        finally:
            os.unlink(tmp_path)

    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("PDF analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"PDF analysis failed: {exc}") from exc


@router.post("/fea/run", response_model=FEAResponse)
async def run_fea_analysis(request: FEARequest):
    """Run FEA modal analysis on specified geometry parameters."""
    try:
        from web.services.fea_service import FEAService

        fea_svc = FEAService()

        def _run():
            if request.use_gmsh:
                return fea_svc.run_modal_analysis_gmsh(
                    horn_type=request.horn_type,
                    diameter_mm=request.width_mm,
                    length_mm=request.height_mm,
                    material=request.material,
                    frequency_khz=request.frequency_khz,
                    mesh_density=request.mesh_density,
                )
            else:
                return fea_svc.run_modal_analysis(
                    horn_type=request.horn_type,
                    width_mm=request.width_mm,
                    height_mm=request.height_mm,
                    length_mm=request.length_mm,
                    material=request.material,
                    frequency_khz=request.frequency_khz,
                    mesh_density=request.mesh_density,
                )

        result = await asyncio.to_thread(_run)
        return result
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.error("FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"FEA analysis failed: {exc}") from exc


@router.get("/fea/materials", response_model=list[FEAMaterialResponse])
async def list_fea_materials():
    """List available FEA horn materials."""
    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
        FEA_MATERIALS,
    )

    result = []
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
