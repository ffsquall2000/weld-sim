"""Geometry analysis and FEA simulation endpoints."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import traceback
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
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
    task_id: Optional[str] = None  # Client-generated UUID for early WebSocket connection


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


class HarmonicRequest(BaseModel):
    """Request for FEA harmonic response analysis."""

    horn_type: str = "cylindrical"
    width_mm: float = Field(gt=0, default=25.0)
    height_mm: float = Field(gt=0, default=80.0)
    length_mm: float = Field(gt=0, default=25.0)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(gt=0, default=20.0)
    mesh_density: str = "medium"  # coarse, medium, fine
    freq_range_percent: float = Field(gt=0, le=100, default=20.0)
    n_freq_points: int = Field(gt=0, default=201)
    damping_model: str = "hysteretic"  # hysteretic, rayleigh, modal
    damping_ratio: float = Field(gt=0, default=0.005)
    task_id: Optional[str] = None  # Client-generated UUID for early WebSocket connection


class HarmonicRunResponse(BaseModel):
    """Response from harmonic FEA analysis."""

    task_id: Optional[str] = None
    frequencies_hz: list[float] = []
    gain: float = 0.0
    q_factor: float = 0.0
    contact_face_uniformity: float = 0.0
    resonance_hz: float = 0.0
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0


class StressRequest(BaseModel):
    """Request for harmonic stress analysis (full chain: mesh -> modal -> harmonic -> stress)."""

    horn_type: str = "cylindrical"
    width_mm: float = Field(gt=0, default=25.0)
    height_mm: float = Field(gt=0, default=80.0)
    length_mm: float = Field(gt=0, default=25.0)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(gt=0, default=20.0)
    mesh_density: str = "medium"  # coarse, medium, fine
    freq_range_percent: float = Field(gt=0, le=100, default=20.0)
    n_freq_points: int = Field(gt=0, default=201)
    damping_model: str = "hysteretic"
    damping_ratio: float = Field(gt=0, default=0.005)
    task_id: Optional[str] = None


class StressResponse(BaseModel):
    """Response from harmonic stress analysis."""

    task_id: Optional[str] = None
    max_stress_mpa: float = 0.0
    safety_factor: float = 0.0
    max_displacement_mm: float = 0.0
    contact_face_uniformity: float = 0.0
    resonance_hz: float = 0.0
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0


class FatigueRequest(BaseModel):
    """Request for fatigue life assessment."""

    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(gt=0, default=20.0)
    mesh_density: str = "medium"
    surface_finish: str = "machined"  # polished, ground, machined, as_forged
    characteristic_diameter_mm: float = Field(gt=0, default=25.0)
    reliability_pct: float = Field(ge=50, le=99.9, default=90.0)
    temperature_c: float = 25.0
    Kt_global: float = Field(ge=1.0, default=1.5)
    # Geometry source (parametric or STEP)
    horn_type: str = "cylindrical"
    width_mm: float = Field(gt=0, default=25.0)
    height_mm: float = Field(gt=0, default=80.0)
    length_mm: float = Field(gt=0, default=25.0)
    # Damping for harmonic analysis
    damping_model: str = "hysteretic"
    damping_ratio: float = Field(gt=0, le=1, default=0.005)
    task_id: Optional[str] = None


class FatigueResponse(BaseModel):
    """Response from fatigue life assessment."""

    task_id: Optional[str] = None
    min_safety_factor: float = 0.0
    estimated_life_cycles: float = 0.0
    estimated_hours_at_20khz: float = 0.0  # cycles / (20000 * 3600)
    critical_locations: list[dict] = []
    sn_curve_name: str = ""
    corrected_endurance_mpa: float = 0.0
    max_stress_mpa: float = 0.0
    safety_factor_distribution: list[float] = []  # per-element
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0


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


class FEARunResponse(BaseModel):
    """Wrapper that adds task_id to FEA result for progress tracking."""
    task_id: Optional[str] = None
    mode_shapes: list[ModeShapeResponse] = []
    closest_mode_hz: float = 0
    target_frequency_hz: float = 0
    frequency_deviation_percent: float = 0
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0
    mesh: Optional[dict] = None
    stress_max_mpa: Optional[float] = None
    temperature_max_c: Optional[float] = None


async def _run_fea_subprocess(task_type: str, params: dict, client_task_id: Optional[str] = None):
    """Shared helper: run FEA in subprocess with progress & cancel."""
    from web.services.fea_process_runner import FEAProcessRunner
    from web.services.analysis_manager import analysis_manager

    if task_type == "modal_step":
        steps = ["init", "import_step", "meshing", "assembly", "solving", "classifying", "packaging"]
    elif task_type == "harmonic":
        steps = ["init", "meshing", "assembly", "solving", "packaging"]
    elif task_type == "harmonic_step":
        steps = ["init", "import_step", "meshing", "assembly", "solving", "packaging"]
    elif task_type == "stress":
        steps = ["init", "meshing", "assembly", "modal_solve", "harmonic_solve", "stress_compute", "packaging"]
    elif task_type == "fatigue":
        steps = ["init", "meshing", "assembly", "modal_solve", "harmonic_solve", "stress_compute", "fatigue_assess", "packaging"]
    else:
        steps = ["init", "meshing", "assembly", "solving", "classifying", "packaging"]
    task_id = analysis_manager.create_task(task_type, steps, task_id=client_task_id)

    runner = FEAProcessRunner()
    analysis_manager.set_cancel_hook(task_id, runner)

    async def _on_progress(phase: str, progress: float, message: str):
        step_map = {s: i for i, s in enumerate(steps)}
        step_idx = step_map.get(phase, 0)
        logger.info("FEA progress: task=%s phase=%s progress=%.3f msg=%s",
                     task_id, phase, progress, message)
        await analysis_manager.update_progress(task_id, step_idx, progress, message)

    try:
        result = await runner.run(task_type, params, on_progress=_on_progress)
        await analysis_manager.complete_task(task_id, result)
        result["task_id"] = task_id
        return result
    except TimeoutError as exc:
        await analysis_manager.fail_task(task_id, str(exc))
        raise HTTPException(504, str(exc)) from exc
    except asyncio.CancelledError:
        await analysis_manager.fail_task(task_id, "Cancelled by user")
        raise HTTPException(499, "FEA cancelled by user")
    except RuntimeError as exc:
        await analysis_manager.fail_task(task_id, str(exc))
        raise HTTPException(500, detail=f"FEA analysis failed: {exc}")


@router.post("/fea/run", response_model=FEARunResponse)
async def run_fea_analysis(request: FEARequest):
    """Run FEA modal analysis on specified geometry parameters.

    The computation runs in an isolated subprocess with a 5-minute timeout.
    Connect to ``/ws/analysis/{task_id}`` for real-time progress.
    """
    if not request.use_gmsh:
        # Legacy path (no subprocess isolation)
        try:
            from web.services.fea_service import FEAService
            fea_svc = FEAService()
            result = await asyncio.to_thread(
                fea_svc.run_modal_analysis,
                horn_type=request.horn_type, width_mm=request.width_mm,
                height_mm=request.height_mm, length_mm=request.length_mm,
                material=request.material, frequency_khz=request.frequency_khz,
                mesh_density=request.mesh_density,
            )
            return result
        except Exception as exc:
            logger.error("FEA legacy error: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(500, f"FEA analysis failed: {exc}") from exc

    params = {
        "horn_type": request.horn_type,
        "diameter_mm": request.width_mm,
        "width_mm": request.width_mm,
        "depth_mm": request.length_mm,
        "length_mm": request.height_mm,
        "material": request.material,
        "frequency_khz": request.frequency_khz,
        "mesh_density": request.mesh_density,
    }
    try:
        result = await _run_fea_subprocess("modal", params, client_task_id=request.task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"FEA analysis failed: {exc}") from exc


@router.post("/fea/run-step", response_model=FEARunResponse)
async def run_fea_on_step_file(
    file: UploadFile = File(...),
    material: str = Form("Titanium Ti-6Al-4V"),
    frequency_khz: float = Form(20.0),
    mesh_density: str = Form("medium"),
    task_id: Optional[str] = Form(None),
):
    """Run FEA modal analysis directly on an uploaded STEP file.

    The computation runs in an isolated subprocess with a 5-minute timeout.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".step", ".stp"):
        raise HTTPException(400, f"Only STEP files supported, got: {ext}")

    # Save uploaded file temporarily (subprocess needs a file path)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    params = {
        "step_file_path": tmp_path,
        "material": material,
        "frequency_khz": frequency_khz,
        "mesh_density": mesh_density,
    }
    try:
        result = await _run_fea_subprocess("modal_step", params, client_task_id=task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("FEA STEP error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"FEA analysis on STEP file failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/fea/run-harmonic", response_model=HarmonicRunResponse)
async def run_harmonic_analysis(request: HarmonicRequest):
    """Run FEA harmonic response analysis on specified geometry parameters.

    The computation runs in an isolated subprocess with a 5-minute timeout.
    Connect to ``/ws/analysis/{task_id}`` for real-time progress.
    """
    target_hz = request.frequency_khz * 1000.0
    half_range = target_hz * (request.freq_range_percent / 100.0)
    params = {
        "horn_type": request.horn_type,
        "diameter_mm": request.width_mm,
        "width_mm": request.width_mm,
        "depth_mm": request.length_mm,
        "length_mm": request.height_mm,
        "material": request.material,
        "frequency_khz": request.frequency_khz,
        "mesh_density": request.mesh_density,
        "freq_min_hz": target_hz - half_range,
        "freq_max_hz": target_hz + half_range,
        "n_freq_points": request.n_freq_points,
        "damping_model": request.damping_model,
        "damping_ratio": request.damping_ratio,
    }
    try:
        result = await _run_fea_subprocess("harmonic", params, client_task_id=request.task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Harmonic FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Harmonic analysis failed: {exc}") from exc


@router.post("/fea/run-harmonic-step", response_model=HarmonicRunResponse)
async def run_harmonic_on_step_file(
    file: UploadFile = File(...),
    material: str = Form("Titanium Ti-6Al-4V"),
    frequency_khz: float = Form(20.0),
    mesh_density: str = Form("medium"),
    freq_range_percent: float = Form(20.0),
    n_freq_points: int = Form(201),
    damping_model: str = Form("hysteretic"),
    damping_ratio: float = Form(0.005),
    task_id: Optional[str] = Form(None),
):
    """Run FEA harmonic response analysis on an uploaded STEP file.

    The computation runs in an isolated subprocess with a 5-minute timeout.
    Connect to ``/ws/analysis/{task_id}`` for real-time progress.
    """
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".step", ".stp"):
        raise HTTPException(400, f"Only STEP files supported, got: {ext}")

    # Save uploaded file temporarily (subprocess needs a file path)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    target_hz = frequency_khz * 1000.0
    half_range = target_hz * (freq_range_percent / 100.0)
    params = {
        "step_file_path": tmp_path,
        "material": material,
        "frequency_khz": frequency_khz,
        "mesh_density": mesh_density,
        "freq_min_hz": target_hz - half_range,
        "freq_max_hz": target_hz + half_range,
        "n_freq_points": n_freq_points,
        "damping_model": damping_model,
        "damping_ratio": damping_ratio,
    }
    try:
        result = await _run_fea_subprocess("harmonic_step", params, client_task_id=task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Harmonic STEP FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Harmonic analysis on STEP file failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/fea/run-stress", response_model=StressResponse)
async def run_stress_analysis(request: StressRequest):
    """Run harmonic stress analysis (full chain: mesh -> harmonic -> stress).

    The computation runs in an isolated subprocess with a 5-minute timeout.
    Connect to ``/ws/analysis/{task_id}`` for real-time progress.
    """
    target_hz = request.frequency_khz * 1000.0
    half_range = target_hz * (request.freq_range_percent / 100.0)
    params = {
        "horn_type": request.horn_type,
        "diameter_mm": request.width_mm,
        "width_mm": request.width_mm,
        "depth_mm": request.length_mm,
        "length_mm": request.height_mm,
        "material": request.material,
        "frequency_khz": request.frequency_khz,
        "mesh_density": request.mesh_density,
        "freq_min_hz": target_hz - half_range,
        "freq_max_hz": target_hz + half_range,
        "n_freq_points": request.n_freq_points,
        "damping_model": request.damping_model,
        "damping_ratio": request.damping_ratio,
    }
    try:
        result = await _run_fea_subprocess("stress", params, client_task_id=request.task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Stress FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Stress analysis failed: {exc}") from exc


@router.post("/fea/run-fatigue", response_model=FatigueResponse)
async def run_fatigue_analysis(request: FatigueRequest):
    """Run fatigue life assessment (full chain: mesh -> modal -> harmonic -> stress -> fatigue).

    The computation runs in an isolated subprocess with a 5-minute timeout.
    Connect to ``/ws/analysis/{task_id}`` for real-time progress.
    """
    target_hz = request.frequency_khz * 1000.0
    half_range = target_hz * 0.20  # 20% range for harmonic sweep
    params = {
        "horn_type": request.horn_type,
        "diameter_mm": request.width_mm,
        "width_mm": request.width_mm,
        "depth_mm": request.length_mm,
        "length_mm": request.height_mm,
        "material": request.material,
        "frequency_khz": request.frequency_khz,
        "mesh_density": request.mesh_density,
        "freq_min_hz": target_hz - half_range,
        "freq_max_hz": target_hz + half_range,
        "n_freq_points": 201,
        "damping_model": request.damping_model,
        "damping_ratio": request.damping_ratio,
        # Fatigue-specific params
        "surface_finish": request.surface_finish,
        "characteristic_diameter_mm": request.characteristic_diameter_mm,
        "reliability_pct": request.reliability_pct,
        "temperature_c": request.temperature_c,
        "Kt_global": request.Kt_global,
    }
    try:
        result = await _run_fea_subprocess("fatigue", params, client_task_id=request.task_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Fatigue FEA error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Fatigue analysis failed: {exc}") from exc


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
