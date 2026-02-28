"""Knurl FEA endpoints for geometry generation, analysis, comparison, and STEP export.

Ties together knurl geometry generation (horn_generator), adaptive meshing
(GmshMesher with knurl refinement), and FEA analysis (SolverA) to provide
end-to-end knurl horn analysis with Three.js-compatible mesh output.

Includes STEP file export via :class:`StepExportService` and Pareto-optimal
knurl optimization via :class:`KnurlFEAOptimizer`.
"""
from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knurl-fea", tags=["knurl-fea"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class KnurlParams(BaseModel):
    """Knurl pattern parameters."""

    type: str = "linear"  # linear | cross_hatch | diamond
    pitch_mm: float = Field(default=1.0, gt=0)
    depth_mm: float = Field(default=0.3, gt=0)
    tooth_width_mm: float = Field(default=0.5, gt=0)


class HornDimensions(BaseModel):
    """Parametric horn dimensions."""

    horn_type: str = "cylindrical"  # cylindrical | flat
    width_mm: float = Field(default=25.0, gt=0)
    height_mm: float = Field(default=80.0, gt=0)
    length_mm: float = Field(default=25.0, gt=0)


class KnurlFEAGenerateRequest(BaseModel):
    """Request for knurl geometry preview generation."""

    horn: HornDimensions = Field(default_factory=HornDimensions)
    knurl: KnurlParams = Field(default_factory=KnurlParams)
    step_file_path: Optional[str] = None  # Alternative: use STEP file
    mesh_density: str = "medium"


class MeshPreview(BaseModel):
    """Mesh data for Three.js visualization."""

    vertices: list[list[float]]  # [[x, y, z], ...]
    faces: list[list[int]]  # [[a, b, c], ...]
    node_count: int
    element_count: int


class KnurlFEAGenerateResponse(BaseModel):
    """Response from knurl geometry preview generation."""

    mesh: MeshPreview
    knurl_info: dict
    horn_type: str
    mesh_stats: dict


class KnurlFEAAnalyzeRequest(BaseModel):
    """Request for full FEA analysis on knurl horn."""

    horn: HornDimensions = Field(default_factory=HornDimensions)
    knurl: KnurlParams = Field(default_factory=KnurlParams)
    step_file_path: Optional[str] = None
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(default=20.0, gt=0)
    mesh_density: str = "medium"
    n_modes: int = Field(default=20, ge=1)
    task_id: Optional[str] = None


class ModeResult(BaseModel):
    """Single modal analysis result."""

    frequency_hz: float
    mode_type: str = "unknown"
    participation_factor: float = 0.0
    effective_mass_ratio: float = 0.0


class KnurlFEAAnalyzeResponse(BaseModel):
    """Response from full FEA analysis on knurl horn."""

    task_id: Optional[str] = None
    mode_shapes: list[ModeResult] = []
    closest_mode_hz: float = 0.0
    target_frequency_hz: float = 0.0
    frequency_deviation_percent: float = 0.0
    stress_max_mpa: Optional[float] = None
    amplitude_uniformity: Optional[float] = None
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0
    mesh: Optional[MeshPreview] = None
    knurl_info: dict = {}


class KnurlFEACompareRequest(BaseModel):
    """Request for with/without knurl comparison."""

    horn: HornDimensions = Field(default_factory=HornDimensions)
    knurl: KnurlParams = Field(default_factory=KnurlParams)
    step_file_path: Optional[str] = None
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(default=20.0, gt=0)
    mesh_density: str = "medium"
    n_modes: int = Field(default=20, ge=1)
    task_id: Optional[str] = None


class ComparisonResult(BaseModel):
    """Side-by-side modal/amplitude comparison result."""

    closest_mode_hz: float = 0.0
    frequency_deviation_percent: float = 0.0
    node_count: int = 0
    element_count: int = 0
    mode_shapes: list[ModeResult] = []
    solve_time_s: float = 0.0


class KnurlFEACompareResponse(BaseModel):
    """Response from with/without knurl comparison."""

    task_id: Optional[str] = None
    target_frequency_hz: float = 0.0
    with_knurl: ComparisonResult = Field(default_factory=ComparisonResult)
    without_knurl: ComparisonResult = Field(default_factory=ComparisonResult)
    frequency_shift_hz: float = 0.0
    frequency_shift_percent: float = 0.0
    knurl_info: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_knurl_dict(knurl: KnurlParams) -> dict:
    """Convert KnurlParams to the dict format used by mesher/horn_generator."""
    return {
        "type": knurl.type,
        "pitch_mm": knurl.pitch_mm,
        "depth_mm": knurl.depth_mm,
        "tooth_width_mm": knurl.tooth_width_mm,
    }


def _build_dimensions(horn: HornDimensions) -> dict:
    """Convert HornDimensions to the dict format used by GmshMesher."""
    if horn.horn_type == "cylindrical":
        return {
            "diameter_mm": horn.width_mm,
            "length_mm": horn.height_mm,
        }
    else:
        return {
            "width_mm": horn.width_mm,
            "depth_mm": horn.length_mm,
            "length_mm": horn.height_mm,
        }


def _mesh_to_preview(mesh) -> MeshPreview:
    """Convert an FEAMesh to a MeshPreview for Three.js rendering."""
    vertices = (mesh.nodes * 1000.0).tolist()  # meters -> mm for display
    faces = mesh.surface_tris.tolist()
    return MeshPreview(
        vertices=vertices,
        faces=faces,
        node_count=mesh.nodes.shape[0],
        element_count=mesh.elements.shape[0],
    )


def _generate_knurl_mesh(
    horn: HornDimensions,
    knurl: KnurlParams,
    mesh_density: str = "medium",
    step_file_path: str | None = None,
):
    """Generate a mesh with knurl-aware refinement.

    This is the shared core logic used by both generate and analyze endpoints.

    Returns
    -------
    FEAMesh
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import (
            GmshMesher,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Gmsh is required for knurl FEA but is not installed."
        ) from exc

    mesher = GmshMesher()
    knurl_dict = _build_knurl_dict(knurl)

    if step_file_path:
        return mesher.mesh_from_step(
            step_path=step_file_path,
            mesh_density=mesh_density,
            knurl_info=knurl_dict,
        )
    else:
        dimensions = _build_dimensions(horn)
        return mesher.mesh_parametric_horn(
            horn_type=horn.horn_type,
            dimensions=dimensions,
            mesh_density=mesh_density,
            knurl_info=knurl_dict,
        )


def _run_modal_analysis(mesh, material: str, frequency_khz: float, n_modes: int = 20):
    """Run modal analysis on a mesh.

    Returns
    -------
    dict with modal results
    """
    import time

    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
            ModalConfig,
        )
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import (
            SolverA,
        )
    except ImportError as exc:
        raise RuntimeError(
            "FEA solver dependencies are not installed."
        ) from exc

    target_hz = frequency_khz * 1000.0
    config = ModalConfig(
        mesh=mesh,
        material_name=material,
        n_modes=n_modes,
        target_frequency_hz=target_hz,
    )

    solver = SolverA()
    t0 = time.perf_counter()
    result = solver.modal_analysis(config)
    solve_time = time.perf_counter() - t0

    return {**result, "solve_time_s": solve_time}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=KnurlFEAGenerateResponse)
async def generate_knurl_preview(request: KnurlFEAGenerateRequest):
    """Generate knurl geometry preview mesh for Three.js display.

    Returns mesh vertices and faces suitable for 3D visualization.
    Uses knurl-aware mesh refinement for accurate representation
    of knurl features.
    """
    try:
        mesh = await asyncio.to_thread(
            _generate_knurl_mesh,
            horn=request.horn,
            knurl=request.knurl,
            mesh_density=request.mesh_density,
            step_file_path=request.step_file_path,
        )

        preview = _mesh_to_preview(mesh)
        knurl_dict = _build_knurl_dict(request.knurl)

        return KnurlFEAGenerateResponse(
            mesh=preview,
            knurl_info=knurl_dict,
            horn_type=request.horn.horn_type,
            mesh_stats=mesh.mesh_stats,
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Knurl FEA generate error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            500, f"Knurl geometry generation failed: {exc}"
        ) from exc


@router.post("/analyze", response_model=KnurlFEAAnalyzeResponse)
async def analyze_knurl_horn(request: KnurlFEAAnalyzeRequest):
    """Run full FEA analysis on a knurl horn.

    Performs knurl-aware meshing followed by modal analysis to find
    resonant frequencies and mode shapes. Returns modal results plus
    mesh preview for visualization.
    """
    try:

        def _run():
            mesh = _generate_knurl_mesh(
                horn=request.horn,
                knurl=request.knurl,
                mesh_density=request.mesh_density,
                step_file_path=request.step_file_path,
            )
            modal_result = _run_modal_analysis(
                mesh=mesh,
                material=request.material,
                frequency_khz=request.frequency_khz,
                n_modes=request.n_modes,
            )
            return mesh, modal_result

        mesh, modal_result = await asyncio.to_thread(_run)

        # Extract mode shapes
        target_hz = request.frequency_khz * 1000.0
        modes = []
        freqs = modal_result.get("frequencies_hz", [])
        mode_types = modal_result.get("mode_types", [])
        participation = modal_result.get("participation_factors", [])
        mass_ratios = modal_result.get("effective_mass_ratios", [])

        for i, f in enumerate(freqs):
            modes.append(ModeResult(
                frequency_hz=f,
                mode_type=mode_types[i] if i < len(mode_types) else "unknown",
                participation_factor=(
                    participation[i] if i < len(participation) else 0.0
                ),
                effective_mass_ratio=(
                    mass_ratios[i] if i < len(mass_ratios) else 0.0
                ),
            ))

        # Find closest mode to target
        closest_hz = 0.0
        if freqs:
            closest_hz = min(freqs, key=lambda f: abs(f - target_hz))

        deviation_pct = 0.0
        if target_hz > 0 and closest_hz > 0:
            deviation_pct = abs(closest_hz - target_hz) / target_hz * 100.0

        preview = _mesh_to_preview(mesh)
        knurl_dict = _build_knurl_dict(request.knurl)

        return KnurlFEAAnalyzeResponse(
            task_id=request.task_id,
            mode_shapes=modes,
            closest_mode_hz=closest_hz,
            target_frequency_hz=target_hz,
            frequency_deviation_percent=deviation_pct,
            stress_max_mpa=modal_result.get("stress_max_mpa"),
            amplitude_uniformity=modal_result.get("amplitude_uniformity"),
            node_count=mesh.nodes.shape[0],
            element_count=mesh.elements.shape[0],
            solve_time_s=modal_result.get("solve_time_s", 0.0),
            mesh=preview,
            knurl_info=knurl_dict,
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Knurl FEA analyze error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            500, f"Knurl FEA analysis failed: {exc}"
        ) from exc


@router.post("/compare", response_model=KnurlFEACompareResponse)
async def compare_knurl_horn(request: KnurlFEACompareRequest):
    """Compare FEA results with and without knurl.

    Runs modal analysis twice -- once with knurl-aware mesh refinement
    and once without -- then returns side-by-side comparison of modal
    frequencies and amplitude characteristics.
    """
    try:

        def _run_comparison():
            try:
                from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import (
                    GmshMesher,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Gmsh is required for knurl FEA but is not installed."
                ) from exc

            mesher = GmshMesher()
            dimensions = _build_dimensions(request.horn)
            knurl_dict = _build_knurl_dict(request.knurl)

            # Mesh WITH knurl
            if request.step_file_path:
                mesh_with = mesher.mesh_from_step(
                    step_path=request.step_file_path,
                    mesh_density=request.mesh_density,
                    knurl_info=knurl_dict,
                )
            else:
                mesh_with = mesher.mesh_parametric_horn(
                    horn_type=request.horn.horn_type,
                    dimensions=dimensions,
                    mesh_density=request.mesh_density,
                    knurl_info=knurl_dict,
                )

            # Mesh WITHOUT knurl
            if request.step_file_path:
                mesh_without = mesher.mesh_from_step(
                    step_path=request.step_file_path,
                    mesh_density=request.mesh_density,
                    knurl_info=None,
                )
            else:
                mesh_without = mesher.mesh_parametric_horn(
                    horn_type=request.horn.horn_type,
                    dimensions=dimensions,
                    mesh_density=request.mesh_density,
                    knurl_info=None,
                )

            # Run modal on both
            result_with = _run_modal_analysis(
                mesh=mesh_with,
                material=request.material,
                frequency_khz=request.frequency_khz,
                n_modes=request.n_modes,
            )
            result_without = _run_modal_analysis(
                mesh=mesh_without,
                material=request.material,
                frequency_khz=request.frequency_khz,
                n_modes=request.n_modes,
            )

            return mesh_with, mesh_without, result_with, result_without

        (
            mesh_with,
            mesh_without,
            result_with,
            result_without,
        ) = await asyncio.to_thread(_run_comparison)

        target_hz = request.frequency_khz * 1000.0

        def _build_comparison(modal_result, mesh) -> ComparisonResult:
            freqs = modal_result.get("frequencies_hz", [])
            mode_types = modal_result.get("mode_types", [])
            participation = modal_result.get("participation_factors", [])
            mass_ratios = modal_result.get("effective_mass_ratios", [])

            modes = []
            for i, f in enumerate(freqs):
                modes.append(ModeResult(
                    frequency_hz=f,
                    mode_type=(
                        mode_types[i] if i < len(mode_types) else "unknown"
                    ),
                    participation_factor=(
                        participation[i] if i < len(participation) else 0.0
                    ),
                    effective_mass_ratio=(
                        mass_ratios[i] if i < len(mass_ratios) else 0.0
                    ),
                ))

            closest_hz = 0.0
            if freqs:
                closest_hz = min(freqs, key=lambda f: abs(f - target_hz))

            deviation_pct = 0.0
            if target_hz > 0 and closest_hz > 0:
                deviation_pct = (
                    abs(closest_hz - target_hz) / target_hz * 100.0
                )

            return ComparisonResult(
                closest_mode_hz=closest_hz,
                frequency_deviation_percent=deviation_pct,
                node_count=mesh.nodes.shape[0],
                element_count=mesh.elements.shape[0],
                mode_shapes=modes,
                solve_time_s=modal_result.get("solve_time_s", 0.0),
            )

        comp_with = _build_comparison(result_with, mesh_with)
        comp_without = _build_comparison(result_without, mesh_without)

        freq_shift = comp_with.closest_mode_hz - comp_without.closest_mode_hz
        freq_shift_pct = 0.0
        if comp_without.closest_mode_hz > 0:
            freq_shift_pct = (
                freq_shift / comp_without.closest_mode_hz * 100.0
            )

        return KnurlFEACompareResponse(
            task_id=request.task_id,
            target_frequency_hz=target_hz,
            with_knurl=comp_with,
            without_knurl=comp_without,
            frequency_shift_hz=freq_shift,
            frequency_shift_percent=freq_shift_pct,
            knurl_info=_build_knurl_dict(request.knurl),
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "Knurl FEA compare error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            500, f"Knurl FEA comparison failed: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# STEP Export models & endpoint
# ---------------------------------------------------------------------------


class StepExportRequest(BaseModel):
    """Request for STEP file export of a knurl horn geometry."""

    horn: HornDimensions = Field(default_factory=HornDimensions)
    knurl: KnurlParams = Field(default_factory=KnurlParams)
    step_file_path: Optional[str] = None
    filename: Optional[str] = None  # Custom output filename


class StepExportResponse(BaseModel):
    """Response with download URL for the exported STEP file."""

    filename: str
    download_url: str
    file_size_bytes: int


@router.post("/export-step", response_model=StepExportResponse)
async def export_knurl_step(request: StepExportRequest):
    """Export knurl horn geometry as a STEP file for download.

    Generates the knurl geometry using the horn_generator and exports
    the resulting CadQuery solid to a STEP file via StepExportService.
    Returns a download URL that can be used to retrieve the file.
    """
    try:

        def _generate_and_export():
            from web.services.step_export_service import StepExportService

            try:
                from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
                    HornGenerator,
                    HornParams,
                    KnurlParams as HGKnurlParams,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "CadQuery/horn_generator is required for STEP export "
                    "but is not installed."
                ) from exc

            knurl_dict = _build_knurl_dict(request.knurl)

            if request.step_file_path:
                # Apply knurl to existing STEP file
                generator = HornGenerator()
                knurl_p = HGKnurlParams(
                    knurl_type=knurl_dict["type"],
                    pitch_mm=knurl_dict["pitch_mm"],
                    depth_mm=knurl_dict["depth_mm"],
                    tooth_width_mm=knurl_dict.get("tooth_width_mm", 0.5),
                )
                solid = generator.apply_knurl_to_step(
                    request.step_file_path, knurl_p
                )
            else:
                # Generate parametric horn with knurl
                params = HornParams(
                    horn_type=request.horn.horn_type,
                    width_mm=request.horn.width_mm,
                    height_mm=request.horn.height_mm,
                    length_mm=request.horn.length_mm,
                    knurl_type=knurl_dict["type"],
                    knurl_pitch_mm=knurl_dict["pitch_mm"],
                    knurl_depth_mm=knurl_dict["depth_mm"],
                    knurl_tooth_width_mm=knurl_dict.get(
                        "tooth_width_mm", 0.5
                    ),
                )
                result = HornGenerator().generate(params)
                if not result.has_cad_export:
                    raise RuntimeError(
                        "CadQuery is required for STEP export but is not "
                        "available. The horn was generated with the numpy "
                        "fallback which does not produce STEP output."
                    )
                # Re-generate using CadQuery directly to get the solid
                generator = HornGenerator()
                body = generator._cq_create_body(params)
                if params.knurl_type != "none":
                    body = generator._cq_apply_knurl(body, params)
                solid = body

            # Generate a unique filename
            export_name = request.filename or f"knurl_horn_{uuid.uuid4().hex[:8]}"

            service = StepExportService()
            file_path = service.export(solid, export_name)
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            actual_filename = os.path.basename(file_path)

            return actual_filename, file_size

        import os

        filename, file_size = await asyncio.to_thread(_generate_and_export)
        download_url = f"/api/v1/knurl-fea/download-step/{filename}"

        return StepExportResponse(
            filename=filename,
            download_url=download_url,
            file_size_bytes=file_size,
        )
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        logger.error(
            "STEP export error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            500, f"STEP export failed: {exc}"
        ) from exc


@router.get("/download-step/{filename}")
async def download_step_file(filename: str):
    """Download a previously exported STEP file.

    The filename is returned by the ``/export-step`` endpoint.
    """
    from web.services.step_export_service import StepExportService

    service = StepExportService()
    file_path = service.get_file(filename)

    if file_path is None:
        raise HTTPException(
            404, f"STEP file not found: {filename}. It may have expired."
        )

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/step",
    )


# ---------------------------------------------------------------------------
# Knurl FEA Optimization models & endpoint
# ---------------------------------------------------------------------------


class KnurlFEAOptimizeRequest(BaseModel):
    """Request for Pareto-optimal knurl optimization via FEA."""

    horn: HornDimensions = Field(default_factory=HornDimensions)
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(default=20.0, gt=0)
    n_candidates: int = Field(default=10, ge=1, le=50)
    task_id: Optional[str] = None


class OptimizationCandidate(BaseModel):
    """A single optimization candidate result."""

    knurl_type: str
    pitch_mm: float
    depth_mm: float
    tooth_width_mm: float
    closest_mode_hz: float
    frequency_deviation_percent: float
    amplitude_uniformity: float
    node_count: int
    element_count: int
    solve_time_s: float
    mode_count: int
    analytical_score: float
    fea_score: float


class OptimizationSummary(BaseModel):
    """Summary statistics for the optimization run."""

    total_grid_size: int
    candidates_evaluated: int
    pareto_front_size: int
    total_time_s: float
    target_frequency_khz: float
    material: str


class KnurlFEAOptimizeResponse(BaseModel):
    """Response from the knurl FEA optimization endpoint."""

    task_id: Optional[str] = None
    candidates: list[OptimizationCandidate] = []
    pareto_front: list[OptimizationCandidate] = []
    best_frequency_match: Optional[OptimizationCandidate] = None
    best_uniformity: Optional[OptimizationCandidate] = None
    summary: OptimizationSummary


@router.post("/optimize", response_model=KnurlFEAOptimizeResponse)
async def optimize_knurl_fea(request: KnurlFEAOptimizeRequest):
    """Find Pareto-optimal knurl configurations via FEA.

    Generates a parameter grid, pre-screens with analytical scoring,
    then runs full FEA on the top candidates to find configurations
    that optimally balance frequency match and amplitude uniformity.
    """
    try:
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()

        horn_config = {
            "horn_type": request.horn.horn_type,
            "width_mm": request.horn.width_mm,
            "height_mm": request.horn.height_mm,
            "length_mm": request.horn.length_mm,
        }

        result = await optimizer.optimize(
            horn_config=horn_config,
            material=request.material,
            target_freq_khz=request.frequency_khz,
            n_candidates=request.n_candidates,
        )

        return KnurlFEAOptimizeResponse(
            task_id=request.task_id,
            candidates=[
                OptimizationCandidate(**c) for c in result["candidates"]
            ],
            pareto_front=[
                OptimizationCandidate(**c) for c in result["pareto_front"]
            ],
            best_frequency_match=(
                OptimizationCandidate(**result["best_frequency_match"])
                if result.get("best_frequency_match")
                else None
            ),
            best_uniformity=(
                OptimizationCandidate(**result["best_uniformity"])
                if result.get("best_uniformity")
                else None
            ),
            summary=OptimizationSummary(**result["summary"]),
        )
    except Exception as exc:
        logger.error(
            "Knurl FEA optimize error: %s\n%s", exc, traceback.format_exc()
        )
        raise HTTPException(
            500, f"Knurl FEA optimization failed: {exc}"
        ) from exc
