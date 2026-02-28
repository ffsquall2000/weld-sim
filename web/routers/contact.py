"""Contact analysis and thermal simulation API endpoints.

Provides endpoints for:
- Contact mechanics analysis (penalty / Nitsche formulations) via Docker FEniCSx
- Docker/FEniCSx availability check
- Anvil geometry preview for Three.js
- Thermal analysis (friction heating + heat conduction)
- Combined contact + thermal full-analysis pipeline
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contact", tags=["contact"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class HornGeometry(BaseModel):
    """Horn geometry parameters."""

    horn_type: str = "cylindrical"  # cylindrical | flat
    width_mm: float = Field(default=25.0, gt=0)
    height_mm: float = Field(default=80.0, gt=0)
    length_mm: float = Field(default=25.0, gt=0)


class WorkpieceParams(BaseModel):
    """Workpiece material and thickness."""

    material: str = "ABS"
    thickness_mm: float = Field(default=3.0, gt=0)


class AnvilParamsInput(BaseModel):
    """Anvil type and parameters."""

    anvil_type: str = "flat"  # flat | groove | knurled | contour
    width_mm: float = Field(default=50.0, gt=0)
    depth_mm: float = Field(default=30.0, gt=0)
    height_mm: float = Field(default=20.0, gt=0)
    # Groove params
    groove_width_mm: float = Field(default=5.0, gt=0)
    groove_depth_mm: float = Field(default=3.0, gt=0)
    groove_count: int = Field(default=3, ge=0)
    # Knurl params
    knurl_pitch_mm: float = Field(default=1.0, gt=0)
    knurl_depth_mm: float = Field(default=0.3, gt=0)
    # Contour params
    contour_radius_mm: float = Field(default=25.0, gt=0)


class ContactAnalyzeRequest(BaseModel):
    """Request for contact analysis."""

    horn: HornGeometry = Field(default_factory=HornGeometry)
    workpiece: WorkpieceParams = Field(default_factory=WorkpieceParams)
    anvil: AnvilParamsInput = Field(default_factory=AnvilParamsInput)
    contact_type: str = "penalty"  # penalty | nitsche
    frequency_hz: float = Field(default=20000.0, gt=0)
    amplitude_um: float = Field(default=30.0, gt=0)
    friction_coefficient: float = Field(default=0.3, ge=0, le=1.0)
    penalty_stiffness: float = Field(default=1e12, gt=0)
    nitsche_parameter: float = Field(default=100.0, gt=0)
    horn_material: str = "Titanium Ti-6Al-4V"
    anvil_material: str = "Tool Steel"


class ContactPressureResult(BaseModel):
    """Contact pressure result data."""

    max_MPa: float = 0.0
    mean_MPa: float = 0.0
    distribution: list[float] = []


class SlipDistanceResult(BaseModel):
    """Slip distance result data."""

    max_um: float = 0.0
    mean_um: float = 0.0
    distribution: list[float] = []


class DeformationResult(BaseModel):
    """Deformation result data."""

    max_um: float = 0.0
    field: list[float] = []


class StressResult(BaseModel):
    """Stress result data."""

    von_mises_max_MPa: float = 0.0
    field: list[float] = []


class ContactSummary(BaseModel):
    """Contact analysis summary."""

    contact_area_mm2: float = 0.0
    total_force_N: float = 0.0
    newton_iterations: int = 0
    converged: bool = False
    solve_time_s: float = 0.0
    contact_type: str = "unknown"


class WeldQualityEstimate(BaseModel):
    """Weld quality estimate based on contact analysis."""

    score: float = Field(default=0.0, ge=0, le=100)
    rating: str = "unknown"  # excellent | good | fair | poor
    notes: list[str] = []


class ContactAnalyzeResponse(BaseModel):
    """Response from contact analysis."""

    status: str = "success"
    contact_pressure: ContactPressureResult = Field(
        default_factory=ContactPressureResult
    )
    slip_distance: SlipDistanceResult = Field(default_factory=SlipDistanceResult)
    deformation: DeformationResult = Field(default_factory=DeformationResult)
    stress: StressResult = Field(default_factory=StressResult)
    summary: ContactSummary = Field(default_factory=ContactSummary)
    weld_quality: WeldQualityEstimate = Field(default_factory=WeldQualityEstimate)


class DockerCheckResponse(BaseModel):
    """Response from Docker/FEniCSx availability check."""

    available: bool = False
    image: str = ""
    version: str = ""
    message: str = ""


class AnvilPreviewRequest(BaseModel):
    """Request for anvil geometry preview."""

    anvil_type: str = "flat"
    width_mm: float = Field(default=50.0, gt=0)
    depth_mm: float = Field(default=30.0, gt=0)
    height_mm: float = Field(default=20.0, gt=0)
    groove_width_mm: float = Field(default=5.0, gt=0)
    groove_depth_mm: float = Field(default=3.0, gt=0)
    groove_count: int = Field(default=3, ge=0)
    knurl_pitch_mm: float = Field(default=1.0, gt=0)
    knurl_depth_mm: float = Field(default=0.3, gt=0)
    contour_radius_mm: float = Field(default=25.0, gt=0)


class AnvilPreviewResponse(BaseModel):
    """Anvil mesh preview for Three.js."""

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    anvil_type: str = ""
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    contact_face: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _estimate_weld_quality(contact_result: dict) -> dict:
    """Estimate weld quality from contact analysis results.

    Scoring is based on:
    - Contact pressure uniformity
    - Sufficient contact area
    - Convergence
    """
    summary = contact_result.get("summary", {})
    pressure = contact_result.get("contact_pressure", {})

    score = 0.0
    notes: list[str] = []

    # Check convergence
    if summary.get("converged", False):
        score += 30.0
        notes.append("Solver converged successfully")
    else:
        notes.append("Solver did not converge -- results may be unreliable")

    # Contact area score (target: > 50% of nominal)
    area = summary.get("contact_area_mm2", 0.0)
    if area > 0:
        score += min(25.0, area / 10.0)  # scaled
        notes.append(f"Contact area: {area:.1f} mm^2")
    else:
        notes.append("No contact area detected")

    # Pressure quality
    max_p = pressure.get("max_MPa", 0.0)
    mean_p = pressure.get("mean_MPa", 0.0)
    if max_p > 0 and mean_p > 0:
        uniformity = mean_p / max_p
        score += uniformity * 25.0
        notes.append(f"Pressure uniformity: {uniformity:.1%}")
    else:
        notes.append("No contact pressure detected")

    # Force adequacy
    force = summary.get("total_force_N", 0.0)
    if force > 100:
        score += min(20.0, force / 100.0)
        notes.append(f"Total contact force: {force:.1f} N")
    else:
        notes.append("Contact force may be insufficient")

    score = min(100.0, max(0.0, score))

    if score >= 80:
        rating = "excellent"
    elif score >= 60:
        rating = "good"
    elif score >= 40:
        rating = "fair"
    else:
        rating = "poor"

    return {
        "score": round(score, 1),
        "rating": rating,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/analyze", response_model=ContactAnalyzeResponse)
async def analyze_contact(request: ContactAnalyzeRequest):
    """Run contact mechanics analysis using FEniCSx Docker solver.

    Supports penalty and Nitsche contact formulations for the
    horn-workpiece-anvil assembly. Returns contact pressure map,
    slip distance, deformation, stress, and weld quality estimate.
    """
    if request.contact_type not in ("penalty", "nitsche"):
        raise HTTPException(
            400,
            f"Invalid contact_type: {request.contact_type!r}. "
            f"Must be 'penalty' or 'nitsche'.",
        )

    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_fenicsx import (
            ContactSolver,
        )

        solver = ContactSolver()

        config = {
            "horn_material": request.horn_material,
            "workpiece_material": request.workpiece.material,
            "anvil_material": request.anvil_material,
            "frequency_hz": request.frequency_hz,
            "amplitude_um": request.amplitude_um,
            "contact_type": request.contact_type,
            "friction_coefficient": request.friction_coefficient,
            "penalty_stiffness": request.penalty_stiffness,
            "nitsche_parameter": request.nitsche_parameter,
        }

        result = await solver.analyze(config)

        if result.get("status") == "error":
            raise HTTPException(
                500,
                f"Contact analysis failed: {result.get('error', 'Unknown error')}",
            )

        # Estimate weld quality
        weld_quality = _estimate_weld_quality(result)

        return ContactAnalyzeResponse(
            status="success",
            contact_pressure=ContactPressureResult(
                **result.get("contact_pressure", {})
            ),
            slip_distance=SlipDistanceResult(**result.get("slip_distance", {})),
            deformation=DeformationResult(**result.get("deformation", {})),
            stress=StressResult(**result.get("stress", {})),
            summary=ContactSummary(**result.get("summary", {})),
            weld_quality=WeldQualityEstimate(**weld_quality),
        )

    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(
            500,
            f"Contact solver not available: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Contact analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Contact analysis failed: {exc}") from exc


@router.post("/check-docker", response_model=DockerCheckResponse)
async def check_docker():
    """Check if Docker and FEniCSx image are available.

    Returns availability status, image name, and version info.
    """
    try:
        from web.services.fenicsx_runner import FEniCSxRunner, DOCKER_IMAGE

        runner = FEniCSxRunner()
        available = await runner.check_available()

        return DockerCheckResponse(
            available=available,
            image=DOCKER_IMAGE,
            version="stable" if available else "",
            message=(
                "Docker and FEniCSx image are available"
                if available
                else "Docker or FEniCSx image not found"
            ),
        )
    except ImportError:
        return DockerCheckResponse(
            available=False,
            image="",
            version="",
            message="FEniCSxRunner module not available",
        )
    except Exception as exc:
        logger.error("Docker check error: %s\n%s", exc, traceback.format_exc())
        return DockerCheckResponse(
            available=False,
            image="",
            version="",
            message=f"Error checking Docker: {exc}",
        )


@router.post("/anvil-preview", response_model=AnvilPreviewResponse)
async def anvil_preview(request: AnvilPreviewRequest):
    """Generate anvil geometry preview mesh for Three.js rendering.

    Supports flat, groove, knurled, and contour anvil types.
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator import (
            AnvilGenerator,
            AnvilParams,
        )

        params = AnvilParams(
            anvil_type=request.anvil_type,
            width_mm=request.width_mm,
            depth_mm=request.depth_mm,
            height_mm=request.height_mm,
            groove_width_mm=request.groove_width_mm,
            groove_depth_mm=request.groove_depth_mm,
            groove_count=request.groove_count,
            knurl_pitch_mm=request.knurl_pitch_mm,
            knurl_depth_mm=request.knurl_depth_mm,
            contour_radius_mm=request.contour_radius_mm,
        )

        generator = AnvilGenerator()
        result = await asyncio.to_thread(generator.generate, params)

        mesh_preview = result.get("mesh_preview", {})

        return AnvilPreviewResponse(
            vertices=mesh_preview.get("vertices", []),
            faces=mesh_preview.get("faces", []),
            anvil_type=request.anvil_type,
            volume_mm3=result.get("volume_mm3", 0.0),
            surface_area_mm2=result.get("surface_area_mm2", 0.0),
            contact_face=result.get("contact_face", {}),
        )

    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(
            500,
            f"Anvil generator not available: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Anvil preview error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Anvil preview generation failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Thermal analysis models
# ---------------------------------------------------------------------------


class ThermalAnalyzeRequest(BaseModel):
    """Request for thermal analysis."""

    # Geometry / material
    workpiece_material: str = "ABS"
    horn_material: str = "Titanium Ti-6Al-4V"

    # Welding parameters
    frequency_hz: float = Field(default=20000.0, gt=0)
    amplitude_um: float = Field(default=30.0, gt=0)
    weld_time_s: float = Field(default=0.5, gt=0)
    contact_pressure_mpa: float = Field(default=1.0, gt=0)
    friction_coefficient: float = Field(default=0.3, ge=0, le=1.0)

    # Thermal parameters
    initial_temp_c: float = Field(default=25.0)
    n_time_steps: int = Field(default=50, ge=1, le=1000)
    interface_thickness_mm: float = Field(default=1.0, gt=0)

    # Optional mesh path
    mesh_path: str = ""


class MeltZoneResult(BaseModel):
    """Melt zone prediction data."""

    melt_temp_c: float = 0.0
    melt_fraction: float = 0.0
    melt_volume_mm3: float = 0.0
    n_melt_nodes: int = 0
    n_total_nodes: int = 0


class ThermalHistoryPoint(BaseModel):
    """Single point in the thermal history."""

    time_s: float = 0.0
    max_temp_c: float = 0.0
    mean_temp_c: float = 0.0
    min_temp_c: float = 0.0


class ThermalAnalyzeResponse(BaseModel):
    """Response from thermal analysis."""

    status: str = "success"
    # Temperature results
    max_temperature_c: float = 0.0
    mean_temperature_c: float = 0.0
    min_temperature_c: float = 0.0
    initial_temperature_c: float = 25.0
    # Distribution
    temperature_distribution: list[float] = []
    # Melt zone
    melt_zone: MeltZoneResult = Field(default_factory=MeltZoneResult)
    # Thermal history
    thermal_history: list[ThermalHistoryPoint] = []
    max_temp_history: list[float] = []
    mean_temp_history: list[float] = []
    # Heat source info
    heat_generation_rate_w_m3: float = 0.0
    surface_heat_flux_w_m2: float = 0.0
    # Timing
    solve_time_s: float = 0.0
    weld_time_s: float = 0.0
    n_time_steps: int = 0


class FullAnalysisRequest(BaseModel):
    """Request for combined contact + thermal analysis."""

    # Horn
    horn: HornGeometry = Field(default_factory=HornGeometry)
    horn_material: str = "Titanium Ti-6Al-4V"

    # Workpiece
    workpiece: WorkpieceParams = Field(default_factory=WorkpieceParams)

    # Anvil
    anvil: AnvilParamsInput = Field(default_factory=AnvilParamsInput)
    anvil_material: str = "Tool Steel"

    # Contact parameters
    contact_type: str = "penalty"
    friction_coefficient: float = Field(default=0.3, ge=0, le=1.0)
    penalty_stiffness: float = Field(default=1e12, gt=0)
    nitsche_parameter: float = Field(default=100.0, gt=0)

    # Welding parameters
    frequency_hz: float = Field(default=20000.0, gt=0)
    amplitude_um: float = Field(default=30.0, gt=0)
    weld_time_s: float = Field(default=0.5, gt=0)

    # Thermal parameters
    initial_temp_c: float = Field(default=25.0)
    n_time_steps: int = Field(default=50, ge=1, le=1000)


class FullAnalysisResponse(BaseModel):
    """Response from combined contact + thermal analysis."""

    status: str = "success"
    # Contact results
    contact: Optional[ContactAnalyzeResponse] = None
    # Thermal results
    thermal: Optional[ThermalAnalyzeResponse] = None
    # Combined quality
    weld_quality: WeldQualityEstimate = Field(default_factory=WeldQualityEstimate)
    # Total timing
    total_solve_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Thermal endpoint
# ---------------------------------------------------------------------------


@router.post("/thermal", response_model=ThermalAnalyzeResponse)
async def analyze_thermal(request: ThermalAnalyzeRequest):
    """Run thermal analysis for ultrasonic welding.

    Solves the transient heat equation with friction heating at the
    contact interface. Returns temperature distribution, melt zone
    prediction, and thermal history over the weld duration.
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.thermal_solver import (
            ThermalSolver,
        )

        solver = ThermalSolver()

        config = {
            "materials": {
                "workpiece": {"name": request.workpiece_material},
            },
            "frequency_hz": request.frequency_hz,
            "amplitude_um": request.amplitude_um,
            "contact_pressure_pa": request.contact_pressure_mpa * 1e6,
            "friction_coefficient": request.friction_coefficient,
            "weld_time_s": request.weld_time_s,
            "n_time_steps": request.n_time_steps,
            "initial_temp_c": request.initial_temp_c,
            "interface_thickness_m": request.interface_thickness_mm * 1e-3,
            "mesh_path": request.mesh_path,
        }

        result = await solver.analyze(config)

        if result.get("status") == "error":
            raise HTTPException(
                500,
                f"Thermal analysis failed: {result.get('error', 'Unknown error')}",
            )

        # Build melt zone
        melt_zone_data = result.get("melt_zone", {})
        melt_zone = MeltZoneResult(**melt_zone_data)

        # Build thermal history
        history_data = result.get("thermal_history", [])
        thermal_history = [ThermalHistoryPoint(**h) for h in history_data]

        return ThermalAnalyzeResponse(
            status="success",
            max_temperature_c=result.get("max_temperature_c", 0.0),
            mean_temperature_c=result.get("mean_temperature_c", 0.0),
            min_temperature_c=result.get("min_temperature_c", 0.0),
            initial_temperature_c=result.get("initial_temperature_c", 25.0),
            temperature_distribution=result.get("temperature_distribution", []),
            melt_zone=melt_zone,
            thermal_history=thermal_history,
            max_temp_history=result.get("max_temp_history", []),
            mean_temp_history=result.get("mean_temp_history", []),
            heat_generation_rate_w_m3=result.get("heat_generation_rate_w_m3", 0.0),
            surface_heat_flux_w_m2=result.get("surface_heat_flux_w_m2", 0.0),
            solve_time_s=result.get("solve_time_s", 0.0),
            weld_time_s=result.get("weld_time_s", 0.0),
            n_time_steps=result.get("n_time_steps", 0),
        )

    except HTTPException:
        raise
    except ImportError as exc:
        raise HTTPException(
            500,
            f"Thermal solver not available: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Thermal analysis error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(500, f"Thermal analysis failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Full analysis endpoint (contact + thermal)
# ---------------------------------------------------------------------------


@router.post("/full-analysis", response_model=FullAnalysisResponse)
async def full_analysis(request: FullAnalysisRequest):
    """Run combined contact + thermal analysis.

    First runs the contact mechanics analysis to obtain contact pressure,
    then feeds the pressure into the thermal solver for friction heating
    simulation. Returns combined results with overall weld quality.
    """
    import time as _time

    t_start = _time.monotonic()

    if request.contact_type not in ("penalty", "nitsche"):
        raise HTTPException(
            400,
            f"Invalid contact_type: {request.contact_type!r}. "
            f"Must be 'penalty' or 'nitsche'.",
        )

    contact_response = None
    thermal_response = None
    contact_pressure_pa = 1e6  # default fallback

    # ---- Step 1: Contact analysis ----
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_fenicsx import (
            ContactSolver,
        )

        contact_solver = ContactSolver()
        contact_config = {
            "horn_material": request.horn_material,
            "workpiece_material": request.workpiece.material,
            "anvil_material": request.anvil_material,
            "frequency_hz": request.frequency_hz,
            "amplitude_um": request.amplitude_um,
            "contact_type": request.contact_type,
            "friction_coefficient": request.friction_coefficient,
            "penalty_stiffness": request.penalty_stiffness,
            "nitsche_parameter": request.nitsche_parameter,
        }

        contact_result = await contact_solver.analyze(contact_config)

        if contact_result.get("status") == "success":
            weld_quality = _estimate_weld_quality(contact_result)
            contact_response = ContactAnalyzeResponse(
                status="success",
                contact_pressure=ContactPressureResult(
                    **contact_result.get("contact_pressure", {})
                ),
                slip_distance=SlipDistanceResult(
                    **contact_result.get("slip_distance", {})
                ),
                deformation=DeformationResult(
                    **contact_result.get("deformation", {})
                ),
                stress=StressResult(**contact_result.get("stress", {})),
                summary=ContactSummary(**contact_result.get("summary", {})),
                weld_quality=WeldQualityEstimate(**weld_quality),
            )
            # Extract contact pressure for thermal analysis
            mean_p = contact_result.get("contact_pressure", {}).get("mean_MPa", 1.0)
            contact_pressure_pa = mean_p * 1e6
        else:
            logger.warning(
                "Contact analysis failed: %s -- proceeding with default pressure",
                contact_result.get("error", "unknown"),
            )
    except ImportError as exc:
        logger.warning("Contact solver not available: %s", exc)
    except Exception as exc:
        logger.warning("Contact analysis error: %s", exc)

    # ---- Step 2: Thermal analysis ----
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.thermal_solver import (
            ThermalSolver,
        )

        thermal_solver = ThermalSolver()
        thermal_config = {
            "materials": {
                "workpiece": {"name": request.workpiece.material},
            },
            "frequency_hz": request.frequency_hz,
            "amplitude_um": request.amplitude_um,
            "contact_pressure_pa": contact_pressure_pa,
            "friction_coefficient": request.friction_coefficient,
            "weld_time_s": request.weld_time_s,
            "n_time_steps": request.n_time_steps,
            "initial_temp_c": request.initial_temp_c,
        }

        thermal_result = await thermal_solver.analyze(thermal_config)

        if thermal_result.get("status") == "success":
            melt_zone_data = thermal_result.get("melt_zone", {})
            history_data = thermal_result.get("thermal_history", [])

            thermal_response = ThermalAnalyzeResponse(
                status="success",
                max_temperature_c=thermal_result.get("max_temperature_c", 0.0),
                mean_temperature_c=thermal_result.get("mean_temperature_c", 0.0),
                min_temperature_c=thermal_result.get("min_temperature_c", 0.0),
                initial_temperature_c=thermal_result.get(
                    "initial_temperature_c", 25.0
                ),
                temperature_distribution=thermal_result.get(
                    "temperature_distribution", []
                ),
                melt_zone=MeltZoneResult(**melt_zone_data),
                thermal_history=[
                    ThermalHistoryPoint(**h) for h in history_data
                ],
                max_temp_history=thermal_result.get("max_temp_history", []),
                mean_temp_history=thermal_result.get("mean_temp_history", []),
                heat_generation_rate_w_m3=thermal_result.get(
                    "heat_generation_rate_w_m3", 0.0
                ),
                surface_heat_flux_w_m2=thermal_result.get(
                    "surface_heat_flux_w_m2", 0.0
                ),
                solve_time_s=thermal_result.get("solve_time_s", 0.0),
                weld_time_s=thermal_result.get("weld_time_s", 0.0),
                n_time_steps=thermal_result.get("n_time_steps", 0),
            )
        else:
            logger.warning(
                "Thermal analysis failed: %s",
                thermal_result.get("error", "unknown"),
            )
    except ImportError as exc:
        logger.warning("Thermal solver not available: %s", exc)
    except Exception as exc:
        logger.warning("Thermal analysis error: %s", exc)

    # ---- Compute combined quality ----
    total_time = _time.monotonic() - t_start

    # Use contact weld quality if available, otherwise estimate from thermal
    combined_quality = WeldQualityEstimate()
    if contact_response and contact_response.weld_quality:
        combined_quality = contact_response.weld_quality

    # Determine overall status
    status = "success"
    if contact_response is None and thermal_response is None:
        raise HTTPException(
            500,
            "Both contact and thermal analyses failed. "
            "Check Docker/FEniCSx availability.",
        )

    return FullAnalysisResponse(
        status=status,
        contact=contact_response,
        thermal=thermal_response,
        weld_quality=combined_quality,
        total_solve_time_s=round(total_time, 3),
    )
