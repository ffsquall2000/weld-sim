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
