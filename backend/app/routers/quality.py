"""V2 API endpoint for welding quality assessment.

Evaluates ultrasonic metal welding simulation results against
industry-standard criteria and returns a detailed quality report.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter

from backend.app.schemas.quality import (
    AnalysisSection,
    CriterionResult,
    QualityAssessmentRequest,
    QualityAssessmentResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/quality", tags=["quality"])

# ---------------------------------------------------------------------------
# Criterion specification helpers
# ---------------------------------------------------------------------------


@dataclass
class CriterionSpec:
    """Defines the acceptable / optimal ranges for a single criterion."""

    name: str
    description: str
    param_key: str
    unit: str
    acceptable_min: float
    acceptable_max: float
    optimal_min: Optional[float] = None
    optimal_max: Optional[float] = None
    weight: float = 1.0  # relative scoring weight


@dataclass
class ApplicationProfile:
    """Complete set of criteria and metadata for an application type."""

    label: str
    criteria: List[CriterionSpec] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-application criteria definitions
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, ApplicationProfile] = {
    "li_battery_tab": ApplicationProfile(
        label="Li-Ion Battery Tab Welding",
        criteria=[
            CriterionSpec(
                name="amplitude",
                description="Vibration amplitude",
                param_key="amplitude_um",
                unit="um",
                acceptable_min=15.0,
                acceptable_max=50.0,
                optimal_min=20.0,
                optimal_max=40.0,
                weight=1.5,
            ),
            CriterionSpec(
                name="pressure",
                description="Clamping pressure",
                param_key="pressure_mpa",
                unit="MPa",
                acceptable_min=1.0,
                acceptable_max=15.0,
                optimal_min=3.0,
                optimal_max=10.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="energy",
                description="Welding energy",
                param_key="energy_j",
                unit="J",
                acceptable_min=50.0,
                acceptable_max=3000.0,
                optimal_min=200.0,
                optimal_max=2000.0,
                weight=2.0,
            ),
            CriterionSpec(
                name="time",
                description="Weld duration",
                param_key="time_ms",
                unit="ms",
                acceptable_min=100.0,
                acceptable_max=1500.0,
                optimal_min=200.0,
                optimal_max=1000.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="frequency_deviation",
                description="Frequency deviation from nominal",
                param_key="frequency_deviation_pct",
                unit="%",
                acceptable_min=0.0,
                acceptable_max=3.0,
                optimal_min=0.0,
                optimal_max=1.5,
                weight=1.0,
            ),
            CriterionSpec(
                name="temperature_rise",
                description="Temperature rise during welding",
                param_key="temperature_rise_c",
                unit="C",
                acceptable_min=0.0,
                acceptable_max=120.0,
                optimal_min=0.0,
                optimal_max=80.0,
                weight=2.0,
            ),
            CriterionSpec(
                name="amplitude_uniformity",
                description="Amplitude distribution uniformity",
                param_key="amplitude_uniformity_pct",
                unit="%",
                acceptable_min=85.0,
                acceptable_max=100.0,
                optimal_min=92.0,
                optimal_max=100.0,
                weight=1.5,
            ),
        ],
    ),
    "busbar": ApplicationProfile(
        label="Busbar Welding",
        criteria=[
            CriterionSpec(
                name="amplitude",
                description="Vibration amplitude",
                param_key="amplitude_um",
                unit="um",
                acceptable_min=25.0,
                acceptable_max=50.0,
                optimal_min=30.0,
                optimal_max=45.0,
                weight=1.5,
            ),
            CriterionSpec(
                name="pressure",
                description="Clamping pressure",
                param_key="pressure_mpa",
                unit="MPa",
                acceptable_min=0.3,
                acceptable_max=1.0,
                optimal_min=0.4,
                optimal_max=0.8,
                weight=1.0,
            ),
            CriterionSpec(
                name="energy",
                description="Welding energy",
                param_key="energy_j",
                unit="J",
                acceptable_min=100.0,
                acceptable_max=1000.0,
                optimal_min=200.0,
                optimal_max=800.0,
                weight=2.0,
            ),
            CriterionSpec(
                name="time",
                description="Weld duration",
                param_key="time_ms",
                unit="ms",
                acceptable_min=200.0,
                acceptable_max=1500.0,
                optimal_min=400.0,
                optimal_max=1200.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="temperature_rise",
                description="Temperature rise during welding",
                param_key="temperature_rise_c",
                unit="C",
                acceptable_min=0.0,
                acceptable_max=200.0,
                optimal_min=0.0,
                optimal_max=120.0,
                weight=1.5,
            ),
            CriterionSpec(
                name="stress_safety_factor",
                description="Mechanical stress safety factor",
                param_key="stress_safety_factor",
                unit="",
                acceptable_min=2.0,
                acceptable_max=100.0,
                optimal_min=3.0,
                optimal_max=100.0,
                weight=2.0,
            ),
        ],
    ),
    "collector": ApplicationProfile(
        label="Collector Plate Welding",
        criteria=[
            CriterionSpec(
                name="amplitude",
                description="Vibration amplitude",
                param_key="amplitude_um",
                unit="um",
                acceptable_min=15.0,
                acceptable_max=45.0,
                optimal_min=20.0,
                optimal_max=35.0,
                weight=1.5,
            ),
            CriterionSpec(
                name="pressure",
                description="Clamping pressure",
                param_key="pressure_mpa",
                unit="MPa",
                acceptable_min=0.1,
                acceptable_max=0.8,
                optimal_min=0.2,
                optimal_max=0.5,
                weight=1.0,
            ),
            CriterionSpec(
                name="energy",
                description="Welding energy",
                param_key="energy_j",
                unit="J",
                acceptable_min=30.0,
                acceptable_max=500.0,
                optimal_min=80.0,
                optimal_max=350.0,
                weight=2.0,
            ),
            CriterionSpec(
                name="time",
                description="Weld duration",
                param_key="time_ms",
                unit="ms",
                acceptable_min=100.0,
                acceptable_max=800.0,
                optimal_min=200.0,
                optimal_max=500.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="temperature_rise",
                description="Temperature rise during welding",
                param_key="temperature_rise_c",
                unit="C",
                acceptable_min=0.0,
                acceptable_max=150.0,
                optimal_min=0.0,
                optimal_max=100.0,
                weight=1.5,
            ),
        ],
    ),
    "general_metal": ApplicationProfile(
        label="General Metal Welding",
        criteria=[
            CriterionSpec(
                name="amplitude",
                description="Vibration amplitude",
                param_key="amplitude_um",
                unit="um",
                acceptable_min=15.0,
                acceptable_max=50.0,
                optimal_min=20.0,
                optimal_max=45.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="pressure",
                description="Clamping pressure",
                param_key="pressure_mpa",
                unit="MPa",
                acceptable_min=0.1,
                acceptable_max=1.5,
                optimal_min=0.2,
                optimal_max=1.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="energy",
                description="Welding energy",
                param_key="energy_j",
                unit="J",
                acceptable_min=10.0,
                acceptable_max=2000.0,
                optimal_min=50.0,
                optimal_max=1500.0,
                weight=1.0,
            ),
            CriterionSpec(
                name="time",
                description="Weld duration",
                param_key="time_ms",
                unit="ms",
                acceptable_min=50.0,
                acceptable_max=2000.0,
                optimal_min=100.0,
                optimal_max=1500.0,
                weight=1.0,
            ),
        ],
    ),
}

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _evaluate_criterion(
    spec: CriterionSpec,
    value: Optional[float],
) -> CriterionResult:
    """Evaluate a single value against its specification."""

    if value is None:
        return CriterionResult(
            name=spec.name,
            description=spec.description,
            value=None,
            unit=spec.unit,
            min_threshold=spec.acceptable_min,
            max_threshold=spec.acceptable_max,
            status="not_tested",
            explanation=(
                f"{spec.description} was not provided in the simulation results. "
                f"Acceptable range is {spec.acceptable_min}-{spec.acceptable_max} {spec.unit}."
            ),
        )

    # Determine status -------------------------------------------------------
    optimal_min = spec.optimal_min if spec.optimal_min is not None else spec.acceptable_min
    optimal_max = spec.optimal_max if spec.optimal_max is not None else spec.acceptable_max

    if optimal_min <= value <= optimal_max:
        status = "pass"
        explanation = (
            f"{spec.description} of {value} {spec.unit} is within the optimal range "
            f"({optimal_min}-{optimal_max} {spec.unit}). "
            "This value supports high-quality weld formation."
        )
    elif spec.acceptable_min <= value <= spec.acceptable_max:
        status = "warning"
        if value < optimal_min:
            explanation = (
                f"{spec.description} of {value} {spec.unit} is below the optimal range "
                f"({optimal_min}-{optimal_max} {spec.unit}) but within acceptable limits "
                f"({spec.acceptable_min}-{spec.acceptable_max} {spec.unit}). "
                "Consider increasing the value for improved weld quality."
            )
        else:
            explanation = (
                f"{spec.description} of {value} {spec.unit} is above the optimal range "
                f"({optimal_min}-{optimal_max} {spec.unit}) but within acceptable limits "
                f"({spec.acceptable_min}-{spec.acceptable_max} {spec.unit}). "
                "Consider decreasing the value to reduce potential side-effects."
            )
    else:
        status = "fail"
        if value < spec.acceptable_min:
            explanation = (
                f"{spec.description} of {value} {spec.unit} is below the minimum acceptable "
                f"threshold of {spec.acceptable_min} {spec.unit}. "
                "This is likely to result in insufficient weld formation or bond strength."
            )
        else:
            explanation = (
                f"{spec.description} of {value} {spec.unit} exceeds the maximum acceptable "
                f"threshold of {spec.acceptable_max} {spec.unit}. "
                "This may cause material damage, excessive heat, or other defects."
            )

    return CriterionResult(
        name=spec.name,
        description=spec.description,
        value=value,
        unit=spec.unit,
        min_threshold=spec.acceptable_min,
        max_threshold=spec.acceptable_max,
        status=status,
        explanation=explanation,
    )


def _compute_score(results: List[CriterionResult], specs: List[CriterionSpec]) -> float:
    """Compute weighted 0-100 quality score from criterion results.

    - pass     -> full weight
    - warning  -> half weight
    - fail     -> 0
    - not_tested -> excluded from scoring
    """
    spec_map = {s.name: s for s in specs}
    total_weight = 0.0
    earned = 0.0

    for r in results:
        if r.status == "not_tested":
            continue
        w = spec_map.get(r.name, CriterionSpec("", "", "", "", 0, 0)).weight
        total_weight += w
        if r.status == "pass":
            earned += w
        elif r.status == "warning":
            earned += w * 0.5
        # fail contributes 0

    if total_weight == 0.0:
        return 0.0
    return round(earned / total_weight * 100.0, 1)


def _build_recommendations(
    results: List[CriterionResult],
    body: QualityAssessmentRequest,
) -> List[str]:
    """Generate actionable recommendations from non-passing criteria."""
    recs: List[str] = []

    for r in results:
        if r.status == "fail":
            if r.name == "amplitude":
                if r.value is not None and r.min_threshold is not None and r.value < r.min_threshold:
                    recs.append(
                        f"Increase vibration amplitude to at least {r.min_threshold} {r.unit}. "
                        "Insufficient amplitude leads to poor interfacial heating and weak bonds."
                    )
                elif r.value is not None and r.max_threshold is not None:
                    recs.append(
                        f"Reduce vibration amplitude to below {r.max_threshold} {r.unit}. "
                        "Excessive amplitude can cause material cracking or surface damage."
                    )
            elif r.name == "pressure":
                if r.value is not None and r.min_threshold is not None and r.value < r.min_threshold:
                    recs.append(
                        f"Increase clamping pressure to at least {r.min_threshold} {r.unit}. "
                        "Low pressure results in poor contact and energy transfer."
                    )
                elif r.value is not None and r.max_threshold is not None:
                    recs.append(
                        f"Reduce clamping pressure to below {r.max_threshold} {r.unit}. "
                        "Excessive pressure can cause material deformation and restrict vibration."
                    )
            elif r.name == "energy":
                if r.value is not None and r.min_threshold is not None and r.value < r.min_threshold:
                    recs.append(
                        f"Increase welding energy above {r.min_threshold} {r.unit}. "
                        "Insufficient energy produces under-welded joints with low peel strength."
                    )
                elif r.value is not None and r.max_threshold is not None:
                    recs.append(
                        f"Reduce welding energy below {r.max_threshold} {r.unit}. "
                        "Over-welding degrades joint quality through excessive material softening."
                    )
            elif r.name == "time":
                if r.value is not None and r.min_threshold is not None and r.value < r.min_threshold:
                    recs.append(
                        f"Increase weld time above {r.min_threshold} {r.unit}. "
                        "Short weld times may not allow sufficient diffusion bonding."
                    )
                elif r.value is not None and r.max_threshold is not None:
                    recs.append(
                        f"Reduce weld time below {r.max_threshold} {r.unit}. "
                        "Excessive weld time leads to unnecessary heat build-up and energy waste."
                    )
            elif r.name == "temperature_rise":
                recs.append(
                    f"Temperature rise of {r.value} {r.unit} exceeds the maximum safe limit of "
                    f"{r.max_threshold} {r.unit}. Consider reducing energy, amplitude, or weld time, "
                    "and verify that horn and anvil cooling are adequate."
                )
            elif r.name == "frequency_deviation":
                recs.append(
                    f"Frequency deviation of {r.value}% exceeds the {r.max_threshold}% limit. "
                    "Check the horn-booster-converter stack tuning and ensure proper mechanical coupling."
                )
            elif r.name == "amplitude_uniformity":
                recs.append(
                    f"Amplitude uniformity of {r.value}% is below the required {r.min_threshold}%. "
                    "Re-evaluate horn geometry or consider a slotted horn design for better uniformity."
                )
            elif r.name == "stress_safety_factor":
                recs.append(
                    f"Stress safety factor of {r.value} is below the minimum of {r.min_threshold}. "
                    "The joint may not withstand service loads. Increase weld energy or contact area."
                )
            else:
                recs.append(
                    f"{r.description} ({r.value} {r.unit}) is outside acceptable limits. "
                    "Review and adjust this parameter."
                )
        elif r.status == "warning":
            recs.append(
                f"Consider optimizing {r.description.lower()}: current value {r.value} {r.unit} "
                f"is acceptable but outside the optimal range."
            )

    if not recs:
        recs.append(
            "All tested parameters are within optimal ranges. "
            "No corrective action required."
        )

    return recs


def _build_application_notes(
    body: QualityAssessmentRequest,
    results: List[CriterionResult],
) -> str:
    """Generate application-specific guidance and warnings."""
    notes_parts: List[str] = []
    app = body.application_type

    # Material-pair warnings ------------------------------------------------
    mat_upper = (body.material_upper or "").upper()
    mat_lower = (body.material_lower or "").upper()
    is_cu_al = (
        ("CU" in mat_upper and "AL" in mat_lower)
        or ("AL" in mat_upper and "CU" in mat_lower)
    )

    if app == "li_battery_tab":
        notes_parts.append(
            "Li-Ion battery tab welding requires strict thermal control to prevent "
            "separator damage and electrolyte decomposition."
        )
        # IMC warning for Cu-Al pairs
        if is_cu_al:
            energy_result = next((r for r in results if r.name == "energy"), None)
            if energy_result and energy_result.value is not None:
                if energy_result.value > 120.0:
                    notes_parts.append(
                        "WARNING: Cu-Al material combination with high welding energy "
                        f"({energy_result.value} J) increases the risk of brittle "
                        "intermetallic compound (IMC) formation (CuAl2, Cu9Al4). "
                        "IMC layers degrade electrical conductivity and long-term "
                        "mechanical reliability. Consider reducing energy or using "
                        "an intermediate Ni layer."
                    )
                else:
                    notes_parts.append(
                        "Cu-Al material pair detected. Current energy level is acceptable, "
                        "but monitor intermetallic compound formation during production "
                        "validation."
                    )

        # Temperature / separator warning
        temp_result = next((r for r in results if r.name == "temperature_rise"), None)
        if temp_result and temp_result.value is not None and temp_result.value > 80.0:
            notes_parts.append(
                f"Temperature rise of {temp_result.value} C approaches the safety "
                "limit for battery separator materials. Ensure adequate heat sinking "
                "and consider reducing weld time."
            )

        notes_parts.append(
            "For production, validate peel strength per IPC/WHMA-A-620 or equivalent "
            "standard and perform 100% visual inspection for tab alignment."
        )

    elif app == "busbar":
        notes_parts.append(
            "Busbar welding requires high joint strength and low electrical resistance. "
            "Ensure that the contact area is sufficiently large to carry rated current."
        )
        sf_result = next(
            (r for r in results if r.name == "stress_safety_factor"), None
        )
        if sf_result and sf_result.value is not None and sf_result.value < 2.5:
            notes_parts.append(
                "The stress safety factor is marginal for busbar applications "
                "subject to thermal cycling and vibration loads. "
                "Consider increasing weld energy or adding supplementary fastening."
            )
        if is_cu_al:
            notes_parts.append(
                "Cu-Al busbar joints are susceptible to IMC growth under thermal "
                "cycling. Consider bi-metal transition strips or protective coatings "
                "for long service life."
            )

    elif app == "collector":
        notes_parts.append(
            "Collector plate welding demands uniform weld quality across multiple "
            "layers. Ensure consistent clamping across the weld zone."
        )
        if body.total_thickness_mm is not None and body.total_thickness_mm > 1.0:
            notes_parts.append(
                f"Total stack thickness of {body.total_thickness_mm} mm may require "
                "staged welding or higher energy settings for full penetration."
            )

    else:  # general_metal
        notes_parts.append(
            "General metal ultrasonic welding assessment applied. For application-"
            "specific guidance, specify the application_type (li_battery_tab, "
            "busbar, or collector)."
        )
        if body.total_thickness_mm is not None and body.total_thickness_mm > 3.0:
            notes_parts.append(
                f"Total thickness of {body.total_thickness_mm} mm is relatively high "
                "for ultrasonic metal welding. Verify that sufficient energy is "
                "delivered to the faying interface."
            )

    return " ".join(notes_parts)


def _section_summary(items: List[CriterionResult]) -> str:
    """Produce a short summary line for an analysis section."""
    tested = [i for i in items if i.status != "not_tested"]
    if not tested:
        return "No parameters were available for evaluation in this section."
    passes = sum(1 for i in tested if i.status == "pass")
    warnings = sum(1 for i in tested if i.status == "warning")
    fails = sum(1 for i in tested if i.status == "fail")
    parts: List[str] = []
    if passes:
        parts.append(f"{passes} passed")
    if warnings:
        parts.append(f"{warnings} warning(s)")
    if fails:
        parts.append(f"{fails} failed")
    return f"{len(tested)} criteria evaluated: " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Main assessment function
# ---------------------------------------------------------------------------


def _run_assessment(body: QualityAssessmentRequest) -> QualityAssessmentResponse:
    """Core assessment logic."""
    profile = _PROFILES.get(body.application_type)
    if profile is None:
        # Fall back to general_metal for unknown application types
        profile = _PROFILES["general_metal"]

    # Merge parameters and metrics into a single lookup dict
    values: Dict[str, float] = dict(body.parameters)
    if body.metrics:
        values.update(body.metrics)

    # Evaluate each criterion -------------------------------------------------
    all_results: List[CriterionResult] = []
    for spec in profile.criteria:
        value = values.get(spec.param_key)
        result = _evaluate_criterion(spec, value)
        all_results.append(result)

    # Build sections ----------------------------------------------------------
    process_param_names = {"amplitude", "pressure", "energy", "time"}
    thermal_names = {"temperature_rise", "frequency_deviation"}
    mechanical_names = {"amplitude_uniformity", "stress_safety_factor"}

    process_items = [r for r in all_results if r.name in process_param_names]
    thermal_items = [r for r in all_results if r.name in thermal_names]
    mechanical_items = [r for r in all_results if r.name in mechanical_names]
    # Anything that doesn't fit the categories above
    other_items = [
        r
        for r in all_results
        if r.name not in process_param_names | thermal_names | mechanical_names
    ]

    sections: List[AnalysisSection] = []
    if process_items:
        sections.append(
            AnalysisSection(
                title="Process Parameters",
                items=process_items,
                summary=_section_summary(process_items),
            )
        )
    if thermal_items:
        sections.append(
            AnalysisSection(
                title="Thermal & Frequency Analysis",
                items=thermal_items,
                summary=_section_summary(thermal_items),
            )
        )
    if mechanical_items:
        sections.append(
            AnalysisSection(
                title="Mechanical & Uniformity",
                items=mechanical_items,
                summary=_section_summary(mechanical_items),
            )
        )
    if other_items:
        sections.append(
            AnalysisSection(
                title="Additional Criteria",
                items=other_items,
                summary=_section_summary(other_items),
            )
        )

    # Score & verdict ---------------------------------------------------------
    score = _compute_score(all_results, profile.criteria)

    if score >= 85.0:
        verdict = "pass"
        verdict_desc = (
            f"Welding parameters score {score}/100 and meet {profile.label} "
            "quality requirements. The process is within specification."
        )
    elif score >= 50.0:
        verdict = "warning"
        verdict_desc = (
            f"Welding parameters score {score}/100 for {profile.label}. "
            "Some parameters are outside optimal ranges. Review recommendations "
            "before production use."
        )
    else:
        verdict = "fail"
        verdict_desc = (
            f"Welding parameters score {score}/100 for {profile.label}. "
            "Critical parameters are outside acceptable limits. The weld is "
            "unlikely to meet quality requirements without adjustment."
        )

    # Also force fail if any single criterion is a hard fail
    has_fail = any(r.status == "fail" for r in all_results)
    if has_fail and verdict == "pass":
        verdict = "warning"
        verdict_desc = (
            f"Welding parameters score {score}/100 for {profile.label}. "
            "Overall score is high but at least one criterion failed. "
            "Review the failed criterion before production use."
        )

    recommendations = _build_recommendations(all_results, body)
    app_notes = _build_application_notes(body, all_results)

    return QualityAssessmentResponse(
        overall_score=score,
        overall_verdict=verdict,
        verdict_description=verdict_desc,
        sections=sections,
        recommendations=recommendations,
        application_notes=app_notes,
    )


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/assess", response_model=QualityAssessmentResponse)
async def assess_welding_quality(
    body: QualityAssessmentRequest,
) -> QualityAssessmentResponse:
    """Evaluate welding simulation results against industry quality standards.

    Accepts welding parameters (amplitude, pressure, energy, time, etc.) and
    an application type, then returns a detailed assessment report with
    per-criterion pass/warning/fail status, an overall quality score (0-100),
    actionable recommendations, and application-specific notes.
    """
    logger.info(
        "Quality assessment requested for application_type=%s",
        body.application_type,
    )
    return _run_assessment(body)
