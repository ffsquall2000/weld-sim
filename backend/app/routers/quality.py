"""Quality assessment endpoints with application-specific thresholds.

BUG-8 fix: Provides realistic ultrasonic welding quality thresholds
instead of unreasonably low placeholder values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter

router = APIRouter(tags=["quality"])

# ---------------------------------------------------------------------------
# Application-specific quality profiles with realistic thresholds
# ---------------------------------------------------------------------------

QUALITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "li_battery_tab": {
        "name": "Li-Battery Tab Welding",
        "criteria": {
            "amplitude": {
                "unit": "um",
                "acceptable_range": [15, 50],
                "optimal_range": [20, 40],
                "weight": 1.5,
            },
            "pressure": {
                "unit": "MPa",
                "acceptable_range": [1, 15],
                "optimal_range": [3, 10],
                "weight": 1.0,
            },
            "energy": {
                "unit": "J",
                "acceptable_range": [50, 3000],
                "optimal_range": [200, 2000],
                "weight": 2.0,
            },
            "time": {
                "unit": "ms",
                "acceptable_range": [100, 1500],
                "optimal_range": [200, 1000],
                "weight": 1.0,
            },
            "frequency_deviation": {
                "unit": "%",
                "acceptable_range": [0, 3],
                "optimal_range": [0, 1.5],
                "weight": 1.5,
            },
            "temperature_rise": {
                "unit": "C",
                "acceptable_range": [0, 120],
                "optimal_range": [0, 80],
                "weight": 1.0,
            },
            "amplitude_uniformity": {
                "unit": "%",
                "acceptable_range": [85, 100],
                "optimal_range": [92, 100],
                "weight": 1.5,
            },
            "max_von_mises_stress": {
                "unit": "MPa",
                "acceptable_range": [0, 350],
                "optimal_range": [0, 200],
                "weight": 1.5,
            },
            "fatigue_life": {
                "unit": "cycles",
                "acceptable_range": [1e7, None],
                "optimal_range": [1e9, None],
                "weight": 1.0,
            },
        },
    },
    "li_battery_busbar": {
        "name": "Li-Battery Busbar Welding",
        "criteria": {
            "amplitude": {
                "unit": "um",
                "acceptable_range": [25, 50],
                "optimal_range": [30, 45],
                "weight": 1.5,
            },
            "pressure": {
                "unit": "MPa",
                "acceptable_range": [0.3, 1.0],
                "optimal_range": [0.4, 0.8],
                "weight": 1.0,
            },
            "energy": {
                "unit": "J",
                "acceptable_range": [100, 1000],
                "optimal_range": [200, 800],
                "weight": 2.0,
            },
            "time": {
                "unit": "ms",
                "acceptable_range": [200, 1500],
                "optimal_range": [400, 1200],
                "weight": 1.0,
            },
            "temperature_rise": {
                "unit": "C",
                "acceptable_range": [0, 200],
                "optimal_range": [0, 120],
                "weight": 1.0,
            },
            "max_von_mises_stress": {
                "unit": "MPa",
                "acceptable_range": [0, 400],
                "optimal_range": [0, 250],
                "weight": 1.5,
            },
        },
    },
    "general_metal": {
        "name": "General Metal Welding",
        "criteria": {
            "amplitude": {
                "unit": "um",
                "acceptable_range": [15, 80],
                "optimal_range": [25, 60],
                "weight": 1.5,
            },
            "pressure": {
                "unit": "MPa",
                "acceptable_range": [1, 30],
                "optimal_range": [5, 20],
                "weight": 1.0,
            },
            "energy": {
                "unit": "J",
                "acceptable_range": [100, 5000],
                "optimal_range": [300, 3000],
                "weight": 2.0,
            },
            "frequency_deviation": {
                "unit": "%",
                "acceptable_range": [0, 5],
                "optimal_range": [0, 2],
                "weight": 1.5,
            },
            "max_von_mises_stress": {
                "unit": "MPa",
                "acceptable_range": [0, 500],
                "optimal_range": [0, 300],
                "weight": 1.5,
            },
            "fatigue_life": {
                "unit": "cycles",
                "acceptable_range": [1e6, None],
                "optimal_range": [1e8, None],
                "weight": 1.0,
            },
        },
    },
}


def _evaluate_criterion(
    value: float,
    acceptable_range: list,
    optimal_range: list,
) -> str:
    """Evaluate a single criterion: PASS, MARGINAL, or FAIL."""
    lo_acc = acceptable_range[0] if acceptable_range[0] is not None else float("-inf")
    hi_acc = acceptable_range[1] if acceptable_range[1] is not None else float("inf")
    lo_opt = optimal_range[0] if optimal_range[0] is not None else float("-inf")
    hi_opt = optimal_range[1] if optimal_range[1] is not None else float("inf")

    if value < lo_acc or value > hi_acc:
        return "FAIL"
    if lo_opt <= value <= hi_opt:
        return "PASS"
    return "MARGINAL"


def assess_quality(
    metrics: Dict[str, float],
    application_type: str = "li_battery_tab",
) -> Dict[str, Any]:
    """Assess quality based on application-specific thresholds."""
    profile = QUALITY_PROFILES.get(application_type, QUALITY_PROFILES["general_metal"])
    criteria = profile["criteria"]

    results = []
    total_score = 0.0
    total_weight = 0.0

    for name, spec in criteria.items():
        if name not in metrics:
            results.append({
                "name": name,
                "unit": spec["unit"],
                "value": None,
                "status": "N/A",
                "acceptable_range": spec["acceptable_range"],
                "optimal_range": spec["optimal_range"],
            })
            continue

        value = metrics[name]
        status = _evaluate_criterion(
            value, spec["acceptable_range"], spec["optimal_range"]
        )

        score_map = {"PASS": 1.0, "MARGINAL": 0.5, "FAIL": 0.0}
        weight = spec.get("weight", 1.0)
        total_score += score_map[status] * weight
        total_weight += weight

        results.append({
            "name": name,
            "unit": spec["unit"],
            "value": value,
            "status": status,
            "acceptable_range": spec["acceptable_range"],
            "optimal_range": spec["optimal_range"],
        })

    quality_score = round(total_score / total_weight * 100, 1) if total_weight > 0 else 0
    overall = "PASS" if quality_score >= 80 else ("MARGINAL" if quality_score >= 50 else "FAIL")

    return {
        "application_type": application_type,
        "profile_name": profile["name"],
        "criteria": results,
        "quality_score": quality_score,
        "overall_status": overall,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/quality/profiles")
async def list_quality_profiles() -> Dict[str, Any]:
    """List all available quality assessment profiles."""
    return {
        "profiles": {
            k: {"name": v["name"], "criteria_count": len(v["criteria"])}
            for k, v in QUALITY_PROFILES.items()
        }
    }


@router.get("/quality/profiles/{application_type}")
async def get_quality_profile(application_type: str) -> Dict[str, Any]:
    """Get detailed quality criteria for an application type."""
    profile = QUALITY_PROFILES.get(application_type)
    if not profile:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Unknown application type '{application_type}'. "
                   f"Available: {list(QUALITY_PROFILES.keys())}",
        )
    return profile


@router.post("/quality/assess")
async def assess_weld_quality(
    body: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess weld quality against application-specific thresholds."""
    metrics = body.get("metrics", {})
    app_type = body.get("application_type", "li_battery_tab")
    # Normalize field names: strip unit suffixes so e.g. "amplitude_um" → "amplitude"
    normalized = _normalize_metric_names(metrics)
    return assess_quality(normalized, app_type)


# ---------------------------------------------------------------------------
# Field name normalization
# ---------------------------------------------------------------------------

# Map from common suffixed field names to the short criteria names
_FIELD_ALIASES: Dict[str, str] = {
    "amplitude_um": "amplitude",
    "pressure_mpa": "pressure",
    "energy_j": "energy",
    "time_ms": "time",
    "frequency_deviation_pct": "frequency_deviation",
    "frequency_deviation_percent": "frequency_deviation",
    "temperature_rise_c": "temperature_rise",
    "amplitude_uniformity_pct": "amplitude_uniformity",
    "max_von_mises_stress_mpa": "max_von_mises_stress",
    "von_mises_stress_mpa": "max_von_mises_stress",
    "fatigue_life_cycles": "fatigue_life",
    "fatigue_cycle_estimate": "fatigue_life",
}


def _normalize_metric_names(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize metric field names to match quality criteria keys."""
    result: Dict[str, Any] = {}
    for key, value in metrics.items():
        normalized_key = _FIELD_ALIASES.get(key, key)
        result[normalized_key] = value
    return result
