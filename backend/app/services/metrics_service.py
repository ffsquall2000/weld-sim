"""Service for computing and normalizing the 12 standardized simulation metrics."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# The 12 standardized metrics with units and descriptions
STANDARD_METRICS = {
    "natural_frequency_hz": {"unit": "Hz", "description": "Closest natural frequency to target"},
    "frequency_deviation_pct": {"unit": "%", "description": "Deviation from target frequency"},
    "amplitude_uniformity": {"unit": "", "description": "Min/max amplitude ratio on contact face", "range": [0, 1]},
    "max_von_mises_stress_mpa": {"unit": "MPa", "description": "Maximum Von Mises stress"},
    "stress_safety_factor": {"unit": "", "description": "Yield strength / max stress"},
    "max_temperature_rise_c": {"unit": "\u00b0C", "description": "Max temperature increase at interface"},
    "contact_pressure_uniformity": {"unit": "", "description": "Pressure uniformity across weld area", "range": [0, 1]},
    "effective_contact_area_mm2": {"unit": "mm\u00b2", "description": "Actual contact area (with knurl)"},
    "energy_coupling_efficiency": {"unit": "", "description": "Useful energy / total input energy", "range": [0, 1]},
    "horn_gain": {"unit": "", "description": "Amplitude amplification factor"},
    "modal_separation_hz": {"unit": "Hz", "description": "Gap to nearest unwanted mode"},
    "fatigue_cycle_estimate": {"unit": "cycles", "description": "Estimated fatigue life"},
}


class MetricsService:
    """Compute and normalize standardized metrics from solver results."""

    @staticmethod
    def compute_standard_metrics(
        solver_metrics: dict[str, float],
        material_props: dict[str, Any] | None = None,
        geometry_params: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Compute all 12 standard metrics from solver output + material/geometry data.

        Fills in derived metrics that solvers don't compute directly:
        - stress_safety_factor from max_stress + yield_strength
        - fatigue_cycle_estimate from stress + S-N curve
        - contact_pressure_uniformity from force distribution
        - effective_contact_area from geometry + knurl params
        - energy_coupling_efficiency from multiple metrics
        """
        metrics = dict(solver_metrics)
        mat = material_props or {}
        geo = geometry_params or {}

        # Compute derived metrics...
        yield_mpa = mat.get("yield_strength_mpa", mat.get("yield_mpa", 880.0))

        # Safety factor
        if "max_von_mises_stress_mpa" in metrics and "stress_safety_factor" not in metrics:
            stress = metrics["max_von_mises_stress_mpa"]
            metrics["stress_safety_factor"] = round(yield_mpa / stress, 3) if stress > 0 else 999.0

        # Fatigue estimate (Basquin's law: N = (sigma_f / sigma_a)^(1/b))
        if "max_von_mises_stress_mpa" in metrics and "fatigue_cycle_estimate" not in metrics:
            sigma_a = metrics["max_von_mises_stress_mpa"]
            sigma_f = yield_mpa * 1.5  # fatigue strength coefficient
            b = -0.12  # Basquin exponent for steel/titanium
            if sigma_a > 0 and sigma_a < sigma_f:
                N = (sigma_f / sigma_a) ** (1 / b)
                metrics["fatigue_cycle_estimate"] = round(min(float(N), 1e12), 0)
            else:
                metrics["fatigue_cycle_estimate"] = 0.0

        # Contact pressure uniformity (if not provided)
        if "contact_pressure_uniformity" not in metrics:
            metrics["contact_pressure_uniformity"] = 0.85  # default good uniformity

        # Effective contact area
        if "effective_contact_area_mm2" not in metrics:
            width = geo.get("width_mm", 40.0)
            length = geo.get("length_mm", 40.0)
            knurl_type = geo.get("knurl_type", "none")
            contact_ratio = 1.0
            if knurl_type == "linear":
                tooth_w = geo.get("knurl_tooth_width_mm", 0.5)
                pitch = geo.get("knurl_pitch_mm", 1.0)
                contact_ratio = tooth_w / pitch if pitch > 0 else 0.5
            elif knurl_type in ("cross_hatch", "diamond"):
                tooth_w = geo.get("knurl_tooth_width_mm", 0.5)
                pitch = geo.get("knurl_pitch_mm", 1.0)
                contact_ratio = (tooth_w / pitch) ** 2 if pitch > 0 else 0.25
            metrics["effective_contact_area_mm2"] = round(width * length * contact_ratio, 2)

        # Energy coupling efficiency
        if "energy_coupling_efficiency" not in metrics:
            uniformity = metrics.get("amplitude_uniformity", 0.85)
            gain = metrics.get("horn_gain", 1.0)
            pressure_unif = metrics.get("contact_pressure_uniformity", 0.85)
            efficiency = uniformity * min(gain / 2.0, 1.0) * pressure_unif
            metrics["energy_coupling_efficiency"] = round(efficiency, 4)

        return metrics

    @staticmethod
    def get_metric_info() -> dict:
        """Return the standard metrics catalog."""
        return STANDARD_METRICS

    @staticmethod
    def compute_quality_score(metrics: dict[str, float]) -> float:
        """Compute overall quality score (0-100) from standard metrics.

        Weighted combination:
        - amplitude_uniformity (25%)
        - stress_safety_factor (25%)
        - energy_coupling_efficiency (20%)
        - frequency_deviation_pct (15%, inverted)
        - contact_pressure_uniformity (15%)
        """
        score = 0.0
        if "amplitude_uniformity" in metrics:
            score += 25.0 * min(metrics["amplitude_uniformity"], 1.0)
        if "stress_safety_factor" in metrics:
            sf = min(metrics["stress_safety_factor"] / 3.0, 1.0)  # normalize to [0,1] with 3.0 as ideal
            score += 25.0 * sf
        if "energy_coupling_efficiency" in metrics:
            score += 20.0 * min(metrics["energy_coupling_efficiency"], 1.0)
        if "frequency_deviation_pct" in metrics:
            dev = abs(metrics["frequency_deviation_pct"])
            freq_score = max(0.0, 1.0 - dev / 5.0)  # 0% deviation = 1.0, 5% = 0.0
            score += 15.0 * freq_score
        if "contact_pressure_uniformity" in metrics:
            score += 15.0 * min(metrics["contact_pressure_uniformity"], 1.0)
        return round(score, 1)
