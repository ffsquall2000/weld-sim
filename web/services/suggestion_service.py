"""Real-time welding parameter suggestion engine.

Compares actual experiment parameters against simulation recommendations,
detects deviations and quality trends, and generates prioritized suggestions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Suggestion:
    """A single parameter adjustment suggestion."""
    parameter: str
    current_value: float
    suggested_value: float
    unit: str
    reason: str
    priority: str  # critical | high | medium | low
    confidence: float  # 0-1


class SuggestionService:
    """Generate parameter adjustment suggestions during welding experiments."""

    # Safety window multipliers (parameter -> (lower_ratio, upper_ratio))
    SAFETY_WINDOWS = {
        "amplitude_um": (0.85, 1.15),
        "pressure_mpa": (0.80, 1.20),
        "energy_j": (0.80, 1.30),
        "time_ms": (0.70, 1.50),
    }

    PARAMETER_UNITS = {
        "amplitude_um": "\u03bcm",
        "pressure_mpa": "MPa",
        "pressure_n": "N",
        "energy_j": "J",
        "time_ms": "ms",
        "frequency_khz": "kHz",
        "power_w": "W",
    }

    def generate_suggestions(
        self,
        experiment_params: dict,
        simulation_recipe: dict,
        trial_history: Optional[list[dict]] = None,
    ) -> dict:
        """Generate suggestions by comparing experiment vs simulation.

        Args:
            experiment_params: Current welding parameters from the experiment.
                Keys: amplitude_um, pressure_mpa, energy_j, time_ms, etc.
            simulation_recipe: Recommended parameters from simulation.
                Contains 'parameters' dict and 'safety_window' dict.
            trial_history: Optional list of previous trial results.
                Each dict has: trial_number, parameters, quality metrics.

        Returns:
            Dict with:
            - suggestions: list of Suggestion dicts sorted by priority
            - deviations: dict of parameter deviations from recommendation
            - safety_status: dict of which parameters are within safety window
            - quality_trend: analysis of quality over trials (if history provided)
        """
        sim_params = simulation_recipe.get("parameters", {})
        safety_window = simulation_recipe.get("safety_window", {})

        # 1. Compute deviations
        deviations = self._compute_deviations(experiment_params, sim_params)

        # 2. Check safety windows
        safety_status = self._check_safety_windows(
            experiment_params, sim_params, safety_window
        )

        # 3. Analyze quality trend (if history available)
        quality_trend = {}
        if trial_history and len(trial_history) >= 2:
            quality_trend = self._analyze_quality_trend(trial_history)

        # 4. Generate suggestions
        suggestions = []

        # Out-of-range parameter suggestions
        for param, status in safety_status.items():
            if not status["within_window"]:
                suggestion = self._parameter_adjustment_suggestion(
                    param,
                    experiment_params.get(param, 0),
                    sim_params.get(param, 0),
                    status,
                )
                if suggestion:
                    suggestions.append(suggestion)

        # Significant deviation suggestions (even within window)
        for param, dev in deviations.items():
            if abs(dev["percent"]) > 10 and param not in [
                s["parameter"] for s in suggestions
            ]:
                suggestions.append({
                    "parameter": param,
                    "current_value": dev["actual"],
                    "suggested_value": dev["recommended"],
                    "unit": self.PARAMETER_UNITS.get(param, ""),
                    "reason": (
                        f"Parameter deviates {dev['percent']:.1f}% "
                        "from simulation recommendation"
                    ),
                    "priority": "medium",
                    "confidence": 0.7,
                })

        # Quality trend suggestions
        if quality_trend.get("declining"):
            suggestions.extend(
                self._quality_trend_suggestions(quality_trend, sim_params)
            )

        # Risk-based suggestions from simulation
        risk = simulation_recipe.get("risk_assessment", {})
        suggestions.extend(
            self._risk_based_suggestions(risk, experiment_params, sim_params)
        )

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: priority_order.get(s["priority"], 99))

        return {
            "suggestions": suggestions,
            "deviations": deviations,
            "safety_status": safety_status,
            "quality_trend": quality_trend,
        }

    def _compute_deviations(self, actual: dict, recommended: dict) -> dict:
        """Compute percentage deviations from recommended values."""
        deviations = {}
        for param in ["amplitude_um", "pressure_mpa", "energy_j", "time_ms"]:
            act_val = actual.get(param)
            rec_val = recommended.get(param)
            if act_val is not None and rec_val is not None and rec_val > 0:
                pct = (act_val - rec_val) / rec_val * 100
                deviations[param] = {
                    "actual": act_val,
                    "recommended": rec_val,
                    "difference": round(act_val - rec_val, 3),
                    "percent": round(pct, 1),
                }
        return deviations

    def _check_safety_windows(
        self, actual: dict, recommended: dict, safety_window: dict
    ) -> dict:
        """Check if parameters are within safety windows."""
        status = {}
        for param, (lo_ratio, hi_ratio) in self.SAFETY_WINDOWS.items():
            rec_val = recommended.get(param)
            act_val = actual.get(param)
            if rec_val is None or act_val is None:
                continue

            # Use explicit safety window if available
            if param in safety_window:
                lo, hi = safety_window[param]
            else:
                lo = rec_val * lo_ratio
                hi = rec_val * hi_ratio

            within = lo <= act_val <= hi
            status[param] = {
                "within_window": within,
                "current": act_val,
                "window_low": round(lo, 2),
                "window_high": round(hi, 2),
                "deviation_from_center": (
                    round((act_val - rec_val) / rec_val * 100, 1)
                    if rec_val > 0
                    else 0
                ),
            }
        return status

    def _parameter_adjustment_suggestion(
        self,
        param: str,
        current: float,
        recommended: float,
        status: dict,
    ) -> Optional[dict]:
        """Generate adjustment suggestion for out-of-range parameter."""
        if recommended <= 0:
            return None

        deviation_pct = abs((current - recommended) / recommended * 100)
        if deviation_pct > 30:
            priority = "critical"
            confidence = 0.9
        elif deviation_pct > 15:
            priority = "high"
            confidence = 0.8
        else:
            priority = "medium"
            confidence = 0.7

        direction = "increase" if current < recommended else "decrease"

        return {
            "parameter": param,
            "current_value": round(current, 3),
            "suggested_value": round(recommended, 3),
            "unit": self.PARAMETER_UNITS.get(param, ""),
            "reason": (
                f"Parameter is outside safety window "
                f"[{status['window_low']}, {status['window_high']}]. "
                f"Recommend to {direction} towards {recommended:.1f}."
            ),
            "priority": priority,
            "confidence": confidence,
        }

    def _analyze_quality_trend(self, trials: list[dict]) -> dict:
        """Analyze quality metrics over trial history."""
        if len(trials) < 2:
            return {"declining": False, "stable": True}

        # Extract quality scores (if available)
        scores = []
        for t in trials:
            q = t.get("quality", t.get("quality_score", t.get("peel_force_n")))
            if q is not None:
                scores.append(float(q))

        if len(scores) < 2:
            return {
                "declining": False,
                "stable": True,
                "reason": "Insufficient quality data",
            }

        # Simple trend: compare last 3 vs first 3 (or available)
        n = min(3, len(scores) // 2)
        early = sum(scores[:n]) / n
        recent = sum(scores[-n:]) / n

        if early > 0:
            trend_pct = (recent - early) / early * 100
        else:
            trend_pct = 0

        declining = trend_pct < -10
        improving = trend_pct > 10
        stable = not declining and not improving

        return {
            "declining": declining,
            "improving": improving,
            "stable": stable,
            "trend_percent": round(trend_pct, 1),
            "trial_count": len(trials),
            "quality_scores": scores[-5:],  # last 5
        }

    def _quality_trend_suggestions(
        self, trend: dict, sim_params: dict
    ) -> list[dict]:
        """Generate suggestions based on quality trend."""
        suggestions = []
        if trend.get("declining"):
            suggestions.append({
                "parameter": "energy_j",
                "current_value": 0,
                "suggested_value": sim_params.get("energy_j", 0),
                "unit": "J",
                "reason": (
                    f"Quality declining by "
                    f"{abs(trend.get('trend_percent', 0)):.1f}% "
                    "over recent trials. Review energy input and horn condition."
                ),
                "priority": "high",
                "confidence": 0.6,
            })
            suggestions.append({
                "parameter": "amplitude_um",
                "current_value": 0,
                "suggested_value": sim_params.get("amplitude_um", 0),
                "unit": "\u03bcm",
                "reason": (
                    "Quality decline may indicate horn wear. "
                    "Check amplitude consistency."
                ),
                "priority": "medium",
                "confidence": 0.5,
            })
        return suggestions

    def _risk_based_suggestions(
        self, risk: dict, actual: dict, sim_params: dict
    ) -> list[dict]:
        """Generate suggestions based on simulation risk assessment."""
        suggestions = []

        if risk.get("overweld_risk") in ("high", "critical"):
            suggestions.append({
                "parameter": "energy_j",
                "current_value": actual.get("energy_j", 0),
                "suggested_value": sim_params.get("energy_j", 0) * 0.85,
                "unit": "J",
                "reason": (
                    "High overweld risk detected. "
                    "Consider reducing energy by 15%."
                ),
                "priority": "high",
                "confidence": 0.8,
            })

        if risk.get("underweld_risk") in ("high", "critical"):
            suggestions.append({
                "parameter": "amplitude_um",
                "current_value": actual.get("amplitude_um", 0),
                "suggested_value": sim_params.get("amplitude_um", 0) * 1.1,
                "unit": "\u03bcm",
                "reason": (
                    "High underweld risk. "
                    "Consider increasing amplitude by 10%."
                ),
                "priority": "high",
                "confidence": 0.7,
            })

        if risk.get("edge_damage_risk") in ("high", "critical"):
            suggestions.append({
                "parameter": "pressure_mpa",
                "current_value": actual.get("pressure_mpa", 0),
                "suggested_value": sim_params.get("pressure_mpa", 0) * 0.9,
                "unit": "MPa",
                "reason": (
                    "Edge damage risk is high. "
                    "Reduce pressure and check horn chamfer."
                ),
                "priority": "high",
                "confidence": 0.7,
            })

        if risk.get("perforation_risk") in ("high", "critical"):
            suggestions.append({
                "parameter": "pressure_mpa",
                "current_value": actual.get("pressure_mpa", 0),
                "suggested_value": sim_params.get("pressure_mpa", 0) * 0.85,
                "unit": "MPa",
                "reason": (
                    "Perforation risk detected. "
                    "Reduce pressure to prevent material breakthrough."
                ),
                "priority": "critical",
                "confidence": 0.85,
            })

        return suggestions
