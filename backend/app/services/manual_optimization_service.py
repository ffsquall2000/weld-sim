"""Manual guided optimization service.

Combines real-time parameter suggestions with domain knowledge rules
to guide users through iterative parameter adjustment.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class ManualOptimizationService:
    """Service for manual guided optimization workflow."""

    def __init__(self):
        from backend.app.domain.knowledge_rules import KnowledgeRuleEngine

        self._rule_engine = KnowledgeRuleEngine()
        # Default safety window multipliers
        self._safety_multipliers = {
            "amplitude_um": (0.85, 1.15),
            "pressure_mpa": (0.80, 1.20),
            "energy_j": (0.80, 1.30),
            "time_ms": (0.70, 1.50),
        }

    def create_session(
        self, project_id: str, baseline_params: dict[str, Any]
    ) -> dict:
        """Create a new manual optimization session.

        Args:
            project_id: Associated project ID.
            baseline_params: Starting parameter values from simulation.

        Returns:
            Session dict with id, baseline, iteration_count=0, history=[]
        """
        session_id = str(uuid4())
        return {
            "session_id": session_id,
            "project_id": project_id,
            "strategy": "manual_guided",
            "baseline_params": baseline_params,
            "current_params": dict(baseline_params),
            "iteration_count": 0,
            "history": [],
            "created_at": datetime.utcnow().isoformat(),
            "status": "active",
        }

    def generate_suggestions(
        self,
        current_params: dict[str, float],
        baseline_params: dict[str, float],
        context: dict[str, Any],
        trial_history: list[dict] | None = None,
    ) -> dict:
        """Generate parameter adjustment suggestions.

        Combines:
        1. Deviation analysis (current vs baseline)
        2. Knowledge rule evaluation
        3. Quality trend analysis from trial history

        Args:
            current_params: Current experiment parameters.
            baseline_params: Recommended parameters from simulation.
            context: Material/application context for knowledge rules.
            trial_history: Optional list of previous trial results.

        Returns:
            Dict with suggestions, deviations, safety_status,
            knowledge_recommendations, quality_trend
        """
        suggestions: list[dict] = []
        deviations: dict[str, dict] = {}
        safety_status: dict[str, dict] = {}

        # 1. Compute deviations and safety window status
        for param, baseline_val in baseline_params.items():
            if param not in current_params or baseline_val == 0:
                continue
            current_val = current_params[param]
            deviation_pct = (current_val - baseline_val) / baseline_val * 100
            deviations[param] = {
                "current": current_val,
                "baseline": baseline_val,
                "deviation_pct": round(deviation_pct, 2),
            }

            # Check safety window
            lo_mult, hi_mult = self._safety_multipliers.get(
                param, (0.8, 1.2)
            )
            safe_min = baseline_val * lo_mult
            safe_max = baseline_val * hi_mult
            in_window = safe_min <= current_val <= safe_max
            safety_status[param] = {
                "in_window": in_window,
                "safe_min": round(safe_min, 3),
                "safe_max": round(safe_max, 3),
            }

            # Generate suggestion if out of window or significant deviation
            if not in_window:
                priority = "critical" if abs(deviation_pct) > 30 else "high"
                direction = (
                    "decrease" if current_val > safe_max else "increase"
                )
                target = safe_max if current_val > safe_max else safe_min
                suggestions.append(
                    {
                        "parameter": param,
                        "current_value": current_val,
                        "suggested_value": round(target, 3),
                        "unit": self._get_unit(param),
                        "reason": (
                            f"Parameter is outside safety window "
                            f"({safe_min:.3f} - {safe_max:.3f}). "
                            f"{direction.capitalize()} to bring within "
                            f"safe range."
                        ),
                        "priority": priority,
                        "confidence": 0.9,
                        "direction": direction,
                    }
                )
            elif abs(deviation_pct) > 10:
                suggestions.append(
                    {
                        "parameter": param,
                        "current_value": current_val,
                        "suggested_value": round(baseline_val, 3),
                        "unit": self._get_unit(param),
                        "reason": (
                            f"Deviation of {deviation_pct:.1f}% "
                            f"from recommended value."
                        ),
                        "priority": "medium",
                        "confidence": 0.7,
                        "direction": (
                            "decrease" if deviation_pct > 0 else "increase"
                        ),
                    }
                )

        # 2. Knowledge rule evaluation
        knowledge_recommendations: list[dict] = []
        if context:
            matched_rules = self._rule_engine.evaluate(context)
            for rule in matched_rules:
                knowledge_recommendations.append(
                    {
                        "rule_id": rule["rule_id"],
                        "description": rule["description"],
                        "adjustments": rule.get("adjustments", {}),
                        "recommendations": rule.get("recommendations", []),
                        "priority": rule.get("priority", 0),
                    }
                )
                # Convert rule adjustments to suggestions
                for adj_param, factor in rule.get("adjustments", {}).items():
                    base_param = adj_param.replace("_factor", "").replace(
                        "_offset", ""
                    )
                    if base_param in baseline_params:
                        if adj_param.endswith("_factor"):
                            suggested = baseline_params[base_param] * factor
                        else:
                            suggested = baseline_params[base_param] + factor
                        suggestions.append(
                            {
                                "parameter": base_param,
                                "current_value": current_params.get(
                                    base_param, baseline_params[base_param]
                                ),
                                "suggested_value": round(suggested, 3),
                                "unit": self._get_unit(base_param),
                                "reason": rule["description"],
                                "priority": (
                                    "high"
                                    if rule.get("priority", 0) > 5
                                    else "medium"
                                ),
                                "confidence": 0.8,
                                "source": "knowledge_rule",
                                "rule_id": rule["rule_id"],
                            }
                        )

        # 3. Quality trend analysis
        quality_trend = (
            self._analyze_quality_trend(trial_history)
            if trial_history
            else {
                "trend": "insufficient_data",
                "message": (
                    "Not enough trial data for trend analysis. "
                    "At least 3 trials recommended."
                ),
            }
        )

        # Sort suggestions by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(
            key=lambda s: priority_order.get(s.get("priority", "low"), 3)
        )

        # Deduplicate by parameter (keep highest priority)
        seen_params: set[str] = set()
        unique_suggestions: list[dict] = []
        for s in suggestions:
            if s["parameter"] not in seen_params:
                seen_params.add(s["parameter"])
                unique_suggestions.append(s)

        return {
            "suggestions": unique_suggestions,
            "deviations": deviations,
            "safety_status": safety_status,
            "knowledge_recommendations": knowledge_recommendations,
            "quality_trend": quality_trend,
        }

    def record_iteration(
        self,
        session: dict,
        params: dict[str, float],
        results: dict[str, Any],
        notes: str = "",
    ) -> dict:
        """Record a manual optimization iteration.

        Args:
            session: Current session dict.
            params: Parameters used in this iteration.
            results: Measurement results (e.g., peel_force, resistance,
                     quality_score).
            notes: Optional user notes.

        Returns:
            Updated session dict.
        """
        iteration = {
            "iteration": session["iteration_count"] + 1,
            "params": dict(params),
            "results": dict(results),
            "notes": notes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        session["history"].append(iteration)
        session["iteration_count"] += 1
        session["current_params"] = dict(params)
        return session

    def get_trend_analysis(self, session: dict) -> dict:
        """Analyze parameter and quality trends across iterations.

        Returns:
            Dict with parameter_trends, quality_trends, best_iteration,
            recommendation
        """
        history = session.get("history", [])
        if len(history) < 2:
            return {
                "parameter_trends": {},
                "quality_trends": {},
                "best_iteration": None,
                "recommendation": (
                    "Need at least 2 iterations for trend analysis."
                ),
            }

        # Track parameter trends
        param_trends: dict[str, dict] = {}
        all_params: set[str] = set()
        for entry in history:
            all_params.update(entry.get("params", {}).keys())

        for param in all_params:
            values = [
                e["params"].get(param)
                for e in history
                if param in e.get("params", {})
            ]
            if len(values) >= 2:
                if values[-1] > values[-2]:
                    trend_direction = "increasing"
                elif values[-1] < values[-2]:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                param_trends[param] = {
                    "values": values,
                    "current": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "trend": trend_direction,
                }

        # Track quality trends
        quality_trends: dict[str, dict] = {}
        quality_keys: set[str] = set()
        for entry in history:
            quality_keys.update(entry.get("results", {}).keys())

        best_iteration: Optional[int] = None
        best_score = -float("inf")

        for key in quality_keys:
            values = [
                e["results"].get(key)
                for e in history
                if key in e.get("results", {})
            ]
            if len(values) >= 2:
                if values[-1] > values[-2]:
                    q_trend = "improving"
                elif values[-1] < values[-2]:
                    q_trend = "declining"
                else:
                    q_trend = "stable"
                quality_trends[key] = {
                    "values": values,
                    "current": values[-1],
                    "best": max(values),
                    "trend": q_trend,
                }

        # Find best iteration (by quality_score if available, else first
        # quality metric)
        for i, entry in enumerate(history):
            score = entry.get("results", {}).get(
                "quality_score",
                entry.get("results", {}).get("peel_force_n", 0),
            )
            if score > best_score:
                best_score = score
                best_iteration = i + 1

        # Generate recommendation
        recent_quality = quality_trends.get(
            "quality_score",
            quality_trends.get(next(iter(quality_trends), ""), {}),
        )
        if recent_quality:
            trend_val = recent_quality.get("trend")
            if trend_val == "improving":
                recommendation = (
                    "Quality is improving. "
                    "Continue current adjustment direction."
                )
            elif trend_val == "declining":
                recommendation = (
                    "Quality is declining. Consider reverting to "
                    "iteration {} parameters.".format(best_iteration)
                )
            else:
                recommendation = (
                    "Quality is stable. Consider fine-tuning "
                    "individual parameters."
                )
        else:
            recommendation = (
                "Insufficient quality data for recommendation."
            )

        return {
            "parameter_trends": param_trends,
            "quality_trends": quality_trends,
            "best_iteration": best_iteration,
            "recommendation": recommendation,
        }

    def _analyze_quality_trend(self, trial_history: list[dict]) -> dict:
        """Analyze quality trend from trial history."""
        if not trial_history or len(trial_history) < 2:
            return {
                "trend": "insufficient_data",
                "message": "Need more trial data.",
            }

        scores: list[float] = []
        for trial in trial_history:
            score = (
                trial.get("quality_score")
                or trial.get("peel_force_n")
                or trial.get("weld_strength", 0)
            )
            scores.append(float(score))

        if len(scores) < 3:
            return {
                "trend": "insufficient_data",
                "message": "Need at least 3 data points.",
            }

        # Simple trend: compare last 3 values
        recent = scores[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return {
                "trend": "improving",
                "message": (
                    "Quality is consistently improving "
                    "across recent trials."
                ),
            }
        elif recent[-1] < recent[-2] < recent[-3]:
            return {
                "trend": "declining",
                "message": (
                    "Quality is declining. Review parameter changes."
                ),
            }
        else:
            # Check overall direction
            half = len(scores) // 2
            avg_first_half = sum(scores[:half]) / max(half, 1)
            avg_second_half = sum(scores[half:]) / max(
                len(scores) - half, 1
            )
            if avg_second_half > avg_first_half * 1.05:
                return {
                    "trend": "improving",
                    "message": "Overall quality trend is positive.",
                }
            elif avg_second_half < avg_first_half * 0.95:
                return {
                    "trend": "declining",
                    "message": "Overall quality trend is negative.",
                }
            return {
                "trend": "stable",
                "message": "Quality is relatively stable.",
            }

    @staticmethod
    def _get_unit(param: str) -> str:
        """Get measurement unit for a parameter."""
        units = {
            "amplitude_um": "\u03bcm",
            "pressure_mpa": "MPa",
            "pressure_n": "N",
            "energy_j": "J",
            "time_ms": "ms",
            "frequency_khz": "kHz",
            "temperature_c": "\u00b0C",
        }
        return units.get(param, "")
