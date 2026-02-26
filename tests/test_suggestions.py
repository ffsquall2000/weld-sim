"""Tests for real-time suggestion engine."""
import pytest
from web.services.suggestion_service import SuggestionService


class TestSuggestionService:
    def setup_method(self):
        self.svc = SuggestionService()

    def test_within_window_no_critical_suggestions(self):
        """Parameters within safety window -> no critical/high suggestions."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
                "safety_window": {"amplitude_um": [25, 35], "pressure_mpa": [1.6, 2.4], "energy_j": [80, 130], "time_ms": [140, 300]},
                "risk_assessment": {},
            },
        )
        critical = [s for s in result["suggestions"] if s["priority"] == "critical"]
        assert len(critical) == 0

    def test_out_of_range_generates_suggestions(self):
        """Parameters outside window -> suggestions generated."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 50, "pressure_mpa": 5.0, "energy_j": 200, "time_ms": 500},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
                "safety_window": {"amplitude_um": [25, 35], "pressure_mpa": [1.6, 2.4], "energy_j": [80, 130], "time_ms": [140, 300]},
                "risk_assessment": {},
            },
        )
        assert len(result["suggestions"]) > 0

    def test_deviations_computed(self):
        """Deviations should be computed for each parameter."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 36, "pressure_mpa": 2.0},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0},
                "safety_window": {},
                "risk_assessment": {},
            },
        )
        assert "amplitude_um" in result["deviations"]
        assert result["deviations"]["amplitude_um"]["percent"] == pytest.approx(20.0, abs=0.5)

    def test_risk_based_suggestions(self):
        """High risk -> appropriate suggestions."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
                "safety_window": {},
                "risk_assessment": {"overweld_risk": "high", "edge_damage_risk": "critical"},
            },
        )
        assert len(result["suggestions"]) > 0

    def test_suggestions_sorted_by_priority(self):
        """Suggestions should be sorted: critical first, then high, medium, low."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 50, "pressure_mpa": 5.0, "energy_j": 200, "time_ms": 500},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
                "safety_window": {},
                "risk_assessment": {"overweld_risk": "high", "perforation_risk": "critical"},
            },
        )
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        priorities = [priority_order.get(s["priority"], 99) for s in result["suggestions"]]
        assert priorities == sorted(priorities)

    def test_quality_trend_declining(self):
        """Declining quality trend -> high priority suggestions."""
        result = self.svc.generate_suggestions(
            experiment_params={"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
            simulation_recipe={
                "parameters": {"amplitude_um": 30, "pressure_mpa": 2.0, "energy_j": 100, "time_ms": 200},
                "safety_window": {},
                "risk_assessment": {},
            },
            trial_history=[
                {"quality_score": 95}, {"quality_score": 90}, {"quality_score": 85},
                {"quality_score": 70}, {"quality_score": 60},
            ],
        )
        assert result["quality_trend"]["declining"] == True
