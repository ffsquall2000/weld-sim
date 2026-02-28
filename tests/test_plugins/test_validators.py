from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.li_battery.validators import (
    PhysicsValidator, SafetyValidator, ConsistencyValidator, validate_recipe,
)

def _make_recipe(**overrides):
    defaults = {
        "recipe_id": "test", "application": "li_battery_tab",
        "inputs": {"upper_material": "Al", "upper_layers": 40,
                   "upper_thickness_mm": 0.012, "lower_material": "Cu", "weld_area_mm2": 125.0},
        "parameters": {"amplitude_um": 30.0, "pressure_n": 37.5, "pressure_mpa": 5.0,
                        "energy_j": 60.0, "time_ms": 200, "frequency_khz": 20.0},
        "safety_window": {},
        "risk_assessment": {"overweld_risk": "low", "underweld_risk": "low", "perforation_risk": "low"},
    }
    defaults.update(overrides)
    return WeldRecipe(**defaults)

class TestPhysicsValidator:
    def test_valid_recipe_passes(self):
        recipe = _make_recipe()
        v = PhysicsValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

    def test_extreme_amplitude_fails(self):
        recipe = _make_recipe(parameters={
            "amplitude_um": 100.0, "pressure_n": 200.0, "pressure_mpa": 0.3,
            "energy_j": 60.0, "time_ms": 200, "frequency_khz": 20.0})
        v = PhysicsValidator()
        result = v.validate(recipe)
        assert result["status"] in ("warning", "fail")

class TestSafetyValidator:
    def test_valid_recipe_passes(self):
        recipe = _make_recipe()
        v = SafetyValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

    def test_high_risk_warns(self):
        recipe = _make_recipe(risk_assessment={
            "overweld_risk": "high", "underweld_risk": "low", "perforation_risk": "high"})
        v = SafetyValidator()
        result = v.validate(recipe)
        assert result["status"] in ("warning", "fail")

class TestConsistencyValidator:
    def test_consistent_recipe_passes(self):
        recipe = _make_recipe()
        v = ConsistencyValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

class TestValidateRecipe:
    def test_full_validation(self):
        recipe = _make_recipe()
        result = validate_recipe(recipe)
        assert isinstance(result, ValidationResult)
        assert "physics" in result.validators
        assert "safety" in result.validators
        assert "consistency" in result.validators
