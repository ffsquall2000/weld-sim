from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.models import (
    WeldRecipe, ValidationResult, ValidationStatus,
    MaterialInfo, SonotrodeInfo, WeldInputs, RiskLevel,
)

class TestWeldRecipe:
    def test_create_recipe(self):
        recipe = WeldRecipe(
            recipe_id="R001", application="li_battery_tab", inputs={},
            parameters={"amplitude_um": 30.0, "pressure_n": 200.0, "energy_j": 50.0, "time_ms": 200},
            safety_window={"amplitude_um": [25.0, 35.0], "pressure_n": [150.0, 250.0]},
        )
        assert recipe.recipe_id == "R001"
        assert recipe.parameters["amplitude_um"] == 30.0

    def test_recipe_to_dict(self):
        recipe = WeldRecipe(recipe_id="R002", application="general_metal", inputs={}, parameters={"amplitude_um": 25.0})
        d = recipe.to_dict()
        assert d["recipe_id"] == "R002"
        assert "created_at" in d

class TestValidationResult:
    def test_pass_result(self):
        result = ValidationResult(status=ValidationStatus.PASS, validators={"physics": {"status": "pass", "messages": []}})
        assert result.is_passed()

    def test_fail_result(self):
        result = ValidationResult(status=ValidationStatus.FAIL, validators={"physics": {"status": "fail", "messages": ["Power density too high"]}})
        assert not result.is_passed()

class TestMaterialInfo:
    def test_create_material(self):
        mat = MaterialInfo(name="Copper C11000", material_type="Cu", thickness_mm=0.2, layers=1)
        assert mat.material_type == "Cu"
        assert mat.total_thickness_mm == 0.2

    def test_multi_layer_thickness(self):
        mat = MaterialInfo(name="Al foil", material_type="Al", thickness_mm=0.012, layers=40)
        assert abs(mat.total_thickness_mm - 0.48) < 1e-6

class TestWeldInputs:
    def test_create_inputs(self):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0, weld_length_mm=25.0, frequency_khz=20.0, max_power_w=3000,
        )
        assert inputs.weld_area_mm2 == 125.0
