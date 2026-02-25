from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.calculator import LiBatteryCalculator
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.core.models import WeldInputs, MaterialInfo, WeldRecipe

class TestLiBatteryCalculator:
    @pytest.fixture
    def calculator(self):
        mat_db = MaterialDBPlugin()
        mat_db.activate({})
        return LiBatteryCalculator(material_db=mat_db)

    def test_calculate_tab_welding(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al foil", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0, weld_length_mm=25.0, frequency_khz=20.0, max_power_w=3500)
        recipe = calculator.calculate(inputs)
        assert isinstance(recipe, WeldRecipe)
        assert "amplitude_um" in recipe.parameters
        assert "pressure_n" in recipe.parameters
        assert "energy_j" in recipe.parameters
        assert recipe.parameters["amplitude_um"] > 0
        assert recipe.parameters["pressure_n"] > 0
        assert recipe.parameters["energy_j"] > 0

    def test_calculate_cu_cu(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Cu foil", material_type="Cu", thickness_mm=0.008, layers=50),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0, weld_length_mm=20.0, frequency_khz=20.0, max_power_w=3500)
        recipe = calculator.calculate(inputs)
        assert recipe.parameters["amplitude_um"] > 0

    def test_safety_window_included(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al foil", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0, weld_length_mm=25.0)
        recipe = calculator.calculate(inputs)
        assert "amplitude_um" in recipe.safety_window
        sw = recipe.safety_window["amplitude_um"]
        assert sw[0] < recipe.parameters["amplitude_um"] < sw[1]

    def test_risk_assessment(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0, weld_length_mm=25.0)
        recipe = calculator.calculate(inputs)
        assert "overweld_risk" in recipe.risk_assessment
        assert "underweld_risk" in recipe.risk_assessment
