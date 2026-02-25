from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.core.event_bus import EventBus

class TestGeneralMetalPlugin:
    @pytest.fixture
    def plugin(self):
        mat_db = MaterialDBPlugin()
        mat_db.activate({})
        p = GeneralMetalPlugin()
        p.activate({"material_db": mat_db})
        return p

    def test_get_info(self):
        p = GeneralMetalPlugin()
        info = p.get_info()
        assert info.name == "general_metal"

    def test_supported_applications(self, plugin):
        apps = plugin.get_supported_applications()
        assert "general_metal" in apps

    def test_calculate_cu_steel(self, plugin):
        inputs = {
            "application": "general_metal",
            "upper_material_type": "Cu", "upper_thickness_mm": 0.5, "upper_layers": 1,
            "lower_material_type": "Steel", "lower_thickness_mm": 1.0,
            "weld_width_mm": 6.0, "weld_length_mm": 20.0,
            "frequency_khz": 20.0, "max_power_w": 4000,
        }
        recipe = plugin.calculate_parameters(inputs)
        assert isinstance(recipe, WeldRecipe)
        assert recipe.parameters["amplitude_um"] > 0
        assert recipe.parameters["energy_j"] > 0

    def test_calculate_and_validate(self, plugin):
        inputs = {
            "application": "general_metal",
            "upper_material_type": "Al", "upper_thickness_mm": 0.2, "upper_layers": 1,
            "lower_material_type": "Al", "lower_thickness_mm": 0.5,
            "weld_width_mm": 5.0, "weld_length_mm": 15.0,
        }
        recipe = plugin.calculate_parameters(inputs)
        result = plugin.validate_parameters(recipe)
        assert isinstance(result, ValidationResult)
