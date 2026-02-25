from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.core.event_bus import EventBus

class TestLiBatteryPlugin:
    @pytest.fixture
    def plugin(self):
        mat_db = MaterialDBPlugin()
        mat_db.activate({})
        event_bus = EventBus()
        p = LiBatteryPlugin()
        p.activate({"config": None, "event_bus": event_bus, "logger": None, "material_db": mat_db})
        return p

    def test_get_info(self):
        p = LiBatteryPlugin()
        info = p.get_info()
        assert info.name == "li_battery"

    def test_supported_applications(self, plugin):
        apps = plugin.get_supported_applications()
        assert "li_battery_tab" in apps
        assert "li_battery_busbar" in apps

    def test_calculate_and_validate(self, plugin):
        inputs = {
            "application": "li_battery_tab",
            "upper_material_type": "Al", "upper_thickness_mm": 0.012, "upper_layers": 40,
            "lower_material_type": "Cu", "lower_thickness_mm": 0.3,
            "weld_width_mm": 5.0, "weld_length_mm": 25.0,
            "frequency_khz": 20.0, "max_power_w": 3500,
        }
        recipe = plugin.calculate_parameters(inputs)
        assert isinstance(recipe, WeldRecipe)
        result = plugin.validate_parameters(recipe)
        assert isinstance(result, ValidationResult)
