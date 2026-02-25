"""Li-battery welding parameter engine plugin."""
from __future__ import annotations

from typing import Any

from ultrasonic_weld_master.core.plugin_api import ParameterEnginePlugin, PluginInfo
from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe, MaterialInfo, ValidationResult
from ultrasonic_weld_master.plugins.li_battery.calculator import LiBatteryCalculator
from ultrasonic_weld_master.plugins.li_battery.validators import validate_recipe


class LiBatteryPlugin(ParameterEnginePlugin):
    def __init__(self):
        self._calculator = None
        self._material_db = None
        self._event_bus = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="li_battery", version="1.0.0",
            description="Lithium battery ultrasonic welding parameter engine",
            author="UltrasonicWeldMaster", dependencies=["material_db"])

    def activate(self, context: Any) -> None:
        self._material_db = context.get("material_db") if isinstance(context, dict) else None
        self._event_bus = context.get("event_bus") if isinstance(context, dict) else None
        if self._material_db:
            self._calculator = LiBatteryCalculator(material_db=self._material_db)

    def deactivate(self) -> None:
        self._calculator = None

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "application": {"type": "string", "enum": self.get_supported_applications()},
                "upper_material_type": {"type": "string"},
                "upper_thickness_mm": {"type": "number", "minimum": 0.001},
                "upper_layers": {"type": "integer", "minimum": 1, "maximum": 200},
                "lower_material_type": {"type": "string"},
                "lower_thickness_mm": {"type": "number", "minimum": 0.01},
                "weld_width_mm": {"type": "number", "minimum": 1},
                "weld_length_mm": {"type": "number", "minimum": 1},
                "frequency_khz": {"type": "number", "default": 20},
                "max_power_w": {"type": "number", "default": 3500},
            },
            "required": ["application", "upper_material_type", "upper_thickness_mm",
                         "upper_layers", "lower_material_type", "lower_thickness_mm",
                         "weld_width_mm", "weld_length_mm"],
        }

    def calculate_parameters(self, inputs: dict) -> WeldRecipe:
        weld_inputs = WeldInputs(
            application=inputs["application"],
            upper_material=MaterialInfo(
                name=inputs["upper_material_type"], material_type=inputs["upper_material_type"],
                thickness_mm=inputs["upper_thickness_mm"], layers=inputs.get("upper_layers", 1)),
            lower_material=MaterialInfo(
                name=inputs["lower_material_type"], material_type=inputs["lower_material_type"],
                thickness_mm=inputs["lower_thickness_mm"], layers=1),
            weld_width_mm=inputs["weld_width_mm"], weld_length_mm=inputs["weld_length_mm"],
            frequency_khz=inputs.get("frequency_khz", 20.0),
            max_power_w=inputs.get("max_power_w", 3500))
        recipe = self._calculator.calculate(weld_inputs)
        if self._event_bus:
            self._event_bus.emit("calculation.completed", recipe.to_dict())
        return recipe

    def validate_parameters(self, recipe: WeldRecipe) -> ValidationResult:
        result = validate_recipe(recipe)
        if self._event_bus:
            self._event_bus.emit("validation.completed", result.to_dict())
        return result

    def get_supported_applications(self) -> list:
        return ["li_battery_tab", "li_battery_busbar", "li_battery_collector"]
