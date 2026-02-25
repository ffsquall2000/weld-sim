"""General metal welding plugin for arbitrary material combinations."""
from __future__ import annotations

from typing import Any, Optional

from ultrasonic_weld_master.core.plugin_api import ParameterEnginePlugin, PluginInfo
from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe, MaterialInfo, ValidationResult
from ultrasonic_weld_master.plugins.general_metal.calculator import GeneralMetalCalculator
from ultrasonic_weld_master.plugins.li_battery.validators import validate_recipe


class GeneralMetalPlugin(ParameterEnginePlugin):
    def __init__(self):
        self._calculator: Optional[GeneralMetalCalculator] = None
        self._material_db = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="general_metal", version="1.0.0",
            description="General metal ultrasonic welding parameter engine",
            author="UltrasonicWeldMaster", dependencies=["material_db"],
        )

    def activate(self, context: Any) -> None:
        self._material_db = context.get("material_db")
        self._calculator = GeneralMetalCalculator(material_db=self._material_db)

    def deactivate(self) -> None:
        self._calculator = None

    def get_input_schema(self) -> dict:
        return {
            "application": {"type": "string"},
            "upper_material_type": {"type": "string"},
            "upper_thickness_mm": {"type": "float"},
            "upper_layers": {"type": "int", "default": 1},
            "lower_material_type": {"type": "string"},
            "lower_thickness_mm": {"type": "float"},
            "weld_width_mm": {"type": "float"},
            "weld_length_mm": {"type": "float"},
            "frequency_khz": {"type": "float", "default": 20.0},
            "max_power_w": {"type": "float", "default": 3500},
        }

    def calculate_parameters(self, inputs: dict) -> WeldRecipe:
        weld_inputs = WeldInputs(
            application=inputs.get("application", "general_metal"),
            upper_material=MaterialInfo(
                name=inputs.get("upper_material_type", ""),
                material_type=inputs.get("upper_material_type", ""),
                thickness_mm=inputs.get("upper_thickness_mm", 0.1),
                layers=inputs.get("upper_layers", 1),
            ),
            lower_material=MaterialInfo(
                name=inputs.get("lower_material_type", ""),
                material_type=inputs.get("lower_material_type", ""),
                thickness_mm=inputs.get("lower_thickness_mm", 0.5),
                layers=1,
            ),
            weld_width_mm=inputs.get("weld_width_mm", 5.0),
            weld_length_mm=inputs.get("weld_length_mm", 20.0),
            frequency_khz=inputs.get("frequency_khz", 20.0),
            max_power_w=inputs.get("max_power_w", 3500),
        )
        return self._calculator.calculate(weld_inputs)

    def validate_parameters(self, recipe: WeldRecipe) -> ValidationResult:
        return validate_recipe(recipe)

    def get_supported_applications(self) -> list:
        return ["general_metal"]
