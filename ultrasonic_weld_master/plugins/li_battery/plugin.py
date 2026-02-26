"""Li-battery welding parameter engine plugin."""
from __future__ import annotations

from typing import Any

from ultrasonic_weld_master.core.plugin_api import ParameterEnginePlugin, PluginInfo
from ultrasonic_weld_master.core.models import (
    WeldInputs, WeldRecipe, MaterialInfo, ValidationResult,
    SonotrodeInfo, AnvilInfo, CylinderInfo, BoosterInfo,
)
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

    def _build_sonotrode(self, inputs: dict) -> SonotrodeInfo | None:
        ht = inputs.get("horn_type")
        if ht is None:
            return None
        return SonotrodeInfo(
            sonotrode_type=ht,
            horn_gain=inputs.get("horn_gain", 1.0),
            mode=inputs.get("horn_mode", "longitudinal"),
            resonant_freq_khz=inputs.get("horn_resonant_freq_khz", 20.0),
            knurl_type=inputs.get("knurl_type", "linear"),
            knurl_pitch_mm=inputs.get("knurl_pitch_mm", 1.0),
            knurl_tooth_width_mm=inputs.get("knurl_tooth_width_mm", 0.5),
            knurl_depth_mm=inputs.get("knurl_depth_mm", 0.3),
            knurl_direction=inputs.get("knurl_direction", "perpendicular"),
            custom_contact_ratio=inputs.get("knurl_custom_contact_ratio", 0.5),
            contact_width_mm=inputs.get("weld_width_mm", 5.0),
            contact_length_mm=inputs.get("weld_length_mm", 25.0),
            chamfer_radius_mm=inputs.get("chamfer_radius_mm", 0.0),
            chamfer_angle_deg=inputs.get("chamfer_angle_deg", 45.0),
            edge_treatment=inputs.get("edge_treatment", "none"),
        )

    def calculate_parameters(self, inputs: dict) -> WeldRecipe:
        sonotrode = self._build_sonotrode(inputs)
        anvil = AnvilInfo(
            anvil_type=inputs.get("anvil_type", "fixed_flat"),
            resonant_freq_khz=inputs.get("anvil_resonant_freq_khz", 0.0),
        ) if inputs.get("anvil_type") else None
        cylinder = CylinderInfo(
            bore_mm=inputs.get("cylinder_bore_mm", 50.0),
            min_air_bar=inputs.get("cylinder_min_air_bar", 1.0),
            max_air_bar=inputs.get("cylinder_max_air_bar", 6.0),
            efficiency=inputs.get("cylinder_efficiency", 0.90),
        ) if inputs.get("cylinder_bore_mm") else None
        booster = BoosterInfo(
            gain_ratio=inputs.get("booster_gain", 1.5),
            rated_amplitude_um=inputs.get("booster_rated_amplitude_um", 70.0),
        ) if inputs.get("booster_gain") else None

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
            max_power_w=inputs.get("max_power_w", 3500),
            sonotrode=sonotrode, anvil=anvil,
            cylinder=cylinder, booster=booster)
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
