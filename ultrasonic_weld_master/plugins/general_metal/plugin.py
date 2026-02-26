"""General metal welding plugin for arbitrary material combinations."""
from __future__ import annotations

from typing import Any, Optional

from ultrasonic_weld_master.core.plugin_api import ParameterEnginePlugin, PluginInfo
from ultrasonic_weld_master.core.models import (
    WeldInputs, WeldRecipe, MaterialInfo, ValidationResult,
    SonotrodeInfo, AnvilInfo, CylinderInfo, BoosterInfo,
)
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
            sonotrode=sonotrode, anvil=anvil,
            cylinder=cylinder, booster=booster,
        )
        return self._calculator.calculate(weld_inputs)

    def validate_parameters(self, recipe: WeldRecipe) -> ValidationResult:
        return validate_recipe(recipe)

    def get_supported_applications(self) -> list:
        return ["general_metal"]
