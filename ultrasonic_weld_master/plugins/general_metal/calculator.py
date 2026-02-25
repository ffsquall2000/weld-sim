"""General metal welding parameter calculator using simplified physics model."""
from __future__ import annotations

import math
import uuid
from typing import Any

from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

# Conservative base parameters for arbitrary material combinations
BASE_DEFAULTS = {
    "amplitude_um": 32, "pressure_mpa": 0.35,
    "energy_density_j_mm2": 0.55, "time_ms": 250,
}


class GeneralMetalCalculator:
    def __init__(self, material_db: Any):
        self._material_db = material_db
        self._physics = PhysicsModel()

    def calculate(self, inputs: WeldInputs) -> WeldRecipe:
        upper_mat = self._material_db.get_material(inputs.upper_material.material_type) or {}
        lower_mat = self._material_db.get_material(inputs.lower_material.material_type) or {}
        combo = self._material_db.get_combination_properties(
            inputs.upper_material.material_type, inputs.lower_material.material_type)

        z1 = float(upper_mat.get("acoustic_impedance", 20e6))
        z2 = float(lower_mat.get("acoustic_impedance", 40e6))
        impedance_eff = self._physics.acoustic_impedance_match(z1, z2)

        base = dict(BASE_DEFAULTS)
        friction = combo.get("friction_coefficient", 0.3)

        # Hardness-based amplitude correction
        upper_hv = upper_mat.get("hardness_hv", 50)
        lower_hv = lower_mat.get("hardness_hv", 50)
        hardness_factor = max(upper_hv, lower_hv) / 50.0
        base["amplitude_um"] *= min(max(hardness_factor, 0.8), 1.5)

        # Impedance mismatch correction
        if impedance_eff < 0.8:
            base["amplitude_um"] *= 1 + 0.15 * (1 - impedance_eff)
            base["energy_density_j_mm2"] *= 1 + 0.1 * (1 - impedance_eff)

        # Layer count correction
        n_layers = inputs.upper_material.layers
        if n_layers > 1:
            layer_factor = 1.0 + 0.01 * (n_layers - 1)
            base["amplitude_um"] *= min(layer_factor, 1.5)
            base["energy_density_j_mm2"] *= min(layer_factor, 1.8)
            base["time_ms"] *= min(layer_factor, 1.6)

        # Area correction
        area = inputs.weld_area_mm2
        if area > 100:
            base["pressure_mpa"] *= min(math.sqrt(area / 100), 1.4)

        # Clamp values
        base["amplitude_um"] = max(15, min(base["amplitude_um"], 65))
        base["pressure_mpa"] = max(0.1, min(base["pressure_mpa"], 0.9))
        base["time_ms"] = max(80, min(base["time_ms"], 1000))

        # Build output
        energy_j = base["energy_density_j_mm2"] * area
        pressure_n = base["pressure_mpa"] * area

        parameters = {
            "amplitude_um": round(base["amplitude_um"], 1),
            "pressure_n": round(pressure_n, 0),
            "pressure_mpa": round(base["pressure_mpa"], 3),
            "energy_j": round(energy_j, 1),
            "time_ms": round(base["time_ms"]),
            "frequency_khz": inputs.frequency_khz,
            "control_mode": "energy",
        }

        amp = parameters["amplitude_um"]
        safety_window = {
            "amplitude_um": [round(amp * 0.8, 1), round(amp * 1.2, 1)],
            "pressure_n": [round(pressure_n * 0.75, 0), round(pressure_n * 1.25, 0)],
            "energy_j": [round(energy_j * 0.75, 1), round(energy_j * 1.35, 1)],
            "time_ms": [round(base["time_ms"] * 0.65), round(base["time_ms"] * 1.6)],
        }

        risk = {"overweld_risk": "medium", "underweld_risk": "medium", "perforation_risk": "low"}
        recommendations = [
            "General metal welding: start trial at 75%% of recommended energy.",
            "Increase energy in 5%% steps until acceptable bond strength.",
        ]

        return WeldRecipe(
            recipe_id=uuid.uuid4().hex[:12], application=inputs.application,
            inputs={"upper_material": inputs.upper_material.material_type,
                    "upper_thickness_mm": inputs.upper_material.thickness_mm,
                    "upper_layers": inputs.upper_material.layers,
                    "lower_material": inputs.lower_material.material_type,
                    "lower_thickness_mm": inputs.lower_material.thickness_mm,
                    "weld_area_mm2": area, "frequency_khz": inputs.frequency_khz},
            parameters=parameters, safety_window=safety_window,
            risk_assessment=risk, recommendations=recommendations)
