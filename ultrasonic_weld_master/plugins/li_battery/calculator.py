"""Li-battery welding parameter calculator with 3-layer model."""
from __future__ import annotations

import math
import uuid
from typing import Any, Optional

from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

BASE_PARAMS = {
    "Al-Cu": {"amplitude_um": 30, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 200},
    "Al-Al": {"amplitude_um": 28, "pressure_mpa": 0.25, "energy_density_j_mm2": 0.4, "time_ms": 180},
    "Cu-Cu": {"amplitude_um": 35, "pressure_mpa": 0.35, "energy_density_j_mm2": 0.6, "time_ms": 250},
    "Cu-Ni": {"amplitude_um": 38, "pressure_mpa": 0.40, "energy_density_j_mm2": 0.7, "time_ms": 280},
    "Cu-Al": {"amplitude_um": 30, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 200},
    "Ni-Cu": {"amplitude_um": 38, "pressure_mpa": 0.40, "energy_density_j_mm2": 0.7, "time_ms": 280},
}
DEFAULT_BASE = {"amplitude_um": 32, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 220}


class LiBatteryCalculator:
    def __init__(self, material_db: Any):
        self._material_db = material_db
        self._physics = PhysicsModel()

    def calculate(self, inputs: WeldInputs) -> WeldRecipe:
        upper_mat = self._material_db.get_material(inputs.upper_material.material_type) or {}
        lower_mat = self._material_db.get_material(inputs.lower_material.material_type) or {}
        combo_props = self._material_db.get_combination_properties(
            inputs.upper_material.material_type, inputs.lower_material.material_type)

        physics_data = self._layer1_physics(inputs, upper_mat, lower_mat, combo_props)
        corrected = self._layer2_empirical(inputs, physics_data, combo_props)
        return self._layer3_output(inputs, corrected, physics_data)

    def _layer1_physics(self, inputs, upper_mat, lower_mat, combo):
        z1 = upper_mat.get("acoustic_impedance", 17e6)
        z2 = lower_mat.get("acoustic_impedance", 41e6)
        impedance_efficiency = self._physics.acoustic_impedance_match(z1, z2)
        friction_coeff = combo.get("friction_coefficient", 0.3)
        combo_key = f"{inputs.upper_material.material_type}-{inputs.lower_material.material_type}"
        base = BASE_PARAMS.get(combo_key, DEFAULT_BASE)

        pd = self._physics.interface_power_density(
            frequency_hz=inputs.frequency_khz * 1000, amplitude_um=base["amplitude_um"],
            pressure_mpa=base["pressure_mpa"], friction_coeff=friction_coeff,
            contact_area_mm2=inputs.weld_area_mm2)

        energy_ratios = self._physics.multilayer_energy_attenuation(
            n_layers=inputs.upper_material.layers, material_impedance=z1,
            layer_thickness_mm=inputs.upper_material.thickness_mm)
        bottom_energy_ratio = energy_ratios[-1] if energy_ratios else 1.0

        delta_t = self._physics.interface_temperature_rise(
            power_density_w_mm2=pd, weld_time_s=base["time_ms"] / 1000.0,
            thermal_conductivity_1=upper_mat.get("thermal_conductivity", 200),
            thermal_conductivity_2=lower_mat.get("thermal_conductivity", 200),
            density_1=upper_mat.get("density_kg_m3", 5000),
            density_2=lower_mat.get("density_kg_m3", 5000),
            specific_heat_1=upper_mat.get("specific_heat_j_kg_k", 500),
            specific_heat_2=lower_mat.get("specific_heat_j_kg_k", 500))

        return {"impedance_efficiency": impedance_efficiency, "power_density_w_mm2": pd,
                "bottom_energy_ratio": bottom_energy_ratio, "interface_temp_rise_c": delta_t,
                "base_params": base, "friction_coeff": friction_coeff}

    def _layer2_empirical(self, inputs, physics, combo):
        base = dict(physics["base_params"])
        n_layers = inputs.upper_material.layers
        layer_factor = 1.0 + 0.008 * max(n_layers - 1, 0)
        base["amplitude_um"] *= min(layer_factor, 1.6)
        base["energy_density_j_mm2"] *= min(layer_factor, 2.0)
        base["time_ms"] *= min(layer_factor, 1.8)

        area = inputs.weld_area_mm2
        if area > 100:
            base["pressure_mpa"] *= min(math.sqrt(area / 100), 1.5)

        eff = physics["impedance_efficiency"]
        if eff < 0.8:
            base["amplitude_um"] *= 1 + 0.2 * (1 - eff)
            base["energy_density_j_mm2"] *= 1 + 0.15 * (1 - eff)

        if combo.get("imc_risk") == "high":
            max_temp = combo.get("max_interface_temp_c", 300)
            if physics["interface_temp_rise_c"] > max_temp * 0.7:
                base["energy_density_j_mm2"] *= 0.85
                base["time_ms"] *= 0.9

        base["amplitude_um"] = max(15, min(base["amplitude_um"], 60))
        base["pressure_mpa"] = max(0.1, min(base["pressure_mpa"], 0.8))
        base["time_ms"] = max(50, min(base["time_ms"], 800))
        return base

    def _layer3_output(self, inputs, params, physics):
        area = inputs.weld_area_mm2
        energy_j = params["energy_density_j_mm2"] * area
        pressure_n = params["pressure_mpa"] * area

        parameters = {
            "amplitude_um": round(params["amplitude_um"], 1),
            "pressure_n": round(pressure_n, 0),
            "pressure_mpa": round(params["pressure_mpa"], 3),
            "energy_j": round(energy_j, 1),
            "time_ms": round(params["time_ms"]),
            "frequency_khz": inputs.frequency_khz,
            "control_mode": "energy",
        }

        amp = parameters["amplitude_um"]
        safety_window = {
            "amplitude_um": [round(amp * 0.85, 1), round(amp * 1.15, 1)],
            "pressure_n": [round(pressure_n * 0.8, 0), round(pressure_n * 1.2, 0)],
            "energy_j": [round(energy_j * 0.8, 1), round(energy_j * 1.3, 1)],
            "time_ms": [round(params["time_ms"] * 0.7), round(params["time_ms"] * 1.5)],
        }

        risk = self._assess_risk(inputs, parameters, physics)
        recommendations = self._generate_recommendations(inputs, parameters, physics, risk)

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

    def _assess_risk(self, inputs, params, physics):
        risks = {}
        temp = physics["interface_temp_rise_c"]
        upper_melt = 660 if inputs.upper_material.material_type == "Al" else 1085
        temp_ratio = temp / upper_melt
        risks["overweld_risk"] = "high" if temp_ratio > 0.5 else ("medium" if temp_ratio > 0.3 else "low")

        ber = physics["bottom_energy_ratio"]
        risks["underweld_risk"] = "high" if ber < 0.3 else ("medium" if ber < 0.5 else "low")

        collapse = self._physics.estimate_collapse_um(
            amplitude_um=params["amplitude_um"], pressure_mpa=params.get("pressure_mpa", 0.3),
            weld_time_s=params["time_ms"] / 1000, n_layers=inputs.upper_material.layers,
            material_yield_mpa=35 if inputs.upper_material.material_type == "Al" else 70)
        total_thickness_um = inputs.upper_material.total_thickness_mm * 1000
        if total_thickness_um > 0 and collapse / total_thickness_um > 0.4:
            risks["perforation_risk"] = "high"
        elif total_thickness_um > 0 and collapse / total_thickness_um > 0.2:
            risks["perforation_risk"] = "medium"
        else:
            risks["perforation_risk"] = "low"
        return risks

    def _generate_recommendations(self, inputs, params, physics, risks):
        recs = []
        if risks.get("overweld_risk") == "high":
            recs.append("Overweld risk is high. Consider reducing energy or weld time.")
        if risks.get("underweld_risk") == "high":
            recs.append("Bottom layers may not bond well. Consider increasing amplitude.")
        if risks.get("perforation_risk") in ("medium", "high"):
            recs.append("Perforation risk detected. Reduce pressure or amplitude.")
        if inputs.upper_material.layers > 30:
            recs.append("High layer count (%d). Recommend trial welding with progressive energy." % inputs.upper_material.layers)
        recs.append("Start trial welding at 80%% of recommended energy, then increase in 5%% steps.")
        return recs
