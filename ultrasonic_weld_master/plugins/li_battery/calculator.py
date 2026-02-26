"""Li-battery welding parameter calculator with 3-layer model."""
from __future__ import annotations

import math
import uuid
from typing import Any, Optional

# Unit conversion constants
MPA_TO_PSI = 145.038  # 1 MPa = 145.038 PSI
BAR_TO_PSI = 14.5038  # 1 bar = 14.5038 PSI

from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

# Industry-calibrated base parameters for SINGLE-LAYER reference at 20 kHz.
# pressure_mpa = average interface pressure over nominal weld area.
# Multi-layer scaling is applied in _layer2_empirical via power-law factors.
BASE_PARAMS = {
    "Al-Cu": {"amplitude_um": 25, "pressure_mpa": 2.0, "energy_density_j_mm2": 0.30, "time_ms": 100},
    "Al-Al": {"amplitude_um": 23, "pressure_mpa": 1.5, "energy_density_j_mm2": 0.25, "time_ms": 90},
    "Cu-Cu": {"amplitude_um": 30, "pressure_mpa": 3.0, "energy_density_j_mm2": 0.45, "time_ms": 130},
    "Cu-Ni": {"amplitude_um": 35, "pressure_mpa": 4.0, "energy_density_j_mm2": 0.55, "time_ms": 160},
    "Cu-Al": {"amplitude_um": 25, "pressure_mpa": 2.0, "energy_density_j_mm2": 0.30, "time_ms": 100},
    "Ni-Cu": {"amplitude_um": 35, "pressure_mpa": 4.0, "energy_density_j_mm2": 0.55, "time_ms": 160},
}
DEFAULT_BASE = {"amplitude_um": 28, "pressure_mpa": 2.5, "energy_density_j_mm2": 0.35, "time_ms": 120}


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

        # Correction 1: Use effective_area instead of nominal area
        effective_area = inputs.effective_area_mm2

        # Correction 2: Effective friction with knurl interaction
        mu_base = combo.get("friction_coefficient", 0.3)
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            hardness_hv = lower_mat.get("hardness_hv", 100)
            friction_coeff = self._physics.effective_friction_coefficient(
                mu_base=mu_base, knurl_depth_mm=s.knurl_depth_mm,
                hardness_hv=hardness_hv, knurl_type=s.knurl_type,
                knurl_direction=s.knurl_direction)
        else:
            friction_coeff = mu_base

        # Chamfer / edge treatment analysis
        chamfer_data = {}
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            kt = self._physics.chamfer_stress_concentration_factor(
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                edge_treatment=s.edge_treatment,
            )
            # Correct effective area for chamfer geometry
            chamfer_area = self._physics.chamfer_contact_area_correction(
                nominal_area_mm2=inputs.weld_area_mm2,
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                contact_length_mm=s.contact_length_mm,
                edge_treatment=s.edge_treatment,
            )
            # Use chamfer-corrected area as effective area baseline
            if s.edge_treatment != "none" and s.chamfer_radius_mm > 0:
                effective_area = min(effective_area, chamfer_area)
            chamfer_data = {"kt": kt, "chamfer_corrected_area_mm2": chamfer_area}
        else:
            chamfer_data = {"kt": 1.0, "chamfer_corrected_area_mm2": inputs.weld_area_mm2}

        combo_key = f"{inputs.upper_material.material_type}-{inputs.lower_material.material_type}"
        base = BASE_PARAMS.get(combo_key, DEFAULT_BASE)

        pd = self._physics.interface_power_density(
            frequency_hz=inputs.frequency_khz * 1000, amplitude_um=base["amplitude_um"],
            pressure_mpa=base["pressure_mpa"], friction_coeff=friction_coeff,
            contact_area_mm2=effective_area)

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
                "base_params": base, "friction_coeff": friction_coeff,
                "effective_area_mm2": effective_area, "mu_base": mu_base,
                "chamfer_data": chamfer_data}

    def _layer2_empirical(self, inputs, physics, combo):
        base = dict(physics["base_params"])
        n_layers = max(inputs.upper_material.layers, 1)

        # --- Power-law layer scaling (industry-calibrated) ---
        # Amplitude scales slowly (foil deformation saturates)
        # Pressure/force scales moderately (more layers need more clamping)
        # Energy scales significantly (each interface needs bonding energy)
        # Time scales moderately (heat accumulates across interfaces)
        if n_layers > 1:
            amp_factor = min(n_layers ** 0.12, 1.6)
            pressure_factor = min(n_layers ** 0.40, 5.0)
            energy_factor = min(n_layers ** 0.55, 8.0)
            time_factor = min(n_layers ** 0.35, 4.0)
            base["amplitude_um"] *= amp_factor
            base["pressure_mpa"] *= pressure_factor
            base["energy_density_j_mm2"] *= energy_factor
            base["time_ms"] *= time_factor

        # Frequency scaling (base params calibrated at 20 kHz)
        freq_ratio = inputs.frequency_khz / 20.0
        if freq_ratio > 1.1 or freq_ratio < 0.9:
            # Higher frequency → lower amplitude, slightly shorter time
            base["amplitude_um"] /= min(max(freq_ratio, 0.5), 2.0)
            base["time_ms"] /= min(max(freq_ratio ** 0.3, 0.7), 1.4)

        # Nominal area scaling: larger welds need proportionally more pressure
        nominal_area = inputs.weld_area_mm2
        if nominal_area > 100:
            base["pressure_mpa"] *= min(math.sqrt(nominal_area / 100), 1.8)

        # Impedance mismatch correction
        eff = physics["impedance_efficiency"]
        if eff < 0.8:
            base["amplitude_um"] *= 1 + 0.2 * (1 - eff)
            base["energy_density_j_mm2"] *= 1 + 0.15 * (1 - eff)

        # IMC risk: reduce energy to avoid brittle intermetallic formation
        if combo.get("imc_risk") == "high":
            max_temp = combo.get("max_interface_temp_c", 300)
            if physics["interface_temp_rise_c"] > max_temp * 0.7:
                base["energy_density_j_mm2"] *= 0.85
                base["time_ms"] *= 0.9

        # Total stack thickness correction: thicker stacks need more energy penetration
        stack_mm = inputs.upper_material.total_thickness_mm
        if stack_mm > 0.5:
            thickness_factor = min(1.0 + 0.3 * (stack_mm - 0.5), 1.8)
            base["energy_density_j_mm2"] *= thickness_factor
            base["time_ms"] *= min(thickness_factor ** 0.5, 1.3)

        # Clamp to physically reasonable ranges
        base["amplitude_um"] = max(15, min(base["amplitude_um"], 60))
        base["pressure_mpa"] = max(0.5, min(base["pressure_mpa"], 25.0))
        base["time_ms"] = max(50, min(base["time_ms"], 1200))
        return base

    # Typical ultrasonic system efficiency: converter + booster + horn losses
    ETA_SYSTEM = 0.65

    def _layer3_output(self, inputs, params, physics):
        effective_area = physics["effective_area_mm2"]
        nominal_area = inputs.weld_area_mm2
        force_n = params["pressure_mpa"] * nominal_area

        # --- PHYSICS-DERIVED ENERGY ---
        # Friction power: P = mu * Force * v_peak = mu * (sigma*Area) * (2*pi*f*A)
        friction_coeff = physics["friction_coeff"]
        interface_power_w = self._physics.interface_power_density(
            frequency_hz=inputs.frequency_khz * 1000,
            amplitude_um=params["amplitude_um"],
            pressure_mpa=params["pressure_mpa"],
            friction_coeff=friction_coeff,
            contact_area_mm2=nominal_area,
        ) * nominal_area

        # Machine power = interface_power / system_efficiency
        machine_power_w = interface_power_w / self.ETA_SYSTEM

        # Weld time from empirical layer scaling (already calibrated)
        time_ms = params["time_ms"]

        # If machine power exceeds max, extend weld time to compensate
        if machine_power_w > inputs.max_power_w > 0:
            scale = inputs.max_power_w / machine_power_w
            machine_power_w = inputs.max_power_w
            interface_power_w = machine_power_w * self.ETA_SYSTEM
            time_ms = time_ms / scale  # extend time to deliver same total energy

        # Energy = friction_power × time (physics-consistent)
        energy_j = interface_power_w * (time_ms / 1000.0)

        # Correction 4: Amplitude chain (booster × horn gain)
        system_gain = 1.0
        amplitude_percent = 0.0
        if inputs.booster is not None:
            system_gain *= inputs.booster.gain_ratio
        if inputs.sonotrode is not None:
            system_gain *= inputs.sonotrode.horn_gain
        actual_amplitude_um = params["amplitude_um"]
        if inputs.booster is not None and inputs.booster.rated_amplitude_um > 0:
            amplitude_percent = actual_amplitude_um / inputs.booster.rated_amplitude_um * 100

        pressure_psi = params["pressure_mpa"] * MPA_TO_PSI

        parameters = {
            "amplitude_um": round(actual_amplitude_um, 1),
            "pressure_n": round(force_n, 0),
            "pressure_mpa": round(params["pressure_mpa"], 3),
            "pressure_psi": round(pressure_psi, 1),
            "energy_j": round(energy_j, 1),
            "time_ms": round(time_ms),
            "interface_power_w": round(interface_power_w, 0),
            "machine_power_w": round(machine_power_w, 0),
            "frequency_khz": inputs.frequency_khz,
            "control_mode": "energy",
        }

        # Correction 3: Cylinder force / air pressure back-calculation
        air_pressure_bar = 0.0
        if inputs.cylinder is not None:
            cyl = inputs.cylinder
            air_pressure_bar = self._physics.required_air_pressure_bar(
                target_force_n=force_n, bore_mm=cyl.bore_mm,
                efficiency=cyl.efficiency)
            parameters["air_pressure_bar"] = round(air_pressure_bar, 2)
            parameters["air_pressure_psi"] = round(air_pressure_bar * BAR_TO_PSI, 1)

        # Correction 6: Device settings card data
        device_settings = {
            "air_pressure_bar": round(air_pressure_bar, 2),
            "air_pressure_psi": round(air_pressure_bar * BAR_TO_PSI, 1),
            "amplitude_um": round(actual_amplitude_um, 1),
            "amplitude_percent": round(amplitude_percent, 1),
            "weld_time_s": round(time_ms / 1000, 3),
            "delay_time_s": 0.20,
            "hold_time_s": 0.50,
            "trigger_mode": "energy",
            "trigger_value_j": round(energy_j, 1),
            "interface_power_w": round(interface_power_w, 0),
            "machine_power_w": round(machine_power_w, 0),
            "effective_area_mm2": round(effective_area, 1),
            "effective_friction": round(friction_coeff, 3),
            "actual_pressure_mpa": round(params["pressure_mpa"], 3),
            "actual_pressure_psi": round(pressure_psi, 1),
            "system_gain": round(system_gain, 2),
        }
        parameters["device_settings"] = device_settings

        amp = parameters["amplitude_um"]
        safety_window = {
            "amplitude_um": [round(amp * 0.85, 1), round(amp * 1.15, 1)],
            "pressure_n": [round(force_n * 0.8, 0), round(force_n * 1.2, 0)],
            "energy_j": [round(energy_j * 0.8, 1), round(energy_j * 1.3, 1)],
            "time_ms": [round(time_ms * 0.7), round(time_ms * 1.5)],
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
                    "weld_area_mm2": nominal_area,
                    "effective_area_mm2": effective_area,
                    "frequency_khz": inputs.frequency_khz},
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

        # Edge damage risk from chamfer/stress concentration
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            kt = self._physics.chamfer_stress_concentration_factor(
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                edge_treatment=s.edge_treatment,
            )
            upper_yield = 35 if inputs.upper_material.material_type == "Al" else 70
            edge_risk_data = self._physics.chamfer_material_damage_risk(
                kt=kt,
                pressure_mpa=params.get("pressure_mpa", 2.0),
                amplitude_um=params["amplitude_um"],
                material_yield_mpa=upper_yield,
                weld_time_s=params["time_ms"] / 1000.0,
            )
            risks["edge_damage_risk"] = edge_risk_data["risk_level"]
            risks["edge_damage_detail"] = edge_risk_data
        else:
            risks["edge_damage_risk"] = "low"

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

        # Correction 5: Mode and frequency warnings
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            freq_diff = abs(s.resonant_freq_khz - inputs.frequency_khz)
            if freq_diff > 0.5:
                recs.append("Horn resonant frequency deviation: %.1f kHz from working frequency. Check horn tuning." % freq_diff)
            if s.mode != "longitudinal":
                recs.append("Non-longitudinal mode (%s) may reduce weld quality." % s.mode)

        # Correction 3: Cylinder pressure range warning
        if inputs.cylinder is not None:
            cyl = inputs.cylinder
            air_bar = params.get("air_pressure_bar", 0)
            if isinstance(air_bar, (int, float)) and air_bar > cyl.max_air_bar:
                recs.append("Required air pressure (%.1f bar) exceeds cylinder max (%.1f bar). Increase bore size or reduce weld area." % (air_bar, cyl.max_air_bar))
            elif isinstance(air_bar, (int, float)) and air_bar < cyl.min_air_bar:
                recs.append("Required air pressure (%.1f bar) is below cylinder min (%.1f bar). Reduce bore size." % (air_bar, cyl.min_air_bar))

        # Knurl pattern recommendation based on application and materials
        n_layers = inputs.upper_material.layers
        upper_type = inputs.upper_material.material_type
        if inputs.sonotrode is None:
            # Provide knurl recommendations when no sonotrode is specified
            if n_layers > 20:
                recs.append(
                    "Recommended knurl: Cross-hatch pattern, pitch 0.8-1.2 mm, "
                    "tooth width 0.4-0.6 mm, depth 0.2-0.4 mm. "
                    "Cross-hatch provides optimal energy coupling for multi-layer stacks."
                )
            elif upper_type == "Al":
                recs.append(
                    "Recommended knurl: Linear perpendicular, pitch 0.8-1.0 mm, "
                    "tooth width 0.4-0.5 mm, depth 0.2-0.3 mm. "
                    "Linear pattern breaks Al oxide layer effectively."
                )
            else:
                recs.append(
                    "Recommended knurl: Diamond pattern, pitch 1.0-1.5 mm, "
                    "tooth width 0.5-0.7 mm, depth 0.3-0.5 mm. "
                    "Diamond pattern provides balanced coupling for harder materials."
                )
        else:
            # Analyze current knurl settings
            s = inputs.sonotrode
            contact_ratio = inputs.effective_area_mm2 / inputs.weld_area_mm2 if inputs.weld_area_mm2 > 0 else 1.0
            if contact_ratio < 0.3:
                recs.append(
                    "Contact ratio %.0f%% is low. Consider increasing tooth width "
                    "or reducing pitch for better energy transfer." % (contact_ratio * 100)
                )
            elif contact_ratio > 0.8:
                recs.append(
                    "Contact ratio %.0f%% is high. Consider finer knurl (smaller pitch) "
                    "for better oxide breaking and grip." % (contact_ratio * 100)
                )
            if n_layers > 20 and s.knurl_type == "linear" and s.knurl_direction == "parallel":
                recs.append(
                    "Parallel linear knurl has low coupling (65%%) for multi-layer. "
                    "Consider cross-hatch or perpendicular orientation."
                )

        # Chamfer / edge treatment recommendations
        if risks.get("edge_damage_risk") in ("high", "critical"):
            detail = risks.get("edge_damage_detail", {})
            recs.append(
                "Edge damage risk is %s (Kt=%.1f, peak stress=%.1f MPa). "
                "Consider adding fillet or compound edge treatment with radius >=0.3 mm."
                % (risks["edge_damage_risk"], detail.get("kt", 3.0), detail.get("peak_stress_mpa", 0))
            )
        elif inputs.sonotrode is not None and inputs.sonotrode.edge_treatment == "none":
            recs.append(
                "No edge treatment specified. Recommend fillet radius 0.2-0.5 mm "
                "to reduce material damage risk."
            )

        recs.append("Start trial welding at 80%% of recommended energy, then increase in 5%% steps.")
        return recs
