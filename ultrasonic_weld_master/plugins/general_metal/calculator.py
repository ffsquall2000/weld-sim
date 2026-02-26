"""General metal welding parameter calculator using simplified physics model."""
from __future__ import annotations

import math
import uuid
from typing import Any

# Unit conversion constants
MPA_TO_PSI = 145.038  # 1 MPa = 145.038 PSI
BAR_TO_PSI = 14.5038  # 1 bar = 14.5038 PSI

from ultrasonic_weld_master.core.models import WeldInputs, WeldRecipe
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

# Industry-calibrated base parameters for SINGLE-LAYER reference at 20 kHz.
# pressure_mpa = average interface pressure over nominal weld area.
BASE_DEFAULTS = {
    "amplitude_um": 28, "pressure_mpa": 2.5,
    "energy_density_j_mm2": 0.40, "time_ms": 120,
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

        # Correction 1: Track areas separately
        effective_area = inputs.effective_area_mm2
        nominal_area = inputs.weld_area_mm2

        # Correction 2: Effective friction with knurl interaction
        mu_base = combo.get("friction_coefficient", 0.3)
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            hardness_hv = lower_mat.get("hardness_hv", 50)
            friction = self._physics.effective_friction_coefficient(
                mu_base=mu_base, knurl_depth_mm=s.knurl_depth_mm,
                hardness_hv=hardness_hv, knurl_type=s.knurl_type,
                knurl_direction=s.knurl_direction)
        else:
            friction = mu_base

        # Chamfer / edge treatment analysis
        chamfer_data = {}
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            kt = self._physics.chamfer_stress_concentration_factor(
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                edge_treatment=s.edge_treatment,
            )
            chamfer_area = self._physics.chamfer_contact_area_correction(
                nominal_area_mm2=inputs.weld_area_mm2,
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                contact_length_mm=s.contact_length_mm,
                edge_treatment=s.edge_treatment,
                chamfer_angle_deg=s.chamfer_angle_deg,
            )
            if s.edge_treatment != "none" and s.chamfer_radius_mm > 0:
                effective_area = min(effective_area, chamfer_area)
            chamfer_data = {"kt": kt, "chamfer_corrected_area_mm2": chamfer_area}
        else:
            chamfer_data = {"kt": 1.0, "chamfer_corrected_area_mm2": inputs.weld_area_mm2}

        # Hardness-based amplitude correction
        upper_hv = upper_mat.get("hardness_hv", 50)
        lower_hv = lower_mat.get("hardness_hv", 50)
        hardness_factor = max(upper_hv, lower_hv) / 50.0
        base["amplitude_um"] *= min(max(hardness_factor, 0.8), 1.5)

        # Impedance mismatch correction
        if impedance_eff < 0.8:
            base["amplitude_um"] *= 1 + 0.15 * (1 - impedance_eff)
            base["energy_density_j_mm2"] *= 1 + 0.1 * (1 - impedance_eff)

        # Power-law layer scaling (same approach as li_battery)
        n_layers = max(inputs.upper_material.layers, 1)
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
            base["amplitude_um"] /= min(max(freq_ratio, 0.5), 2.0)
            base["time_ms"] /= min(max(freq_ratio ** 0.3, 0.7), 1.4)

        # Nominal area scaling: larger welds need proportionally more pressure
        if nominal_area > 100:
            base["pressure_mpa"] *= min(math.sqrt(nominal_area / 100), 1.8)

        # Total stack thickness correction
        stack_mm = inputs.upper_material.total_thickness_mm
        if stack_mm > 0.5:
            thickness_factor = min(1.0 + 0.3 * (stack_mm - 0.5), 1.8)
            base["energy_density_j_mm2"] *= thickness_factor
            base["time_ms"] *= min(thickness_factor ** 0.5, 1.3)

        # Clamp to physically reasonable ranges
        base["amplitude_um"] = max(15, min(base["amplitude_um"], 65))
        base["pressure_mpa"] = max(0.5, min(base["pressure_mpa"], 25.0))
        base["time_ms"] = max(80, min(base["time_ms"], 1200))

        # Build output — force from corrected pressure × nominal area
        force_n = base["pressure_mpa"] * nominal_area

        # --- PHYSICS-DERIVED ENERGY ---
        # Friction power: P = mu * Force * v_peak
        eta_system = 0.65
        interface_power_w = self._physics.interface_power_density(
            frequency_hz=inputs.frequency_khz * 1000,
            amplitude_um=base["amplitude_um"],
            pressure_mpa=base["pressure_mpa"],
            friction_coeff=friction,
            contact_area_mm2=nominal_area,
        ) * nominal_area

        machine_power_w = interface_power_w / eta_system
        time_ms = base["time_ms"]

        # If machine power exceeds max, extend weld time
        if machine_power_w > inputs.max_power_w > 0:
            scale = inputs.max_power_w / machine_power_w
            machine_power_w = inputs.max_power_w
            interface_power_w = machine_power_w * eta_system
            time_ms = time_ms / scale

        # Energy = friction_power × time (physics-consistent)
        energy_j = interface_power_w * (time_ms / 1000.0)

        # Correction 4: Amplitude chain (booster × horn gain)
        system_gain = 1.0
        amplitude_percent = 0.0
        if inputs.booster is not None:
            system_gain *= inputs.booster.gain_ratio
        if inputs.sonotrode is not None:
            system_gain *= inputs.sonotrode.horn_gain
        actual_amplitude_um = base["amplitude_um"]
        if inputs.booster is not None and inputs.booster.rated_amplitude_um > 0:
            amplitude_percent = actual_amplitude_um / inputs.booster.rated_amplitude_um * 100

        pressure_psi = base["pressure_mpa"] * MPA_TO_PSI

        parameters = {
            "amplitude_um": round(actual_amplitude_um, 1),
            "pressure_n": round(force_n, 0),
            "pressure_mpa": round(base["pressure_mpa"], 3),
            "pressure_psi": round(pressure_psi, 1),
            "energy_j": round(energy_j, 1),
            "time_ms": round(time_ms),
            "interface_power_w": round(interface_power_w, 0),
            "machine_power_w": round(machine_power_w, 0),
            "frequency_khz": inputs.frequency_khz,
            "control_mode": "energy",
        }

        # Correction 3: Cylinder air pressure back-calculation
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
            "effective_friction": round(friction, 3),
            "actual_pressure_mpa": round(base["pressure_mpa"], 3),
            "actual_pressure_psi": round(pressure_psi, 1),
            "system_gain": round(system_gain, 2),
        }
        parameters["device_settings"] = device_settings

        amp = parameters["amplitude_um"]
        safety_window = {
            "amplitude_um": [round(amp * 0.8, 1), round(amp * 1.2, 1)],
            "pressure_n": [round(force_n * 0.75, 0), round(force_n * 1.25, 0)],
            "energy_j": [round(energy_j * 0.75, 1), round(energy_j * 1.35, 1)],
            "time_ms": [round(time_ms * 0.65), round(time_ms * 1.6)],
        }

        risk = {"overweld_risk": "medium", "underweld_risk": "medium", "perforation_risk": "low"}

        # Edge damage risk from chamfer/stress concentration
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            kt = self._physics.chamfer_stress_concentration_factor(
                chamfer_radius_mm=s.chamfer_radius_mm,
                contact_width_mm=s.contact_width_mm,
                edge_treatment=s.edge_treatment,
            )
            upper_yield = upper_mat.get("yield_strength_mpa", 70)
            edge_risk_data = self._physics.chamfer_material_damage_risk(
                kt=kt,
                pressure_mpa=base["pressure_mpa"],
                amplitude_um=base["amplitude_um"],
                material_yield_mpa=upper_yield,
                weld_time_s=time_ms / 1000.0,
            )
            risk["edge_damage_risk"] = edge_risk_data["risk_level"]
            risk["edge_damage_detail"] = edge_risk_data
        else:
            risk["edge_damage_risk"] = "low"

        # Correction 5: Mode/frequency warnings + recommendations
        recommendations = [
            "General metal welding: start trial at 75%% of recommended energy.",
            "Increase energy in 5%% steps until acceptable bond strength.",
        ]
        if inputs.sonotrode is not None:
            s = inputs.sonotrode
            freq_diff = abs(s.resonant_freq_khz - inputs.frequency_khz)
            if freq_diff > 0.5:
                recommendations.append("Horn resonant frequency deviation: %.1f kHz from working frequency. Check horn tuning." % freq_diff)
            if s.mode != "longitudinal":
                recommendations.append("Non-longitudinal mode (%s) may reduce weld quality." % s.mode)

        if inputs.cylinder is not None:
            cyl = inputs.cylinder
            if air_pressure_bar > cyl.max_air_bar:
                recommendations.append("Required air pressure (%.1f bar) exceeds cylinder max (%.1f bar). Increase bore size or reduce weld area." % (air_pressure_bar, cyl.max_air_bar))
            elif air_pressure_bar < cyl.min_air_bar:
                recommendations.append("Required air pressure (%.1f bar) is below cylinder min (%.1f bar). Reduce bore size." % (air_pressure_bar, cyl.min_air_bar))

        # Knurl pattern recommendations based on material hardness and layers
        n_layers = max(inputs.upper_material.layers, 1)
        upper_type = inputs.upper_material.material_type
        if inputs.sonotrode is None:
            # Provide knurl recommendations when no sonotrode specified
            if n_layers > 10:
                recommendations.append(
                    "Recommended knurl: Cross-hatch pattern, pitch 1.0-1.5 mm, "
                    "tooth width 0.5-0.8 mm, depth 0.3-0.5 mm. "
                    "Cross-hatch provides optimal energy coupling for multi-layer metal stacks."
                )
            elif max(upper_hv, lower_hv) > 100:
                recommendations.append(
                    "Recommended knurl: Diamond pattern, pitch 1.2-1.8 mm, "
                    "tooth width 0.6-0.9 mm, depth 0.4-0.6 mm. "
                    "Diamond pattern provides high grip force for harder materials (HV>100)."
                )
            else:
                recommendations.append(
                    "Recommended knurl: Linear perpendicular, pitch 0.8-1.2 mm, "
                    "tooth width 0.4-0.6 mm, depth 0.2-0.4 mm. "
                    "Linear pattern is suitable for softer metals with good oxide breaking."
                )
        else:
            # Analyze current knurl settings
            s = inputs.sonotrode
            contact_ratio = effective_area / nominal_area if nominal_area > 0 else 1.0
            if contact_ratio < 0.3:
                recommendations.append(
                    "Contact ratio %.0f%% is low. Consider increasing tooth width "
                    "or reducing pitch for better energy transfer." % (contact_ratio * 100)
                )
            elif contact_ratio > 0.8:
                recommendations.append(
                    "Contact ratio %.0f%% is high. Consider finer knurl (smaller pitch) "
                    "for better oxide breaking and grip." % (contact_ratio * 100)
                )
            if n_layers > 10 and s.knurl_type == "linear" and s.knurl_direction == "parallel":
                recommendations.append(
                    "Parallel linear knurl has low coupling (65%%) for multi-layer. "
                    "Consider cross-hatch or perpendicular orientation."
                )

        # Chamfer / edge treatment recommendations
        if risk.get("edge_damage_risk") in ("high", "critical"):
            detail = risk.get("edge_damage_detail", {})
            recommendations.append(
                "Edge damage risk is %s (Kt=%.1f, peak stress=%.1f MPa). "
                "Consider adding fillet or compound edge treatment with radius >=0.3 mm."
                % (risk["edge_damage_risk"], detail.get("kt", 3.0), detail.get("peak_stress_mpa", 0))
            )
        elif inputs.sonotrode is not None and inputs.sonotrode.edge_treatment == "none":
            recommendations.append(
                "No edge treatment specified. Recommend fillet radius 0.2-0.5 mm "
                "to reduce material damage risk."
            )

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
