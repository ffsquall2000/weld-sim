"""Physics models for ultrasonic metal welding."""
from __future__ import annotations

import math


class PhysicsModel:
    def acoustic_impedance_match(self, z1: float, z2: float) -> float:
        """Energy transmission efficiency: T = 4*Z1*Z2 / (Z1+Z2)^2"""
        if z1 <= 0 or z2 <= 0:
            return 0.0
        return 4 * z1 * z2 / (z1 + z2) ** 2

    def interface_power_density(self, frequency_hz: float, amplitude_um: float,
                                 pressure_mpa: float, friction_coeff: float,
                                 contact_area_mm2: float) -> float:
        """Interface power density in W/mm2."""
        amplitude_m = amplitude_um * 1e-6
        power_w = (2 * math.pi * frequency_hz * friction_coeff
                   * pressure_mpa * 1e6 * amplitude_m * contact_area_mm2 * 1e-6)
        return power_w / contact_area_mm2

    def multilayer_energy_attenuation(self, n_layers: int, material_impedance: float,
                                       layer_thickness_mm: float,
                                       attenuation_coeff: float = 0.02) -> list:
        """Energy ratio reaching each layer. Returns [E1/E0, ..., En/E0]."""
        ratios = []
        for i in range(n_layers):
            depth_mm = (i + 0.5) * layer_thickness_mm
            ratio = math.exp(-attenuation_coeff * depth_mm * n_layers ** 0.3)
            ratios.append(max(ratio, 0.0))
        return ratios

    def interface_temperature_rise(self, power_density_w_mm2: float, weld_time_s: float,
                                    thermal_conductivity_1: float, thermal_conductivity_2: float,
                                    density_1: float, density_2: float,
                                    specific_heat_1: float, specific_heat_2: float) -> float:
        """Estimate interface temperature rise using 1D heat diffusion."""
        e1 = math.sqrt(thermal_conductivity_1 * density_1 * specific_heat_1)
        e2 = math.sqrt(thermal_conductivity_2 * density_2 * specific_heat_2)
        e_eff = 2 * e1 * e2 / (e1 + e2) if (e1 + e2) > 0 else 1.0
        q = power_density_w_mm2 * 1e6
        return q * math.sqrt(weld_time_s / math.pi) / e_eff

    def estimate_collapse_um(self, amplitude_um: float, pressure_mpa: float,
                              weld_time_s: float, n_layers: int,
                              material_yield_mpa: float) -> float:
        """Estimate total collapse/indentation in micrometers."""
        if material_yield_mpa <= 0:
            return 0.0
        ratio = pressure_mpa / material_yield_mpa
        collapse = amplitude_um * ratio * weld_time_s * 1000 * (1 + 0.01 * n_layers)
        return max(collapse, 0.0)

    # --- Correction 2: Knurl-Material Effective Friction ---

    DIRECTION_COUPLING = {
        "linear_perpendicular": 0.90,
        "linear_parallel": 0.65,
        "cross_hatch": 0.85,
        "diamond": 0.85,
        "conical": 0.80,
        "spherical": 0.75,
    }

    def effective_friction_coefficient(
        self,
        mu_base: float,
        knurl_depth_mm: float,
        hardness_hv: float,
        knurl_type: str,
        knurl_direction: str = "perpendicular",
    ) -> float:
        """Compute effective friction including ploughing and direction coupling.

        Args:
            mu_base: Base friction coefficient from material combination.
            knurl_depth_mm: Knurl tooth depth in mm.
            hardness_hv: Vickers hardness of softer material.
            knurl_type: One of linear/cross_hatch/diamond/conical/spherical.
            knurl_direction: perpendicular or parallel (only for linear type).

        Returns:
            Effective friction coefficient.
        """
        mu_ploughing = min(knurl_depth_mm / (hardness_hv * 0.01) * 0.15, 0.2) if hardness_hv > 0 else 0.0

        if knurl_type == "linear":
            key = f"linear_{knurl_direction}"
        else:
            key = knurl_type
        direction_coupling = self.DIRECTION_COUPLING.get(key, 0.85)

        return (mu_base + mu_ploughing) * direction_coupling

    # --- Correction 3: Cylinder Force / Air Pressure ---

    def cylinder_force_range(
        self,
        bore_mm: float,
        min_air_bar: float,
        max_air_bar: float,
        efficiency: float = 0.90,
    ) -> tuple[float, float]:
        """Calculate min/max force from cylinder parameters.

        Returns:
            (min_force_N, max_force_N)
        """
        area_mm2 = math.pi * (bore_mm / 2) ** 2
        # 1 bar = 0.1 MPa = 0.1 N/mm²
        min_force = min_air_bar * area_mm2 * 0.1 * efficiency
        max_force = max_air_bar * area_mm2 * 0.1 * efficiency
        return min_force, max_force

    def required_air_pressure_bar(
        self,
        target_force_n: float,
        bore_mm: float,
        efficiency: float = 0.90,
    ) -> float:
        """Calculate required air pressure in bar for a target force."""
        area_mm2 = math.pi * (bore_mm / 2) ** 2
        if area_mm2 * efficiency <= 0:
            return 0.0
        return target_force_n / (area_mm2 * 0.1 * efficiency)

    # --- Chamfer / Edge Treatment Physics ---

    def chamfer_stress_concentration_factor(
        self,
        chamfer_radius_mm: float,
        contact_width_mm: float,
        edge_treatment: str = "none",
    ) -> float:
        """Stress concentration factor Kt at horn edge.

        Sharp edge Kt ~ 3.0, large fillet Kt -> 1.0.
        Based on Peterson's stress concentration factors for
        stepped/shouldered geometries.

        Args:
            chamfer_radius_mm: Edge radius in mm (0 = sharp).
            contact_width_mm: Horn contact face width in mm.
            edge_treatment: none | chamfer | fillet | compound.

        Returns:
            Kt value (1.0-3.0).
        """
        if edge_treatment == "none" or chamfer_radius_mm <= 0:
            return 3.0

        # Normalised radius: r/w ratio
        r_over_w = chamfer_radius_mm / contact_width_mm if contact_width_mm > 0 else 0.0

        if edge_treatment == "fillet":
            # Fillet provides smooth transition -> lower Kt
            # Kt = 1 + 2 / (1 + 2 * sqrt(r/w))  (empirical fit)
            kt = 1.0 + 2.0 / (1.0 + 2.0 * math.sqrt(max(r_over_w, 0.001)))
        elif edge_treatment == "chamfer":
            # Chamfer still has a corner -> Kt slightly higher than fillet
            kt = 1.0 + 2.2 / (1.0 + 1.5 * math.sqrt(max(r_over_w, 0.001)))
        elif edge_treatment == "compound":
            # Compound: outer chamfer + inner fillet -> best Kt
            kt = 1.0 + 1.8 / (1.0 + 2.5 * math.sqrt(max(r_over_w, 0.001)))
        else:
            kt = 3.0

        return max(1.0, min(kt, 3.0))

    def chamfer_material_damage_risk(
        self,
        kt: float,
        pressure_mpa: float,
        amplitude_um: float,
        material_yield_mpa: float,
        weld_time_s: float,
    ) -> dict:
        """Assess material damage risk from horn edge stress concentration.

        Args:
            kt: Stress concentration factor from chamfer_stress_concentration_factor().
            pressure_mpa: Nominal welding pressure in MPa.
            amplitude_um: Welding amplitude in um.
            material_yield_mpa: Yield strength of the softer workpiece material.
            weld_time_s: Welding time in seconds.

        Returns:
            Dict with risk_level, peak_stress_mpa, contact_area_modification,
            energy_redistribution_factor.
        """
        # Peak stress at edge = Kt x nominal pressure
        peak_stress_mpa = kt * pressure_mpa

        # Dynamic stress from ultrasonic vibration adds cyclic loading
        # Approximate dynamic stress contribution from amplitude
        dynamic_stress_mpa = 0.1 * amplitude_um * pressure_mpa / 10.0
        total_peak_stress = peak_stress_mpa + dynamic_stress_mpa

        # Damage ratio: peak stress / yield strength
        damage_ratio = total_peak_stress / material_yield_mpa if material_yield_mpa > 0 else 10.0

        # Time factor: longer welds accumulate more fatigue damage
        time_factor = min(1.0 + 0.5 * weld_time_s, 2.0)
        effective_damage = damage_ratio * time_factor

        if effective_damage > 2.0:
            risk_level = "critical"
        elif effective_damage > 1.2:
            risk_level = "high"
        elif effective_damage > 0.7:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Contact area modification: sharp edges cause localised deformation
        # reducing effective contact area
        contact_area_mod = max(0.7, 1.0 - 0.1 * (kt - 1.0))

        # Energy redistribution: sharp edges concentrate energy at periphery
        # instead of uniform distribution
        energy_redistribution = max(0.6, 1.0 - 0.15 * (kt - 1.0))

        return {
            "risk_level": risk_level,
            "peak_stress_mpa": round(total_peak_stress, 2),
            "damage_ratio": round(effective_damage, 3),
            "contact_area_modification": round(contact_area_mod, 3),
            "energy_redistribution_factor": round(energy_redistribution, 3),
        }

    def chamfer_contact_area_correction(
        self,
        nominal_area_mm2: float,
        chamfer_radius_mm: float,
        contact_width_mm: float,
        contact_length_mm: float,
        edge_treatment: str = "none",
    ) -> float:
        """Corrected contact area accounting for chamfer geometry.

        Chamfer removes material from the edge, reducing the effective
        contact area. The removed area depends on the chamfer radius and
        the perimeter of the contact face.

        Args:
            nominal_area_mm2: Original contact face area in mm2.
            chamfer_radius_mm: Chamfer radius in mm.
            contact_width_mm: Contact face width in mm.
            contact_length_mm: Contact face length in mm.
            edge_treatment: none | chamfer | fillet | compound.

        Returns:
            Corrected contact area in mm2.
        """
        if edge_treatment == "none" or chamfer_radius_mm <= 0:
            return nominal_area_mm2

        # Perimeter of contact face
        perimeter_mm = 2 * (contact_width_mm + contact_length_mm)

        if edge_treatment == "fillet":
            # Fillet: quarter-circle profile removes pi/4 * r^2 at each corner (4 corners)
            # Plus linear reduction along edges
            corner_loss = 4 * (1.0 - math.pi / 4.0) * chamfer_radius_mm ** 2
            total_loss = corner_loss
        elif edge_treatment == "chamfer":
            # Chamfer: 45 deg cut removes triangular strip along perimeter
            # Strip width = r, depth = r * tan(angle), area ~ r * perimeter
            # But corners overlap -- subtract corner overlap
            strip_loss = chamfer_radius_mm * perimeter_mm
            corner_overlap = 4 * chamfer_radius_mm ** 2  # 4 corner overlaps
            total_loss = strip_loss - corner_overlap
        elif edge_treatment == "compound":
            # Compound: smaller effective loss (outer chamfer catches debris,
            # inner fillet preserves contact)
            strip_loss = 0.5 * chamfer_radius_mm * perimeter_mm
            total_loss = strip_loss
        else:
            total_loss = 0

        corrected = nominal_area_mm2 - max(total_loss, 0)
        return max(corrected, nominal_area_mm2 * 0.5)  # never less than 50% of nominal
