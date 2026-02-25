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
