"""Knurl pattern optimization service.

Optimizes knurl geometry for maximum energy coupling and minimum material damage
by sweeping parameters and computing physics metrics for each configuration.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

logger = logging.getLogger(__name__)


@dataclass
class KnurlConfig:
    """A single knurl configuration to evaluate."""
    knurl_type: str
    pitch_mm: float
    tooth_width_mm: float
    depth_mm: float
    direction: str = "perpendicular"


@dataclass
class KnurlScore:
    """Evaluation score for a knurl configuration."""
    config: KnurlConfig
    effective_friction: float
    contact_pressure_mpa: float
    energy_coupling_efficiency: float  # 0-1
    material_damage_index: float  # 0-1 (lower = better)
    overall_score: float  # weighted composite
    rank: int = 0


class KnurlOptimizer:
    """Multi-objective knurl parameter optimizer."""

    # Knurl types to sweep
    KNURL_TYPES = ["linear", "cross_hatch", "diamond", "conical", "spherical"]

    # Parameter ranges for sweep
    PITCH_RANGE = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]  # mm
    TOOTH_WIDTH_RANGE = [0.3, 0.4, 0.5, 0.6, 0.8]  # mm
    DEPTH_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5]  # mm

    def __init__(self):
        self._physics = PhysicsModel()

    def optimize(
        self,
        upper_material: str,
        lower_material: str,
        upper_hardness_hv: float = 50.0,
        lower_hardness_hv: float = 50.0,
        mu_base: float = 0.3,
        weld_area_mm2: float = 75.0,
        pressure_mpa: float = 2.0,
        amplitude_um: float = 30.0,
        frequency_khz: float = 20.0,
        n_layers: int = 1,
        max_results: int = 10,
    ) -> dict:
        """Run parameter sweep and return ranked configurations.

        Returns dict with:
        - recommendations: top-N ranked KnurlScore configs
        - pareto_front: configs on the Pareto frontier (energy vs damage)
        - analysis_summary: statistics about the sweep
        """
        candidates = []

        # Generate all parameter combinations
        for knurl_type in self.KNURL_TYPES:
            directions = ["perpendicular", "parallel"] if knurl_type == "linear" else ["perpendicular"]
            for pitch in self.PITCH_RANGE:
                for tw in self.TOOTH_WIDTH_RANGE:
                    if tw >= pitch:
                        continue  # tooth can't be wider than pitch
                    for depth in self.DEPTH_RANGE:
                        for direction in directions:
                            config = KnurlConfig(
                                knurl_type=knurl_type,
                                pitch_mm=pitch,
                                tooth_width_mm=tw,
                                depth_mm=depth,
                                direction=direction,
                            )
                            score = self._evaluate(
                                config, upper_hardness_hv, lower_hardness_hv,
                                mu_base, weld_area_mm2, pressure_mpa,
                                amplitude_um, frequency_khz, n_layers,
                            )
                            candidates.append(score)

        # Sort by overall_score (descending = better)
        candidates.sort(key=lambda s: s.overall_score, reverse=True)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        # Find Pareto front (maximize coupling, minimize damage)
        pareto = self._pareto_front(candidates)

        top_n = candidates[:max_results]

        return {
            "recommendations": [self._score_to_dict(s) for s in top_n],
            "pareto_front": [self._score_to_dict(s) for s in pareto[:max_results]],
            "analysis_summary": {
                "total_configs_evaluated": len(candidates),
                "pareto_front_size": len(pareto),
                "best_coupling_efficiency": round(candidates[0].energy_coupling_efficiency, 4) if candidates else 0,
                "best_damage_index": round(min(c.material_damage_index for c in candidates), 4) if candidates else 0,
                "upper_material": upper_material,
                "lower_material": lower_material,
            },
        }

    def _evaluate(
        self,
        config: KnurlConfig,
        upper_hv: float,
        lower_hv: float,
        mu_base: float,
        weld_area_mm2: float,
        pressure_mpa: float,
        amplitude_um: float,
        frequency_khz: float,
        n_layers: int,
    ) -> KnurlScore:
        """Evaluate a single knurl configuration."""
        # 1. Effective friction coefficient
        effective_mu = self._physics.effective_friction_coefficient(
            mu_base=mu_base,
            knurl_depth_mm=config.depth_mm,
            hardness_hv=lower_hv,
            knurl_type=config.knurl_type,
            knurl_direction=config.direction,
        )

        # 2. Contact ratio (effective area / nominal area)
        contact_ratio = self._compute_contact_ratio(config)

        # 3. Contact pressure (Hertzian model approximation)
        contact_pressure = pressure_mpa / max(contact_ratio, 0.01)

        # 4. Energy coupling efficiency
        # Higher friction x higher contact area x good direction coupling = better coupling
        freq_hz = frequency_khz * 1000
        interface_power = self._physics.interface_power_density(
            frequency_hz=freq_hz,
            amplitude_um=amplitude_um,
            pressure_mpa=pressure_mpa,
            friction_coeff=effective_mu,
            contact_area_mm2=weld_area_mm2 * contact_ratio,
        )
        # Normalize to 0-1 range (reference: max possible coupling)
        ref_power = self._physics.interface_power_density(
            frequency_hz=freq_hz,
            amplitude_um=amplitude_um,
            pressure_mpa=pressure_mpa,
            friction_coeff=0.5,  # theoretical max friction
            contact_area_mm2=weld_area_mm2,
        )
        energy_coupling = min(interface_power / max(ref_power, 1e-10), 1.0)

        # 5. Material damage index (0-1, lower = better)
        # Damage increases with: depth/hardness ratio, high contact pressure, low contact area
        depth_ratio = config.depth_mm / (lower_hv * 0.01) if lower_hv > 0 else 1.0
        pressure_ratio = contact_pressure / max(lower_hv * 3.0, 1.0)  # rough conversion HV to MPa
        damage_index = min(0.3 * depth_ratio + 0.4 * pressure_ratio + 0.3 * (1 - contact_ratio), 1.0)
        damage_index = max(damage_index, 0.0)

        # 6. Multi-layer bonus: cross_hatch and diamond provide better coupling for multi-layer
        layer_bonus = 0.0
        if n_layers > 5:
            if config.knurl_type in ("cross_hatch", "diamond"):
                layer_bonus = 0.1
            elif config.knurl_type == "linear" and config.direction == "perpendicular":
                layer_bonus = 0.05

        # 7. Overall score: maximize coupling, minimize damage
        overall = 0.6 * energy_coupling + 0.3 * (1 - damage_index) + 0.1 * contact_ratio + layer_bonus

        return KnurlScore(
            config=config,
            effective_friction=round(effective_mu, 4),
            contact_pressure_mpa=round(contact_pressure, 2),
            energy_coupling_efficiency=round(energy_coupling, 4),
            material_damage_index=round(damage_index, 4),
            overall_score=round(overall, 4),
        )

    def _compute_contact_ratio(self, config: KnurlConfig) -> float:
        """Compute contact ratio (effective area / nominal area)."""
        p = config.pitch_mm
        tw = config.tooth_width_mm
        if p <= 0:
            return 1.0

        if config.knurl_type == "linear":
            return tw / p
        elif config.knurl_type in ("cross_hatch", "diamond"):
            return (tw / p) ** 2
        elif config.knurl_type in ("conical", "spherical"):
            tip_r = tw / 2
            return math.pi * tip_r ** 2 / p ** 2
        return 1.0

    def _pareto_front(self, candidates: list[KnurlScore]) -> list[KnurlScore]:
        """Find Pareto optimal configs (maximize coupling, minimize damage)."""
        pareto = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if (other.energy_coupling_efficiency > c.energy_coupling_efficiency and
                        other.material_damage_index < c.material_damage_index):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)
        pareto.sort(key=lambda s: s.overall_score, reverse=True)
        return pareto

    def _score_to_dict(self, score: KnurlScore) -> dict:
        return {
            "knurl_type": score.config.knurl_type,
            "pitch_mm": score.config.pitch_mm,
            "tooth_width_mm": score.config.tooth_width_mm,
            "depth_mm": score.config.depth_mm,
            "direction": score.config.direction,
            "effective_friction": score.effective_friction,
            "contact_pressure_mpa": score.contact_pressure_mpa,
            "energy_coupling_efficiency": score.energy_coupling_efficiency,
            "material_damage_index": score.material_damage_index,
            "overall_score": score.overall_score,
            "rank": score.rank,
        }
