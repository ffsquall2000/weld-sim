"""High-cycle fatigue assessment for ultrasonic welding components.

Provides S-N interpolation, Goodman diagram analysis, Marin correction
factors, and safety factor computation for components operating at
10^9+ cycles (20 kHz ultrasonic frequency).

Workflow
--------
1. Create a ``FatigueConfig`` specifying material, surface finish, Kt, etc.
2. Instantiate ``FatigueAssessor(config)``.
3. Call ``assess(stress_alternating, stress_mean, mesh)`` to get a
   ``FatigueResult`` with per-element safety factors and critical locations.

S-N Curve Convention
--------------------
Power-law form::

    sigma_a = sigma_f' * (2 * N)^b

Solving for *N*::

    N = (sigma_a / sigma_f')^(1/b) / 2

Goodman Criterion
-----------------
::

    sigma_alt / sigma_e + sigma_mean / sigma_UTS = 1 / SF

Marin Factors
-------------
::

    sigma_e_corrected = ka * kb * kc * kd * ke * sigma_e_raw
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import FatigueResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FatigueMaterialData:
    """S-N curve parameters for a single material.

    Attributes
    ----------
    name : str
        Material identifier.
    sigma_f_prime_mpa : float
        Fatigue strength coefficient [MPa].
    b_exponent : float
        Fatigue strength exponent (negative).
    sigma_e_mpa : float
        Endurance limit at 10^9 cycles [MPa].
    sigma_uts_mpa : float
        Ultimate tensile strength [MPa].
    sigma_yield_mpa : float
        Yield strength [MPa].
    """

    name: str
    sigma_f_prime_mpa: float
    b_exponent: float
    sigma_e_mpa: float
    sigma_uts_mpa: float
    sigma_yield_mpa: float


@dataclass
class FatigueConfig:
    """Configuration for a fatigue assessment run.

    Attributes
    ----------
    material : str
        Material name (must exist in ``FatigueAssessor.SN_DATABASE``).
    Kt_global : float
        Global stress concentration factor applied to all elements.
    Kt_regions : dict
        Per-element-set Kt overrides, keyed by element set name.
    target_SF : float
        Target safety factor for pass/fail evaluation.
    n_critical_locations : int
        Number of lowest-SF locations to report.
    include_mean_stress : bool
        If True, use Goodman criterion with mean stress.
    surface_finish : str
        Surface finish category for Marin ka factor.
        One of ``"machined"``, ``"ground"``, ``"polished"``.
    characteristic_diameter_mm : float
        Part diameter for Marin kb (size) factor [mm].
    reliability_pct : float
        Reliability percentage for Marin kc factor.
    temperature_c : float
        Operating temperature for Marin kd factor [deg C].
    """

    material: str = "Ti-6Al-4V"
    Kt_global: float = 1.0
    Kt_regions: dict = field(default_factory=dict)
    target_SF: float = 2.0
    n_critical_locations: int = 10
    include_mean_stress: bool = False
    surface_finish: str = "machined"
    characteristic_diameter_mm: float = 25.0
    reliability_pct: float = 50.0
    temperature_c: float = 25.0


# ---------------------------------------------------------------------------
# Allowed surface finish values
# ---------------------------------------------------------------------------

_SURFACE_FINISH_KA = {
    "machined": 0.80,
    "ground": 0.92,
    "polished": 0.97,
}

# ---------------------------------------------------------------------------
# Reliability kc lookup
# ---------------------------------------------------------------------------

_RELIABILITY_KC = {
    50.0: 1.000,
    90.0: 0.897,
    95.0: 0.868,
    99.0: 0.814,
    99.9: 0.753,
    99.99: 0.702,
}


# ---------------------------------------------------------------------------
# FatigueAssessor
# ---------------------------------------------------------------------------


class FatigueAssessor:
    """Fatigue assessment engine for ultrasonic welding components.

    Parameters
    ----------
    config : FatigueConfig
        Assessment configuration.

    Raises
    ------
    KeyError
        If the configured material is not in ``SN_DATABASE``.
    ValueError
        If the configured surface finish is not recognised.

    Examples
    --------
    >>> cfg = FatigueConfig(material="Ti-6Al-4V", Kt_global=1.0)
    >>> fa = FatigueAssessor(cfg)
    >>> fa.simple_safety_factor(150.0)
    1.76...
    """

    # Built-in S-N material database
    SN_DATABASE: dict[str, FatigueMaterialData] = {
        "Ti-6Al-4V": FatigueMaterialData(
            "Ti-6Al-4V", 1700.0, -0.095, 330.0, 950.0, 880.0
        ),
        "Al 7075-T6": FatigueMaterialData(
            "Al 7075-T6", 1050.0, -0.110, 105.0, 572.0, 503.0
        ),
        "Steel D2": FatigueMaterialData(
            "Steel D2", 2200.0, -0.080, 550.0, 1620.0, 1620.0
        ),
        "CPM 10V": FatigueMaterialData(
            "CPM 10V", 2400.0, -0.085, 510.0, 2100.0, 2100.0
        ),
        "M2 HSS": FatigueMaterialData(
            "M2 HSS", 2500.0, -0.082, 560.0, 2200.0, 2200.0
        ),
    }

    def __init__(self, config: FatigueConfig) -> None:
        if config.material not in self.SN_DATABASE:
            raise KeyError(
                f"Material '{config.material}' not found in SN_DATABASE. "
                f"Available: {list(self.SN_DATABASE.keys())}"
            )
        if config.surface_finish not in _SURFACE_FINISH_KA:
            raise ValueError(
                f"Unknown surface finish '{config.surface_finish}'. "
                f"Must be one of {list(_SURFACE_FINISH_KA.keys())}."
            )
        self._config = config
        self._mat = self.SN_DATABASE[config.material]

    # ------------------------------------------------------------------
    # Marin correction factors
    # ------------------------------------------------------------------

    def marin_factors(self) -> dict[str, float]:
        """Compute all five Marin correction factors.

        Returns
        -------
        dict[str, float]
            Keys ``ka``, ``kb``, ``kc``, ``kd``, ``ke``.
        """
        ka = self._marin_ka()
        kb = self._marin_kb()
        kc = self._marin_kc()
        kd = self._marin_kd()
        ke = 1.0  # miscellaneous, default

        return {"ka": ka, "kb": kb, "kc": kc, "kd": kd, "ke": ke}

    def _marin_ka(self) -> float:
        """Surface finish factor."""
        return _SURFACE_FINISH_KA[self._config.surface_finish]

    def _marin_kb(self) -> float:
        """Size factor.

        For d < 8 mm: kb = 1.0.
        For d >= 8 mm: kb = 1.189 * d^(-0.097).
        """
        d = self._config.characteristic_diameter_mm
        if d < 8.0:
            return 1.0
        return 1.189 * d ** (-0.097)

    def _marin_kc(self) -> float:
        """Reliability factor.

        Uses tabulated values for common reliability levels.  For
        intermediate values, linearly interpolates between the two
        nearest tabulated points.
        """
        pct = self._config.reliability_pct

        # Exact match
        if pct in _RELIABILITY_KC:
            return _RELIABILITY_KC[pct]

        # Linear interpolation between nearest points
        sorted_pcts = sorted(_RELIABILITY_KC.keys())
        if pct <= sorted_pcts[0]:
            return _RELIABILITY_KC[sorted_pcts[0]]
        if pct >= sorted_pcts[-1]:
            return _RELIABILITY_KC[sorted_pcts[-1]]

        for i in range(len(sorted_pcts) - 1):
            if sorted_pcts[i] <= pct <= sorted_pcts[i + 1]:
                p0, p1 = sorted_pcts[i], sorted_pcts[i + 1]
                k0, k1 = _RELIABILITY_KC[p0], _RELIABILITY_KC[p1]
                frac = (pct - p0) / (p1 - p0)
                return k0 + frac * (k1 - k0)

        return 1.0  # fallback (should not be reached)

    def _marin_kd(self) -> float:
        """Temperature factor.

        For T < 450 deg C: kd = 1.0.
        For T >= 450 deg C: kd = 1.0 - 0.0058 * (T - 450).
        Clamped to [0.5, 1.0].
        """
        T = self._config.temperature_c
        if T < 450.0:
            return 1.0
        kd = 1.0 - 0.0058 * (T - 450.0)
        return max(kd, 0.5)

    # ------------------------------------------------------------------
    # Corrected endurance limit
    # ------------------------------------------------------------------

    def corrected_endurance_limit(self) -> float:
        """Apply Marin correction factors to the raw endurance limit.

        Returns
        -------
        float
            Corrected endurance limit [MPa].
        """
        factors = self.marin_factors()
        product = (
            factors["ka"]
            * factors["kb"]
            * factors["kc"]
            * factors["kd"]
            * factors["ke"]
        )
        return product * self._mat.sigma_e_mpa

    # ------------------------------------------------------------------
    # S-N life calculation
    # ------------------------------------------------------------------

    def sn_life(self, sigma_a_mpa: float) -> float:
        """Compute cycles to failure from S-N curve.

        Uses the power-law form::

            N = (sigma_a / sigma_f')^(1/b) / 2

        Parameters
        ----------
        sigma_a_mpa : float
            Alternating stress amplitude [MPa].

        Returns
        -------
        float
            Number of cycles to failure.  Returns ``inf`` for zero stress.
        """
        sigma_a = abs(sigma_a_mpa)
        if sigma_a == 0.0:
            return float("inf")

        ratio = sigma_a / self._mat.sigma_f_prime_mpa
        N = ratio ** (1.0 / self._mat.b_exponent) / 2.0
        return N

    # ------------------------------------------------------------------
    # Safety factors
    # ------------------------------------------------------------------

    def simple_safety_factor(self, sigma_alt_mpa: float) -> float:
        """Compute safety factor without mean stress.

        ::

            SF = sigma_e_corrected / (Kt * sigma_alt)

        Parameters
        ----------
        sigma_alt_mpa : float
            Alternating Von Mises stress [MPa].

        Returns
        -------
        float
            Safety factor.  Returns ``inf`` for zero stress.
        """
        if sigma_alt_mpa == 0.0:
            return float("inf")

        se = self.corrected_endurance_limit()
        return se / (self._config.Kt_global * sigma_alt_mpa)

    def goodman_safety_factor(
        self, sigma_alt_mpa: float, sigma_mean_mpa: float
    ) -> float:
        """Compute Goodman safety factor with mean stress.

        ::

            SF = 1 / (sigma_alt / sigma_e + sigma_mean / sigma_UTS)

        Parameters
        ----------
        sigma_alt_mpa : float
            Alternating Von Mises stress [MPa].
        sigma_mean_mpa : float
            Mean Von Mises stress [MPa].

        Returns
        -------
        float
            Safety factor.  Returns ``inf`` if both stresses are zero.
        """
        se = self.corrected_endurance_limit()
        sigma_uts = self._mat.sigma_uts_mpa

        denominator = 0.0
        if se > 0.0:
            denominator += sigma_alt_mpa / se
        if sigma_uts > 0.0:
            denominator += sigma_mean_mpa / sigma_uts

        if denominator == 0.0:
            return float("inf")

        return 1.0 / denominator

    # ------------------------------------------------------------------
    # Critical location finder
    # ------------------------------------------------------------------

    def find_critical_locations(
        self,
        safety_factors: NDArray[np.float64],
        mesh: object,
        n_top: int = 10,
    ) -> list[dict]:
        """Find the n_top elements with lowest safety factor.

        Parameters
        ----------
        safety_factors : (n_elements,)
            Per-element safety factor array.
        mesh : object
            Mesh with ``.nodes`` and ``.elements`` attributes.  Element
            centroids are computed as the mean of the element's node
            coordinates.
        n_top : int
            Number of critical locations to return.

        Returns
        -------
        list[dict]
            List of dicts sorted ascending by safety factor.  Each dict
            contains ``element_id``, ``safety_factor``, ``x``, ``y``, ``z``.
        """
        n_elements = len(safety_factors)
        n_return = min(n_top, n_elements)

        if n_return == 0:
            return []

        # Indices sorted by safety factor ascending
        sorted_idx = np.argsort(safety_factors)[:n_return]

        locations = []
        for idx in sorted_idx:
            # Compute element centroid from node coordinates
            node_ids = mesh.elements[idx]
            coords = mesh.nodes[node_ids]
            centroid = coords.mean(axis=0)

            locations.append(
                {
                    "element_id": int(idx),
                    "safety_factor": float(safety_factors[idx]),
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]),
                }
            )

        return locations

    # ------------------------------------------------------------------
    # Full assessment pipeline
    # ------------------------------------------------------------------

    def assess(
        self,
        stress_alternating: NDArray[np.float64],
        stress_mean: Optional[NDArray[np.float64]] = None,
        mesh: Optional[object] = None,
    ) -> FatigueResult:
        """Run fatigue assessment on a per-element stress field.

        Parameters
        ----------
        stress_alternating : (n_elements,)
            Per-element Von Mises alternating stress [MPa].
        stress_mean : (n_elements,) or None
            Per-element Von Mises mean stress [MPa].  Required when
            ``config.include_mean_stress`` is True.
        mesh : object or None
            Mesh with ``.nodes``, ``.elements``, and optionally
            ``.element_sets`` for Kt region overrides.

        Returns
        -------
        FatigueResult
            Assessment results including per-element safety factors,
            minimum SF, critical location, and estimated life.
        """
        n_elements = len(stress_alternating)

        # Build per-element Kt array
        kt_array = np.full(n_elements, self._config.Kt_global, dtype=np.float64)

        if mesh is not None and hasattr(mesh, "element_sets"):
            for set_name, kt_val in self._config.Kt_regions.items():
                if set_name in mesh.element_sets:
                    elem_ids = mesh.element_sets[set_name]
                    kt_array[elem_ids] = kt_val

        # Effective alternating stress (with Kt)
        effective_alt = kt_array * np.asarray(stress_alternating, dtype=np.float64)

        # Compute per-element safety factors
        safety_factors = np.empty(n_elements, dtype=np.float64)

        if self._config.include_mean_stress and stress_mean is not None:
            mean_arr = np.asarray(stress_mean, dtype=np.float64)
            for i in range(n_elements):
                safety_factors[i] = self.goodman_safety_factor(
                    effective_alt[i], mean_arr[i]
                )
        else:
            se = self.corrected_endurance_limit()
            for i in range(n_elements):
                if effective_alt[i] == 0.0:
                    safety_factors[i] = float("inf")
                else:
                    safety_factors[i] = se / effective_alt[i]

        # Find critical element
        crit_idx = int(np.argmin(safety_factors))
        min_sf = float(safety_factors[crit_idx])

        # Critical location coordinates
        if mesh is not None:
            node_ids = mesh.elements[crit_idx]
            coords = mesh.nodes[node_ids]
            critical_location = coords.mean(axis=0).astype(np.float64)
        else:
            critical_location = np.zeros(3, dtype=np.float64)

        # Estimated life at critical element
        crit_stress = float(effective_alt[crit_idx])
        estimated_life = self.sn_life(crit_stress)

        # Log summary
        logger.info(
            "Fatigue assessment: material=%s, min_SF=%.3f, "
            "estimated_life=%.2e cycles, critical_element=%d",
            self._config.material,
            min_sf,
            estimated_life,
            crit_idx,
        )

        return FatigueResult(
            safety_factors=safety_factors,
            min_safety_factor=min_sf,
            critical_location=critical_location,
            estimated_life_cycles=estimated_life,
            sn_curve_name=self._config.material,
        )
