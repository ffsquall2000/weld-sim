"""Tests for fatigue assessment module.

Covers S-N interpolation, Marin correction factors, Goodman diagram,
simple safety factor, critical location identification, and the full
``assess()`` pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.fatigue import (
    FatigueAssessor,
    FatigueConfig,
    FatigueMaterialData,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import FatigueResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_mesh(n_elements: int = 10, n_nodes: int = 20):
    """Create a minimal mock mesh with element centroids for testing.

    Returns an object with ``.nodes``, ``.elements``, and ``.element_sets``
    attributes matching the FEAMesh interface used by fatigue module.
    """

    @dataclass
    class _MockMesh:
        nodes: np.ndarray
        elements: np.ndarray
        element_sets: dict = field(default_factory=dict)

    nodes = np.random.default_rng(42).uniform(0, 0.1, (n_nodes, 3))
    # Each element references 4 node indices (simplified TET4 for test)
    elements = np.zeros((n_elements, 4), dtype=int)
    for e in range(n_elements):
        elements[e] = np.random.default_rng(e).choice(n_nodes, 4, replace=False)
    return _MockMesh(nodes=nodes, elements=elements)


# ===================================================================
# 1. S-N Database
# ===================================================================


class TestSNDatabase:
    """Verify the built-in S-N material database."""

    EXPECTED_MATERIALS = ["Ti-6Al-4V", "Al 7075-T6", "Steel D2", "CPM 10V", "M2 HSS"]

    def test_all_materials_present(self):
        for name in self.EXPECTED_MATERIALS:
            assert name in FatigueAssessor.SN_DATABASE, f"{name} missing from SN_DATABASE"

    def test_material_fields(self):
        mat = FatigueAssessor.SN_DATABASE["Ti-6Al-4V"]
        assert isinstance(mat, FatigueMaterialData)
        assert mat.sigma_f_prime_mpa == 1700.0
        assert mat.b_exponent == -0.095
        assert mat.sigma_e_mpa == 330.0
        assert mat.sigma_uts_mpa == 950.0
        assert mat.sigma_yield_mpa == 880.0

    def test_al7075_values(self):
        mat = FatigueAssessor.SN_DATABASE["Al 7075-T6"]
        assert mat.sigma_f_prime_mpa == 1050.0
        assert mat.b_exponent == -0.110
        assert mat.sigma_e_mpa == 105.0
        assert mat.sigma_uts_mpa == 572.0

    def test_b_exponent_negative(self):
        """All b exponents must be negative (decreasing S-N curve)."""
        for name, mat in FatigueAssessor.SN_DATABASE.items():
            assert mat.b_exponent < 0, f"{name} has non-negative b"

    def test_sigma_e_less_than_uts(self):
        """Endurance limit must be less than UTS."""
        for name, mat in FatigueAssessor.SN_DATABASE.items():
            assert mat.sigma_e_mpa < mat.sigma_uts_mpa, (
                f"{name}: sigma_e ({mat.sigma_e_mpa}) >= UTS ({mat.sigma_uts_mpa})"
            )


# ===================================================================
# 2. S-N Life Calculation
# ===================================================================


class TestSNLife:
    """Test S-N curve: N = (sigma_a / sigma_f')^(1/b) / 2."""

    def test_known_life_ti64(self):
        """Verify S-N life formula: N = (sigma_a / sigma_f')^(1/b) / 2.

        The sigma_f' and b parameters are independent S-N curve-fit
        coefficients; the tabulated sigma_e is a separately measured
        endurance limit.  This test verifies the power-law formula itself.
        """
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        mat = FatigueAssessor.SN_DATABASE["Ti-6Al-4V"]
        # Use a known stress and verify against hand calculation
        sigma_a = 500.0
        expected_N = (sigma_a / mat.sigma_f_prime_mpa) ** (1.0 / mat.b_exponent) / 2.0
        life = fa.sn_life(sigma_a)
        assert life == pytest.approx(expected_N, rel=1e-10)

    def test_high_stress_short_life(self):
        """Higher stress -> fewer cycles."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        life_high = fa.sn_life(800.0)
        life_low = fa.sn_life(400.0)
        assert life_high < life_low

    def test_zero_stress_infinite_life(self):
        """Zero stress should give infinite (or very large) life."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        life = fa.sn_life(0.0)
        assert life == float("inf")

    def test_negative_stress_uses_absolute(self):
        """Negative stress amplitude is physically meaningless; use absolute."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        life_pos = fa.sn_life(500.0)
        life_neg = fa.sn_life(-500.0)
        assert life_pos == life_neg

    def test_analytical_sn_formula(self):
        """Verify N = (sigma_a / sigma_f')^(1/b) / 2 analytically."""
        config = FatigueConfig(material="Steel D2")
        fa = FatigueAssessor(config)
        mat = FatigueAssessor.SN_DATABASE["Steel D2"]
        sigma_a = 700.0
        expected_N = (sigma_a / mat.sigma_f_prime_mpa) ** (1.0 / mat.b_exponent) / 2.0
        assert fa.sn_life(sigma_a) == pytest.approx(expected_N, rel=1e-10)


# ===================================================================
# 3. Marin Correction Factors
# ===================================================================


class TestMarinFactors:
    """Test Marin surface/size/reliability/temperature correction."""

    def test_machined_surface_ka(self):
        config = FatigueConfig(surface_finish="machined")
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert 0.7 <= factors["ka"] <= 0.9

    def test_ground_surface_ka(self):
        config = FatigueConfig(surface_finish="ground")
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert 0.9 <= factors["ka"] <= 0.95

    def test_polished_surface_ka(self):
        config = FatigueConfig(surface_finish="polished")
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert 0.95 <= factors["ka"] <= 1.0

    def test_size_factor_small_diameter(self):
        """For d < 8 mm, kb = 1.0."""
        config = FatigueConfig(characteristic_diameter_mm=5.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kb"] == 1.0

    def test_size_factor_large_diameter(self):
        """For d >= 8 mm, kb = 1.189 * d^(-0.097)."""
        d = 25.0
        config = FatigueConfig(characteristic_diameter_mm=d)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        expected_kb = 1.189 * d ** (-0.097)
        assert factors["kb"] == pytest.approx(expected_kb, rel=1e-6)

    def test_reliability_50pct(self):
        config = FatigueConfig(reliability_pct=50.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kc"] == 1.0

    def test_reliability_90pct(self):
        config = FatigueConfig(reliability_pct=90.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kc"] == pytest.approx(0.897, rel=1e-3)

    def test_reliability_99pct(self):
        config = FatigueConfig(reliability_pct=99.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kc"] == pytest.approx(0.814, rel=1e-3)

    def test_temperature_below_threshold(self):
        """For T < 450 C, kd = 1.0."""
        config = FatigueConfig(temperature_c=25.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kd"] == 1.0

    def test_temperature_above_threshold(self):
        """For T >= 450 C, kd < 1.0."""
        config = FatigueConfig(temperature_c=500.0)
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["kd"] < 1.0

    def test_ke_default(self):
        config = FatigueConfig()
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert factors["ke"] == 1.0

    def test_all_factors_present(self):
        config = FatigueConfig()
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        assert set(factors.keys()) == {"ka", "kb", "kc", "kd", "ke"}


# ===================================================================
# 4. Corrected Endurance Limit
# ===================================================================


class TestCorrectedEnduranceLimit:
    """Test that corrected endurance = ka*kb*kc*kd*ke * sigma_e."""

    def test_default_config_ti64(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        factors = fa.marin_factors()
        product = factors["ka"] * factors["kb"] * factors["kc"] * factors["kd"] * factors["ke"]
        expected = product * 330.0
        assert fa.corrected_endurance_limit() == pytest.approx(expected, rel=1e-10)

    def test_polished_no_derating(self):
        """Polished, small diameter, 50% reliability, room temp -> minimal derating."""
        config = FatigueConfig(
            material="Ti-6Al-4V",
            surface_finish="polished",
            characteristic_diameter_mm=5.0,
            reliability_pct=50.0,
            temperature_c=25.0,
        )
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        # With polished/small/50%/roomtemp, the correction is close to 1.0
        raw = 330.0
        # Should be between 0.95*raw and raw
        assert 0.95 * raw <= se <= raw

    def test_harsh_conditions_reduce_endurance(self):
        """Bad surface + large diameter + high reliability -> lower endurance."""
        config_easy = FatigueConfig(
            material="Ti-6Al-4V",
            surface_finish="polished",
            characteristic_diameter_mm=5.0,
            reliability_pct=50.0,
        )
        config_harsh = FatigueConfig(
            material="Ti-6Al-4V",
            surface_finish="machined",
            characteristic_diameter_mm=50.0,
            reliability_pct=99.0,
        )
        fa_easy = FatigueAssessor(config_easy)
        fa_harsh = FatigueAssessor(config_harsh)
        assert fa_harsh.corrected_endurance_limit() < fa_easy.corrected_endurance_limit()


# ===================================================================
# 5. Simple Safety Factor (no mean stress)
# ===================================================================


class TestSimpleSafetyFactor:
    """SF = sigma_e_corrected / (Kt * sigma_alt)."""

    def test_basic_calculation(self):
        config = FatigueConfig(material="Ti-6Al-4V", Kt_global=1.0)
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sigma_alt = 100.0
        expected_sf = se / (1.0 * sigma_alt)
        assert fa.simple_safety_factor(sigma_alt) == pytest.approx(expected_sf, rel=1e-10)

    def test_with_Kt(self):
        config = FatigueConfig(material="Ti-6Al-4V", Kt_global=2.5)
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sigma_alt = 100.0
        expected_sf = se / (2.5 * sigma_alt)
        assert fa.simple_safety_factor(sigma_alt) == pytest.approx(expected_sf, rel=1e-10)

    def test_zero_stress_infinite_sf(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = fa.simple_safety_factor(0.0)
        assert sf == float("inf")

    def test_higher_stress_lower_sf(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf_low = fa.simple_safety_factor(100.0)
        sf_high = fa.simple_safety_factor(200.0)
        assert sf_high < sf_low


# ===================================================================
# 6. Goodman Safety Factor (mean + alternating stress)
# ===================================================================


class TestGoodmanSafetyFactor:
    """SF = 1 / (sigma_alt/sigma_e + sigma_mean/sigma_UTS)."""

    def test_pure_alternating(self):
        """With zero mean stress, Goodman reduces to sigma_e / sigma_alt."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sigma_alt = 100.0
        sf = fa.goodman_safety_factor(sigma_alt, 0.0)
        expected = se / sigma_alt
        assert sf == pytest.approx(expected, rel=1e-10)

    def test_with_mean_stress(self):
        """Mean stress reduces safety factor."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sigma_uts = 950.0
        sigma_alt = 100.0
        sigma_mean = 200.0
        expected = 1.0 / (sigma_alt / se + sigma_mean / sigma_uts)
        sf = fa.goodman_safety_factor(sigma_alt, sigma_mean)
        assert sf == pytest.approx(expected, rel=1e-10)

    def test_mean_stress_reduces_sf(self):
        """Adding mean stress should always reduce safety factor."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf_no_mean = fa.goodman_safety_factor(100.0, 0.0)
        sf_with_mean = fa.goodman_safety_factor(100.0, 200.0)
        assert sf_with_mean < sf_no_mean

    def test_zero_stress_infinite(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = fa.goodman_safety_factor(0.0, 0.0)
        assert sf == float("inf")

    def test_analytical_formula(self):
        """Verify against hand calculation for Al 7075-T6."""
        config = FatigueConfig(
            material="Al 7075-T6",
            surface_finish="polished",
            characteristic_diameter_mm=5.0,
            reliability_pct=50.0,
            temperature_c=25.0,
        )
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sigma_uts = 572.0
        sigma_alt = 50.0
        sigma_mean = 100.0
        expected = 1.0 / (sigma_alt / se + sigma_mean / sigma_uts)
        assert fa.goodman_safety_factor(sigma_alt, sigma_mean) == pytest.approx(
            expected, rel=1e-10
        )


# ===================================================================
# 7. Critical Location Identification
# ===================================================================


class TestCriticalLocations:
    """Test find_critical_locations returns sorted lowest-SF elements."""

    def test_returns_correct_count(self):
        config = FatigueConfig(material="Ti-6Al-4V", n_critical_locations=5)
        fa = FatigueAssessor(config)
        sf = np.array([3.0, 1.5, 2.0, 0.8, 4.0, 1.2, 2.5, 5.0, 1.0, 3.5])
        mesh = _simple_mesh(n_elements=10)
        locs = fa.find_critical_locations(sf, mesh, n_top=5)
        assert len(locs) == 5

    def test_sorted_ascending_by_sf(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = np.array([3.0, 1.5, 2.0, 0.8, 4.0])
        mesh = _simple_mesh(n_elements=5)
        locs = fa.find_critical_locations(sf, mesh, n_top=5)
        sf_values = [loc["safety_factor"] for loc in locs]
        assert sf_values == sorted(sf_values)

    def test_critical_element_identified(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = np.array([3.0, 1.5, 2.0, 0.8, 4.0])
        mesh = _simple_mesh(n_elements=5)
        locs = fa.find_critical_locations(sf, mesh, n_top=3)
        # Most critical is element 3 (SF=0.8)
        assert locs[0]["element_id"] == 3
        assert locs[0]["safety_factor"] == pytest.approx(0.8)

    def test_location_has_coordinates(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = np.array([2.0, 1.0, 3.0])
        mesh = _simple_mesh(n_elements=3)
        locs = fa.find_critical_locations(sf, mesh, n_top=1)
        assert "x" in locs[0]
        assert "y" in locs[0]
        assert "z" in locs[0]

    def test_n_top_clamped(self):
        """If n_top > number of elements, return all elements."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        sf = np.array([2.0, 1.0])
        mesh = _simple_mesh(n_elements=2)
        locs = fa.find_critical_locations(sf, mesh, n_top=10)
        assert len(locs) == 2


# ===================================================================
# 8. Full assess() Pipeline
# ===================================================================


class TestAssessPipeline:
    """Integration test for the full assess() method."""

    def test_returns_fatigue_result(self):
        config = FatigueConfig(material="Ti-6Al-4V", Kt_global=1.0)
        fa = FatigueAssessor(config)
        # 10 elements, alternating stress per element
        stress_alt = np.array([100.0, 150.0, 200.0, 250.0, 300.0,
                               120.0, 180.0, 220.0, 280.0, 350.0])
        mesh = _simple_mesh(n_elements=10)
        result = fa.assess(stress_alt, mesh=mesh)
        assert isinstance(result, FatigueResult)

    def test_safety_factors_shape(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.ones(20) * 100.0
        mesh = _simple_mesh(n_elements=20)
        result = fa.assess(stress_alt, mesh=mesh)
        assert result.safety_factors.shape == (20,)

    def test_min_sf_is_minimum(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.array([100.0, 500.0, 200.0])
        mesh = _simple_mesh(n_elements=3)
        result = fa.assess(stress_alt, mesh=mesh)
        assert result.min_safety_factor == pytest.approx(np.min(result.safety_factors))

    def test_critical_location_is_3d(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.array([100.0, 500.0, 200.0])
        mesh = _simple_mesh(n_elements=3)
        result = fa.assess(stress_alt, mesh=mesh)
        assert result.critical_location.shape == (3,)

    def test_sn_curve_name(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.ones(5) * 100.0
        mesh = _simple_mesh(n_elements=5)
        result = fa.assess(stress_alt, mesh=mesh)
        assert result.sn_curve_name == "Ti-6Al-4V"

    def test_with_mean_stress(self):
        """When include_mean_stress is True, mean stress lowers SF."""
        config_no_mean = FatigueConfig(material="Ti-6Al-4V", include_mean_stress=False)
        config_with_mean = FatigueConfig(material="Ti-6Al-4V", include_mean_stress=True)

        fa_no = FatigueAssessor(config_no_mean)
        fa_mean = FatigueAssessor(config_with_mean)

        stress_alt = np.array([100.0, 200.0, 150.0])
        stress_mean = np.array([50.0, 100.0, 75.0])
        mesh = _simple_mesh(n_elements=3)

        result_no = fa_no.assess(stress_alt, mesh=mesh)
        result_mean = fa_mean.assess(stress_alt, stress_mean=stress_mean, mesh=mesh)

        assert result_mean.min_safety_factor < result_no.min_safety_factor

    def test_estimated_life_at_critical(self):
        """Estimated life should correspond to the most critical element."""
        config = FatigueConfig(material="Ti-6Al-4V", Kt_global=1.0)
        fa = FatigueAssessor(config)
        stress_alt = np.array([100.0, 400.0, 200.0])
        mesh = _simple_mesh(n_elements=3)
        result = fa.assess(stress_alt, mesh=mesh)
        # The most critical element has stress 400 MPa
        # Estimated life should match sn_life for that stress
        expected_life = fa.sn_life(config.Kt_global * 400.0)
        assert result.estimated_life_cycles == pytest.approx(expected_life, rel=1e-6)

    def test_assess_without_mesh(self):
        """assess() works without mesh; critical_location defaults to origin."""
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.array([100.0, 200.0, 300.0])
        result = fa.assess(stress_alt)
        assert isinstance(result, FatigueResult)
        np.testing.assert_array_equal(result.critical_location, np.zeros(3))


# ===================================================================
# 9. Kt Regions (per-element-set stress concentration)
# ===================================================================


class TestKtRegions:
    """Test region-specific stress concentration factors."""

    def test_kt_regions_applied(self):
        """Elements in a named set should use their set's Kt."""
        mesh = _simple_mesh(n_elements=6)
        # Mark elements 0,1 as "threaded" with Kt=4.0
        mesh.element_sets = {"threaded": np.array([0, 1])}

        config = FatigueConfig(
            material="Ti-6Al-4V",
            Kt_global=1.0,
            Kt_regions={"threaded": 4.0},
        )
        fa = FatigueAssessor(config)
        stress_alt = np.ones(6) * 100.0

        result = fa.assess(stress_alt, mesh=mesh)
        # Elements 0,1 should have lower SF due to higher Kt
        assert result.safety_factors[0] < result.safety_factors[2]
        assert result.safety_factors[0] == pytest.approx(result.safety_factors[1])


# ===================================================================
# 10. Edge Cases and Validation
# ===================================================================


class TestEdgeCases:
    """Boundary and error conditions."""

    def test_unknown_material_raises(self):
        config = FatigueConfig(material="Unobtanium")
        with pytest.raises(KeyError):
            FatigueAssessor(config)

    def test_unknown_surface_finish_raises(self):
        config = FatigueConfig(surface_finish="laser_peened")
        with pytest.raises(ValueError):
            FatigueAssessor(config)

    def test_single_element(self):
        config = FatigueConfig(material="Ti-6Al-4V")
        fa = FatigueAssessor(config)
        stress_alt = np.array([150.0])
        mesh = _simple_mesh(n_elements=1)
        result = fa.assess(stress_alt, mesh=mesh)
        assert result.safety_factors.shape == (1,)

    def test_all_materials_give_positive_sf(self):
        """All S-N materials should produce positive SF for valid stress."""
        for mat_name in FatigueAssessor.SN_DATABASE:
            config = FatigueConfig(material=mat_name)
            fa = FatigueAssessor(config)
            sf = fa.simple_safety_factor(100.0)
            assert sf > 0, f"{mat_name} gave non-positive SF"

    def test_very_high_stress_low_sf(self):
        """Stress exceeding endurance limit should give SF < 1."""
        config = FatigueConfig(material="Al 7075-T6", Kt_global=1.0)
        fa = FatigueAssessor(config)
        se = fa.corrected_endurance_limit()
        sf = fa.simple_safety_factor(se * 2.0)
        assert sf < 1.0
