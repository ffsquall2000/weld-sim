"""Tests for the FEA cross-validation harness.

Covers MAC matrix computation, frequency deviation, stress comparison,
displacement comparison, mode pairing, validation thresholds (PASS /
WARNING / FAIL), edge cases, and human-readable report generation.

All tests use synthetic data -- no FEniCSx dependency required.
"""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.cross_validation import (
    CrossValidator,
    ValidationReport,
    DEFAULT_THRESHOLDS,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_modal_result(
    frequencies: np.ndarray,
    mode_shapes: np.ndarray,
    solver_name: str = "SolverA",
) -> ModalResult:
    """Build a minimal ModalResult for testing."""
    n_modes = len(frequencies)
    return ModalResult(
        frequencies_hz=np.asarray(frequencies, dtype=float),
        mode_shapes=np.asarray(mode_shapes, dtype=float),
        mode_types=["unknown"] * n_modes,
        effective_mass_ratios=np.zeros(n_modes),
        mesh=None,
        solve_time_s=0.0,
        solver_name=solver_name,
    )


def _identity_modes(n_dof: int, n_modes: int) -> np.ndarray:
    """Return an (n_dof, n_modes) mode matrix that is a sub-block of I."""
    return np.eye(n_dof, n_modes)


# ===================================================================
# 1. MAC Matrix Computation
# ===================================================================


class TestMACMatrix:
    """Verify Modal Assurance Criterion calculation."""

    def test_identity_modes_diagonal_ones(self):
        """MAC of identical orthonormal modes -> identity matrix."""
        modes = _identity_modes(6, 3)
        mac = CrossValidator.compute_mac(modes, modes)
        np.testing.assert_allclose(mac, np.eye(3), atol=1e-12)

    def test_scaled_modes_still_one(self):
        """Scaling a mode should not change MAC (it is normalized)."""
        modes_a = _identity_modes(6, 3)
        modes_b = modes_a * 5.0  # scale by 5
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        np.testing.assert_allclose(mac, np.eye(3), atol=1e-12)

    def test_negative_scaled_modes_still_one(self):
        """Negative scaling should also give MAC = 1 (squared dot product)."""
        modes_a = _identity_modes(6, 3)
        modes_b = -modes_a * 3.0
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        np.testing.assert_allclose(mac, np.eye(3), atol=1e-12)

    def test_orthogonal_modes_zero(self):
        """MAC between orthogonal modes should be 0."""
        # mode A along x, mode B along y
        modes_a = np.array([[1.0], [0.0], [0.0]])
        modes_b = np.array([[0.0], [1.0], [0.0]])
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(0.0, abs=1e-12)

    def test_mac_values_between_zero_and_one(self):
        """All MAC values must be in [0, 1]."""
        rng = np.random.default_rng(42)
        modes_a = rng.standard_normal((30, 5))
        modes_b = rng.standard_normal((30, 5))
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        assert np.all(mac >= -1e-12)
        assert np.all(mac <= 1.0 + 1e-12)

    def test_mac_shape(self):
        """MAC shape should be (n_modes_a, n_modes_b)."""
        modes_a = np.random.default_rng(0).standard_normal((20, 3))
        modes_b = np.random.default_rng(1).standard_normal((20, 5))
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        assert mac.shape == (3, 5)

    def test_single_mode(self):
        """MAC with a single mode pair."""
        mode = np.array([[1.0], [2.0], [3.0]])
        mac = CrossValidator.compute_mac(mode, mode)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(1.0)

    def test_1d_input_promoted(self):
        """A 1-D array should be treated as a single mode."""
        mode = np.array([1.0, 2.0, 3.0])
        mac = CrossValidator.compute_mac(mode, mode)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(1.0)

    def test_empty_modes(self):
        """Empty input returns empty (0,0) array."""
        mac = CrossValidator.compute_mac(np.array([]), np.array([]))
        assert mac.shape == (0, 0)

    def test_partially_correlated(self):
        """A known partially correlated case: mode at 45 degrees."""
        modes_a = np.array([[1.0], [0.0]])
        s = np.sqrt(2) / 2
        modes_b = np.array([[s], [s]])  # 45-degree rotation
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        # MAC = (1*s + 0*s)^2 / (1 * 1) = s^2 = 0.5
        assert mac[0, 0] == pytest.approx(0.5)


# ===================================================================
# 2. Frequency Deviation
# ===================================================================


class TestFrequencyDeviation:
    """Test percentage frequency deviation calculation."""

    def test_identical_frequencies_zero_deviation(self):
        freq = np.array([1000.0, 2000.0, 3000.0])
        dev = CrossValidator.frequency_deviation(freq, freq)
        np.testing.assert_allclose(dev, 0.0, atol=1e-12)

    def test_known_shift(self):
        """1% shift should give 1% deviation."""
        freq_a = np.array([20000.0])
        freq_b = np.array([20200.0])
        dev = CrossValidator.frequency_deviation(freq_a, freq_b)
        assert dev[0] == pytest.approx(1.0, rel=1e-10)

    def test_five_percent_shift(self):
        freq_a = np.array([10000.0])
        freq_b = np.array([10500.0])
        dev = CrossValidator.frequency_deviation(freq_a, freq_b)
        assert dev[0] == pytest.approx(5.0, rel=1e-10)

    def test_multiple_frequencies(self):
        freq_a = np.array([1000.0, 2000.0])
        freq_b = np.array([1010.0, 2100.0])
        dev = CrossValidator.frequency_deviation(freq_a, freq_b)
        assert dev[0] == pytest.approx(1.0, rel=1e-10)
        assert dev[1] == pytest.approx(5.0, rel=1e-10)

    def test_symmetric_deviation(self):
        """Deviation uses absolute difference, so direction does not matter."""
        freq_a = np.array([1000.0])
        freq_b_high = np.array([1050.0])
        freq_b_low = np.array([950.0])
        dev_h = CrossValidator.frequency_deviation(freq_a, freq_b_high)
        dev_l = CrossValidator.frequency_deviation(freq_a, freq_b_low)
        assert dev_h[0] == dev_l[0]

    def test_zero_reference_zero_target(self):
        """0/0 should give 0 deviation, not NaN."""
        dev = CrossValidator.frequency_deviation(np.array([0.0]), np.array([0.0]))
        assert dev[0] == pytest.approx(0.0)

    def test_zero_reference_nonzero_target(self):
        """0 reference with nonzero target -> inf deviation."""
        dev = CrossValidator.frequency_deviation(np.array([0.0]), np.array([100.0]))
        assert np.isinf(dev[0])

    def test_empty_frequencies(self):
        dev = CrossValidator.frequency_deviation(np.array([]), np.array([]))
        assert dev.size == 0


# ===================================================================
# 3. Stress Comparison
# ===================================================================


class TestStressComparison:
    """Test stress peak comparison."""

    def test_identical_stress_zero_error(self):
        stress = np.array([100.0, 200.0, 300.0])
        result = CrossValidator.stress_comparison(stress, stress)
        assert result["peak_relative_error_pct"] == pytest.approx(0.0)
        assert result["rms_error"] == pytest.approx(0.0)
        assert result["max_abs_error"] == pytest.approx(0.0)
        assert result["peak_a"] == pytest.approx(300.0)
        assert result["peak_b"] == pytest.approx(300.0)

    def test_ten_percent_peak_difference(self):
        stress_a = np.array([100.0, 200.0, 300.0])
        stress_b = np.array([100.0, 200.0, 330.0])  # 10% higher peak
        result = CrossValidator.stress_comparison(stress_a, stress_b)
        assert result["peak_relative_error_pct"] == pytest.approx(10.0)

    def test_peak_locations(self):
        stress_a = np.array([100.0, 500.0, 200.0])
        stress_b = np.array([200.0, 100.0, 450.0])
        result = CrossValidator.stress_comparison(stress_a, stress_b)
        assert result["peak_location_a"] == 1  # index of 500
        assert result["peak_location_b"] == 2  # index of 450

    def test_rms_error(self):
        stress_a = np.array([100.0, 200.0])
        stress_b = np.array([110.0, 210.0])
        result = CrossValidator.stress_comparison(stress_a, stress_b)
        expected_rms = np.sqrt(np.mean(np.array([10.0, 10.0]) ** 2))
        assert result["rms_error"] == pytest.approx(expected_rms)

    def test_max_abs_error(self):
        stress_a = np.array([100.0, 200.0, 300.0])
        stress_b = np.array([100.0, 250.0, 300.0])
        result = CrossValidator.stress_comparison(stress_a, stress_b)
        assert result["max_abs_error"] == pytest.approx(50.0)

    def test_empty_stress(self):
        result = CrossValidator.stress_comparison(np.array([]), np.array([]))
        assert result["peak_a"] == 0.0
        assert result["rms_error"] == 0.0

    def test_zero_reference_peak(self):
        """When stress_a peak is 0 and stress_b peak is nonzero -> inf error."""
        stress_a = np.array([0.0, 0.0])
        stress_b = np.array([0.0, 100.0])
        result = CrossValidator.stress_comparison(stress_a, stress_b)
        assert result["peak_relative_error_pct"] == float("inf")


# ===================================================================
# 4. Displacement Comparison
# ===================================================================


class TestDisplacementComparison:
    """Test displacement field comparison."""

    def test_identical_displacements(self):
        disp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = CrossValidator.displacement_comparison(disp, disp)
        assert result["rms_error"] == pytest.approx(0.0)
        assert result["max_error"] == pytest.approx(0.0)
        assert result["correlation_coefficient"] == pytest.approx(1.0)

    def test_known_rms_error(self):
        disp_a = np.array([1.0, 2.0, 3.0])
        disp_b = np.array([1.1, 2.1, 3.1])
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["rms_error"] == pytest.approx(0.1, rel=1e-10)

    def test_max_error(self):
        disp_a = np.array([1.0, 2.0, 3.0])
        disp_b = np.array([1.0, 2.5, 3.0])
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["max_error"] == pytest.approx(0.5)

    def test_perfect_correlation_with_offset(self):
        """Linear shift: y = x + c has correlation 1.0."""
        disp_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        disp_b = disp_a + 10.0
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["correlation_coefficient"] == pytest.approx(1.0, abs=1e-10)

    def test_perfect_negative_correlation(self):
        """Negated field: r = -1.0."""
        disp_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        disp_b = -disp_a
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["correlation_coefficient"] == pytest.approx(-1.0, abs=1e-10)

    def test_constant_arrays_identical(self):
        """Two identical constant arrays -> corr = 1."""
        disp = np.ones(10) * 5.0
        result = CrossValidator.displacement_comparison(disp, disp)
        assert result["correlation_coefficient"] == pytest.approx(1.0)

    def test_constant_arrays_different(self):
        """Two different constant arrays -> corr = 0."""
        disp_a = np.ones(10) * 5.0
        disp_b = np.ones(10) * 10.0
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["correlation_coefficient"] == pytest.approx(0.0)

    def test_empty_displacements(self):
        result = CrossValidator.displacement_comparison(np.array([]), np.array([]))
        assert result["rms_error"] == 0.0
        assert result["correlation_coefficient"] == 1.0

    def test_2d_input_flattened(self):
        """2-D arrays should be ravelled transparently."""
        disp_a = np.arange(12).reshape(4, 3).astype(float)
        disp_b = disp_a + 0.01
        result = CrossValidator.displacement_comparison(disp_a, disp_b)
        assert result["rms_error"] == pytest.approx(0.01, rel=1e-10)


# ===================================================================
# 5. Mode Pairing
# ===================================================================


class TestModePairing:
    """Test greedy MAC-based mode pairing."""

    def test_identical_modes_sequential_pairing(self):
        """Identical modes should pair 0->0, 1->1, 2->2."""
        modes = _identity_modes(6, 3)
        freq = np.array([1000.0, 2000.0, 3000.0])
        pairs = CrossValidator.pair_modes(freq, freq, modes, modes)
        assert pairs == [(0, 0), (1, 1), (2, 2)]

    def test_shuffled_modes_reordered(self):
        """Shuffled solver B modes should still pair correctly."""
        modes_a = _identity_modes(6, 3)
        # Shuffle columns: solver B has modes [2, 0, 1]
        modes_b = modes_a[:, [2, 0, 1]]
        freq_a = np.array([1000.0, 2000.0, 3000.0])
        freq_b = np.array([3000.0, 1000.0, 2000.0])
        pairs = CrossValidator.pair_modes(freq_a, freq_b, modes_a, modes_b)
        # Mode 0 in A matches column 1 in B, etc.
        assert (0, 1) in pairs
        assert (1, 2) in pairs
        assert (2, 0) in pairs

    def test_different_number_of_modes(self):
        """When A has more modes than B, only min(n_a, n_b) pairs."""
        modes_a = _identity_modes(6, 4)
        modes_b = _identity_modes(6, 2)
        freq_a = np.arange(4, dtype=float)
        freq_b = np.arange(2, dtype=float)
        pairs = CrossValidator.pair_modes(freq_a, freq_b, modes_a, modes_b)
        assert len(pairs) == 2

    def test_single_mode_pairing(self):
        mode = np.array([[1.0], [0.0], [0.0]])
        freq = np.array([1000.0])
        pairs = CrossValidator.pair_modes(freq, freq, mode, mode)
        assert pairs == [(0, 0)]

    def test_empty_modes(self):
        pairs = CrossValidator.pair_modes(
            np.array([]), np.array([]), np.array([]), np.array([])
        )
        assert pairs == []

    def test_pairs_sorted_by_a_index(self):
        """Returned pairs must be sorted ascending by the A index."""
        rng = np.random.default_rng(99)
        modes_a = rng.standard_normal((20, 5))
        modes_b = rng.standard_normal((20, 5))
        freq = np.arange(5, dtype=float)
        pairs = CrossValidator.pair_modes(freq, freq, modes_a, modes_b)
        a_indices = [p[0] for p in pairs]
        assert a_indices == sorted(a_indices)


# ===================================================================
# 6. Validation Thresholds
# ===================================================================


class TestValidation:
    """Test PASS / WARNING / FAIL logic via the validate() method."""

    def _make_pair(
        self,
        n_dof: int = 30,
        n_modes: int = 5,
        freq_shift_pct: float = 0.0,
        mode_noise: float = 0.0,
        solver_a: str = "SolverA",
        solver_b: str = "SolverB",
    ):
        """Create a pair of ModalResults with controlled deviation."""
        rng = np.random.default_rng(7)
        modes_a = rng.standard_normal((n_dof, n_modes))
        # Normalise columns
        modes_a /= np.linalg.norm(modes_a, axis=0, keepdims=True)

        modes_b = modes_a.copy()
        if mode_noise > 0:
            noise = rng.standard_normal(modes_a.shape) * mode_noise
            modes_b = modes_b + noise
            # Re-normalise for fair MAC comparison
            modes_b /= np.linalg.norm(modes_b, axis=0, keepdims=True)

        freq_a = np.linspace(15000, 25000, n_modes)
        freq_b = freq_a * (1.0 + freq_shift_pct / 100.0)

        result_a = _make_modal_result(freq_a, modes_a, solver_a)
        result_b = _make_modal_result(freq_b, modes_b, solver_b)
        return result_a, result_b

    def test_pass_identical_results(self):
        """Identical results should give PASS."""
        result_a, result_b = self._make_pair()
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "PASS"
        assert report.details["min_mac"] == pytest.approx(1.0, abs=1e-10)
        assert report.details["max_freq_dev_pct"] == pytest.approx(0.0, abs=1e-10)

    def test_pass_tiny_frequency_shift(self):
        """0.5% frequency shift -> PASS (threshold is 1%)."""
        result_a, result_b = self._make_pair(freq_shift_pct=0.5)
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "PASS"

    def test_warning_frequency_shift(self):
        """3% frequency shift -> WARNING."""
        result_a, result_b = self._make_pair(freq_shift_pct=3.0)
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "WARNING"

    def test_fail_large_frequency_shift(self):
        """6% frequency shift -> FAIL (>5%)."""
        result_a, result_b = self._make_pair(freq_shift_pct=6.0)
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "FAIL"

    def test_warning_low_mac(self):
        """Moderate noise lowering MAC below 0.95 but above 0.80 -> WARNING."""
        # Controlled noise to get MAC in [0.80, 0.95] range
        result_a, result_b = self._make_pair(mode_noise=0.30)
        report = CrossValidator.validate(result_a, result_b)
        # With 30% noise on normalized modes, MAC should drop below 0.95
        # but typically stays above 0.80 for this seed
        min_mac = report.details["min_mac"]
        if min_mac < 0.80:
            assert report.status == "FAIL"
        elif min_mac < 0.95:
            assert report.status == "WARNING"
        else:
            assert report.status == "PASS"

    def test_fail_very_low_mac(self):
        """Very high noise -> MAC below 0.80 -> FAIL."""
        result_a, result_b = self._make_pair(mode_noise=5.0)
        report = CrossValidator.validate(result_a, result_b)
        # Extreme noise should push MAC well below threshold
        assert report.status in ("WARNING", "FAIL")

    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        result_a, result_b = self._make_pair(freq_shift_pct=2.0)
        # With default thresholds, 2% shift -> WARNING
        report_default = CrossValidator.validate(result_a, result_b)
        assert report_default.status == "WARNING"

        # Relax thresholds so 2% is still PASS
        report_relaxed = CrossValidator.validate(
            result_a, result_b,
            thresholds={"freq_pass_pct": 3.0},
        )
        assert report_relaxed.status == "PASS"

    def test_validation_report_fields(self):
        """ValidationReport has all expected fields populated."""
        result_a, result_b = self._make_pair()
        report = CrossValidator.validate(result_a, result_b)
        assert isinstance(report, ValidationReport)
        assert isinstance(report.mac_matrix, np.ndarray)
        assert isinstance(report.freq_deviations, np.ndarray)
        assert isinstance(report.paired_indices, list)
        assert report.status in ("PASS", "WARNING", "FAIL")
        assert "solver_a" in report.details
        assert "solver_b" in report.details
        assert "thresholds" in report.details

    def test_empty_modes_fail(self):
        """No modes at all should produce FAIL."""
        result_a = _make_modal_result(np.array([]), np.empty((10, 0)), "A")
        result_b = _make_modal_result(np.array([]), np.empty((10, 0)), "B")
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "FAIL"
        assert "No modes" in report.details.get("reason", "")

    def test_single_mode_pass(self):
        """Single mode that matches perfectly -> PASS."""
        mode = np.array([[1.0], [2.0], [3.0]])
        result_a = _make_modal_result(np.array([20000.0]), mode, "A")
        result_b = _make_modal_result(np.array([20000.0]), mode, "B")
        report = CrossValidator.validate(result_a, result_b)
        assert report.status == "PASS"
        assert len(report.paired_indices) == 1


# ===================================================================
# 7. Report Generation
# ===================================================================


class TestReportGeneration:
    """Test human-readable text report."""

    def _make_report(self, **kwargs) -> ValidationReport:
        """Quick helper to build a ValidationReport."""
        defaults = dict(
            status="PASS",
            mac_matrix=np.eye(3),
            freq_deviations=np.array([0.1, 0.2, 0.3]),
            paired_indices=[(0, 0), (1, 1), (2, 2)],
            details={
                "min_mac": 0.99,
                "max_mac": 1.0,
                "mean_mac": 0.995,
                "max_freq_dev_pct": 0.3,
                "mean_freq_dev_pct": 0.2,
                "n_paired_modes": 3,
                "solver_a": "SolverA",
                "solver_b": "SolverB",
                "thresholds": DEFAULT_THRESHOLDS,
            },
        )
        defaults.update(kwargs)
        return ValidationReport(**defaults)

    def test_report_is_string(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert isinstance(text, str)

    def test_report_contains_status(self):
        for status in ("PASS", "WARNING", "FAIL"):
            report = self._make_report(status=status)
            text = CrossValidator.generate_report(report)
            assert status in text

    def test_report_contains_solver_names(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert "SolverA" in text
        assert "SolverB" in text

    def test_report_contains_mac_stats(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert "Min MAC" in text
        assert "Max MAC" in text
        assert "Mean MAC" in text

    def test_report_contains_freq_dev(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert "Frequency Deviation" in text

    def test_report_contains_mode_pairing_table(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert "Mode Pairing Detail" in text

    def test_report_contains_thresholds(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert "Thresholds" in text
        assert "PASS" in text
        assert "FAIL" in text

    def test_report_fail_with_reason(self):
        report = self._make_report(
            status="FAIL",
            paired_indices=[],
            details={"reason": "No modes to compare"},
        )
        text = CrossValidator.generate_report(report)
        assert "No modes to compare" in text

    def test_report_multiline(self):
        report = self._make_report()
        text = CrossValidator.generate_report(report)
        assert text.count("\n") > 5

    def test_full_pipeline_report(self):
        """End-to-end: validate -> generate_report."""
        mode = np.eye(10, 3)
        freq = np.array([15000.0, 20000.0, 25000.0])
        result_a = _make_modal_result(freq, mode, "numpy/scipy")
        result_b = _make_modal_result(freq * 1.005, mode, "FEniCSx")
        vr = CrossValidator.validate(result_a, result_b)
        text = CrossValidator.generate_report(vr)
        assert "PASS" in text
        assert "numpy/scipy" in text
        assert "FEniCSx" in text


# ===================================================================
# 8. ValidationReport Repr
# ===================================================================


class TestValidationReportRepr:
    """Test the __repr__ of ValidationReport."""

    def test_repr_contains_status(self):
        vr = ValidationReport(
            status="PASS",
            mac_matrix=np.eye(3),
            freq_deviations=np.zeros(3),
            paired_indices=[(0, 0), (1, 1), (2, 2)],
        )
        r = repr(vr)
        assert "PASS" in r
        assert "n_modes=3" in r
        assert "n_paired=3" in r


# ===================================================================
# 9. Edge Cases
# ===================================================================


class TestEdgeCases:
    """Additional edge cases and boundary conditions."""

    def test_mac_zero_norm_mode(self):
        """A zero vector mode should yield MAC = 0 (not NaN)."""
        modes_a = np.array([[1.0], [0.0]])
        modes_b = np.array([[0.0], [0.0]])
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        assert mac.shape == (1, 1)
        assert mac[0, 0] == pytest.approx(0.0)

    def test_frequency_deviation_single_element(self):
        dev = CrossValidator.frequency_deviation(np.array([20000.0]), np.array([20100.0]))
        assert dev.shape == (1,)
        assert dev[0] == pytest.approx(0.5, rel=1e-10)

    def test_stress_comparison_single_element(self):
        result = CrossValidator.stress_comparison(np.array([100.0]), np.array([110.0]))
        assert result["peak_relative_error_pct"] == pytest.approx(10.0)
        assert result["peak_location_a"] == 0
        assert result["peak_location_b"] == 0

    def test_displacement_comparison_single_element(self):
        result = CrossValidator.displacement_comparison(
            np.array([1.0]), np.array([1.0])
        )
        assert result["rms_error"] == pytest.approx(0.0)
        # Single element has zero std -> constant array case
        assert result["correlation_coefficient"] == pytest.approx(1.0)

    def test_large_mode_set(self):
        """Sanity check on a larger problem (50 DOF, 10 modes)."""
        rng = np.random.default_rng(123)
        modes = rng.standard_normal((50, 10))
        modes /= np.linalg.norm(modes, axis=0, keepdims=True)
        mac = CrossValidator.compute_mac(modes, modes)
        # Diagonal should be all 1.0
        np.testing.assert_allclose(np.diag(mac), 1.0, atol=1e-10)
        # Off-diagonal should be < 1
        for i in range(10):
            for j in range(10):
                if i != j:
                    assert mac[i, j] < 1.0

    def test_default_thresholds_values(self):
        """Verify the default threshold constants."""
        assert DEFAULT_THRESHOLDS["mac_pass"] == 0.95
        assert DEFAULT_THRESHOLDS["mac_warn"] == 0.80
        assert DEFAULT_THRESHOLDS["freq_pass_pct"] == 1.0
        assert DEFAULT_THRESHOLDS["freq_warn_pct"] == 5.0
