"""Tests for mesh convergence automation module.

All tests use mocked mesher and solver objects to avoid requiring Gmsh
or actual FEA solves.  Synthetic convergence data with known analytical
properties is used to verify Richardson extrapolation, convergence
detection, and recommendation logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_convergence import (
    ConvergenceResult,
    MeshConvergenceStudy,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_mesh(n_nodes: int = 100) -> MagicMock:
    """Create a mock FEAMesh with controllable n_dof."""
    mesh = MagicMock()
    mesh.n_dof = n_nodes * 3
    mesh.nodes = np.zeros((n_nodes, 3))
    return mesh


def _make_mock_mesher(dof_schedule: list[int]) -> MagicMock:
    """Create a mock mesher that returns meshes with predetermined DOFs.

    Parameters
    ----------
    dof_schedule : list[int]
        Sequence of n_dof values to return on successive calls.
    """
    mesher = MagicMock()
    call_count = {"i": 0}

    def _mesh_parametric_horn(**kwargs: Any) -> MagicMock:
        idx = call_count["i"]
        call_count["i"] += 1
        n_nodes = dof_schedule[idx] // 3
        return _make_mock_mesh(n_nodes)

    mesher.mesh_parametric_horn.side_effect = _mesh_parametric_horn
    return mesher


def _make_mock_solver(frequencies_schedule: list[float]) -> MagicMock:
    """Create a mock solver that returns predetermined frequencies.

    Parameters
    ----------
    frequencies_schedule : list[float]
        Fundamental frequency to return on each successive call.
    """
    solver = MagicMock()
    call_count = {"i": 0}

    def _modal_analysis(config: Any) -> MagicMock:
        idx = call_count["i"]
        call_count["i"] += 1
        result = MagicMock()
        freq = frequencies_schedule[idx]
        result.frequencies_hz = np.array([freq, freq * 1.5, freq * 2.0])
        result.mode_shapes = np.array([
            np.random.default_rng(idx).uniform(-1, 1, 30),
            np.random.default_rng(idx + 100).uniform(-1, 1, 30),
            np.random.default_rng(idx + 200).uniform(-1, 1, 30),
        ])
        return result

    solver.modal_analysis.side_effect = _modal_analysis
    return solver


def _h2_convergence_values(
    mesh_sizes: list[float], exact: float = 20000.0, coeff: float = 100.0
) -> list[float]:
    """Simulate h^2 convergence to an exact value.

    f(h) = exact + coeff * (h / h_min)^2

    For the finest mesh h_min the error is *coeff*.
    """
    h_min = min(mesh_sizes)
    return [exact + coeff * (h / h_min) ** 2 for h in mesh_sizes]


def _build_study(
    mesh_sizes: list[float],
    values: list[float],
    dof_schedule: list[int] | None = None,
    convergence_threshold: float = 0.5,
) -> tuple[MeshConvergenceStudy, list[float], list[int]]:
    """Build a MeshConvergenceStudy with mocked mesher/solver.

    Returns (study, sorted_values, sorted_dofs) where items are sorted
    coarsest-to-finest (descending mesh_size) matching run_study's internal
    sort.
    """
    # Sort mesh_sizes descending (coarsest first) to match run_study
    order = sorted(range(len(mesh_sizes)), key=lambda i: mesh_sizes[i], reverse=True)
    sorted_sizes = [mesh_sizes[i] for i in order]
    sorted_values = [values[i] for i in order]

    if dof_schedule is None:
        # Approximate: smaller mesh_size -> more DOFs
        dof_schedule_sorted = [int(3000 * (10.0 / ms) ** 2) for ms in sorted_sizes]
    else:
        dof_schedule_sorted = [dof_schedule[i] for i in order]

    mesher = _make_mock_mesher(dof_schedule_sorted)
    solver = _make_mock_solver(sorted_values)

    study = MeshConvergenceStudy(
        mesher=mesher,
        solver=solver,
        horn_type="cylindrical",
        dimensions={"diameter_mm": 50.0, "length_mm": 127.0},
        material_name="Ti-6Al-4V",
        convergence_threshold=convergence_threshold,
    )
    return study, sorted_values, dof_schedule_sorted


# ===================================================================
# 1. Synthetic h^2 convergence (monotonic approach to limit)
# ===================================================================


class TestSyntheticH2Convergence:
    """Verify convergence study with known h^2 convergence behaviour."""

    MESH_SIZES = [10.0, 7.0, 5.0, 3.5, 2.5]
    EXACT = 20000.0

    def _get_result(self) -> ConvergenceResult:
        values = _h2_convergence_values(self.MESH_SIZES, self.EXACT)
        study, _, _ = _build_study(self.MESH_SIZES, values)
        return study.run_study(self.MESH_SIZES, target_quantity="frequency")

    def test_returns_convergence_result(self):
        result = self._get_result()
        assert isinstance(result, ConvergenceResult)

    def test_mesh_sizes_sorted_descending(self):
        result = self._get_result()
        assert result.mesh_sizes == sorted(self.MESH_SIZES, reverse=True)

    def test_values_decrease_toward_exact(self):
        result = self._get_result()
        # h^2 convergence from above: values should decrease
        for i in range(1, len(result.values)):
            assert result.values[i] <= result.values[i - 1]

    def test_n_dof_increases_with_refinement(self):
        result = self._get_result()
        for i in range(1, len(result.n_dof)):
            assert result.n_dof[i] >= result.n_dof[i - 1]

    def test_relative_changes_first_is_nan(self):
        result = self._get_result()
        assert math.isnan(result.relative_changes[0])

    def test_relative_changes_are_positive(self):
        result = self._get_result()
        for i in range(1, len(result.relative_changes)):
            assert result.relative_changes[i] >= 0.0

    def test_relative_changes_decrease(self):
        result = self._get_result()
        # With monotonic h^2 convergence, changes should decrease
        for i in range(2, len(result.relative_changes)):
            assert result.relative_changes[i] < result.relative_changes[i - 1]

    def test_target_quantity_stored(self):
        result = self._get_result()
        assert result.target_quantity == "frequency"


# ===================================================================
# 2. Richardson extrapolation with known analytical convergence
# ===================================================================


class TestRichardsonExtrapolation:
    """Test Richardson extrapolation on data with known convergence order."""

    def test_h2_convergence_gives_order_2(self):
        """h^2 convergence should yield convergence order ~2."""
        mesh_sizes = [10.0, 5.0, 2.5]  # Constant refinement ratio r=2
        exact = 20000.0
        coeff = 100.0
        values = [exact + coeff * (h / 2.5) ** 2 for h in mesh_sizes]

        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            mesh_sizes, values
        )
        assert abs(order - 2.0) < 0.01, f"Expected order ~2.0, got {order}"
        assert abs(ext_val - exact) < 1.0, (
            f"Expected extrapolated ~{exact}, got {ext_val}"
        )

    def test_h3_convergence_gives_order_3(self):
        """h^3 convergence should yield convergence order ~3."""
        mesh_sizes = [8.0, 4.0, 2.0]  # r=2
        exact = 15000.0
        coeff = 50.0
        values = [exact + coeff * (h / 2.0) ** 3 for h in mesh_sizes]

        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            mesh_sizes, values
        )
        assert abs(order - 3.0) < 0.01, f"Expected order ~3.0, got {order}"
        assert abs(ext_val - exact) < 1.0, (
            f"Expected extrapolated ~{exact}, got {ext_val}"
        )

    def test_h1_convergence_gives_order_1(self):
        """h^1 (linear) convergence should yield convergence order ~1."""
        mesh_sizes = [9.0, 3.0, 1.0]  # r=3
        exact = 10000.0
        coeff = 200.0
        values = [exact + coeff * (h / 1.0) for h in mesh_sizes]

        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            mesh_sizes, values
        )
        assert abs(order - 1.0) < 0.01, f"Expected order ~1.0, got {order}"
        assert abs(ext_val - exact) < 1.0, (
            f"Expected extrapolated ~{exact}, got {ext_val}"
        )

    def test_uses_three_finest_meshes(self):
        """With 5 mesh levels, only the 3 finest should be used."""
        mesh_sizes = [20.0, 10.0, 5.0, 2.5, 1.25]  # r=2
        exact = 20000.0
        coeff = 100.0
        # h^2 convergence for all levels
        values = [exact + coeff * (h / 1.25) ** 2 for h in mesh_sizes]

        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            mesh_sizes, values
        )
        # Should still get order ~2 from the 3 finest
        assert abs(order - 2.0) < 0.01

    def test_too_few_points_returns_nan(self):
        """Fewer than 3 points should return nan."""
        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            [5.0, 2.5], [20100.0, 20025.0]
        )
        assert math.isnan(ext_val)
        assert math.isnan(order)

    def test_identical_values_returns_finest_value(self):
        """If all values are identical (already converged), return f1."""
        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            [10.0, 5.0, 2.5], [20000.0, 20000.0, 20000.0]
        )
        # diff21 == 0 -> should return f1
        assert ext_val == 20000.0
        assert math.isnan(order)

    def test_non_monotonic_convergence_returns_nan(self):
        """Oscillating convergence should return nan for both values."""
        # f3 > f2 < f1 (oscillating)
        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            [10.0, 5.0, 2.5], [20100.0, 19990.0, 20050.0]
        )
        assert math.isnan(ext_val)
        assert math.isnan(order)

    def test_non_uniform_refinement_ratio(self):
        """Richardson with non-uniform refinement ratio (r ~= constant).

        The standard Richardson formula uses a single refinement ratio
        r = h2/h1 for the two finest meshes.  With non-uniform spacing
        the estimated order is approximate, so we allow a wider tolerance.
        """
        mesh_sizes = [10.0, 7.0, 5.0]  # r = 7/5 = 1.4
        exact = 20000.0
        coeff = 100.0
        values = [exact + coeff * (h / 5.0) ** 2 for h in mesh_sizes]

        ext_val, order = MeshConvergenceStudy._richardson_extrapolate(
            mesh_sizes, values
        )
        assert abs(order - 2.0) < 0.3, f"Expected order ~2.0, got {order}"
        # Non-uniform ratio introduces approximation error in the
        # extrapolated value; allow a relative error of 0.1%.
        rel_err = abs(ext_val - exact) / exact * 100.0
        assert rel_err < 0.1, (
            f"Expected extrapolated ~{exact}, got {ext_val} ({rel_err:.4f}% error)"
        )


# ===================================================================
# 3. Convergence detection
# ===================================================================


class TestConvergenceDetection:
    """Test _check_convergence with various data patterns."""

    def test_converged_data(self):
        """Values with < 0.5% last change should be detected as converged."""
        values = [20500.0, 20100.0, 20020.0, 20002.0, 20000.5]
        # Last change: |20000.5 - 20002.0| / 20002.0 * 100 = ~0.0075%
        assert MeshConvergenceStudy._check_convergence(values, 0.5) is True

    def test_not_converged_data(self):
        """Values with large last change should not be converged."""
        values = [22000.0, 21000.0, 20500.0, 20200.0]
        # Last change: |20200 - 20500| / 20500 * 100 = ~1.46%
        assert MeshConvergenceStudy._check_convergence(values, 0.5) is False

    def test_exactly_at_threshold(self):
        """Value exactly at threshold boundary (< means not <=)."""
        # Set up so that change = exactly 0.5%
        prev = 20000.0
        curr = prev * (1.0 - 0.005)  # 0.5% change
        assert MeshConvergenceStudy._check_convergence(
            [21000.0, prev, curr], 0.5
        ) is False

    def test_just_below_threshold(self):
        prev = 20000.0
        curr = prev * (1.0 - 0.004)  # 0.4% change
        assert MeshConvergenceStudy._check_convergence(
            [21000.0, prev, curr], 0.5
        ) is True

    def test_single_value_not_converged(self):
        """Single value cannot be converged."""
        assert MeshConvergenceStudy._check_convergence([20000.0], 0.5) is False

    def test_two_identical_values_converged(self):
        """Two identical values -> 0% change -> converged."""
        assert MeshConvergenceStudy._check_convergence(
            [20000.0, 20000.0], 0.5
        ) is True

    def test_zero_previous_value(self):
        """Zero previous value edge case."""
        # prev=0, curr=0 -> converged
        assert MeshConvergenceStudy._check_convergence([0.0, 0.0], 0.5) is True
        # prev=0, curr!=0 -> not converged (infinite change)
        assert MeshConvergenceStudy._check_convergence([0.0, 1.0], 0.5) is False

    def test_custom_threshold(self):
        """Custom threshold of 1.0%."""
        values = [20500.0, 20100.0, 20050.0]
        # Last change ~0.25% -> converged at 1%
        assert MeshConvergenceStudy._check_convergence(values, 1.0) is True
        # But not at 0.1%
        assert MeshConvergenceStudy._check_convergence(values, 0.1) is False


# ===================================================================
# 4. Recommended mesh size logic
# ===================================================================


class TestRecommendMeshSize:
    """Test _recommend_mesh_size selection logic."""

    def test_recommends_first_below_1pct(self):
        """Should return the coarsest mesh with change < 1%."""
        mesh_sizes = [10.0, 7.0, 5.0, 3.5, 2.5]
        # changes:    nan,  5.0%, 2.0%, 0.8%, 0.3%
        changes = [float("nan"), 5.0, 2.0, 0.8, 0.3]
        rec = MeshConvergenceStudy._recommend_mesh_size(mesh_sizes, changes)
        assert rec == 3.5  # First one below 1%

    def test_recommends_finest_when_not_converged(self):
        """If no mesh is below 1%, recommend the finest."""
        mesh_sizes = [10.0, 7.0, 5.0]
        changes = [float("nan"), 5.0, 2.0]
        rec = MeshConvergenceStudy._recommend_mesh_size(mesh_sizes, changes)
        assert rec == 5.0

    def test_all_below_1pct(self):
        """If all changes are below 1%, recommend the coarsest qualifying."""
        mesh_sizes = [10.0, 7.0, 5.0]
        changes = [float("nan"), 0.5, 0.2]
        rec = MeshConvergenceStudy._recommend_mesh_size(mesh_sizes, changes)
        assert rec == 7.0  # First below 1%

    def test_two_mesh_levels(self):
        mesh_sizes = [10.0, 5.0]
        changes = [float("nan"), 0.3]
        rec = MeshConvergenceStudy._recommend_mesh_size(mesh_sizes, changes)
        assert rec == 5.0

    def test_two_mesh_levels_not_converged(self):
        mesh_sizes = [10.0, 5.0]
        changes = [float("nan"), 3.0]
        rec = MeshConvergenceStudy._recommend_mesh_size(mesh_sizes, changes)
        assert rec == 5.0  # Finest tested


# ===================================================================
# 5. Report generation
# ===================================================================


class TestReportGeneration:
    """Test generate_report produces well-formatted text."""

    def _make_result(self, converged: bool = True) -> ConvergenceResult:
        return ConvergenceResult(
            mesh_sizes=[10.0, 7.0, 5.0, 3.5, 2.5],
            n_dof=[300, 600, 1200, 2400, 4800],
            values=[20160.0, 20078.4, 20040.0, 20019.6, 20010.0],
            relative_changes=[float("nan"), 0.405, 0.191, 0.051, 0.0048],
            converged=converged,
            convergence_rate=2.0,
            extrapolated_value=20000.0,
            recommended_mesh_size=3.5,
            target_quantity="frequency",
        )

    def test_report_contains_header(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        report = study.generate_report(result)
        assert "MESH CONVERGENCE STUDY REPORT" in report

    def test_report_contains_table_header(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        report = study.generate_report(result)
        assert "mesh_size" in report
        assert "n_dof" in report
        assert "value" in report
        assert "change%" in report

    def test_report_contains_mesh_sizes(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        report = study.generate_report(result)
        assert "10.00" in report
        assert "2.50" in report

    def test_report_contains_richardson(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        report = study.generate_report(result)
        assert "RICHARDSON EXTRAPOLATION" in report
        assert "convergence order" in report.lower() or "convergence order" in report

    def test_report_converged_status(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result_conv = self._make_result(converged=True)
        report = study.generate_report(result_conv)
        assert "CONVERGED" in report

    def test_report_not_converged_status(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result_nc = self._make_result(converged=False)
        report = study.generate_report(result_nc)
        assert "NOT CONVERGED" in report

    def test_report_recommendation(self):
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        report = study.generate_report(result)
        assert "RECOMMENDATION" in report
        assert "3.50" in report  # recommended mesh size

    def test_report_nan_richardson(self):
        """Report handles nan Richardson gracefully."""
        study, _, _ = _build_study(
            [10.0, 5.0, 2.5],
            _h2_convergence_values([10.0, 5.0, 2.5]),
        )
        result = self._make_result()
        result.convergence_rate = float("nan")
        result.extrapolated_value = float("nan")
        report = study.generate_report(result)
        assert "Not available" in report


# ===================================================================
# 6. Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_only_two_mesh_levels(self):
        """Two mesh levels: no Richardson, but convergence check works."""
        mesh_sizes = [10.0, 5.0]
        exact = 20000.0
        values = [exact + 100.0 * (h / 5.0) ** 2 for h in mesh_sizes]

        study, _, _ = _build_study(mesh_sizes, values)
        result = study.run_study(mesh_sizes)

        assert len(result.mesh_sizes) == 2
        assert len(result.values) == 2
        assert math.isnan(result.convergence_rate)
        assert math.isnan(result.extrapolated_value)

    def test_identical_values_all_levels(self):
        """All levels produce the same value -> converged, Richardson returns f1."""
        mesh_sizes = [10.0, 5.0, 2.5]
        values = [20000.0, 20000.0, 20000.0]

        study, _, _ = _build_study(mesh_sizes, values)
        result = study.run_study(mesh_sizes)

        assert result.converged is True
        # Richardson: identical values -> extrapolated = finest value
        assert result.extrapolated_value == 20000.0

    def test_oscillating_convergence(self):
        """Oscillating values -> Richardson returns nan."""
        mesh_sizes = [10.0, 5.0, 2.5]
        values = [20100.0, 19950.0, 20020.0]  # oscillating

        study, _, _ = _build_study(mesh_sizes, values)
        result = study.run_study(mesh_sizes)

        assert math.isnan(result.convergence_rate)
        assert math.isnan(result.extrapolated_value)

    def test_single_mesh_size_raises(self):
        """Single mesh size should raise ValueError."""
        study, _, _ = _build_study(
            [5.0, 2.5],
            [20100.0, 20025.0],
        )
        with pytest.raises(ValueError, match="At least 2"):
            study.run_study([5.0])

    def test_empty_mesh_sizes_raises(self):
        """Empty mesh sizes should raise ValueError."""
        study, _, _ = _build_study(
            [5.0, 2.5],
            [20100.0, 20025.0],
        )
        with pytest.raises(ValueError, match="At least 2"):
            study.run_study([])

    def test_invalid_target_quantity_raises(self):
        """Invalid target quantity should raise ValueError."""
        study, _, _ = _build_study(
            [5.0, 2.5],
            [20100.0, 20025.0],
        )
        with pytest.raises(ValueError, match="Unknown target_quantity"):
            study.run_study([10.0, 5.0], target_quantity="invalid")

    def test_unsorted_mesh_sizes_are_sorted(self):
        """Mesh sizes provided in random order should be sorted internally."""
        mesh_sizes = [5.0, 2.5, 10.0, 7.0, 3.5]
        exact = 20000.0
        values = _h2_convergence_values(mesh_sizes, exact)

        study, _, _ = _build_study(mesh_sizes, values)
        result = study.run_study(mesh_sizes)

        assert result.mesh_sizes == sorted(mesh_sizes, reverse=True)

    def test_many_refinement_levels(self):
        """Study works with many refinement levels."""
        mesh_sizes = [20.0, 15.0, 10.0, 7.0, 5.0, 3.5, 2.5, 1.5]
        exact = 20000.0
        values = _h2_convergence_values(mesh_sizes, exact, coeff=50.0)

        study, _, _ = _build_study(mesh_sizes, values)
        result = study.run_study(mesh_sizes)

        assert len(result.mesh_sizes) == 8
        assert result.converged is True


# ===================================================================
# 7. Full integration (mocked) -- run_study end-to-end
# ===================================================================


class TestFullStudy:
    """End-to-end test of run_study with mocked mesher/solver."""

    MESH_SIZES = [10.0, 7.0, 5.0, 3.5, 2.5]
    EXACT = 20000.0
    COEFF = 100.0

    def _run(self, threshold: float = 0.5) -> ConvergenceResult:
        values = _h2_convergence_values(self.MESH_SIZES, self.EXACT, self.COEFF)
        study, _, _ = _build_study(
            self.MESH_SIZES, values, convergence_threshold=threshold
        )
        return study.run_study(self.MESH_SIZES)

    def test_converged_with_default_threshold(self):
        result = self._run()
        assert result.converged is True

    def test_not_converged_with_tight_threshold(self):
        result = self._run(threshold=0.001)
        assert result.converged is False

    def test_richardson_order_close_to_2(self):
        result = self._run()
        # Non-uniform refinement ratio (3.5/2.5 = 1.4) causes the order
        # estimate to deviate from the theoretical 2.0; allow 0.3 tolerance.
        assert abs(result.convergence_rate - 2.0) < 0.3

    def test_extrapolated_value_close_to_exact(self):
        result = self._run()
        # Non-uniform refinement ratio introduces approximation error;
        # allow 0.1% relative error.
        rel_err = abs(result.extrapolated_value - self.EXACT) / self.EXACT * 100.0
        assert rel_err < 0.1, (
            f"Extrapolated {result.extrapolated_value} vs exact {self.EXACT} "
            f"({rel_err:.4f}% error)"
        )

    def test_report_generation(self):
        values = _h2_convergence_values(self.MESH_SIZES, self.EXACT, self.COEFF)
        study, _, _ = _build_study(self.MESH_SIZES, values)
        result = study.run_study(self.MESH_SIZES)
        report = study.generate_report(result)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_mesher_called_correct_times(self):
        values = _h2_convergence_values(self.MESH_SIZES, self.EXACT, self.COEFF)
        study, _, _ = _build_study(self.MESH_SIZES, values)
        study.run_study(self.MESH_SIZES)
        assert study.mesher.mesh_parametric_horn.call_count == len(self.MESH_SIZES)

    def test_solver_called_correct_times(self):
        values = _h2_convergence_values(self.MESH_SIZES, self.EXACT, self.COEFF)
        study, _, _ = _build_study(self.MESH_SIZES, values)
        study.run_study(self.MESH_SIZES)
        assert study.solver.modal_analysis.call_count == len(self.MESH_SIZES)


# ===================================================================
# 8. Relative changes computation
# ===================================================================


class TestRelativeChanges:
    """Test _compute_relative_changes helper."""

    def test_basic_changes(self):
        values = [100.0, 90.0, 85.0]
        changes = MeshConvergenceStudy._compute_relative_changes(values)
        assert len(changes) == 3
        assert math.isnan(changes[0])
        assert abs(changes[1] - 10.0) < 0.01  # 10% change
        assert abs(changes[2] - 100.0 * 5.0 / 90.0) < 0.01  # ~5.56%

    def test_identical_values(self):
        values = [100.0, 100.0, 100.0]
        changes = MeshConvergenceStudy._compute_relative_changes(values)
        assert changes[1] == 0.0
        assert changes[2] == 0.0

    def test_zero_previous_value(self):
        values = [0.0, 5.0]
        changes = MeshConvergenceStudy._compute_relative_changes(values)
        assert changes[1] == float("inf")

    def test_both_zero(self):
        values = [0.0, 0.0]
        changes = MeshConvergenceStudy._compute_relative_changes(values)
        assert changes[1] == 0.0

    def test_single_value(self):
        values = [100.0]
        changes = MeshConvergenceStudy._compute_relative_changes(values)
        assert len(changes) == 1
        assert math.isnan(changes[0])
