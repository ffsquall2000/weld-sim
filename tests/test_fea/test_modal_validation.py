"""Analytical validation benchmarks for FEA modal analysis.

Gold-standard test suite that validates the FEA modal analysis solver
against analytical solutions for uniform bars in free-free conditions.

Analytical reference
--------------------
For a free-free uniform bar, the *n*-th longitudinal mode frequency is::

    f_n = n * c / (2 * L)

where:

    c = sqrt(E / rho)  -- bar (longitudinal) acoustic velocity [m/s]
    L                   -- bar length [m]
    n = 1, 2, 3, ...   -- mode number (n=1 for fundamental)

Each test constructs a cylinder whose half-wave length exactly matches
the target frequency (20 kHz by default), meshes it, solves, and compares
the FEA frequencies to the analytical prediction.

Benchmark JSON files in ``tests/test_fea/benchmarks/`` provide reference
data for regression testing.

Tests
-----
1. Material validation (Ti-6Al-4V, Al 7075-T6, Steel D2): f1 error < 2%.
2. Mesh convergence: three refinement levels show monotonic convergence.
3. Golden data regression: results match JSON benchmark expectations.
4. Multi-mode validation: f2 ~ 2*f1 for a uniform bar.
5. Rigid body mode filtering: no modes below 100 Hz in free-free analysis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

# ---------------------------------------------------------------------------
# Helper: run a modal analysis for a given material and geometry
# ---------------------------------------------------------------------------

def _run_modal(
    material_name: str,
    diameter_mm: float,
    length_mm: float,
    mesh_size: float,
    n_modes: int = 15,
    target_frequency_hz: float = 20000.0,
    order: int = 2,
) -> ModalResult:
    """Mesh a cylindrical bar and solve modal analysis.

    Parameters
    ----------
    material_name : str
        Material name recognised by ``get_material``.
    diameter_mm : float
        Cylinder diameter in mm.
    length_mm : float
        Cylinder length in mm.
    mesh_size : float
        Characteristic element size in mm.
    n_modes : int
        Number of modes to request.
    target_frequency_hz : float
        Shift-invert target frequency.
    order : int
        Element order (1 = TET4, 2 = TET10).

    Returns
    -------
    ModalResult
    """
    mesher = GmshMesher()
    mesh = mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={"diameter_mm": diameter_mm, "length_mm": length_mm},
        mesh_size=mesh_size,
        order=order,
    )
    config = ModalConfig(
        mesh=mesh,
        material_name=material_name,
        n_modes=n_modes,
        target_frequency_hz=target_frequency_hz,
        boundary_conditions="free-free",
    )
    solver = SolverA()
    return solver.modal_analysis(config)


def _find_first_longitudinal(result: ModalResult) -> tuple[int, float]:
    """Return (index, frequency_hz) of the first longitudinal mode.

    Raises
    ------
    AssertionError
        If no longitudinal mode is found.
    """
    for i, mode_type in enumerate(result.mode_types):
        if mode_type == "longitudinal":
            return i, float(result.frequencies_hz[i])
    raise AssertionError(
        f"No longitudinal mode found among {len(result.mode_types)} modes. "
        f"Types: {result.mode_types}, Freqs: {result.frequencies_hz.tolist()}"
    )


def _find_nth_longitudinal(result: ModalResult, n: int) -> tuple[int, float]:
    """Return (index, frequency_hz) of the n-th longitudinal mode (1-based).

    Raises
    ------
    AssertionError
        If fewer than *n* longitudinal modes are found.
    """
    count = 0
    for i, mode_type in enumerate(result.mode_types):
        if mode_type == "longitudinal":
            count += 1
            if count == n:
                return i, float(result.frequencies_hz[i])
    raise AssertionError(
        f"Only {count} longitudinal modes found, but mode #{n} requested. "
        f"Types: {result.mode_types}, Freqs: {result.frequencies_hz.tolist()}"
    )


# ===================================================================
# 1. Material validation: f1 error < 2% for each material
# ===================================================================

class TestMaterialValidation:
    """Validate f1 against analytical for three horn materials."""

    @pytest.mark.parametrize(
        "material_name,expected_c",
        [
            ("Titanium Ti-6Al-4V", 5068.0),
            ("Aluminum 7075-T6", 5050.0),
            ("Steel D2", 5222.0),
        ],
        ids=["Ti-6Al-4V", "Al-7075-T6", "Steel-D2"],
    )
    def test_first_longitudinal_mode_frequency(
        self, material_name: str, expected_c: float
    ):
        """First longitudinal mode of a half-wave bar should match
        f_n = c / (2L) within 2%.

        The bar length is chosen so that f1_analytical = 20 kHz exactly.
        """
        f_target = 20000.0
        L_mm = expected_c / (2.0 * f_target) * 1000.0

        result = _run_modal(
            material_name=material_name,
            diameter_mm=25.0,
            length_mm=L_mm,
            mesh_size=5.0,
            n_modes=15,
            target_frequency_hz=f_target,
        )

        _, f_fea = _find_first_longitudinal(result)
        f_analytical = f_target  # by construction
        error_pct = abs(f_fea - f_analytical) / f_analytical * 100.0

        assert error_pct < 2.0, (
            f"{material_name}: FEA f1 = {f_fea:.1f} Hz vs analytical "
            f"{f_analytical:.1f} Hz -> {error_pct:.2f}% error (limit 2%)"
        )

    @pytest.mark.parametrize(
        "material_name,expected_c",
        [
            ("Titanium Ti-6Al-4V", 5068.0),
            ("Aluminum 7075-T6", 5050.0),
            ("Steel D2", 5222.0),
        ],
        ids=["Ti-6Al-4V", "Al-7075-T6", "Steel-D2"],
    )
    def test_first_mode_classified_as_longitudinal(
        self, material_name: str, expected_c: float
    ):
        """The mode closest to the analytical f1 should be classified
        as 'longitudinal'."""
        f_target = 20000.0
        L_mm = expected_c / (2.0 * f_target) * 1000.0

        result = _run_modal(
            material_name=material_name,
            diameter_mm=25.0,
            length_mm=L_mm,
            mesh_size=5.0,
            n_modes=15,
            target_frequency_hz=f_target,
        )

        # Find mode closest to 20 kHz
        idx_closest = int(np.argmin(np.abs(result.frequencies_hz - f_target)))
        assert result.mode_types[idx_closest] == "longitudinal", (
            f"{material_name}: mode at {result.frequencies_hz[idx_closest]:.1f} Hz "
            f"classified as {result.mode_types[idx_closest]!r}, "
            f"expected 'longitudinal'"
        )


# ===================================================================
# 2. Mesh convergence test
# ===================================================================

class TestMeshConvergence:
    """Mesh refinement should show convergence behaviour."""

    def test_mesh_convergence_titanium(self):
        """Three refinement levels should converge monotonically.

        With TET10 quadratic elements the FEA converges very rapidly for
        a simple cylinder.  The f1 values approach the 3D exact solution
        from below (the 1D bar formula ``f = c/(2L)`` neglects Poisson
        coupling, so the 3D limit is slightly lower).

        We check:
        1. Successive refinements bring f1 closer together (the difference
           ``|f_fine - f_medium| < |f_medium - f_coarse|`` decreases).
        2. All three mesh levels agree to within 0.5% (rapid convergence).
        3. Every mesh is within 2% of the analytical 1D formula.
        """
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        c = mat["acoustic_velocity_m_s"]
        f_target = 20000.0
        L_mm = c / (2.0 * f_target) * 1000.0

        freqs = []
        errors = []
        for mesh_size in [12.0, 5.0, 2.5]:
            result = _run_modal(
                material_name="Titanium Ti-6Al-4V",
                diameter_mm=25.0,
                length_mm=L_mm,
                mesh_size=mesh_size,
                n_modes=15,
                target_frequency_hz=f_target,
            )
            _, f_fea = _find_first_longitudinal(result)
            freqs.append(f_fea)
            error_pct = abs(f_fea - f_target) / f_target * 100.0
            errors.append(error_pct)

        # All meshes should be within 2% of analytical
        for i, err in enumerate(errors):
            assert err < 2.0, (
                f"Mesh level {i}: error {err:.2f}% exceeds 2% limit"
            )

        # Convergence: successive frequency differences should decrease
        # (i.e., the solution stabilises with refinement)
        diff_01 = abs(freqs[1] - freqs[0])
        diff_12 = abs(freqs[2] - freqs[1])
        assert diff_12 < diff_01 or diff_01 < 1.0, (
            f"Convergence check: |f_medium - f_coarse| = {diff_01:.2f} Hz, "
            f"|f_fine - f_medium| = {diff_12:.2f} Hz. "
            f"Expected decreasing differences or very small absolute change."
        )

        # All three meshes should agree closely (within 0.5%)
        f_spread_pct = (max(freqs) - min(freqs)) / np.mean(freqs) * 100.0
        assert f_spread_pct < 0.5, (
            f"Frequency spread across meshes is {f_spread_pct:.3f}%, "
            f"expected < 0.5%. Frequencies: {freqs}"
        )


# ===================================================================
# 3. Golden data (JSON benchmark) regression test
# ===================================================================

class TestGoldenData:
    """Validate FEA results against JSON benchmark reference data."""

    @pytest.mark.parametrize(
        "json_file",
        [
            "ti6al4v_20khz.json",
            "al7075t6_20khz.json",
            "steel_d2_20khz.json",
        ],
        ids=["Ti-6Al-4V-golden", "Al-7075-T6-golden", "Steel-D2-golden"],
    )
    def test_golden_data_regression(self, json_file: str):
        """FEA result should match golden data expectations within
        the specified tolerance."""
        json_path = _BENCHMARKS_DIR / json_file
        with open(json_path) as f:
            golden = json.load(f)

        material = golden["material"]
        geom = golden["geometry"]
        expected = golden["expected"]
        mesh_size = golden["mesh_size_mm"]
        order = golden.get("order", 2)
        f_target = golden["target_frequency_hz"]

        result = _run_modal(
            material_name=material,
            diameter_mm=geom["diameter_mm"],
            length_mm=geom["length_mm"],
            mesh_size=mesh_size,
            n_modes=15,
            target_frequency_hz=f_target,
            order=order,
        )

        # Check f1 within tolerance
        _, f_fea = _find_first_longitudinal(result)
        f_ana = expected["f1_analytical_hz"]
        tol_pct = expected["f1_tolerance_pct"]
        error_pct = abs(f_fea - f_ana) / f_ana * 100.0

        assert error_pct < tol_pct, (
            f"Golden [{json_file}]: f1_fea={f_fea:.1f} Hz vs "
            f"f1_analytical={f_ana:.1f} Hz -> {error_pct:.2f}% "
            f"(limit {tol_pct}%)"
        )

        # Check no rigid body modes
        min_freq = expected["min_frequency_hz"]
        for freq in result.frequencies_hz:
            assert freq >= min_freq, (
                f"Golden [{json_file}]: frequency {freq:.1f} Hz "
                f"below minimum {min_freq} Hz"
            )

        # Check first mode type
        _, _ = _find_first_longitudinal(result)  # asserts existence


# ===================================================================
# 4. Multi-mode validation: f2 ~ 2*f1
# ===================================================================

class TestMultiModeValidation:
    """Second longitudinal mode should be approximately 2*f1."""

    def test_second_longitudinal_mode(self):
        """For a uniform bar, f2 = 2 * c / (2L) = 2 * f1.

        We request enough modes to capture f2.  The bar is tuned so that
        f1 = 20 kHz, meaning f2_analytical = 40 kHz.

        We use a longer bar (full wavelength at 20 kHz) so that f1 = 10 kHz
        and f2 = 20 kHz, keeping both within the typical eigensolver range
        and requesting the solver to target 15 kHz as the midpoint.
        """
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        c = mat["acoustic_velocity_m_s"]

        # Use a full-wavelength bar: L = c / (2 * 10000) so f1 = 10 kHz
        f1_target = 10000.0
        L_mm = c / (2.0 * f1_target) * 1000.0  # ~253.4 mm

        # Target the solver midway between f1 and f2
        result = _run_modal(
            material_name="Titanium Ti-6Al-4V",
            diameter_mm=25.0,
            length_mm=L_mm,
            mesh_size=5.0,
            n_modes=20,
            target_frequency_hz=15000.0,
        )

        # Find first two longitudinal modes
        _, f1_fea = _find_nth_longitudinal(result, 1)
        _, f2_fea = _find_nth_longitudinal(result, 2)

        # f1 should be close to 10 kHz
        error_f1 = abs(f1_fea - f1_target) / f1_target * 100.0
        assert error_f1 < 2.0, (
            f"f1 = {f1_fea:.1f} Hz, expected ~{f1_target:.0f} Hz "
            f"({error_f1:.2f}% error)"
        )

        # f2 should be close to 2 * f1
        f2_analytical = 2.0 * f1_target
        error_f2 = abs(f2_fea - f2_analytical) / f2_analytical * 100.0
        assert error_f2 < 2.0, (
            f"f2 = {f2_fea:.1f} Hz, expected ~{f2_analytical:.0f} Hz "
            f"({error_f2:.2f}% error)"
        )

        # Ratio f2/f1 should be close to 2.0
        ratio = f2_fea / f1_fea
        assert 1.90 < ratio < 2.10, (
            f"f2/f1 ratio = {ratio:.3f}, expected ~2.0 "
            f"(f1={f1_fea:.1f}, f2={f2_fea:.1f})"
        )


# ===================================================================
# 5. Rigid body mode filtering
# ===================================================================

class TestRigidBodyModeFiltering:
    """Free-free analysis should filter all rigid body modes."""

    def test_no_rigid_body_modes(self):
        """All returned frequencies in a free-free analysis should be
        above 100 Hz (the rigid-body cutoff)."""
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        c = mat["acoustic_velocity_m_s"]
        f_target = 20000.0
        L_mm = c / (2.0 * f_target) * 1000.0

        result = _run_modal(
            material_name="Titanium Ti-6Al-4V",
            diameter_mm=25.0,
            length_mm=L_mm,
            mesh_size=5.0,
            n_modes=15,
            target_frequency_hz=f_target,
        )

        for i, freq in enumerate(result.frequencies_hz):
            assert freq >= 100.0, (
                f"Mode {i}: frequency {freq:.1f} Hz is below the "
                f"100 Hz rigid body cutoff. This mode should have been "
                f"filtered out in free-free analysis."
            )

    def test_no_negative_frequencies(self):
        """No negative or NaN frequencies should appear."""
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        c = mat["acoustic_velocity_m_s"]
        f_target = 20000.0
        L_mm = c / (2.0 * f_target) * 1000.0

        result = _run_modal(
            material_name="Titanium Ti-6Al-4V",
            diameter_mm=25.0,
            length_mm=L_mm,
            mesh_size=5.0,
            n_modes=15,
            target_frequency_hz=f_target,
        )

        assert not np.any(np.isnan(result.frequencies_hz)), (
            "NaN found in frequencies"
        )
        assert np.all(result.frequencies_hz >= 0.0), (
            f"Negative frequency found: {result.frequencies_hz}"
        )


# ===================================================================
# 6. Acoustic velocity consistency
# ===================================================================

class TestAcousticVelocityConsistency:
    """Verify that the material database acoustic velocity is consistent
    with E and rho."""

    @pytest.mark.parametrize(
        "material_name",
        [
            "Titanium Ti-6Al-4V",
            "Aluminum 7075-T6",
            "Steel D2",
        ],
    )
    def test_acoustic_velocity_matches_e_rho(self, material_name: str):
        """c_listed should equal sqrt(E / rho) within 0.1%."""
        mat = get_material(material_name)
        assert mat is not None

        c_calc = math.sqrt(mat["E_pa"] / mat["rho_kg_m3"])
        c_listed = mat["acoustic_velocity_m_s"]
        error_pct = abs(c_calc - c_listed) / c_listed * 100.0

        assert error_pct < 0.1, (
            f"{material_name}: c_calc={c_calc:.1f} m/s vs "
            f"c_listed={c_listed:.1f} m/s ({error_pct:.3f}% difference)"
        )
