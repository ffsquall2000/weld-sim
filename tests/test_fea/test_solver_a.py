"""Tests for SolverA modal analysis.

Uses coarse TET10 meshes (mesh_size=8.0 mm) for speed.
Tests requiring gmsh are guarded with ``pytest.importorskip("gmsh")``.

Analytical reference
--------------------
For a uniform cylinder in free-free conditions, the first longitudinal
mode frequency is:

    f1 = c / (2 * L)

where *c* is the bar wave speed ``sqrt(E / rho)`` and *L* is the
cylinder length.

For Ti-6Al-4V:
    E = 113.8 GPa, rho = 4430 kg/m^3 -> c = 5068 m/s
    L = 0.1267 m  (half-wave length at 20 kHz)
    f1_analytical = 5068 / (2 * 0.1267) = 19992 Hz ~ 20000 Hz
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA


# ---------------------------------------------------------------------------
# Material constants for analytical checks
# ---------------------------------------------------------------------------
_TI_E = 113.8e9        # Young's modulus [Pa]
_TI_RHO = 4430.0       # Density [kg/m^3]
_TI_C = np.sqrt(_TI_E / _TI_RHO)  # Bar wave speed ~ 5068 m/s

# For a half-wave horn tuned to 20 kHz:
#   L = c / (2 * f) = 5068 / (2 * 20000) = 0.1267 m = 126.7 mm
_HALF_WAVE_LENGTH_MM = 1000.0 * _TI_C / (2.0 * 20000.0)
_HALF_WAVE_LENGTH_M = _TI_C / (2.0 * 20000.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def half_wave_mesh() -> FEAMesh:
    """TET10 mesh of a cylinder tuned for a first longitudinal mode at ~20 kHz.

    Diameter = 25 mm, Length = half-wave length for Ti-6Al-4V at 20 kHz.
    Coarse mesh (8 mm) for speed.
    """
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={
            "diameter_mm": 25.0,
            "length_mm": _HALF_WAVE_LENGTH_MM,
        },
        mesh_size=8.0,
        order=2,
    )


@pytest.fixture(scope="module")
def modal_result(half_wave_mesh: FEAMesh) -> ModalResult:
    """Run modal analysis on the half-wave cylinder mesh.

    Cached at module scope so all tests share the same (potentially slow)
    eigenvalue solve.
    """
    config = ModalConfig(
        mesh=half_wave_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=10,
        target_frequency_hz=20000.0,
        boundary_conditions="free-free",
    )
    solver = SolverA()
    return solver.modal_analysis(config)


@pytest.fixture(scope="module")
def half_wave_K_M(half_wave_mesh: FEAMesh):
    """Pre-assembled K and M for mass-normalisation checks."""
    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
        GlobalAssembler,
    )
    assembler = GlobalAssembler(half_wave_mesh, "Titanium Ti-6Al-4V")
    return assembler.assemble()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestSolverAModal:
    """Tests for SolverA.modal_analysis."""

    def test_returns_modal_result(self, modal_result: ModalResult):
        """Return type should be ModalResult dataclass."""
        assert isinstance(modal_result, ModalResult)

    def test_solver_name(self, modal_result: ModalResult):
        """solver_name should identify the solver."""
        assert modal_result.solver_name == "SolverA"

    def test_solve_time_positive(self, modal_result: ModalResult):
        """Solve time should be a positive number."""
        assert modal_result.solve_time_s > 0.0

    def test_free_free_cylinder_first_mode(self, modal_result: ModalResult):
        """First longitudinal mode of a uniform cylinder should be f1 = c / (2L).

        For Ti-6Al-4V, c ~ 5068 m/s, L ~ 0.1267 m:
            f1_analytical ~ 20000 Hz

        FEA result should be within 2% of analytical.
        """
        f_analytical = _TI_C / (2.0 * _HALF_WAVE_LENGTH_M)

        # Find the first longitudinal mode
        freqs = modal_result.frequencies_hz
        types = modal_result.mode_types

        # The first longitudinal mode is the one classified as "longitudinal"
        # with frequency closest to analytical
        longitudinal_freqs = [
            f for f, t in zip(freqs, types) if t == "longitudinal"
        ]

        assert len(longitudinal_freqs) > 0, (
            f"No longitudinal modes found. Mode types: {types}. "
            f"Frequencies: {freqs}"
        )

        f_first_long = longitudinal_freqs[0]
        rel_error = abs(f_first_long - f_analytical) / f_analytical

        assert rel_error < 0.02, (
            f"First longitudinal mode frequency {f_first_long:.1f} Hz "
            f"differs from analytical {f_analytical:.1f} Hz "
            f"by {rel_error * 100:.2f}% (limit 2%)"
        )

    def test_rigid_body_modes_filtered(self, modal_result: ModalResult):
        """Free-free analysis should NOT include rigid body modes (f < 100 Hz)."""
        for f in modal_result.frequencies_hz:
            assert f >= 100.0, (
                f"Found mode at {f:.1f} Hz which is below 100 Hz cutoff. "
                "Rigid body modes should be filtered."
            )

    def test_modes_sorted_ascending(self, modal_result: ModalResult):
        """Returned frequencies should be in ascending order."""
        freqs = modal_result.frequencies_hz
        for i in range(len(freqs) - 1):
            assert freqs[i] <= freqs[i + 1], (
                f"Frequencies not sorted: f[{i}]={freqs[i]:.1f} > "
                f"f[{i + 1}]={freqs[i + 1]:.1f}"
            )

    def test_mode_shapes_mass_normalized(
        self, modal_result: ModalResult, half_wave_K_M
    ):
        """Each mode should satisfy phi^T * M * phi = 1.0 (within tolerance)."""
        _, M = half_wave_K_M
        mode_shapes = modal_result.mode_shapes  # (n_modes, n_dof)

        for i in range(mode_shapes.shape[0]):
            phi = mode_shapes[i, :]
            m_gen = phi @ (M @ phi)
            assert abs(m_gen - 1.0) < 1e-4, (
                f"Mode {i} generalised mass = {m_gen:.8f}, expected 1.0. "
                f"(tolerance 1e-4)"
            )

    def test_first_mode_classified_longitudinal(
        self, modal_result: ModalResult
    ):
        """For a uniform cylinder, the first longitudinal mode should be classified
        as 'longitudinal'.

        We look for the mode closest to the analytical frequency and check its
        classification.
        """
        f_analytical = _TI_C / (2.0 * _HALF_WAVE_LENGTH_M)
        freqs = modal_result.frequencies_hz
        types = modal_result.mode_types

        # Find the mode closest to the analytical frequency
        idx_closest = np.argmin(np.abs(freqs - f_analytical))
        assert types[idx_closest] == "longitudinal", (
            f"Mode closest to analytical ({freqs[idx_closest]:.1f} Hz) "
            f"is classified as {types[idx_closest]!r}, expected 'longitudinal'."
        )

    def test_effective_mass_ratios_shape(self, modal_result: ModalResult):
        """Effective mass ratios should have shape (n_modes, 3)."""
        n_modes = len(modal_result.frequencies_hz)
        assert modal_result.effective_mass_ratios.shape == (n_modes, 3)

    def test_effective_mass_ratios_non_negative(
        self, modal_result: ModalResult
    ):
        """Effective mass ratios should be non-negative (they are squared)."""
        assert np.all(modal_result.effective_mass_ratios >= 0.0)

    def test_mode_shapes_dimensions(
        self, modal_result: ModalResult, half_wave_mesh: FEAMesh
    ):
        """Mode shapes should have shape (n_modes, n_dof)."""
        n_modes = len(modal_result.frequencies_hz)
        n_dof = half_wave_mesh.n_dof
        assert modal_result.mode_shapes.shape == (n_modes, n_dof)

    def test_mode_types_count(self, modal_result: ModalResult):
        """Number of mode types should match number of frequencies."""
        assert len(modal_result.mode_types) == len(modal_result.frequencies_hz)

    def test_mode_types_valid_values(self, modal_result: ModalResult):
        """All mode types should be one of the recognised categories."""
        valid = {"longitudinal", "flexural", "compound"}
        for mt in modal_result.mode_types:
            assert mt in valid, (
                f"Invalid mode type {mt!r}. Expected one of {valid}"
            )

    def test_mesh_reference(
        self, modal_result: ModalResult, half_wave_mesh: FEAMesh
    ):
        """Result should reference the input mesh."""
        assert modal_result.mesh is half_wave_mesh


class TestSolverABoundaryConditions:
    """Tests for boundary condition handling."""

    def test_clamped_bc(self, half_wave_mesh: FEAMesh):
        """Clamped BC should produce valid results with no rigid body modes."""
        config = ModalConfig(
            mesh=half_wave_mesh,
            material_name="Titanium Ti-6Al-4V",
            n_modes=5,
            target_frequency_hz=20000.0,
            boundary_conditions="clamped",
            fixed_node_sets=["bottom_face"],
        )
        solver = SolverA()
        result = solver.modal_analysis(config)

        assert isinstance(result, ModalResult)
        assert len(result.frequencies_hz) > 0
        # All modes should have positive frequency (no rigid body modes)
        assert np.all(result.frequencies_hz > 0.0)

    def test_invalid_bc_raises(self, half_wave_mesh: FEAMesh):
        """Unsupported boundary condition type should raise ValueError."""
        config = ModalConfig(
            mesh=half_wave_mesh,
            material_name="Titanium Ti-6Al-4V",
            n_modes=5,
            boundary_conditions="pinned",  # unsupported
        )
        solver = SolverA()
        with pytest.raises(ValueError, match="Unsupported boundary_conditions"):
            solver.modal_analysis(config)

    def test_unknown_material_raises(self, half_wave_mesh: FEAMesh):
        """Unknown material should raise ValueError."""
        config = ModalConfig(
            mesh=half_wave_mesh,
            material_name="FakeMaterial_XYZ",
            n_modes=5,
        )
        solver = SolverA()
        with pytest.raises(ValueError, match="Unknown material"):
            solver.modal_analysis(config)


class TestSolverAPlaceholders:
    """Tests for placeholder methods."""

    def test_harmonic_implemented(self):
        """harmonic_analysis should no longer raise NotImplementedError.

        The method is now implemented (Phase 3).  Passing None raises
        AttributeError, not NotImplementedError.
        """
        solver = SolverA()
        with pytest.raises(AttributeError):
            solver.harmonic_analysis(None)

    def test_static_implemented(self):
        """static_analysis should no longer raise NotImplementedError.

        The method is now implemented (Phase 4).  Passing None raises
        AttributeError, not NotImplementedError.
        """
        solver = SolverA()
        with pytest.raises(AttributeError):
            solver.static_analysis(None)


class TestSolverARepr:
    """Test string representation."""

    def test_repr(self):
        solver = SolverA()
        assert "SolverA" in repr(solver)
        assert "eigsh" in repr(solver)
