"""Tests for ModeClassifier.

Uses a coarse TET10 mesh (mesh_size=8.0 mm) of a uniform cylinder tuned
for a first longitudinal mode at approximately 20 kHz (Ti-6Al-4V).

Analytical reference
--------------------
For a free-free uniform bar of length L, the first longitudinal mode
has frequency:

    f1 = c / (2 L)

where c = sqrt(E / rho) is the bar wave speed.  The nodal plane of
this half-wave mode is at L/2.

For Ti-6Al-4V:
    E   = 113.8 GPa
    rho = 4430 kg/m^3
    c   ~ 5068 m/s
    L   = c / (2 * 20000) = 0.1267 m

These tests reuse module-scoped fixtures (mesh, assembled K/M, modal
result) so the expensive mesh generation and eigensolve are only done
once.
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
    GlobalAssembler,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mode_classifier import (
    ClassificationResult,
    ClassifiedMode,
    ModeClassifier,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA


# ---------------------------------------------------------------------------
# Material constants (same as test_solver_a.py)
# ---------------------------------------------------------------------------
_TI_E = 113.8e9
_TI_RHO = 4430.0
_TI_C = np.sqrt(_TI_E / _TI_RHO)  # ~ 5068 m/s

_HALF_WAVE_LENGTH_MM = 1000.0 * _TI_C / (2.0 * 20000.0)
_HALF_WAVE_LENGTH_M = _TI_C / (2.0 * 20000.0)

_TARGET_FREQ = 20000.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def half_wave_mesh() -> FEAMesh:
    """TET10 mesh of a cylinder tuned for first longitudinal mode ~20 kHz."""
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
def half_wave_K_M(half_wave_mesh: FEAMesh):
    """Pre-assembled K and M for the half-wave mesh."""
    assembler = GlobalAssembler(half_wave_mesh, "Titanium Ti-6Al-4V")
    return assembler.assemble()


@pytest.fixture(scope="module")
def modal_result(half_wave_mesh: FEAMesh) -> ModalResult:
    """Modal analysis result from SolverA (module-cached)."""
    config = ModalConfig(
        mesh=half_wave_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=15,
        target_frequency_hz=_TARGET_FREQ,
        boundary_conditions="free-free",
    )
    solver = SolverA()
    return solver.modal_analysis(config)


@pytest.fixture(scope="module")
def classification_result(
    half_wave_mesh: FEAMesh,
    half_wave_K_M,
    modal_result: ModalResult,
) -> ClassificationResult:
    """Full classification result from ModeClassifier (module-cached)."""
    _, M = half_wave_K_M
    classifier = ModeClassifier(
        node_coords=half_wave_mesh.nodes,
        M=M,
        longitudinal_axis="y",
    )
    return classifier.classify(
        frequencies_hz=modal_result.frequencies_hz,
        mode_shapes=modal_result.mode_shapes,
        target_frequency_hz=_TARGET_FREQ,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestModeClassifier:
    """Tests for ModeClassifier classification results."""

    def test_first_longitudinal_mode(
        self, classification_result: ClassificationResult
    ):
        """Uniform cylinder: first mode near target should be classified as
        longitudinal.

        The mode closest to 20 kHz should be longitudinal, and the
        target_mode_index should point to a longitudinal mode.
        """
        target_idx = classification_result.target_mode_index
        assert target_idx >= 0, "No target longitudinal mode found"

        target_mode = classification_result.modes[target_idx]
        assert target_mode.mode_type == "longitudinal", (
            f"Target mode (index {target_idx}) is {target_mode.mode_type!r}, "
            f"expected 'longitudinal'"
        )

        # Frequency should be within 5% of the analytical value
        f_analytical = _TI_C / (2.0 * _HALF_WAVE_LENGTH_M)
        rel_error = abs(target_mode.frequency_hz - f_analytical) / f_analytical
        assert rel_error < 0.05, (
            f"Target longitudinal mode at {target_mode.frequency_hz:.1f} Hz "
            f"differs from analytical {f_analytical:.1f} Hz by "
            f"{rel_error * 100:.2f}%"
        )

    def test_flexural_mode_detected(
        self, classification_result: ClassificationResult
    ):
        """Check that at least one mode is classified as flexural.

        A uniform cylinder should have flexural (bending) modes in
        addition to longitudinal modes.
        """
        flexural_modes = [
            m for m in classification_result.modes
            if m.mode_type == "flexural"
        ]
        assert len(flexural_modes) > 0, (
            "No flexural modes detected. Mode types: "
            f"{[m.mode_type for m in classification_result.modes]}"
        )

    def test_nodal_plane_at_half_length(
        self, classification_result: ClassificationResult
    ):
        """For a half-wave resonator, the nodal plane of the first
        longitudinal mode should be at approximately L/2.

        The free-free half-wave mode has maximum displacement at both
        ends and zero displacement at the midpoint.
        """
        target_idx = classification_result.target_mode_index
        assert target_idx >= 0, "No target longitudinal mode found"

        target_mode = classification_result.modes[target_idx]
        assert target_mode.nodal_plane_y is not None, (
            "Nodal plane not found for target longitudinal mode"
        )

        L = _HALF_WAVE_LENGTH_M
        expected_y = L / 2.0  # midpoint
        actual_y = target_mode.nodal_plane_y

        # Allow 15% tolerance (coarse mesh discretisation + geometry)
        tol = 0.15 * L
        assert abs(actual_y - expected_y) < tol, (
            f"Nodal plane at y={actual_y:.6f} m, "
            f"expected y={expected_y:.6f} m (L/2), "
            f"difference={abs(actual_y - expected_y):.6f} m, "
            f"tolerance={tol:.6f} m"
        )

    def test_parasitic_detection(
        self, classification_result: ClassificationResult
    ):
        """Non-longitudinal modes near the target should be flagged as
        parasitic (CRITICAL or WARNING).

        Also verify that all non-longitudinal modes have a separation_pct
        value, and that the parasitic_modes list is consistent.
        """
        target_idx = classification_result.target_mode_index
        if target_idx < 0:
            pytest.skip("No target longitudinal mode found")

        f_target = classification_result.modes[target_idx].frequency_hz

        for mode in classification_result.modes:
            if mode.mode_type != "longitudinal":
                # separation_pct should be set
                assert mode.separation_pct is not None, (
                    f"Mode {mode.mode_number} ({mode.mode_type}) has "
                    f"separation_pct=None"
                )

                sep = mode.separation_pct
                if sep < 3.0:
                    assert mode.parasitic_flag == "CRITICAL", (
                        f"Mode {mode.mode_number} at {mode.frequency_hz:.1f} Hz "
                        f"has sep={sep:.2f}% but flag={mode.parasitic_flag!r}"
                    )
                elif sep < 5.0:
                    assert mode.parasitic_flag == "WARNING", (
                        f"Mode {mode.mode_number} at {mode.frequency_hz:.1f} Hz "
                        f"has sep={sep:.2f}% but flag={mode.parasitic_flag!r}"
                    )
                else:
                    assert mode.parasitic_flag == "OK", (
                        f"Mode {mode.mode_number} at {mode.frequency_hz:.1f} Hz "
                        f"has sep={sep:.2f}% but flag={mode.parasitic_flag!r}"
                    )

        # parasitic_modes should be the subset within 5%
        for pm in classification_result.parasitic_modes:
            assert pm.parasitic_flag in ("CRITICAL", "WARNING"), (
                f"Parasitic mode {pm.mode_number} has flag={pm.parasitic_flag!r}"
            )
            assert pm.separation_pct is not None
            assert pm.separation_pct < 5.0

    def test_displacement_ratios_sum_to_one(
        self, classification_result: ClassificationResult
    ):
        """R_x + R_y + R_z should equal 1.0 for every mode."""
        for mode in classification_result.modes:
            ratio_sum = float(np.sum(mode.displacement_ratios))
            assert abs(ratio_sum - 1.0) < 1e-10, (
                f"Mode {mode.mode_number}: displacement ratios sum to "
                f"{ratio_sum:.12f}, expected 1.0. "
                f"Ratios: {mode.displacement_ratios}"
            )

    def test_effective_mass_non_negative(
        self, classification_result: ClassificationResult
    ):
        """All effective mass values should be >= 0."""
        for mode in classification_result.modes:
            assert np.all(mode.effective_mass >= 0.0), (
                f"Mode {mode.mode_number}: negative effective mass "
                f"{mode.effective_mass}"
            )

    def test_target_mode_index_valid(
        self, classification_result: ClassificationResult
    ):
        """Target mode should be a longitudinal mode closest to f_target.

        Verify that:
        1. target_mode_index is a valid index.
        2. The indexed mode is longitudinal.
        3. No other longitudinal mode is closer to f_target.
        """
        idx = classification_result.target_mode_index
        modes = classification_result.modes

        assert 0 <= idx < len(modes), (
            f"target_mode_index={idx} out of range [0, {len(modes)})"
        )

        target_mode = modes[idx]
        assert target_mode.mode_type == "longitudinal", (
            f"Target mode is {target_mode.mode_type!r}, expected 'longitudinal'"
        )

        # Verify it is the closest longitudinal mode to f_target
        f_target = _TARGET_FREQ
        target_dist = abs(target_mode.frequency_hz - f_target)

        for i, m in enumerate(modes):
            if m.mode_type == "longitudinal" and i != idx:
                dist = abs(m.frequency_hz - f_target)
                assert dist >= target_dist - 1e-6, (
                    f"Longitudinal mode {i} at {m.frequency_hz:.1f} Hz is "
                    f"closer to target than mode {idx} at "
                    f"{target_mode.frequency_hz:.1f} Hz"
                )


class TestModeClassifierProperties:
    """Tests for ModeClassifier properties and edge cases."""

    def test_total_mass_positive(
        self, classification_result: ClassificationResult
    ):
        """Total mass should be positive and physically reasonable.

        Analytical mass for the cylinder:
        rho * pi * r^2 * L = 4430 * pi * 0.0125^2 * 0.1267 ~ 0.275 kg
        """
        total_mass = classification_result.total_mass_kg
        assert total_mass > 0.0, f"Total mass is {total_mass}, expected > 0"

        # Should be within 15% of analytical (coarse mesh)
        r = 0.0125
        L = _HALF_WAVE_LENGTH_M
        analytical_mass = _TI_RHO * np.pi * r ** 2 * L
        rel_error = abs(total_mass - analytical_mass) / analytical_mass
        assert rel_error < 0.15, (
            f"Total mass {total_mass:.6f} kg differs from analytical "
            f"{analytical_mass:.6f} kg by {rel_error * 100:.2f}%"
        )

    def test_classification_result_structure(
        self, classification_result: ClassificationResult
    ):
        """ClassificationResult should have correct types and structure."""
        assert isinstance(classification_result.modes, list)
        assert isinstance(classification_result.target_mode_index, int)
        assert isinstance(classification_result.parasitic_modes, list)
        assert isinstance(classification_result.total_mass_kg, float)

        for mode in classification_result.modes:
            assert isinstance(mode, ClassifiedMode)
            assert isinstance(mode.mode_number, int)
            assert isinstance(mode.frequency_hz, float)
            assert mode.mode_type in {
                "longitudinal", "flexural", "torsional", "compound"
            }
            assert mode.displacement_ratios.shape == (3,)
            assert mode.effective_mass.shape == (3,)
            assert mode.parasitic_flag in {"OK", "WARNING", "CRITICAL"}

    def test_mode_numbers_sequential(
        self, classification_result: ClassificationResult
    ):
        """Mode numbers should be sequential starting from 1."""
        for i, mode in enumerate(classification_result.modes):
            assert mode.mode_number == i + 1, (
                f"Mode at index {i} has mode_number={mode.mode_number}, "
                f"expected {i + 1}"
            )

    def test_classifier_repr(self, half_wave_mesh: FEAMesh, half_wave_K_M):
        """Repr should be informative."""
        _, M = half_wave_K_M
        classifier = ModeClassifier(half_wave_mesh.nodes, M, "y")
        r = repr(classifier)
        assert "ModeClassifier" in r
        assert "longitudinal_axis" in r

    def test_invalid_longitudinal_axis_raises(
        self, half_wave_mesh: FEAMesh, half_wave_K_M
    ):
        """Invalid longitudinal_axis should raise ValueError."""
        _, M = half_wave_K_M
        with pytest.raises(ValueError, match="longitudinal_axis"):
            ModeClassifier(half_wave_mesh.nodes, M, "w")

    def test_mode_shapes_shape_mismatch_raises(
        self, half_wave_mesh: FEAMesh, half_wave_K_M, modal_result: ModalResult
    ):
        """Mode shapes with wrong shape should raise ValueError."""
        _, M = half_wave_K_M
        classifier = ModeClassifier(half_wave_mesh.nodes, M, "y")

        # Pass truncated mode shapes (wrong n_dof)
        bad_shapes = modal_result.mode_shapes[:, :10]  # truncated
        with pytest.raises(ValueError, match="mode_shapes shape"):
            classifier.classify(
                modal_result.frequencies_hz, bad_shapes, _TARGET_FREQ
            )


class TestModeClassifierSynthetic:
    """Tests using synthetic (hand-crafted) mode shapes.

    These tests do not require gmsh or real FEA results.
    """

    @pytest.fixture
    def simple_bar(self):
        """A simple 1D bar of 11 nodes along Y, with a trivial mass matrix.

        Nodes at y = 0, 0.01, 0.02, ..., 0.10 (100 mm bar).
        All at x=0, z=0.
        """
        import scipy.sparse as sp

        n_nodes = 11
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        coords[:, 1] = np.linspace(0.0, 0.10, n_nodes)  # y-axis

        n_dof = 3 * n_nodes
        # Simple diagonal mass matrix (lumped)
        m_per_node = 0.01  # kg
        M_diag = np.full(n_dof, m_per_node, dtype=np.float64)
        M = sp.diags(M_diag).tocsr()

        return coords, M, n_nodes, n_dof

    def test_synthetic_longitudinal(self, simple_bar):
        """A purely y-direction mode should be classified as longitudinal."""
        coords, M, n_nodes, n_dof = simple_bar

        classifier = ModeClassifier(coords, M, "y")

        # Create a mode shape: sinusoidal displacement in Y only
        phi = np.zeros(n_dof, dtype=np.float64)
        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()
        phi[1::3] = np.sin(np.pi * y_coords / L)

        # Normalize
        norm = np.sqrt(phi @ (M @ phi))
        phi /= norm

        freqs = np.array([20000.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[0].mode_type == "longitudinal"
        assert result.modes[0].displacement_ratios[1] > 0.99  # nearly all in Y

    def test_synthetic_flexural(self, simple_bar):
        """A purely x-direction mode should be classified as flexural."""
        coords, M, n_nodes, n_dof = simple_bar

        classifier = ModeClassifier(coords, M, "y")

        # Create a mode shape: displacement purely in X
        phi = np.zeros(n_dof, dtype=np.float64)
        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()
        phi[0::3] = np.sin(np.pi * y_coords / L)

        norm = np.sqrt(phi @ (M @ phi))
        phi /= norm

        freqs = np.array([15000.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[0].mode_type == "flexural"
        assert result.modes[0].displacement_ratios[0] > 0.99

    def test_synthetic_torsional(self, simple_bar):
        """A purely rotational mode about Y should be classified as torsional.

        For torsional motion about Y: u_x = -z * theta, u_z = x * theta.
        Since nodes are at x=0, z=0, we need to offset them.
        """
        import scipy.sparse as sp

        n_nodes = 20
        coords = np.zeros((n_nodes, 3), dtype=np.float64)

        # Place nodes in a ring at various y positions
        for i in range(n_nodes):
            angle = 2 * np.pi * (i % 5) / 5
            y_pos = 0.05 * (i // 5)
            coords[i] = [0.01 * np.cos(angle), y_pos, 0.01 * np.sin(angle)]

        n_dof = 3 * n_nodes
        M_diag = np.full(n_dof, 0.01, dtype=np.float64)
        M = sp.diags(M_diag).tocsr()

        classifier = ModeClassifier(coords, M, "y")

        # Torsional mode: u_x = -z * theta, u_z = x * theta
        phi = np.zeros(n_dof, dtype=np.float64)
        for i in range(n_nodes):
            x_i, y_i, z_i = coords[i]
            phi[3 * i + 0] = -z_i  # u_x
            phi[3 * i + 2] = x_i   # u_z

        norm = np.sqrt(phi @ (M @ phi))
        if norm > 0:
            phi /= norm

        freqs = np.array([18000.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[0].mode_type == "torsional", (
            f"Expected torsional, got {result.modes[0].mode_type!r}"
        )

    def test_synthetic_displacement_ratios_sum_to_one(self, simple_bar):
        """Displacement ratios should sum to 1.0 even for mixed modes."""
        coords, M, n_nodes, n_dof = simple_bar

        classifier = ModeClassifier(coords, M, "y")

        # Create a mode with mixed X and Y displacement
        phi = np.zeros(n_dof, dtype=np.float64)
        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()
        phi[0::3] = 0.5 * np.sin(np.pi * y_coords / L)
        phi[1::3] = 0.8 * np.cos(np.pi * y_coords / L)
        phi[2::3] = 0.3 * np.sin(2 * np.pi * y_coords / L)

        norm = np.sqrt(phi @ (M @ phi))
        phi /= norm

        freqs = np.array([20000.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        ratio_sum = float(np.sum(result.modes[0].displacement_ratios))
        assert abs(ratio_sum - 1.0) < 1e-10

    def test_synthetic_parasitic_critical(self, simple_bar):
        """A non-longitudinal mode within 3% of target should be CRITICAL."""
        coords, M, n_nodes, n_dof = simple_bar
        classifier = ModeClassifier(coords, M, "y")

        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()

        # Mode 1: longitudinal at 20000 Hz
        phi1 = np.zeros(n_dof, dtype=np.float64)
        phi1[1::3] = np.sin(np.pi * y_coords / L)
        norm1 = np.sqrt(phi1 @ (M @ phi1))
        phi1 /= norm1

        # Mode 2: flexural at 20400 Hz (2% separation -> CRITICAL)
        phi2 = np.zeros(n_dof, dtype=np.float64)
        phi2[0::3] = np.sin(np.pi * y_coords / L)
        norm2 = np.sqrt(phi2 @ (M @ phi2))
        phi2 /= norm2

        freqs = np.array([20000.0, 20400.0])
        shapes = np.vstack([phi1.reshape(1, -1), phi2.reshape(1, -1)])

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        # First mode: longitudinal at target
        assert result.modes[0].mode_type == "longitudinal"
        assert result.modes[0].parasitic_flag == "OK"

        # Second mode: flexural, 2% from target -> CRITICAL
        assert result.modes[1].mode_type == "flexural"
        assert result.modes[1].parasitic_flag == "CRITICAL"
        assert result.modes[1].separation_pct is not None
        assert result.modes[1].separation_pct < 3.0

    def test_synthetic_parasitic_warning(self, simple_bar):
        """A non-longitudinal mode between 3% and 5% should be WARNING."""
        coords, M, n_nodes, n_dof = simple_bar
        classifier = ModeClassifier(coords, M, "y")

        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()

        # Mode 1: longitudinal at 20000 Hz
        phi1 = np.zeros(n_dof, dtype=np.float64)
        phi1[1::3] = np.sin(np.pi * y_coords / L)
        norm1 = np.sqrt(phi1 @ (M @ phi1))
        phi1 /= norm1

        # Mode 2: flexural at 20800 Hz (4% separation -> WARNING)
        phi2 = np.zeros(n_dof, dtype=np.float64)
        phi2[0::3] = np.sin(np.pi * y_coords / L)
        norm2 = np.sqrt(phi2 @ (M @ phi2))
        phi2 /= norm2

        freqs = np.array([20000.0, 20800.0])
        shapes = np.vstack([phi1.reshape(1, -1), phi2.reshape(1, -1)])

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[1].parasitic_flag == "WARNING"
        assert 3.0 <= result.modes[1].separation_pct < 5.0

    def test_synthetic_parasitic_ok(self, simple_bar):
        """A non-longitudinal mode beyond 5% should be OK."""
        coords, M, n_nodes, n_dof = simple_bar
        classifier = ModeClassifier(coords, M, "y")

        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()

        # Mode 1: longitudinal at 20000 Hz
        phi1 = np.zeros(n_dof, dtype=np.float64)
        phi1[1::3] = np.sin(np.pi * y_coords / L)
        norm1 = np.sqrt(phi1 @ (M @ phi1))
        phi1 /= norm1

        # Mode 2: flexural at 22000 Hz (10% separation -> OK)
        phi2 = np.zeros(n_dof, dtype=np.float64)
        phi2[0::3] = np.sin(np.pi * y_coords / L)
        norm2 = np.sqrt(phi2 @ (M @ phi2))
        phi2 /= norm2

        freqs = np.array([20000.0, 22000.0])
        shapes = np.vstack([phi1.reshape(1, -1), phi2.reshape(1, -1)])

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[1].parasitic_flag == "OK"
        assert result.modes[1].separation_pct >= 5.0

    def test_synthetic_nodal_plane_location(self, simple_bar):
        """Nodal plane of a half-wave sinusoid should be at L/2."""
        coords, M, n_nodes, n_dof = simple_bar
        classifier = ModeClassifier(coords, M, "y")

        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()

        # Half-wave sin: zero at y=0 and y=L, maximum at L/2
        # For a free-free bar first longitudinal mode: cos(pi y/L)
        # which has a zero crossing at L/2
        phi = np.zeros(n_dof, dtype=np.float64)
        phi[1::3] = np.cos(np.pi * y_coords / L)

        norm = np.sqrt(phi @ (M @ phi))
        phi /= norm

        freqs = np.array([20000.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        assert result.modes[0].mode_type == "longitudinal"
        assert result.modes[0].nodal_plane_y is not None

        expected_y = L / 2.0
        actual_y = result.modes[0].nodal_plane_y
        assert abs(actual_y - expected_y) < 0.01 * L, (
            f"Nodal plane at y={actual_y:.6f}, expected {expected_y:.6f}"
        )

    def test_effective_mass_longitudinal_mode(self, simple_bar):
        """For a purely y-direction mode, M_eff_y should be the dominant
        component."""
        coords, M, n_nodes, n_dof = simple_bar
        classifier = ModeClassifier(coords, M, "y")

        phi = np.zeros(n_dof, dtype=np.float64)
        y_coords = coords[:, 1]
        L = y_coords.max() - y_coords.min()
        # Uniform translation in Y (rigid body) -> maximum M_eff_y
        phi[1::3] = 1.0

        norm = np.sqrt(phi @ (M @ phi))
        phi /= norm

        freqs = np.array([100.0])
        shapes = phi.reshape(1, -1)

        result = classifier.classify(freqs, shapes, target_frequency_hz=20000.0)

        eff = result.modes[0].effective_mass
        # M_eff_y should be the total mass (all mass participates)
        assert eff[1] > eff[0] + eff[2], (
            f"M_eff_y={eff[1]} should dominate: {eff}"
        )
        # M_eff_x and M_eff_z should be near zero
        assert eff[0] < 1e-10
        assert eff[2] < 1e-10
