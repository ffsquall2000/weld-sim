"""Tests for pre-stressed modal analysis in SolverA.

Pre-stressed modal analysis computes the geometric stiffness matrix K_sigma
from a static preload, then solves the modified eigenvalue problem:

    (K + K_sigma) * phi = omega^2 * M * phi

Physics:
- Tensile preload increases effective stiffness -> natural frequencies increase
- Compressive preload decreases effective stiffness -> natural frequencies decrease
- Zero preload -> K_sigma = 0 -> frequencies unchanged from clamped analysis

Uses a Ti-6Al-4V uniform cylinder: D=25 mm, L=80 mm, mesh_size=6.0 mm.
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA


# ---------------------------------------------------------------------------
# Material and geometry constants
# ---------------------------------------------------------------------------
_TI_E = 113.8e9        # Young's modulus [Pa]
_TI_NU = 0.342         # Poisson's ratio
_TI_RHO = 4430.0       # Density [kg/m^3]

_DIAMETER_MM = 25.0
_LENGTH_MM = 80.0
_RADIUS_M = _DIAMETER_MM / 2.0 / 1000.0
_LENGTH_M = _LENGTH_MM / 1000.0
_AREA_M2 = np.pi * _RADIUS_M ** 2

_FORCE_N = 10000.0   # 10 kN tensile / compressive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cylinder_mesh() -> FEAMesh:
    """TET10 mesh of a uniform cylinder: D=25mm, L=80mm."""
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={
            "diameter_mm": _DIAMETER_MM,
            "length_mm": _LENGTH_MM,
        },
        mesh_size=6.0,
        order=2,
    )


@pytest.fixture(scope="module")
def solver() -> SolverA:
    """Shared SolverA instance."""
    return SolverA()


@pytest.fixture(scope="module")
def clamped_result(cylinder_mesh: FEAMesh, solver: SolverA) -> ModalResult:
    """Clamped modal result for reference (baseline)."""
    config = ModalConfig(
        mesh=cylinder_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=5,
        target_frequency_hz=20000.0,
        boundary_conditions="clamped",
        fixed_node_sets=["bottom_face"],
    )
    return solver.modal_analysis(config)


@pytest.fixture(scope="module")
def tension_prestressed_result(
    cylinder_mesh: FEAMesh, solver: SolverA
) -> ModalResult:
    """Pre-stressed modal result with 10 kN tensile preload.

    Bottom face fixed, top face loaded with 10 kN in Y direction.
    """
    config = ModalConfig(
        mesh=cylinder_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=5,
        target_frequency_hz=20000.0,
        boundary_conditions="pre-stressed",
        fixed_node_sets=["bottom_face"],
        pre_stress_loads=[
            {
                "type": "force",
                "node_set": "top_face",
                "direction": [0, 1, 0],
                "magnitude": _FORCE_N,
            },
        ],
        pre_stress_bcs=[
            {
                "type": "fixed",
                "node_set": "bottom_face",
            },
        ],
    )
    return solver.modal_analysis(config)


@pytest.fixture(scope="module")
def compression_prestressed_result(
    cylinder_mesh: FEAMesh, solver: SolverA
) -> ModalResult:
    """Pre-stressed modal result with 10 kN compressive preload.

    Bottom face fixed, top face loaded with -10 kN in Y direction.
    """
    config = ModalConfig(
        mesh=cylinder_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=5,
        target_frequency_hz=20000.0,
        boundary_conditions="pre-stressed",
        fixed_node_sets=["bottom_face"],
        pre_stress_loads=[
            {
                "type": "force",
                "node_set": "top_face",
                "direction": [0, -1, 0],
                "magnitude": _FORCE_N,
            },
        ],
        pre_stress_bcs=[
            {
                "type": "fixed",
                "node_set": "bottom_face",
            },
        ],
    )
    return solver.modal_analysis(config)


@pytest.fixture(scope="module")
def zero_prestressed_result(
    cylinder_mesh: FEAMesh, solver: SolverA
) -> ModalResult:
    """Pre-stressed modal result with zero preload.

    All pre_stress_loads magnitudes set to 0 -> K_sigma should be ~0.
    """
    config = ModalConfig(
        mesh=cylinder_mesh,
        material_name="Titanium Ti-6Al-4V",
        n_modes=5,
        target_frequency_hz=20000.0,
        boundary_conditions="pre-stressed",
        fixed_node_sets=["bottom_face"],
        pre_stress_loads=[
            {
                "type": "force",
                "node_set": "top_face",
                "direction": [0, 1, 0],
                "magnitude": 0.0,
            },
        ],
        pre_stress_bcs=[
            {
                "type": "fixed",
                "node_set": "bottom_face",
            },
        ],
    )
    return solver.modal_analysis(config)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------
class TestPreStressedModalResult:
    """Test that pre-stressed modal analysis returns valid ModalResult."""

    def test_returns_modal_result(self, tension_prestressed_result: ModalResult):
        """Pre-stressed modal analysis should return a ModalResult."""
        assert isinstance(tension_prestressed_result, ModalResult)

    def test_has_positive_frequencies(
        self, tension_prestressed_result: ModalResult
    ):
        """All frequencies should be positive."""
        assert len(tension_prestressed_result.frequencies_hz) > 0
        assert np.all(tension_prestressed_result.frequencies_hz > 0.0)

    def test_frequencies_sorted(self, tension_prestressed_result: ModalResult):
        """Frequencies should be sorted in ascending order."""
        freqs = tension_prestressed_result.frequencies_hz
        for i in range(len(freqs) - 1):
            assert freqs[i] <= freqs[i + 1], (
                f"Frequencies not sorted: f[{i}]={freqs[i]:.1f} > "
                f"f[{i + 1}]={freqs[i + 1]:.1f}"
            )

    def test_solver_name(self, tension_prestressed_result: ModalResult):
        """Solver name should be SolverA."""
        assert tension_prestressed_result.solver_name == "SolverA"

    def test_solve_time_positive(self, tension_prestressed_result: ModalResult):
        """Solve time should be positive."""
        assert tension_prestressed_result.solve_time_s > 0.0

    def test_mode_shapes_dimensions(
        self, tension_prestressed_result: ModalResult, cylinder_mesh: FEAMesh
    ):
        """Mode shapes should have shape (n_modes, n_dof)."""
        n_modes = len(tension_prestressed_result.frequencies_hz)
        n_dof = cylinder_mesh.n_dof
        assert tension_prestressed_result.mode_shapes.shape == (n_modes, n_dof)

    def test_mode_types_valid(self, tension_prestressed_result: ModalResult):
        """All mode types should be recognized categories."""
        valid = {"longitudinal", "flexural", "compound"}
        for mt in tension_prestressed_result.mode_types:
            assert mt in valid


class TestTensileStiffening:
    """Tensile preload should increase natural frequencies (stress stiffening)."""

    def test_tension_increases_frequencies(
        self,
        clamped_result: ModalResult,
        tension_prestressed_result: ModalResult,
    ):
        """At least the first mode frequency should increase under tension.

        A tensile preload on a clamped bar adds positive geometric stiffness,
        which increases the effective stiffness and thus the natural frequencies.
        """
        f_clamped = clamped_result.frequencies_hz[0]
        f_tension = tension_prestressed_result.frequencies_hz[0]

        assert f_tension > f_clamped, (
            f"Tensile preload should increase frequency: "
            f"f_clamped={f_clamped:.1f} Hz, f_tension={f_tension:.1f} Hz"
        )


class TestCompressiveSoftening:
    """Compressive preload should decrease natural frequencies (stress softening)."""

    def test_compression_decreases_frequencies(
        self,
        clamped_result: ModalResult,
        compression_prestressed_result: ModalResult,
    ):
        """At least the first mode frequency should decrease under compression.

        A compressive preload adds negative geometric stiffness, which reduces
        the effective stiffness and thus the natural frequencies.
        """
        f_clamped = clamped_result.frequencies_hz[0]
        f_compression = compression_prestressed_result.frequencies_hz[0]

        assert f_compression < f_clamped, (
            f"Compressive preload should decrease frequency: "
            f"f_clamped={f_clamped:.1f} Hz, "
            f"f_compression={f_compression:.1f} Hz"
        )


class TestZeroPreStress:
    """Zero preload should produce same frequencies as clamped analysis."""

    def test_zero_prestress_matches_clamped(
        self,
        clamped_result: ModalResult,
        zero_prestressed_result: ModalResult,
    ):
        """Zero preload -> K_sigma = 0 -> frequencies match clamped baseline.

        With zero applied force, the stress tensor is zero everywhere,
        so K_sigma = 0 and the eigenvalue problem reduces to clamped case.
        Tolerance set to 0.1% to account for numerical differences.
        """
        f_clamped = clamped_result.frequencies_hz
        f_zero = zero_prestressed_result.frequencies_hz

        n_compare = min(len(f_clamped), len(f_zero))
        for i in range(n_compare):
            rel_error = abs(f_zero[i] - f_clamped[i]) / f_clamped[i]
            assert rel_error < 0.001, (
                f"Mode {i}: zero-prestress freq {f_zero[i]:.1f} Hz "
                f"differs from clamped {f_clamped[i]:.1f} Hz "
                f"by {rel_error * 100:.4f}% (limit 0.1%)"
            )


class TestGeometricStiffnessSymmetry:
    """K_sigma should be symmetric."""

    def test_k_sigma_is_symmetric(
        self, cylinder_mesh: FEAMesh, solver: SolverA
    ):
        """The geometric stiffness matrix should be symmetric."""
        # Run a static analysis to get stress tensor
        static_config = StaticConfig(
            mesh=cylinder_mesh,
            material_name="Titanium Ti-6Al-4V",
            loads=[
                {
                    "type": "force",
                    "node_set": "top_face",
                    "direction": [0, 1, 0],
                    "magnitude": _FORCE_N,
                },
            ],
            boundary_conditions=[
                {
                    "type": "fixed",
                    "node_set": "bottom_face",
                },
            ],
        )
        static_result = solver.static_analysis(static_config)

        mat = get_material("Titanium Ti-6Al-4V")
        K_sigma = solver._compute_geometric_stiffness(
            cylinder_mesh, static_result.stress_tensor, mat
        )

        # Check symmetry: K_sigma should equal K_sigma^T
        diff = K_sigma - K_sigma.T
        max_asym = abs(diff).max()
        # Normalize by the max value in K_sigma
        k_max = abs(K_sigma).max()
        if k_max > 0:
            rel_asym = max_asym / k_max
        else:
            rel_asym = max_asym

        assert rel_asym < 1e-10, (
            f"K_sigma is not symmetric: max |K - K^T| / max |K| = {rel_asym:.2e}"
        )


class TestGeometricStiffnessZeroStress:
    """Zero stress should produce a zero geometric stiffness matrix."""

    def test_zero_stress_gives_zero_k_sigma(
        self, cylinder_mesh: FEAMesh, solver: SolverA
    ):
        """When all element stresses are zero, K_sigma should be zero."""
        n_elements = cylinder_mesh.elements.shape[0]
        zero_stress = np.zeros((n_elements, 6), dtype=np.float64)

        mat = get_material("Titanium Ti-6Al-4V")
        K_sigma = solver._compute_geometric_stiffness(
            cylinder_mesh, zero_stress, mat
        )

        assert abs(K_sigma).max() < 1e-30, (
            f"K_sigma should be zero for zero stress, "
            f"but max |K_sigma| = {abs(K_sigma).max():.2e}"
        )


class TestPreStressedConfig:
    """Test that ModalConfig accepts pre-stressed parameters."""

    def test_config_accepts_pre_stress_fields(self, cylinder_mesh: FEAMesh):
        """ModalConfig should accept pre_stress_loads and pre_stress_bcs."""
        config = ModalConfig(
            mesh=cylinder_mesh,
            material_name="Titanium Ti-6Al-4V",
            n_modes=5,
            target_frequency_hz=20000.0,
            boundary_conditions="pre-stressed",
            fixed_node_sets=["bottom_face"],
            pre_stress_loads=[
                {
                    "type": "force",
                    "node_set": "top_face",
                    "direction": [0, 1, 0],
                    "magnitude": 1000.0,
                },
            ],
            pre_stress_bcs=[
                {
                    "type": "fixed",
                    "node_set": "bottom_face",
                },
            ],
        )
        assert config.boundary_conditions == "pre-stressed"
        assert len(config.pre_stress_loads) == 1
        assert len(config.pre_stress_bcs) == 1

    def test_config_defaults_empty_lists(self, cylinder_mesh: FEAMesh):
        """pre_stress_loads and pre_stress_bcs should default to empty lists."""
        config = ModalConfig(
            mesh=cylinder_mesh,
            material_name="Titanium Ti-6Al-4V",
        )
        assert config.pre_stress_loads == []
        assert config.pre_stress_bcs == []


class TestFrequencyOrdering:
    """Verify tension > clamped > compression for the same mode."""

    def test_frequency_ordering(
        self,
        clamped_result: ModalResult,
        tension_prestressed_result: ModalResult,
        compression_prestressed_result: ModalResult,
    ):
        """For the first mode: f_tension > f_clamped > f_compression.

        This verifies the complete physical picture: tensile preload stiffens
        while compressive preload softens the structure.
        """
        f_t = tension_prestressed_result.frequencies_hz[0]
        f_c = clamped_result.frequencies_hz[0]
        f_comp = compression_prestressed_result.frequencies_hz[0]

        assert f_t > f_c > f_comp, (
            f"Expected f_tension > f_clamped > f_compression, "
            f"got {f_t:.1f} > {f_c:.1f} > {f_comp:.1f}"
        )
