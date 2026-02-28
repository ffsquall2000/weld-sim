"""Tests for harmonic stress analysis module.

Tests the extracted _compute_element_stresses() helper, the
harmonic_stress_analysis() method on SolverA, and the
HarmonicStressResult dataclass.

Uses a Ti-6Al-4V uniform cylindrical bar, diameter 25 mm, length 80 mm,
mesh_size=6.0, order=2.
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    HarmonicConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    HarmonicResult,
    HarmonicStressResult,
    StaticResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIAMETER_MM = 25.0
_LENGTH_MM = 80.0
_MATERIAL = "Titanium Ti-6Al-4V"
_YIELD_MPA = 880.0  # Ti-6Al-4V yield strength


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
    """Reusable solver instance."""
    return SolverA()


@pytest.fixture(scope="module")
def static_result(cylinder_mesh: FEAMesh, solver: SolverA) -> StaticResult:
    """Run a static analysis (tensile load) to verify refactored code."""
    config = StaticConfig(
        mesh=cylinder_mesh,
        material_name=_MATERIAL,
        loads=[
            {
                "type": "force",
                "node_set": "top_face",
                "direction": [0, 1, 0],
                "magnitude": 10000.0,  # 10 kN
            },
        ],
        boundary_conditions=[
            {
                "type": "fixed",
                "node_set": "bottom_face",
            },
        ],
    )
    return solver.static_analysis(config)


@pytest.fixture(scope="module")
def harmonic_config(cylinder_mesh: FEAMesh) -> HarmonicConfig:
    """Harmonic config for a Ti-6Al-4V cylinder around 20 kHz."""
    return HarmonicConfig(
        mesh=cylinder_mesh,
        material_name=_MATERIAL,
        freq_min_hz=18000.0,
        freq_max_hz=22000.0,
        n_freq_points=21,  # small for test speed
        damping_model="hysteretic",
        damping_ratio=0.005,
    )


@pytest.fixture(scope="module")
def harmonic_result(
    harmonic_config: HarmonicConfig, solver: SolverA
) -> HarmonicResult:
    """Run harmonic analysis on the cylinder mesh."""
    return solver.harmonic_analysis(harmonic_config)


@pytest.fixture(scope="module")
def stress_result(
    harmonic_result: HarmonicResult,
    harmonic_config: HarmonicConfig,
    solver: SolverA,
) -> HarmonicStressResult:
    """Run harmonic stress analysis at 20 kHz."""
    return solver.harmonic_stress_analysis(
        harmonic_result, harmonic_config, target_freq_hz=20000.0
    )


# ---------------------------------------------------------------------------
# Tests for _compute_element_stresses helper
# ---------------------------------------------------------------------------
class TestComputeElementStresses:
    """Test the reusable _compute_element_stresses() helper."""

    def test_returns_correct_shapes(
        self, cylinder_mesh: FEAMesh, solver: SolverA
    ):
        """_compute_element_stresses should return arrays of correct shape."""
        n_dof = cylinder_mesh.n_dof
        n_elements = cylinder_mesh.elements.shape[0]

        # Create a simple displacement field (e.g., unit extension in Y)
        displacement = np.zeros(n_dof, dtype=np.float64)
        # Small uniform Y displacement at all nodes
        displacement[1::3] = 1e-6

        stress_vm, stress_tensor, max_stress_mpa = solver._compute_element_stresses(
            cylinder_mesh, displacement, _MATERIAL
        )

        assert stress_vm.shape == (n_elements,), (
            f"Expected stress_vm shape ({n_elements},), got {stress_vm.shape}"
        )
        assert stress_tensor.shape == (n_elements, 6), (
            f"Expected stress_tensor shape ({n_elements}, 6), got {stress_tensor.shape}"
        )
        assert isinstance(max_stress_mpa, float)

    def test_zero_displacement_gives_zero_stress(
        self, cylinder_mesh: FEAMesh, solver: SolverA
    ):
        """Zero displacement field should yield zero stress everywhere."""
        n_dof = cylinder_mesh.n_dof
        displacement = np.zeros(n_dof, dtype=np.float64)

        stress_vm, stress_tensor, max_stress_mpa = solver._compute_element_stresses(
            cylinder_mesh, displacement, _MATERIAL
        )

        assert np.allclose(stress_vm, 0.0), "Expected zero VM stress for zero displacement"
        assert np.allclose(stress_tensor, 0.0), "Expected zero stress tensor for zero displacement"
        assert max_stress_mpa == 0.0

    def test_von_mises_matches_known_formula(self):
        """Verify Von Mises formula for a known uniaxial stress state.

        For pure uniaxial stress sigma_xx = S, all other components zero:
            VM = sqrt(0.5 * ((S-0)^2 + (0-0)^2 + (0-S)^2)) = S
        """
        # Voigt: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
        S = 100e6  # 100 MPa
        stress_tensor = np.array([[S, 0.0, 0.0, 0.0, 0.0, 0.0]])

        sxx = stress_tensor[:, 0]
        syy = stress_tensor[:, 1]
        szz = stress_tensor[:, 2]
        txy = stress_tensor[:, 3]
        tyz = stress_tensor[:, 4]
        txz = stress_tensor[:, 5]

        stress_vm = np.sqrt(
            0.5 * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (txy ** 2 + tyz ** 2 + txz ** 2)
            )
        )

        np.testing.assert_allclose(stress_vm[0], S, rtol=1e-12)

    def test_von_mises_pure_shear(self):
        """Verify Von Mises for pure shear: tau_xy = T.

        VM = sqrt(3) * T
        """
        T = 50e6  # 50 MPa shear
        stress_tensor = np.array([[0.0, 0.0, 0.0, T, 0.0, 0.0]])

        sxx = stress_tensor[:, 0]
        syy = stress_tensor[:, 1]
        szz = stress_tensor[:, 2]
        txy = stress_tensor[:, 3]
        tyz = stress_tensor[:, 4]
        txz = stress_tensor[:, 5]

        stress_vm = np.sqrt(
            0.5 * (
                (sxx - syy) ** 2
                + (syy - szz) ** 2
                + (szz - sxx) ** 2
                + 6.0 * (txy ** 2 + tyz ** 2 + txz ** 2)
            )
        )

        expected = np.sqrt(3.0) * T
        np.testing.assert_allclose(stress_vm[0], expected, rtol=1e-12)


class TestStaticAnalysisRefactored:
    """Test that static_analysis() still works after refactoring."""

    def test_static_result_fields(self, static_result: StaticResult):
        """Static analysis should still return valid results after refactoring."""
        assert isinstance(static_result, StaticResult)
        assert static_result.displacement is not None
        assert static_result.stress_vm is not None
        assert static_result.stress_tensor is not None
        assert static_result.max_stress_mpa > 0.0
        assert static_result.solver_name == "SolverA"

    def test_static_vm_positive(self, static_result: StaticResult):
        """Von Mises stress should be non-negative."""
        assert np.all(static_result.stress_vm >= 0.0)

    def test_static_max_matches(self, static_result: StaticResult):
        """max_stress_mpa should match max(stress_vm) / 1e6."""
        expected = np.max(static_result.stress_vm) / 1e6
        assert abs(static_result.max_stress_mpa - expected) < 1e-6


# ---------------------------------------------------------------------------
# Tests for harmonic_stress_analysis()
# ---------------------------------------------------------------------------
class TestHarmonicStressAnalysis:
    """Test the harmonic_stress_analysis() method."""

    def test_returns_harmonic_stress_result(
        self, stress_result: HarmonicStressResult
    ):
        """Should return a HarmonicStressResult instance."""
        assert isinstance(stress_result, HarmonicStressResult)

    def test_all_fields_populated(
        self, stress_result: HarmonicStressResult, cylinder_mesh: FEAMesh
    ):
        """All fields in HarmonicStressResult should be populated."""
        n_elements = cylinder_mesh.elements.shape[0]
        n_nodes = cylinder_mesh.nodes.shape[0]

        assert stress_result.stress_vm.shape == (n_elements,)
        assert stress_result.stress_tensor.shape == (n_elements, 6)
        assert stress_result.max_stress_mpa >= 0.0
        assert stress_result.safety_factor > 0.0
        assert stress_result.displacement_amplitude.shape == (n_nodes,)
        assert stress_result.max_displacement_mm >= 0.0
        assert 0.0 <= stress_result.contact_face_uniformity <= 1.0
        assert stress_result.mesh is not None
        assert stress_result.solve_time_s >= 0.0
        assert stress_result.solver_name == "SolverA"

    def test_stress_vm_non_negative(
        self, stress_result: HarmonicStressResult
    ):
        """Von Mises stress from harmonic analysis should be non-negative."""
        assert np.all(stress_result.stress_vm >= 0.0)

    def test_displacement_amplitude_non_negative(
        self, stress_result: HarmonicStressResult
    ):
        """Displacement amplitude should be non-negative."""
        assert np.all(stress_result.displacement_amplitude >= 0.0)

    def test_safety_factor_from_yield(
        self, stress_result: HarmonicStressResult
    ):
        """safety_factor should equal yield_strength / max_vm_stress."""
        if stress_result.max_stress_mpa > 0.0:
            expected = _YIELD_MPA / stress_result.max_stress_mpa
            assert abs(stress_result.safety_factor - expected) < 1e-6, (
                f"safety_factor {stress_result.safety_factor} != "
                f"yield/max_vm = {expected}"
            )
        else:
            assert stress_result.safety_factor == float("inf")

    def test_max_displacement_mm_consistent(
        self, stress_result: HarmonicStressResult
    ):
        """max_displacement_mm should be max(displacement_amplitude) * 1000."""
        expected_mm = float(np.max(stress_result.displacement_amplitude)) * 1000.0
        assert abs(stress_result.max_displacement_mm - expected_mm) < 1e-10


class TestHarmonicStressResonancePeak:
    """Test that harmonic stress at resonance uses the peak frequency."""

    def test_default_uses_resonance_peak(
        self,
        harmonic_result: HarmonicResult,
        harmonic_config: HarmonicConfig,
        solver: SolverA,
    ):
        """When target_freq_hz is None, should use resonance peak."""
        result = solver.harmonic_stress_analysis(
            harmonic_result, harmonic_config, target_freq_hz=None
        )
        assert isinstance(result, HarmonicStressResult)
        # The stress should be non-negative
        assert np.all(result.stress_vm >= 0.0)
        # At resonance, there should be some displacement
        assert result.max_displacement_mm > 0.0
