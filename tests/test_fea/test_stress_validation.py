"""Tests for stress recovery module.

Tests cover:
1. Von Mises stress computation (vectorized, various shapes)
2. Principal stress computation (eigenvalue decomposition)
3. Gauss-point stress recovery (sigma = D * B * u_e at all 4 Gauss points)
4. SPR nodal extrapolation (averaging at shared nodes)
5. Hotspot identification (top-N critical stress locations)
6. Full pipeline via full_stress_recovery()
7. StressField dataclass validation
8. Patch test: uniform strain produces uniform stress everywhere

Uses a single TET10 element with known displacement to verify
stress computation analytically, plus a small multi-element mesh
to test SPR averaging.
"""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.stress_recovery import (
    StressField,
    StressRecovery,
    full_stress_recovery,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_E = 113.8e9       # Young's modulus [Pa] (Ti-6Al-4V)
_NU = 0.342        # Poisson's ratio
_RHO = 4430.0      # density [kg/m^3]


# ---------------------------------------------------------------------------
# Helper: build a reference single-element mesh
# ---------------------------------------------------------------------------
def _make_single_tet10_mesh() -> FEAMesh:
    """Create a mesh with a single TET10 element (unit tetrahedron).

    Node ordering follows the TET10Element convention:
    - Corners at (1,0,0), (0,1,0), (0,0,1), (0,0,0)
    - Mid-edge nodes at midpoints.
    """
    nodes = np.array([
        [1.0, 0.0, 0.0],   # 0
        [0.0, 1.0, 0.0],   # 1
        [0.0, 0.0, 1.0],   # 2
        [0.0, 0.0, 0.0],   # 3
        [0.5, 0.5, 0.0],   # 4  mid-edge 0-1
        [0.0, 0.5, 0.5],   # 5  mid-edge 1-2
        [0.5, 0.0, 0.5],   # 6  mid-edge 0-2
        [0.5, 0.0, 0.0],   # 7  mid-edge 0-3
        [0.0, 0.5, 0.0],   # 8  mid-edge 1-3
        [0.0, 0.0, 0.5],   # 9  mid-edge 2-3
    ], dtype=np.float64)

    elements = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)

    return FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET10",
        node_sets={"all": np.arange(10)},
        element_sets={"all": np.array([0])},
        surface_tris=np.zeros((0, 3), dtype=np.int64),
        mesh_stats={"n_nodes": 10, "n_elements": 1},
    )


def _make_two_tet10_mesh() -> FEAMesh:
    """Create a mesh with 2 TET10 elements sharing a face.

    Element 0 uses nodes 0-9 (unit tet).
    Element 1 shares the face at nodes 1,2,3 (and mid-edge 5,8,9)
    and has a new apex node at (-1, 0, 0) with new mid-edge nodes.

    This tests SPR averaging at shared nodes.
    """
    nodes = np.array([
        # Element 0 nodes
        [1.0, 0.0, 0.0],   # 0
        [0.0, 1.0, 0.0],   # 1  (shared)
        [0.0, 0.0, 1.0],   # 2  (shared)
        [0.0, 0.0, 0.0],   # 3  (shared)
        [0.5, 0.5, 0.0],   # 4  mid-edge 0-1
        [0.0, 0.5, 0.5],   # 5  mid-edge 1-2 (shared)
        [0.5, 0.0, 0.5],   # 6  mid-edge 0-2
        [0.5, 0.0, 0.0],   # 7  mid-edge 0-3
        [0.0, 0.5, 0.0],   # 8  mid-edge 1-3 (shared)
        [0.0, 0.0, 0.5],   # 9  mid-edge 2-3 (shared)
        # Element 1 new nodes
        [-1.0, 0.0, 0.0],  # 10 new apex (replaces node 0's position)
        [-0.5, 0.5, 0.0],  # 11 mid-edge 10-1
        [-0.5, 0.0, 0.5],  # 12 mid-edge 10-2
        [-0.5, 0.0, 0.0],  # 13 mid-edge 10-3
    ], dtype=np.float64)

    # Element 0: same as single-element
    # Element 1: apex=10, face=1,2,3, mid-edges: 11(10-1), 5(1-2), 12(10-2),
    #            13(10-3), 8(1-3), 9(2-3)
    elements = np.array([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 1, 2, 3, 11, 5, 12, 13, 8, 9],
    ], dtype=np.int64)

    return FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET10",
        node_sets={"all": np.arange(14)},
        element_sets={"all": np.array([0, 1])},
        surface_tris=np.zeros((0, 3), dtype=np.int64),
        mesh_stats={"n_nodes": 14, "n_elements": 2},
    )


def _uniform_strain_displacement(
    nodes: np.ndarray, eps_x: float, nu: float
) -> np.ndarray:
    """Build displacement vector for uniform uniaxial strain in X.

    eps_x in X, -nu*eps_x in Y and Z (Poisson contraction).
    This produces a uniform stress field everywhere in the element.
    """
    n_nodes = nodes.shape[0]
    u = np.zeros(3 * n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        u[3 * i] = eps_x * nodes[i, 0]
        u[3 * i + 1] = -nu * eps_x * nodes[i, 1]
        u[3 * i + 2] = -nu * eps_x * nodes[i, 2]
    return u


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def single_mesh():
    return _make_single_tet10_mesh()


@pytest.fixture
def two_elem_mesh():
    return _make_two_tet10_mesh()


@pytest.fixture
def recovery():
    return StressRecovery()


# ---------------------------------------------------------------------------
# Test: Von Mises stress computation
# ---------------------------------------------------------------------------
class TestVonMises:
    """Test vectorized Von Mises stress computation."""

    def test_uniaxial_tension(self, recovery):
        """Pure uniaxial tension: sigma_vm = |sigma_xx|."""
        # Voigt: [sxx, syy, szz, txy, tyz, txz]
        stress = np.array([100e6, 0, 0, 0, 0, 0], dtype=np.float64)
        vm = recovery.von_mises(stress)
        assert abs(vm - 100e6) < 1.0, (
            f"Expected 100 MPa, got {vm / 1e6:.4f} MPa"
        )

    def test_hydrostatic(self, recovery):
        """Hydrostatic stress: sigma_vm = 0."""
        stress = np.array([50e6, 50e6, 50e6, 0, 0, 0], dtype=np.float64)
        vm = recovery.von_mises(stress)
        assert vm < 1.0, (
            f"Hydrostatic stress should give VM ~ 0, got {vm:.4f}"
        )

    def test_pure_shear(self, recovery):
        """Pure shear: sigma_vm = sqrt(3) * tau."""
        tau = 50e6
        stress = np.array([0, 0, 0, tau, 0, 0], dtype=np.float64)
        vm = recovery.von_mises(stress)
        expected = np.sqrt(3.0) * tau
        assert abs(vm - expected) / expected < 1e-10, (
            f"Expected {expected / 1e6:.4f} MPa, got {vm / 1e6:.4f} MPa"
        )

    def test_vectorized_batch(self, recovery):
        """Von Mises should work on (N, 6) shaped input."""
        stresses = np.array([
            [100e6, 0, 0, 0, 0, 0],
            [50e6, 50e6, 50e6, 0, 0, 0],
            [0, 0, 0, 50e6, 0, 0],
        ], dtype=np.float64)
        vm = recovery.von_mises(stresses)
        assert vm.shape == (3,)
        assert abs(vm[0] - 100e6) < 1.0
        assert vm[1] < 1.0
        assert abs(vm[2] - np.sqrt(3.0) * 50e6) < 1.0

    def test_multidimensional_input(self, recovery):
        """Von Mises should work on (..., 6) shaped inputs like (E, 4, 6)."""
        stresses = np.zeros((3, 4, 6), dtype=np.float64)
        stresses[0, :, 0] = 100e6  # uniaxial in first element
        vm = recovery.von_mises(stresses)
        assert vm.shape == (3, 4)
        assert np.allclose(vm[0, :], 100e6, atol=1.0)
        assert np.allclose(vm[1, :], 0.0, atol=1e-10)

    def test_always_non_negative(self, recovery):
        """Von Mises should always be >= 0."""
        rng = np.random.default_rng(42)
        stresses = rng.standard_normal((100, 6)) * 1e8
        vm = recovery.von_mises(stresses)
        assert np.all(vm >= 0.0)


# ---------------------------------------------------------------------------
# Test: Principal stresses
# ---------------------------------------------------------------------------
class TestPrincipalStresses:
    """Test principal stress computation."""

    def test_uniaxial_principal(self, recovery):
        """For uniaxial tension, principal = [sigma_xx, 0, 0] sorted."""
        stress = np.array([100e6, 0, 0, 0, 0, 0], dtype=np.float64)
        vals, dirs = recovery.principal_stresses(stress)
        assert vals.shape == (3,)
        assert dirs.shape == (3, 3)
        # Sorted descending
        assert vals[0] >= vals[1] >= vals[2]
        assert abs(vals[0] - 100e6) < 1.0
        assert abs(vals[1]) < 1.0
        assert abs(vals[2]) < 1.0

    def test_hydrostatic_principal(self, recovery):
        """Hydrostatic: all principal stresses equal."""
        p = 50e6
        stress = np.array([p, p, p, 0, 0, 0], dtype=np.float64)
        vals, dirs = recovery.principal_stresses(stress)
        assert np.allclose(vals, p, atol=1.0)

    def test_pure_shear_principal(self, recovery):
        """Pure shear tau_xy: principals = [tau, 0, -tau]."""
        tau = 50e6
        stress = np.array([0, 0, 0, tau, 0, 0], dtype=np.float64)
        vals, dirs = recovery.principal_stresses(stress)
        assert abs(vals[0] - tau) < 1.0
        assert abs(vals[1]) < 1.0
        assert abs(vals[2] + tau) < 1.0

    def test_batch_principal(self, recovery):
        """Principal stresses should work on (N, 6) shaped input."""
        stresses = np.array([
            [100e6, 0, 0, 0, 0, 0],
            [50e6, 50e6, 50e6, 0, 0, 0],
        ], dtype=np.float64)
        vals, dirs = recovery.principal_stresses(stresses)
        assert vals.shape == (2, 3)
        assert dirs.shape == (2, 3, 3)

    def test_directions_orthogonal(self, recovery):
        """Principal directions should be orthogonal unit vectors."""
        stress = np.array([100e6, 50e6, 30e6, 10e6, 5e6, 15e6], dtype=np.float64)
        vals, dirs = recovery.principal_stresses(stress)
        # dirs is (3, 3): each column is a direction
        # Check orthogonality: dirs^T @ dirs should be identity
        product = dirs.T @ dirs
        assert np.allclose(product, np.eye(3), atol=1e-10), (
            "Principal directions should be orthogonal"
        )


# ---------------------------------------------------------------------------
# Test: Gauss-point stress recovery
# ---------------------------------------------------------------------------
class TestGaussPointStressRecovery:
    """Test recovery of stress at all 4 Gauss points."""

    def test_output_shape(self, recovery, single_mesh):
        """Output should be (n_elements, 4, 6)."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        result = recovery.recover_gauss_stresses(single_mesh, u, _E, _NU)
        assert result.shape == (1, 4, 6)

    def test_uniform_strain_produces_uniform_stress(
        self, recovery, single_mesh
    ):
        """Uniform strain should produce identical stress at all Gauss points."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        gauss_stresses = recovery.recover_gauss_stresses(
            single_mesh, u, _E, _NU
        )
        # All 4 Gauss points should give the same stress.
        # Use atol based on the stress magnitude to handle near-zero
        # components where rtol would be meaningless.
        max_stress = np.max(np.abs(gauss_stresses[0, 0]))
        for gp_idx in range(1, 4):
            assert np.allclose(
                gauss_stresses[0, gp_idx], gauss_stresses[0, 0],
                atol=max_stress * 1e-10,
            ), (
                f"Gauss point {gp_idx} stress differs from point 0 "
                f"under uniform strain"
            )

    def test_stress_matches_analytical(self, recovery, single_mesh):
        """Gauss-point stress should match D * [eps_x, -nu*eps_x, ...]."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        gauss_stresses = recovery.recover_gauss_stresses(
            single_mesh, u, _E, _NU
        )
        D = TET10Element.isotropic_elasticity_matrix(_E, _NU)
        expected_strain = np.array([
            eps_x, -_NU * eps_x, -_NU * eps_x, 0, 0, 0
        ])
        expected_stress = D @ expected_strain
        assert np.allclose(
            gauss_stresses[0, 0], expected_stress, rtol=1e-8
        ), (
            f"Stress at Gauss point 0 does not match analytical.\n"
            f"Got:      {gauss_stresses[0, 0]}\n"
            f"Expected: {expected_stress}"
        )

    def test_multi_element(self, recovery, two_elem_mesh):
        """Multi-element mesh should return (2, 4, 6)."""
        eps_x = 0.001
        u = _uniform_strain_displacement(two_elem_mesh.nodes, eps_x, _NU)
        result = recovery.recover_gauss_stresses(two_elem_mesh, u, _E, _NU)
        assert result.shape == (2, 4, 6)

    def test_consistent_with_element_stress_at_point(
        self, recovery, single_mesh
    ):
        """Results should match TET10Element.stress_at_point for each GP."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        gauss_stresses = recovery.recover_gauss_stresses(
            single_mesh, u, _E, _NU
        )

        elem = TET10Element()
        D = TET10Element.isotropic_elasticity_matrix(_E, _NU)
        coords = single_mesh.nodes[single_mesh.elements[0]]
        u_e = u  # single element, all DOFs

        for gp_idx in range(4):
            xi, eta, zeta = elem.GAUSS_POINTS[gp_idx, :3]
            expected = elem.stress_at_point(coords, u_e, D, xi, eta, zeta)
            assert np.allclose(
                gauss_stresses[0, gp_idx], expected, rtol=1e-12
            ), f"Gauss point {gp_idx} inconsistent with stress_at_point"


# ---------------------------------------------------------------------------
# Test: SPR nodal extrapolation
# ---------------------------------------------------------------------------
class TestNodalExtrapolation:
    """Test extrapolation of Gauss-point stresses to nodes."""

    def test_output_shape_single(self, recovery, single_mesh):
        """Output should be (n_nodes, 6)."""
        gauss_stresses = np.ones((1, 4, 6), dtype=np.float64) * 1e6
        nodal = recovery.extrapolate_to_nodes(single_mesh, gauss_stresses)
        assert nodal.shape == (10, 6)

    def test_output_shape_multi(self, recovery, two_elem_mesh):
        """Output should be (n_nodes, 6) for multi-element mesh."""
        gauss_stresses = np.ones((2, 4, 6), dtype=np.float64) * 1e6
        nodal = recovery.extrapolate_to_nodes(two_elem_mesh, gauss_stresses)
        assert nodal.shape == (14, 6)

    def test_uniform_stress_preserved(self, recovery, single_mesh):
        """Uniform Gauss-point stress should map to the same at all nodes."""
        sigma = np.array([100e6, 50e6, 50e6, 0, 0, 0])
        gauss_stresses = np.tile(sigma, (1, 4, 1))
        nodal = recovery.extrapolate_to_nodes(single_mesh, gauss_stresses)
        for i in range(10):
            assert np.allclose(nodal[i], sigma, rtol=1e-6), (
                f"Node {i} stress should match uniform Gauss stress"
            )

    def test_shared_nodes_averaged(self, recovery, two_elem_mesh):
        """Shared nodes should have averaged stress from both elements."""
        # Set different stresses for each element
        gauss_stresses = np.zeros((2, 4, 6), dtype=np.float64)
        gauss_stresses[0, :, 0] = 100e6  # Element 0: sxx = 100 MPa
        gauss_stresses[1, :, 0] = 200e6  # Element 1: sxx = 200 MPa

        nodal = recovery.extrapolate_to_nodes(two_elem_mesh, gauss_stresses)

        # Node 0 belongs only to element 0 => should be ~100 MPa
        assert abs(nodal[0, 0] - 100e6) / 100e6 < 0.3, (
            f"Unshared node 0 should be near 100 MPa, got {nodal[0, 0] / 1e6:.1f}"
        )
        # Node 10 belongs only to element 1 => should be ~200 MPa
        assert abs(nodal[10, 0] - 200e6) / 200e6 < 0.3, (
            f"Unshared node 10 should be near 200 MPa, got {nodal[10, 0] / 1e6:.1f}"
        )

        # Shared nodes (1, 2, 3, 5, 8, 9) should be averaged
        shared_nodes = [1, 2, 3, 5, 8, 9]
        for node_id in shared_nodes:
            # Should be between 100 and 200 MPa (averaging)
            val = nodal[node_id, 0]
            assert 80e6 < val < 220e6, (
                f"Shared node {node_id} should have averaged stress, "
                f"got {val / 1e6:.1f} MPa"
            )


# ---------------------------------------------------------------------------
# Test: Hotspot identification
# ---------------------------------------------------------------------------
class TestHotspotIdentification:
    """Test identification of critical stress locations."""

    def test_returns_list_of_dicts(self, recovery, single_mesh):
        """Hotspots should return list of dicts with required keys."""
        stress_vm = np.array([50e6], dtype=np.float64)  # 1 element
        gauss_vm = np.array([[[50e6, 48e6, 52e6, 49e6]]], dtype=np.float64)
        gauss_vm = gauss_vm.reshape(1, 4)
        hotspots = recovery.find_hotspots(single_mesh, gauss_vm, n_top=3)
        assert isinstance(hotspots, list)
        assert len(hotspots) >= 1
        required_keys = {
            "element_id", "gauss_point", "x", "y", "z",
            "stress_mpa", "safety_factor",
        }
        for hs in hotspots:
            assert required_keys.issubset(hs.keys()), (
                f"Missing keys: {required_keys - set(hs.keys())}"
            )

    def test_sorted_descending(self, recovery, single_mesh):
        """Hotspots should be sorted by stress descending."""
        gauss_vm = np.array([[50e6, 80e6, 30e6, 60e6]], dtype=np.float64)
        hotspots = recovery.find_hotspots(single_mesh, gauss_vm, n_top=4)
        stresses = [h["stress_mpa"] for h in hotspots]
        assert stresses == sorted(stresses, reverse=True), (
            f"Hotspots not sorted descending: {stresses}"
        )

    def test_n_top_limit(self, recovery, two_elem_mesh):
        """Should return at most n_top hotspots."""
        gauss_vm = np.random.default_rng(42).uniform(
            10e6, 100e6, size=(2, 4)
        )
        hotspots = recovery.find_hotspots(two_elem_mesh, gauss_vm, n_top=3)
        assert len(hotspots) <= 3

    def test_coordinates_are_physical(self, recovery, single_mesh):
        """Hotspot coordinates should be valid physical coordinates."""
        gauss_vm = np.array([[50e6, 80e6, 30e6, 60e6]], dtype=np.float64)
        hotspots = recovery.find_hotspots(single_mesh, gauss_vm, n_top=1)
        hs = hotspots[0]
        # Gauss point 1 (index) has the max stress
        # Coordinates should be within the element bounding box
        assert -0.1 <= hs["x"] <= 1.1
        assert -0.1 <= hs["y"] <= 1.1
        assert -0.1 <= hs["z"] <= 1.1

    def test_safety_factor_computed(self, recovery, single_mesh):
        """Safety factor should be yield_strength / stress when provided."""
        gauss_vm = np.array([[100e6, 200e6, 50e6, 150e6]], dtype=np.float64)
        hotspots = recovery.find_hotspots(
            single_mesh, gauss_vm, n_top=1, yield_strength_mpa=880.0
        )
        hs = hotspots[0]
        # Max stress is 200 MPa => SF = 880 / 200 = 4.4
        expected_sf = 880.0 / 200.0
        assert abs(hs["safety_factor"] - expected_sf) < 0.01

    def test_safety_factor_inf_without_yield(self, recovery, single_mesh):
        """Safety factor should be inf when yield_strength is not provided."""
        gauss_vm = np.array([[100e6, 200e6, 50e6, 150e6]], dtype=np.float64)
        hotspots = recovery.find_hotspots(single_mesh, gauss_vm, n_top=1)
        hs = hotspots[0]
        assert hs["safety_factor"] == float("inf")


# ---------------------------------------------------------------------------
# Test: StressField dataclass
# ---------------------------------------------------------------------------
class TestStressFieldDataclass:
    """Test the StressField dataclass structure."""

    def test_fields_present(self):
        """StressField should have all required fields."""
        sf = StressField(
            gauss_stresses=np.zeros((1, 4, 6)),
            nodal_stresses=np.zeros((10, 6)),
            von_mises_gauss=np.zeros((1, 4)),
            von_mises_nodal=np.zeros((10,)),
            principal_values=np.zeros((10, 3)),
            principal_directions=np.zeros((10, 3, 3)),
            hotspots=[],
        )
        assert sf.gauss_stresses is not None
        assert sf.nodal_stresses is not None
        assert sf.von_mises_gauss is not None
        assert sf.von_mises_nodal is not None
        assert sf.principal_values is not None
        assert sf.principal_directions is not None
        assert sf.hotspots is not None


# ---------------------------------------------------------------------------
# Test: Full pipeline
# ---------------------------------------------------------------------------
class TestFullStressRecovery:
    """Test the full_stress_recovery convenience function."""

    def test_full_pipeline_single_element(self, single_mesh):
        """Full pipeline should work on a single-element mesh."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        sf = full_stress_recovery(single_mesh, u, _E, _NU, n_hotspots=3)

        assert isinstance(sf, StressField)
        assert sf.gauss_stresses.shape == (1, 4, 6)
        assert sf.nodal_stresses.shape == (10, 6)
        assert sf.von_mises_gauss.shape == (1, 4)
        assert sf.von_mises_nodal.shape == (10,)
        assert sf.principal_values.shape == (10, 3)
        assert sf.principal_directions.shape == (10, 3, 3)
        assert len(sf.hotspots) <= 3

    def test_full_pipeline_multi_element(self, two_elem_mesh):
        """Full pipeline should work on a multi-element mesh."""
        eps_x = 0.001
        u = _uniform_strain_displacement(two_elem_mesh.nodes, eps_x, _NU)
        sf = full_stress_recovery(two_elem_mesh, u, _E, _NU, n_hotspots=5)

        assert isinstance(sf, StressField)
        assert sf.gauss_stresses.shape == (2, 4, 6)
        assert sf.nodal_stresses.shape == (14, 6)
        assert sf.von_mises_gauss.shape == (2, 4)
        assert sf.von_mises_nodal.shape == (14,)

    def test_von_mises_consistency(self, single_mesh):
        """Von Mises from gauss stresses should match direct computation."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        sf = full_stress_recovery(single_mesh, u, _E, _NU)

        recovery = StressRecovery()
        expected_vm = recovery.von_mises(sf.gauss_stresses)
        assert np.allclose(sf.von_mises_gauss, expected_vm, rtol=1e-12)

    def test_hotspots_contain_max_stress(self, single_mesh):
        """The first hotspot should contain the maximum stress."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        sf = full_stress_recovery(single_mesh, u, _E, _NU, n_hotspots=5)

        if len(sf.hotspots) > 0:
            max_vm_mpa = np.max(sf.von_mises_gauss) / 1e6
            # The first hotspot should have the highest stress
            assert abs(sf.hotspots[0]["stress_mpa"] - max_vm_mpa) < 1.0


# ---------------------------------------------------------------------------
# Test: Patch test (uniform strain must be exact)
# ---------------------------------------------------------------------------
class TestPatchTest:
    """Patch test: uniform strain field must be exactly reproduced."""

    def test_patch_uniform_tension(self, recovery, single_mesh):
        """Uniform tension: stress should be uniform at all nodes and GPs."""
        eps_x = 0.001
        u = _uniform_strain_displacement(single_mesh.nodes, eps_x, _NU)
        D = TET10Element.isotropic_elasticity_matrix(_E, _NU)
        expected_strain = np.array([
            eps_x, -_NU * eps_x, -_NU * eps_x, 0, 0, 0
        ])
        expected_stress = D @ expected_strain

        gauss_stresses = recovery.recover_gauss_stresses(
            single_mesh, u, _E, _NU
        )

        # All Gauss points should match.
        # Use atol based on stress magnitude since near-zero components
        # have floating-point noise at ~1e-8 level.
        max_stress = np.max(np.abs(expected_stress))
        for gp in range(4):
            assert np.allclose(
                gauss_stresses[0, gp], expected_stress,
                atol=max_stress * 1e-8,
            ), f"GP {gp} failed patch test"

        # Nodal stresses should also match (uniform field)
        nodal = recovery.extrapolate_to_nodes(single_mesh, gauss_stresses)
        for n in range(10):
            assert np.allclose(
                nodal[n], expected_stress,
                atol=max_stress * 1e-6,
            ), f"Node {n} failed patch test"


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_displacement(self, recovery, single_mesh):
        """Zero displacement should give zero stress everywhere."""
        u = np.zeros(30, dtype=np.float64)
        gauss_stresses = recovery.recover_gauss_stresses(
            single_mesh, u, _E, _NU
        )
        assert np.allclose(gauss_stresses, 0.0, atol=1e-20)

    def test_von_mises_single_value(self, recovery):
        """Von Mises should work on a single (6,) stress vector."""
        stress = np.array([100e6, 0, 0, 0, 0, 0], dtype=np.float64)
        vm = recovery.von_mises(stress)
        assert isinstance(vm, (float, np.floating))
        assert abs(vm - 100e6) < 1.0

    def test_find_hotspots_empty(self, recovery, single_mesh):
        """Find hotspots with zero stress should still work."""
        gauss_vm = np.zeros((1, 4), dtype=np.float64)
        hotspots = recovery.find_hotspots(single_mesh, gauss_vm, n_top=3)
        assert isinstance(hotspots, list)
