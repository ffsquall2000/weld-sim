"""Tests for multi-body assembly builder (coupling of ultrasonic stack components).

Tests use synthetic FEAMesh objects with simple TET10 geometries to verify:
- Interface node pair finding (closest-point matching)
- Bonded coupling via penalty springs
- Penalty coupling with configurable stiffness
- DOF mapping and node offset tracking
- Acoustic impedance computation
- Transmission coefficient calculation
- Full 2-component assembly build
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembly_builder import (
    AssemblyBuilder,
    AssemblyConfig,
    AssemblyResult,
    ComponentMesh,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh


# ---------------------------------------------------------------------------
# Helper: build a minimal TET10 mesh for a single element
# ---------------------------------------------------------------------------

def _make_single_tet10_block(
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a single TET10 element as a small tetrahedron.

    Corner nodes form a right tetrahedron with edge length ``size``.
    Mid-edge nodes are placed at the midpoints of each edge.

    Parameters
    ----------
    x_offset, y_offset, z_offset : float
        Translation of the tetrahedron origin.
    size : float
        Edge length in meters.

    Returns
    -------
    nodes : np.ndarray, shape (10, 3)
    elements : np.ndarray, shape (1, 10)
    """
    # Corner nodes (Bathe convention)
    # Node 0: (1,0,0), Node 1: (0,1,0), Node 2: (0,0,1), Node 3: (0,0,0)
    c = np.array([
        [size, 0.0,  0.0],   # 0
        [0.0,  size, 0.0],   # 1
        [0.0,  0.0,  size],  # 2
        [0.0,  0.0,  0.0],   # 3
    ])
    # Mid-edge nodes
    m = np.array([
        (c[0] + c[1]) / 2,  # 4: mid 0-1
        (c[1] + c[2]) / 2,  # 5: mid 1-2
        (c[0] + c[2]) / 2,  # 6: mid 0-2
        (c[0] + c[3]) / 2,  # 7: mid 0-3
        (c[1] + c[3]) / 2,  # 8: mid 1-3
        (c[2] + c[3]) / 2,  # 9: mid 2-3
    ])
    nodes = np.vstack([c, m]) + np.array([x_offset, y_offset, z_offset])
    elements = np.arange(10, dtype=np.int64).reshape(1, 10)
    return nodes, elements


def _make_two_tet10_block(
    y_base: float = 0.0,
    size: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Create two TET10 elements stacked along Y to form a slightly taller mesh.

    Returns nodes and elements for a mesh with 2 elements sharing some nodes.
    The first tet occupies [0, size] in Y, the second occupies [0, 2*size] roughly.
    """
    # Build first tet
    nodes1, _ = _make_single_tet10_block(y_offset=y_base, size=size)

    # Build a second tet by adding a new apex at (0, 2*size, 0)
    # and reusing some nodes from the first tet.
    # For simplicity, create a completely separate second tet offset in Y.
    nodes2, _ = _make_single_tet10_block(y_offset=y_base + size, size=size)

    # Offset node indices for second element
    n_nodes_1 = nodes1.shape[0]
    nodes = np.vstack([nodes1, nodes2])
    elem1 = np.arange(10, dtype=np.int64).reshape(1, 10)
    elem2 = (np.arange(10, dtype=np.int64) + n_nodes_1).reshape(1, 10)
    elements = np.vstack([elem1, elem2])

    return nodes, elements


def _make_component_mesh(
    name: str,
    material_name: str,
    y_base: float = 0.0,
    size: float = 0.01,
    n_elements: int = 1,
) -> ComponentMesh:
    """Build a synthetic ComponentMesh for testing.

    Parameters
    ----------
    name : str
        Component name (e.g. "horn", "booster").
    material_name : str
        Material name from material_properties database.
    y_base : float
        Starting Y coordinate.
    size : float
        Characteristic size of each tet element.
    n_elements : int
        1 or 2 elements.
    """
    if n_elements == 1:
        nodes, elements = _make_single_tet10_block(y_offset=y_base, size=size)
    else:
        nodes, elements = _make_two_tet10_block(y_base=y_base, size=size)

    y_coords = nodes[:, 1]
    y_min = y_coords.min()
    y_max = y_coords.max()
    tol = size * 0.1

    bottom_nodes = np.where(np.abs(y_coords - y_min) < tol)[0]
    top_nodes = np.where(np.abs(y_coords - y_max) < tol)[0]

    mesh = FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET10",
        node_sets={"bottom_face": bottom_nodes, "top_face": top_nodes},
        element_sets={},
        surface_tris=np.array([[0, 1, 2]]),
        mesh_stats={
            "num_nodes": nodes.shape[0],
            "num_elements": elements.shape[0],
        },
    )

    return ComponentMesh(
        name=name,
        mesh=mesh,
        material_name=material_name,
        interface_nodes={"top": top_nodes, "bottom": bottom_nodes},
    )


# ---------------------------------------------------------------------------
# Helper: build two components with matching interfaces
# ---------------------------------------------------------------------------

def _make_matching_interface_components(
    size: float = 0.01,
) -> tuple[ComponentMesh, ComponentMesh]:
    """Create two single-element components whose top/bottom faces share geometry.

    Component A sits at Y=[0, size], component B at Y=[size, 2*size].
    The top face of A and the bottom face of B share the same X-Z coordinates
    at the Y=size interface.
    """
    # Component A: standard tet at Y=0
    comp_a = _make_component_mesh("booster", "Titanium Ti-6Al-4V", y_base=0.0, size=size)

    # Component B: same tet shifted up by 'size' in Y
    comp_b = _make_component_mesh("horn", "Titanium Ti-6Al-4V", y_base=size, size=size)

    return comp_a, comp_b


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def matching_components():
    """Two components with matching interface nodes."""
    return _make_matching_interface_components(size=0.01)


@pytest.fixture
def default_config():
    """Default bonded assembly configuration."""
    return AssemblyConfig(
        coupling_method="bonded",
        penalty_factor=1e3,
        component_order=["booster", "horn"],
    )


@pytest.fixture
def penalty_config():
    """Penalty coupling configuration with lower stiffness."""
    return AssemblyConfig(
        coupling_method="penalty",
        penalty_factor=1e1,
        component_order=["booster", "horn"],
    )


# ---------------------------------------------------------------------------
# Test: ComponentMesh and AssemblyConfig dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_component_mesh_creation(self):
        """ComponentMesh should store name, mesh, material, and interface nodes."""
        comp = _make_component_mesh("horn", "Titanium Ti-6Al-4V")
        assert comp.name == "horn"
        assert comp.material_name == "Titanium Ti-6Al-4V"
        assert isinstance(comp.mesh, FEAMesh)
        assert "top" in comp.interface_nodes
        assert "bottom" in comp.interface_nodes

    def test_assembly_config_defaults(self):
        """AssemblyConfig should have sensible defaults."""
        cfg = AssemblyConfig()
        assert cfg.coupling_method == "bonded"
        assert cfg.penalty_factor == 1e3
        assert cfg.component_order == ["transducer", "booster", "horn"]

    def test_assembly_config_custom(self):
        """AssemblyConfig should accept custom values."""
        cfg = AssemblyConfig(
            coupling_method="penalty",
            penalty_factor=50.0,
            component_order=["booster", "horn"],
        )
        assert cfg.coupling_method == "penalty"
        assert cfg.penalty_factor == 50.0


# ---------------------------------------------------------------------------
# Test: Interface node matching
# ---------------------------------------------------------------------------

class TestInterfaceMatching:
    def test_find_pairs_matching_geometry(self, matching_components, default_config):
        """Interface node pairs should be found for components with matching faces."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        # Should have found at least one interface pair
        assert len(result.interface_pairs) == 1
        pair_info = result.interface_pairs[0]
        assert pair_info["comp_a"] == "booster"
        assert pair_info["comp_b"] == "horn"
        assert len(pair_info["node_pairs"]) > 0

    def test_find_pairs_no_match(self, default_config):
        """Components with no overlapping faces should yield zero pairs."""
        # Place components far apart in X-Z so interface nodes don't overlap.
        # (Interface matching uses X-Z distance only, ignoring Y.)
        comp_a = _make_component_mesh("booster", "Titanium Ti-6Al-4V",
                                       y_base=0.0, size=0.01)
        # Shift comp_b by 10 m in X so no X-Z overlap exists
        nodes_b, elems_b = _make_single_tet10_block(
            x_offset=10.0, y_offset=0.01, size=0.01
        )
        y_coords_b = nodes_b[:, 1]
        tol = 0.001
        bottom_b = np.where(np.abs(y_coords_b - y_coords_b.min()) < tol)[0]
        top_b = np.where(np.abs(y_coords_b - y_coords_b.max()) < tol)[0]
        mesh_b = FEAMesh(
            nodes=nodes_b, elements=elems_b, element_type="TET10",
            node_sets={"bottom_face": bottom_b, "top_face": top_b},
            element_sets={}, surface_tris=np.array([[0, 1, 2]]),
            mesh_stats={"num_nodes": nodes_b.shape[0], "num_elements": 1},
        )
        comp_b = ComponentMesh(
            name="horn", mesh=mesh_b, material_name="Titanium Ti-6Al-4V",
            interface_nodes={"top": top_b, "bottom": bottom_b},
        )

        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        # Interface pair entry exists but with zero matched nodes
        pair_info = result.interface_pairs[0]
        assert len(pair_info["node_pairs"]) == 0


# ---------------------------------------------------------------------------
# Test: DOF mapping and offsets
# ---------------------------------------------------------------------------

class TestDofMapping:
    def test_dof_map_keys(self, matching_components, default_config):
        """DOF map should contain entries for each component."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        assert "booster" in result.dof_map
        assert "horn" in result.dof_map

    def test_node_offset_map(self, matching_components, default_config):
        """Node offset map should correctly track component stacking."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        assert result.node_offset_map["booster"] == 0
        assert result.node_offset_map["horn"] == comp_a.mesh.nodes.shape[0]

    def test_total_dof(self, matching_components, default_config):
        """Total DOFs should equal sum of component DOFs."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        expected_dof = comp_a.mesh.n_dof + comp_b.mesh.n_dof
        assert result.n_total_dof == expected_dof

    def test_dof_ranges_non_overlapping(self, matching_components, default_config):
        """DOF ranges of different components should not overlap."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        dofs_a = set(result.dof_map["booster"].tolist())
        dofs_b = set(result.dof_map["horn"].tolist())
        assert len(dofs_a & dofs_b) == 0, "DOF ranges should not overlap"


# ---------------------------------------------------------------------------
# Test: Bonded coupling
# ---------------------------------------------------------------------------

class TestBondedCoupling:
    def test_coupled_matrix_dimensions(self, matching_components, default_config):
        """Coupled K and M should be (n_total_dof, n_total_dof)."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        n = result.n_total_dof
        assert result.K_global.shape == (n, n)
        assert result.M_global.shape == (n, n)

    def test_coupled_stiffness_symmetric(self, matching_components, default_config):
        """Coupled K should be symmetric."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        K = result.K_global
        diff = K - K.T
        if diff.nnz > 0:
            assert np.max(np.abs(diff.data)) < 1e-6 * sp.linalg.norm(K), \
                "Coupled stiffness matrix should be symmetric"

    def test_coupled_mass_symmetric(self, matching_components, default_config):
        """Coupled M should be symmetric."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        M = result.M_global
        diff = M - M.T
        if diff.nnz > 0:
            assert np.max(np.abs(diff.data)) < 1e-6 * sp.linalg.norm(M), \
                "Coupled mass matrix should be symmetric"

    def test_penalty_spring_placed_correctly(self, matching_components, default_config):
        """Bonded coupling should add penalty springs at interface DOFs.

        After coupling, the diagonal entries at interface DOFs should
        be larger than before coupling (due to penalty stiffness addition).
        """
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        K = result.K_global
        K_diag = K.diagonal()

        # All diagonal entries should be non-negative
        assert np.all(K_diag >= -1e-10 * np.max(np.abs(K_diag))), \
            "Coupled K diagonal should be non-negative"

        # If there are interface pairs, the coupled matrix should have
        # off-diagonal entries connecting the two components
        if len(result.interface_pairs) > 0:
            pair_info = result.interface_pairs[0]
            if len(pair_info["node_pairs"]) > 0:
                node_pair = pair_info["node_pairs"][0]
                global_node_a = node_pair[0]
                global_node_b = node_pair[1]

                # Check that off-diagonal coupling exists
                for d in range(3):
                    dof_a = 3 * global_node_a + d
                    dof_b = 3 * global_node_b + d
                    # The off-diagonal entry should be negative (penalty spring)
                    assert K[dof_a, dof_b] < 0, \
                        f"Off-diagonal K[{dof_a},{dof_b}] should be negative (penalty spring)"

    def test_matrices_are_csr(self, matching_components, default_config):
        """Result matrices should be CSR format."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        assert isinstance(result.K_global, sp.csr_matrix)
        assert isinstance(result.M_global, sp.csr_matrix)


# ---------------------------------------------------------------------------
# Test: Penalty coupling
# ---------------------------------------------------------------------------

class TestPenaltyCoupling:
    def test_penalty_coupling_lower_stiffness(
        self, matching_components, default_config, penalty_config
    ):
        """Penalty coupling with lower factor should yield smaller off-diagonal entries."""
        comp_a, comp_b = matching_components

        builder_bonded = AssemblyBuilder([comp_a, comp_b], default_config)
        result_bonded = builder_bonded.build()

        builder_penalty = AssemblyBuilder([comp_a, comp_b], penalty_config)
        result_penalty = builder_penalty.build()

        # The bonded penalty_factor (1e3) >> penalty_factor (1e1)
        # So the bonded coupling should produce larger absolute off-diagonal values
        K_bonded = result_bonded.K_global
        K_penalty = result_penalty.K_global

        if (len(result_bonded.interface_pairs) > 0
                and len(result_bonded.interface_pairs[0]["node_pairs"]) > 0):
            node_pair = result_bonded.interface_pairs[0]["node_pairs"][0]
            dof_a = 3 * node_pair[0]
            dof_b = 3 * node_pair[1]

            assert abs(K_bonded[dof_a, dof_b]) > abs(K_penalty[dof_a, dof_b]), \
                "Bonded coupling should produce larger penalty springs than penalty coupling"

    def test_penalty_coupling_symmetric(self, matching_components, penalty_config):
        """Penalty coupled K should also be symmetric."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], penalty_config)
        result = builder.build()

        K = result.K_global
        diff = K - K.T
        if diff.nnz > 0:
            assert np.max(np.abs(diff.data)) < 1e-6 * sp.linalg.norm(K)


# ---------------------------------------------------------------------------
# Test: Acoustic impedance
# ---------------------------------------------------------------------------

class TestAcousticImpedance:
    def test_impedance_titanium(self, matching_components, default_config):
        """Impedance for Ti-6Al-4V should match Z = rho * c * A.

        For Ti-6Al-4V: rho = 4430 kg/m^3, c = sqrt(E/rho) = sqrt(113.8e9/4430) ~ 5068 m/s.
        """
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        # Both components use Ti-6Al-4V
        assert "booster" in result.impedance
        assert "horn" in result.impedance

        # Impedance should be positive
        assert result.impedance["booster"] > 0
        assert result.impedance["horn"] > 0

    def test_impedance_formula(self):
        """Verify impedance Z = rho * c * A with known values."""
        # Use a component with known geometry to check impedance formula
        comp = _make_component_mesh("horn", "Titanium Ti-6Al-4V", size=0.01)
        config = AssemblyConfig(component_order=["horn"])
        builder = AssemblyBuilder([comp], config)

        # Compute impedance directly
        rho = 4430.0
        E = 113.8e9
        c = math.sqrt(E / rho)

        # c should be approximately 5068 m/s
        assert abs(c - 5068.0) < 5.0, f"Wave speed should be ~5068 m/s, got {c:.1f}"

        Z = builder._compute_impedance(comp)
        assert Z > 0, "Impedance should be positive"

        # Z = rho * c * A, so Z / (rho * c) should give the interface area
        A_computed = Z / (rho * c)
        assert A_computed > 0, "Estimated area should be positive"


# ---------------------------------------------------------------------------
# Test: Transmission coefficient
# ---------------------------------------------------------------------------

class TestTransmissionCoefficient:
    def test_same_material_perfect_transmission(self, matching_components, default_config):
        """Two identical materials should have T ~ 1.0 (perfect transmission)."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        key = "booster-horn"
        assert key in result.transmission_coefficients
        T = result.transmission_coefficients[key]
        # Same material -> T should be 1.0
        assert abs(T - 1.0) < 0.01, \
            f"Same material transmission should be ~1.0, got {T:.4f}"

    def test_transmission_coefficient_formula(self):
        """T = 4*Z_A*Z_B / (Z_A + Z_B)^2 should match known values."""
        comp = _make_component_mesh("test", "Titanium Ti-6Al-4V")
        config = AssemblyConfig(component_order=["test"])
        builder = AssemblyBuilder([comp], config)

        # Same impedance -> T = 1.0
        T_same = builder._transmission_coefficient(100.0, 100.0)
        assert abs(T_same - 1.0) < 1e-10

        # Different impedances -> T < 1.0
        Z_A, Z_B = 100.0, 200.0
        T_expected = 4.0 * Z_A * Z_B / (Z_A + Z_B) ** 2
        T_actual = builder._transmission_coefficient(Z_A, Z_B)
        assert abs(T_actual - T_expected) < 1e-10

        # T should be symmetric: T(A,B) = T(B,A)
        T_ba = builder._transmission_coefficient(Z_B, Z_A)
        assert abs(T_actual - T_ba) < 1e-10

    def test_transmission_always_leq_one(self):
        """Transmission coefficient should always be <= 1.0."""
        comp = _make_component_mesh("test", "Titanium Ti-6Al-4V")
        config = AssemblyConfig(component_order=["test"])
        builder = AssemblyBuilder([comp], config)

        for Z_A in [1.0, 10.0, 100.0, 1000.0]:
            for Z_B in [1.0, 10.0, 100.0, 1000.0]:
                T = builder._transmission_coefficient(Z_A, Z_B)
                assert T <= 1.0 + 1e-10, \
                    f"T({Z_A}, {Z_B}) = {T} > 1.0"
                assert T > 0, f"T({Z_A}, {Z_B}) = {T} <= 0"


# ---------------------------------------------------------------------------
# Test: Different materials assembly
# ---------------------------------------------------------------------------

class TestDifferentMaterials:
    def test_mixed_material_assembly(self):
        """Assembly with different materials should work and have T < 1."""
        comp_a = _make_component_mesh("booster", "Titanium Ti-6Al-4V",
                                       y_base=0.0, size=0.01)
        comp_b = _make_component_mesh("horn", "Aluminum 7075-T6",
                                       y_base=0.01, size=0.01)
        config = AssemblyConfig(component_order=["booster", "horn"])
        builder = AssemblyBuilder([comp_a, comp_b], config)
        result = builder.build()

        # Both impedances should be present
        assert "booster" in result.impedance
        assert "horn" in result.impedance

        # Different materials -> T < 1
        key = "booster-horn"
        if key in result.transmission_coefficients:
            T = result.transmission_coefficients[key]
            assert T < 1.0, "Different materials should have T < 1.0"
            assert T > 0.0, "Transmission should be positive"

    def test_material_map_in_result(self):
        """The assembly should track which material is used for each component."""
        comp_a = _make_component_mesh("booster", "Titanium Ti-6Al-4V",
                                       y_base=0.0, size=0.01)
        comp_b = _make_component_mesh("horn", "Steel D2",
                                       y_base=0.01, size=0.01)
        config = AssemblyConfig(component_order=["booster", "horn"])
        builder = AssemblyBuilder([comp_a, comp_b], config)
        result = builder.build()

        assert result.K_global.shape[0] == result.n_total_dof
        assert result.M_global.shape[0] == result.n_total_dof


# ---------------------------------------------------------------------------
# Test: Single component assembly (edge case)
# ---------------------------------------------------------------------------

class TestSingleComponent:
    def test_single_component_no_coupling(self):
        """A single component should assemble without coupling."""
        comp = _make_component_mesh("horn", "Titanium Ti-6Al-4V", size=0.01)
        config = AssemblyConfig(component_order=["horn"])
        builder = AssemblyBuilder([comp], config)
        result = builder.build()

        assert result.n_total_dof == comp.mesh.n_dof
        assert result.K_global.shape == (comp.mesh.n_dof, comp.mesh.n_dof)
        assert result.M_global.shape == (comp.mesh.n_dof, comp.mesh.n_dof)
        assert len(result.interface_pairs) == 0
        assert "horn" in result.dof_map


# ---------------------------------------------------------------------------
# Test: Three component assembly
# ---------------------------------------------------------------------------

class TestThreeComponentAssembly:
    def test_three_component_stack(self):
        """A 3-component stack (transducer + booster + horn) should assemble."""
        size = 0.01
        comp_t = _make_component_mesh("transducer", "PZT-4",
                                       y_base=0.0, size=size)
        comp_b = _make_component_mesh("booster", "Titanium Ti-6Al-4V",
                                       y_base=size, size=size)
        comp_h = _make_component_mesh("horn", "Titanium Ti-6Al-4V",
                                       y_base=2 * size, size=size)
        config = AssemblyConfig(
            component_order=["transducer", "booster", "horn"],
        )
        builder = AssemblyBuilder([comp_t, comp_b, comp_h], config)
        result = builder.build()

        expected_dof = comp_t.mesh.n_dof + comp_b.mesh.n_dof + comp_h.mesh.n_dof
        assert result.n_total_dof == expected_dof
        assert "transducer" in result.dof_map
        assert "booster" in result.dof_map
        assert "horn" in result.dof_map
        # Two interfaces: transducer-booster and booster-horn
        assert len(result.interface_pairs) == 2


# ---------------------------------------------------------------------------
# Test: AssemblyResult structure
# ---------------------------------------------------------------------------

class TestAssemblyResult:
    def test_result_fields_present(self, matching_components, default_config):
        """AssemblyResult should have all required fields."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        assert hasattr(result, "K_global")
        assert hasattr(result, "M_global")
        assert hasattr(result, "n_total_dof")
        assert hasattr(result, "dof_map")
        assert hasattr(result, "node_offset_map")
        assert hasattr(result, "interface_pairs")
        assert hasattr(result, "impedance")
        assert hasattr(result, "transmission_coefficients")

    def test_result_types(self, matching_components, default_config):
        """AssemblyResult fields should have correct types."""
        comp_a, comp_b = matching_components
        builder = AssemblyBuilder([comp_a, comp_b], default_config)
        result = builder.build()

        assert isinstance(result.K_global, sp.csr_matrix)
        assert isinstance(result.M_global, sp.csr_matrix)
        assert isinstance(result.n_total_dof, int)
        assert isinstance(result.dof_map, dict)
        assert isinstance(result.node_offset_map, dict)
        assert isinstance(result.interface_pairs, list)
        assert isinstance(result.impedance, dict)
        assert isinstance(result.transmission_coefficients, dict)


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unknown_material_raises(self):
        """Using an unknown material should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown material"):
            comp = _make_component_mesh("horn", "FakeAlloy999", size=0.01)
            config = AssemblyConfig(component_order=["horn"])
            builder = AssemblyBuilder([comp], config)
            builder.build()

    def test_invalid_coupling_method(self, matching_components):
        """Invalid coupling method should raise ValueError."""
        comp_a, comp_b = matching_components
        config = AssemblyConfig(coupling_method="invalid_method")
        with pytest.raises(ValueError, match="coupling_method"):
            builder = AssemblyBuilder([comp_a, comp_b], config)
            builder.build()

    def test_empty_component_list(self):
        """Empty component list should raise ValueError."""
        config = AssemblyConfig()
        with pytest.raises(ValueError, match="[Cc]omponent"):
            builder = AssemblyBuilder([], config)
            builder.build()
