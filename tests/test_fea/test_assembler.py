"""Tests for global matrix assembly."""
from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
    GlobalAssembler,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coarse_cylinder_mesh() -> FEAMesh:
    """A coarse TET10 cylinder mesh for fast tests (~few hundred nodes)."""
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
        mesh_size=8.0,
        order=2,
    )


@pytest.fixture(scope="module")
def coarse_assembled(coarse_cylinder_mesh: FEAMesh) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Pre-assembled K and M for the coarse cylinder mesh."""
    assembler = GlobalAssembler(coarse_cylinder_mesh, "Titanium Ti-6Al-4V")
    return assembler.assemble()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestGlobalAssembler:
    def test_matrix_dimensions(
        self, coarse_cylinder_mesh: FEAMesh, coarse_assembled: tuple
    ):
        """K and M should be (3*n_nodes, 3*n_nodes)."""
        K, M = coarse_assembled
        n_dof = coarse_cylinder_mesh.n_dof
        assert K.shape == (n_dof, n_dof)
        assert M.shape == (n_dof, n_dof)

    def test_matrices_are_sparse_csr(self, coarse_assembled: tuple):
        """Returned matrices should be CSR format."""
        K, M = coarse_assembled
        assert isinstance(K, sp.csr_matrix)
        assert isinstance(M, sp.csr_matrix)

    def test_stiffness_symmetric(self, coarse_assembled: tuple):
        """Global K should be symmetric."""
        K, _ = coarse_assembled
        diff = K - K.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-6 * sp.linalg.norm(K)

    def test_mass_symmetric(self, coarse_assembled: tuple):
        """Global M should be symmetric."""
        _, M = coarse_assembled
        diff = M - M.T
        assert diff.nnz == 0 or np.max(np.abs(diff.data)) < 1e-6 * sp.linalg.norm(M)

    def test_stiffness_positive_semidefinite(self, coarse_assembled: tuple):
        """Free-free: K should have 6 zero eigenvalues (rigid body modes)."""
        K, _ = coarse_assembled
        # Convert to dense for eigenvalue check (small mesh)
        K_dense = K.toarray()
        eigvals = np.linalg.eigvalsh(K_dense)
        # Sort eigenvalues
        eigvals = np.sort(eigvals)
        # First 6 should be ~zero (rigid body modes)
        max_eig = eigvals[-1]
        n_zero = np.sum(np.abs(eigvals) < 1e-6 * max_eig)
        assert n_zero == 6, (
            f"Expected 6 zero eigenvalues (rigid body modes), got {n_zero}. "
            f"Smallest 10 eigenvalues: {eigvals[:10]}"
        )
        # Rest should be positive
        assert np.all(eigvals[6:] > 0), "Non-rigid-body eigenvalues should be positive"

    def test_mass_positive_definite(self, coarse_assembled: tuple):
        """M should be positive definite (all eigenvalues > 0)."""
        _, M = coarse_assembled
        M_dense = M.toarray()
        eigvals = np.linalg.eigvalsh(M_dense)
        assert np.all(eigvals > 0), (
            f"Mass matrix should be positive definite. "
            f"Min eigenvalue: {eigvals.min()}"
        )

    def test_total_mass_correct(self, coarse_cylinder_mesh: FEAMesh, coarse_assembled: tuple):
        """Total mass from M should match rho * volume.

        Analytical volume of cylinder: pi * r^2 * L.
        Total mass via M: u^T * M * u where u is unit translation.
        """
        _, M = coarse_assembled
        n_nodes = coarse_cylinder_mesh.nodes.shape[0]

        # Unit translation in x-direction
        u_x = np.zeros(coarse_cylinder_mesh.n_dof)
        for i in range(n_nodes):
            u_x[3 * i] = 1.0

        total_mass_from_M = u_x @ (M @ u_x)

        # Analytical: rho * pi * r^2 * L
        rho = 4430.0  # Titanium density
        r = 0.0125    # 25mm diameter -> 12.5mm radius -> 0.0125m
        L = 0.080     # 80mm -> 0.08m
        analytical_volume = np.pi * r**2 * L
        analytical_mass = rho * analytical_volume

        # Mesh approximation of a cylinder has discretization error.
        # For a coarse mesh, allow up to 10% tolerance.
        rel_error = abs(total_mass_from_M - analytical_mass) / analytical_mass
        assert rel_error < 0.10, (
            f"Total mass mismatch: M gives {total_mass_from_M:.6f} kg, "
            f"analytical {analytical_mass:.6f} kg, "
            f"relative error {rel_error:.4f}"
        )

    def test_shared_node_assembly(
        self, coarse_cylinder_mesh: FEAMesh, coarse_assembled: tuple
    ):
        """Elements sharing nodes should have their contributions summed.

        Verify that diagonal entries of K are larger than a single element's
        diagonal entry, confirming assembly adds contributions from multiple
        elements.
        """
        K, _ = coarse_assembled

        # Find a node that is shared by multiple elements
        # (interior nodes in a mesh are shared by many elements)
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import (
            TET10Element,
        )
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
            get_material,
        )

        mat = get_material("Titanium Ti-6Al-4V")
        elem = TET10Element()
        D = TET10Element.isotropic_elasticity_matrix(mat["E_pa"], mat["nu"])

        mesh = coarse_cylinder_mesh
        # Take the first element's first node
        first_elem_nodes = mesh.elements[0]
        test_node = first_elem_nodes[0]
        test_dof = 3 * test_node

        # Compute single-element stiffness for this element
        coords = mesh.nodes[first_elem_nodes]
        Ke = elem.stiffness_matrix(coords, D)

        # The single element's diagonal for this node's x-DOF
        local_idx = 0  # first node in element, x-DOF
        single_elem_diag = Ke[local_idx, local_idx]

        # The global diagonal should be >= single element (usually much larger
        # for interior nodes shared by multiple elements)
        global_diag = K[test_dof, test_dof]
        assert global_diag >= single_elem_diag * (1 - 1e-10), (
            f"Global diagonal ({global_diag:.6e}) should be >= "
            f"single element diagonal ({single_elem_diag:.6e})"
        )

    def test_performance_benchmark(self):
        """A moderate mesh should assemble in a reasonable time."""
        mesher = GmshMesher()
        mesh = mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=3.0,
            order=2,
        )

        assembler = GlobalAssembler(mesh, "Titanium Ti-6Al-4V")

        start = time.perf_counter()
        K, M = assembler.assemble()
        elapsed = time.perf_counter() - start

        # Should complete in < 30 seconds (generous for CI)
        assert elapsed < 30.0, (
            f"Assembly took {elapsed:.2f}s for {mesh.nodes.shape[0]} nodes, "
            f"{mesh.elements.shape[0]} elements"
        )
        # Sanity check: matrices exist and are non-empty
        assert K.nnz > 0
        assert M.nnz > 0

    def test_diagonal_positive(self, coarse_assembled: tuple):
        """All diagonal entries of K and M should be non-negative / positive."""
        K, M = coarse_assembled
        K_diag = K.diagonal()
        M_diag = M.diagonal()

        # Stiffness diagonal should be non-negative
        assert np.all(K_diag >= -1e-10 * np.max(K_diag)), (
            f"Some K diagonal entries are negative: min={K_diag.min()}"
        )

        # Mass diagonal should be strictly positive
        assert np.all(M_diag > 0), (
            f"Some M diagonal entries are non-positive: min={M_diag.min()}"
        )

    def test_material_not_found_raises(self, coarse_cylinder_mesh: FEAMesh):
        """Unknown material should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown material"):
            GlobalAssembler(coarse_cylinder_mesh, "NonexistentMaterial123")

    def test_tet10_only_check(self):
        """Assembler should reject non-TET10 meshes."""
        # Create a fake TET4 mesh
        fake_mesh = FEAMesh(
            nodes=np.zeros((4, 3)),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="TET4",
            node_sets={},
            element_sets={},
            surface_tris=np.array([[0, 1, 2]]),
            mesh_stats={},
        )
        with pytest.raises(ValueError, match="TET10"):
            GlobalAssembler(fake_mesh, "Titanium Ti-6Al-4V")
