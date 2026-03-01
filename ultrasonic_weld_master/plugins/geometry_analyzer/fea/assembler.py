"""Global sparse matrix assembly for FEA.

Assembles element-level stiffness (K) and consistent mass (M) matrices
into global sparse matrices in CSR format.  Uses COO (coordinate) format
for efficient incremental assembly, then converts to CSR for downstream
solvers.

Algorithm
---------
1. Pre-allocate COO arrays sized for ``n_elements * 30 * 30`` entries
   (TET10 has 30 DOFs per element, so 900 entries per element).
2. Loop over elements:
   a. Extract element node indices from ``mesh.elements``.
   b. Gather the 10 node coordinates from ``mesh.nodes``.
   c. Compute element K_e (30x30) and M_e (30x30) via ``TET10Element``.
   d. Build the DOF mapping: node ``idx`` maps to DOFs
      ``[3*idx, 3*idx+1, 3*idx+2]``.
   e. Scatter K_e and M_e into the COO arrays.
3. Build ``scipy.sparse.coo_matrix``, convert to ``csr_matrix``.
4. Symmetrize via ``(A + A.T) / 2`` for numerical cleanup.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)

logger = logging.getLogger(__name__)

# Number of nodes and DOFs per TET10 element
_NODES_PER_ELEM = 10
_DOFS_PER_ELEM = 30
_ENTRIES_PER_ELEM = _DOFS_PER_ELEM * _DOFS_PER_ELEM  # 900


class GlobalAssembler:
    """Assemble global stiffness and mass matrices from mesh + element + material.

    The assembler works exclusively with TET10 (10-node quadratic tetrahedral)
    meshes.  It uses the ``TET10Element`` class for element-level computations
    and the material property database for elastic constants and density.

    Parameters
    ----------
    mesh : FEAMesh
        The finite element mesh (from ``GmshMesher``).  Must have
        ``element_type == "TET10"``.
    material_name : str
        Material name for property lookup (e.g., ``"Titanium Ti-6Al-4V"``).
        Must be a recognized name or alias in the material database.

    Raises
    ------
    ValueError
        If the mesh is not TET10, or the material is not found.
    """

    def __init__(self, mesh: FEAMesh, material_name: str) -> None:
        if mesh.element_type != "TET10":
            raise ValueError(
                f"GlobalAssembler requires TET10 mesh, got {mesh.element_type!r}. "
                "Only 10-node quadratic tetrahedral elements are supported."
            )

        mat = get_material(material_name)
        if mat is None:
            raise ValueError(
                f"Unknown material {material_name!r}. "
                "Use material_properties.list_materials() for available names."
            )

        self._mesh = mesh
        self._material_name = material_name
        self._E: float = mat["E_pa"]
        self._nu: float = mat["nu"]
        self._rho: float = mat["rho_kg_m3"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assemble(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Assemble global K and M matrices.

        Returns
        -------
        K : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
            Global stiffness matrix.
        M : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
            Global consistent mass matrix.
        """
        t0 = time.perf_counter()

        mesh = self._mesh
        n_nodes = mesh.nodes.shape[0]
        n_dof = mesh.n_dof
        n_elements = mesh.elements.shape[0]

        # Build the constitutive matrix once
        D = TET10Element.isotropic_elasticity_matrix(self._E, self._nu)
        elem = TET10Element()

        # Pre-allocate COO arrays
        nnz_estimate = n_elements * _ENTRIES_PER_ELEM
        rows = np.empty(nnz_estimate, dtype=np.int64)
        cols = np.empty(nnz_estimate, dtype=np.int64)
        vals_K = np.empty(nnz_estimate, dtype=np.float64)
        vals_M = np.empty(nnz_estimate, dtype=np.float64)

        # Pre-compute the local DOF index arrays for scattering.
        # For a single element, local DOF i maps to global DOF dof_map[i].
        # dof_map is built per-element from the node indices.
        local_i = np.empty(_ENTRIES_PER_ELEM, dtype=np.int64)
        local_j = np.empty(_ENTRIES_PER_ELEM, dtype=np.int64)
        for a in range(_DOFS_PER_ELEM):
            for b in range(_DOFS_PER_ELEM):
                idx = a * _DOFS_PER_ELEM + b
                local_i[idx] = a
                local_j[idx] = b

        offset = 0
        n_skipped = 0

        for e in range(n_elements):
            # Element node indices (0-based)
            node_indices = mesh.elements[e]  # (10,)

            # Node coordinates for this element
            coords = mesh.nodes[node_indices]  # (10, 3)

            # Compute element matrices
            try:
                Ke = elem.stiffness_matrix(coords, D)  # (30, 30)
                Me = elem.mass_matrix(coords, self._rho)  # (30, 30)
            except ValueError:
                # Skip elements with non-positive Jacobian (degenerate elements)
                n_skipped += 1
                continue

            # Build global DOF mapping for this element
            # Node idx -> DOFs [3*idx, 3*idx+1, 3*idx+2]
            dof_map = np.empty(_DOFS_PER_ELEM, dtype=np.int64)
            for i in range(_NODES_PER_ELEM):
                base = 3 * node_indices[i]
                dof_map[3 * i] = base
                dof_map[3 * i + 1] = base + 1
                dof_map[3 * i + 2] = base + 2

            # Scatter into COO arrays
            start = offset
            end = offset + _ENTRIES_PER_ELEM

            rows[start:end] = dof_map[local_i]
            cols[start:end] = dof_map[local_j]
            vals_K[start:end] = Ke.ravel()
            vals_M[start:end] = Me.ravel()

            offset = end

        if n_skipped > 0:
            logger.warning(
                "Skipped %d degenerate elements out of %d total",
                n_skipped,
                n_elements,
            )

        # Trim arrays if any elements were skipped
        if offset < nnz_estimate:
            rows = rows[:offset]
            cols = cols[:offset]
            vals_K = vals_K[:offset]
            vals_M = vals_M[:offset]

        # Build COO matrices and convert to CSR
        K_coo = sp.coo_matrix((vals_K, (rows, cols)), shape=(n_dof, n_dof))
        M_coo = sp.coo_matrix((vals_M, (rows, cols)), shape=(n_dof, n_dof))

        K_csr = K_coo.tocsr()
        M_csr = M_coo.tocsr()

        # Symmetrize for numerical cleanup
        # (element matrices are symmetric, but floating-point scatter
        # can introduce tiny asymmetries)
        K_csr = (K_csr + K_csr.T) / 2.0
        M_csr = (M_csr + M_csr.T) / 2.0

        # Eliminate explicit zeros to save memory
        K_csr.eliminate_zeros()
        M_csr.eliminate_zeros()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Assembled global matrices: %d DOFs, %d elements, "
            "K nnz=%d, M nnz=%d, time=%.3fs",
            n_dof,
            n_elements - n_skipped,
            K_csr.nnz,
            M_csr.nnz,
            elapsed,
        )

        return K_csr, M_csr

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mesh(self) -> FEAMesh:
        """The finite element mesh."""
        return self._mesh

    @property
    def material_name(self) -> str:
        """The material name used for assembly."""
        return self._material_name

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom."""
        return self._mesh.n_dof

    def __repr__(self) -> str:
        return (
            f"GlobalAssembler(n_nodes={self._mesh.nodes.shape[0]}, "
            f"n_elements={self._mesh.elements.shape[0]}, "
            f"n_dof={self.n_dof}, "
            f"material={self._material_name!r})"
        )
