"""Multi-body assembly builder for ultrasonic welding stack coupling.

Couples multiple ultrasonic component meshes (transducer, booster, horn)
into a single system by merging their stiffness and mass matrices at
interface nodes.

Coupling methods
----------------
- **Bonded** (shared DOF via penalty): Adds very stiff penalty springs
  between matching interface node pairs to enforce displacement continuity
  u_i = u_j.  Penalty stiffness = ``penalty_factor * max(diag(K))``.
- **Penalty** (MPC penalty): Same spring-based coupling with a
  configurable (potentially lower) penalty stiffness to model joint
  compliance.

Interface matching
------------------
Components are stacked along the Y-axis (longitudinal direction).  For
two adjacent components:
- comp_A's ``top_face`` (max Y) connects to comp_B's ``bottom_face``
  (min Y).
- Node pairs are found by closest-point matching in the X-Z plane.

Acoustic impedance
------------------
For each component: ``Z = rho * c * A`` where:
- ``c = sqrt(E / rho)`` is the longitudinal bar wave speed
- ``A`` is the interface cross-section area estimated from the bounding
  circle of interface nodes

Transmission coefficient between adjacent components:
``T = 4 * Z_A * Z_B / (Z_A + Z_B)^2``
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
    GlobalAssembler,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComponentMesh:
    """A single component in the ultrasonic welding stack.

    Parameters
    ----------
    name : str
        Component identifier, e.g. ``"horn"``, ``"booster"``,
        ``"transducer"``.
    mesh : FEAMesh
        The finite element mesh for this component.
    material_name : str
        Material name for property lookup (must be recognised by
        ``get_material``).
    interface_nodes : dict[str, np.ndarray]
        Mapping from interface label to local node indices.
        Typically ``{"top": array, "bottom": array}`` where ``"top"``
        is the max-Y face and ``"bottom"`` is the min-Y face.
    """

    name: str
    mesh: FEAMesh
    material_name: str
    interface_nodes: dict[str, np.ndarray]


@dataclass
class AssemblyConfig:
    """Configuration for multi-body assembly coupling.

    Parameters
    ----------
    coupling_method : str
        ``"bonded"`` for very stiff penalty springs (shared DOF),
        ``"penalty"`` for configurable penalty contact.
    penalty_factor : float
        Multiplier applied to ``max(diag(K_component))`` to compute the
        penalty stiffness.  A larger value enforces tighter coupling.
    component_order : list[str]
        Bottom-to-top stacking order of component names.
    """

    coupling_method: str = "bonded"
    penalty_factor: float = 1e3
    component_order: list[str] = field(
        default_factory=lambda: ["transducer", "booster", "horn"]
    )


@dataclass
class AssemblyResult:
    """Result of the assembly coupling procedure.

    Parameters
    ----------
    K_global : scipy.sparse.csr_matrix
        Coupled global stiffness matrix.
    M_global : scipy.sparse.csr_matrix
        Coupled global mass matrix.
    n_total_dof : int
        Total degrees of freedom across all components.
    dof_map : dict[str, np.ndarray]
        Mapping from component name to its global DOF indices.
    node_offset_map : dict[str, int]
        Mapping from component name to its node index offset in the
        global system.
    interface_pairs : list[dict]
        One entry per adjacent component pair.  Each dict has keys
        ``"comp_a"``, ``"comp_b"``, ``"node_pairs"`` where
        ``node_pairs`` is an ``(N, 2)`` array of global node indices.
    impedance : dict[str, float]
        Acoustic impedance ``Z`` for each component [Pa*s/m].
    transmission_coefficients : dict[str, float]
        Transmission coefficient ``T`` for each adjacent pair.  Key
        format is ``"comp_a-comp_b"``.
    """

    K_global: sp.csr_matrix
    M_global: sp.csr_matrix
    n_total_dof: int
    dof_map: dict[str, np.ndarray]
    node_offset_map: dict[str, int]
    interface_pairs: list[dict]
    impedance: dict[str, float]
    transmission_coefficients: dict[str, float]


# ---------------------------------------------------------------------------
# Assembly builder
# ---------------------------------------------------------------------------


class AssemblyBuilder:
    """Couple multiple ultrasonic component meshes into a single system.

    The builder takes a list of ``ComponentMesh`` objects and an
    ``AssemblyConfig``, assembles K and M for each component
    independently, places them into an expanded global system, and
    couples adjacent interfaces via penalty springs.

    Parameters
    ----------
    components : list[ComponentMesh]
        Components to assemble.  They will be ordered according to
        ``config.component_order``.
    config : AssemblyConfig
        Assembly configuration.

    Raises
    ------
    ValueError
        If the component list is empty or contains duplicate names.
    """

    _VALID_COUPLING_METHODS = {"bonded", "penalty"}

    def __init__(
        self,
        components: list[ComponentMesh],
        config: AssemblyConfig,
    ) -> None:
        self._config = config
        self._components_by_name: dict[str, ComponentMesh] = {}

        for comp in components:
            if comp.name in self._components_by_name:
                raise ValueError(
                    f"Duplicate component name {comp.name!r}. "
                    "Each component must have a unique name."
                )
            self._components_by_name[comp.name] = comp

        # Order components according to config (filter to those present)
        self._ordered: list[ComponentMesh] = []
        for name in config.component_order:
            if name in self._components_by_name:
                self._ordered.append(self._components_by_name[name])

        # Append any components not in the explicit order
        ordered_names = {c.name for c in self._ordered}
        for comp in components:
            if comp.name not in ordered_names:
                self._ordered.append(comp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> AssemblyResult:
        """Build coupled global K and M from individual component matrices.

        Steps
        -----
        1. Validate inputs.
        2. Assemble K, M for each component independently.
        3. Compute DOF offsets (component stacking).
        4. Place each component's K, M into the expanded global system.
        5. Couple interfaces (bonded or penalty).
        6. Compute acoustic impedance and transmission coefficients.

        Returns
        -------
        AssemblyResult
            The coupled system matrices and metadata.

        Raises
        ------
        ValueError
            If no components are provided or the coupling method is
            invalid.
        """
        self._validate()

        n_components = len(self._ordered)

        # --- Step 1: assemble each component ---
        comp_matrices: dict[str, tuple[sp.csr_matrix, sp.csr_matrix]] = {}
        for comp in self._ordered:
            K_c, M_c = self._assemble_component(comp)
            comp_matrices[comp.name] = (K_c, M_c)

        # --- Step 2: compute DOF offsets ---
        node_offset_map: dict[str, int] = {}
        dof_offset_map: dict[str, int] = {}
        dof_map: dict[str, np.ndarray] = {}

        running_node_offset = 0
        running_dof_offset = 0
        n_total_dof = 0

        for comp in self._ordered:
            n_nodes = comp.mesh.nodes.shape[0]
            n_dof = comp.mesh.n_dof

            node_offset_map[comp.name] = running_node_offset
            dof_offset_map[comp.name] = running_dof_offset
            dof_map[comp.name] = np.arange(
                running_dof_offset, running_dof_offset + n_dof, dtype=np.int64
            )

            running_node_offset += n_nodes
            running_dof_offset += n_dof
            n_total_dof += n_dof

        # --- Step 3: expand into global system (LIL for efficient insertion) ---
        K_global = sp.lil_matrix((n_total_dof, n_total_dof), dtype=np.float64)
        M_global = sp.lil_matrix((n_total_dof, n_total_dof), dtype=np.float64)

        for comp in self._ordered:
            K_c, M_c = comp_matrices[comp.name]
            offset = dof_offset_map[comp.name]
            n_dof_c = comp.mesh.n_dof
            end = offset + n_dof_c

            # Place component matrices into the global system
            K_c_coo = K_c.tocoo()
            for r, c, v in zip(K_c_coo.row, K_c_coo.col, K_c_coo.data):
                K_global[offset + r, offset + c] += v

            M_c_coo = M_c.tocoo()
            for r, c, v in zip(M_c_coo.row, M_c_coo.col, M_c_coo.data):
                M_global[offset + r, offset + c] += v

        # --- Step 4: couple interfaces ---
        interface_pairs: list[dict] = []

        # Compute a representative penalty stiffness from the max diagonal
        max_diag_K = 0.0
        for comp in self._ordered:
            K_c, _ = comp_matrices[comp.name]
            diag_c = K_c.diagonal()
            if len(diag_c) > 0:
                max_diag_K = max(max_diag_K, np.max(np.abs(diag_c)))

        if max_diag_K == 0.0:
            max_diag_K = 1.0  # fallback for degenerate cases

        penalty = self._config.penalty_factor * max_diag_K

        for idx in range(n_components - 1):
            comp_a = self._ordered[idx]
            comp_b = self._ordered[idx + 1]

            offset_a = node_offset_map[comp_a.name]
            offset_b = node_offset_map[comp_b.name]

            pairs = self._find_interface_pairs(
                comp_a, comp_b, offset_a, offset_b
            )

            interface_pairs.append({
                "comp_a": comp_a.name,
                "comp_b": comp_b.name,
                "node_pairs": pairs,
            })

            if len(pairs) > 0:
                if self._config.coupling_method == "bonded":
                    self._apply_bonded_coupling(
                        K_global, M_global, pairs, penalty
                    )
                else:  # penalty
                    self._apply_penalty_coupling(
                        K_global, M_global, pairs, penalty
                    )

        # Convert to CSR
        K_csr = K_global.tocsr()
        M_csr = M_global.tocsr()

        # Symmetrize for numerical cleanup
        K_csr = (K_csr + K_csr.T) / 2.0
        M_csr = (M_csr + M_csr.T) / 2.0

        K_csr.eliminate_zeros()
        M_csr.eliminate_zeros()

        # --- Step 5: compute acoustic impedance & transmission ---
        impedance: dict[str, float] = {}
        for comp in self._ordered:
            impedance[comp.name] = self._compute_impedance(comp)

        transmission_coefficients: dict[str, float] = {}
        for idx in range(n_components - 1):
            comp_a = self._ordered[idx]
            comp_b = self._ordered[idx + 1]
            Z_A = impedance[comp_a.name]
            Z_B = impedance[comp_b.name]
            key = f"{comp_a.name}-{comp_b.name}"
            transmission_coefficients[key] = self._transmission_coefficient(
                Z_A, Z_B
            )

        logger.info(
            "Assembly complete: %d components, %d total DOFs, "
            "%d interface pairs, K nnz=%d",
            n_components,
            n_total_dof,
            sum(len(p["node_pairs"]) for p in interface_pairs),
            K_csr.nnz,
        )

        return AssemblyResult(
            K_global=K_csr,
            M_global=M_csr,
            n_total_dof=n_total_dof,
            dof_map=dof_map,
            node_offset_map=node_offset_map,
            interface_pairs=interface_pairs,
            impedance=impedance,
            transmission_coefficients=transmission_coefficients,
        )

    # ------------------------------------------------------------------
    # Private: validate
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        """Validate builder state before assembly.

        Raises
        ------
        ValueError
            If no components, or invalid coupling method.
        """
        if len(self._ordered) == 0:
            raise ValueError(
                "Component list is empty.  At least one ComponentMesh "
                "is required for assembly."
            )

        if self._config.coupling_method not in self._VALID_COUPLING_METHODS:
            raise ValueError(
                f"Invalid coupling_method {self._config.coupling_method!r}. "
                f"Must be one of {sorted(self._VALID_COUPLING_METHODS)}."
            )

    # ------------------------------------------------------------------
    # Private: per-component assembly
    # ------------------------------------------------------------------

    def _assemble_component(
        self, comp: ComponentMesh
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Assemble K, M for a single component using ``GlobalAssembler``.

        Parameters
        ----------
        comp : ComponentMesh
            The component to assemble.

        Returns
        -------
        K : scipy.sparse.csr_matrix
            Component stiffness matrix.
        M : scipy.sparse.csr_matrix
            Component mass matrix.

        Raises
        ------
        ValueError
            If the material is not found.
        """
        mat = get_material(comp.material_name)
        if mat is None:
            raise ValueError(
                f"Unknown material {comp.material_name!r} for component "
                f"{comp.name!r}. Use material_properties.list_materials() "
                "for available names."
            )

        assembler = GlobalAssembler(comp.mesh, comp.material_name)
        K, M = assembler.assemble()

        logger.debug(
            "Component %r assembled: %d DOFs, K nnz=%d, M nnz=%d",
            comp.name,
            comp.mesh.n_dof,
            K.nnz,
            M.nnz,
        )

        return K, M

    # ------------------------------------------------------------------
    # Private: interface node matching
    # ------------------------------------------------------------------

    def _find_interface_pairs(
        self,
        comp_a: ComponentMesh,
        comp_b: ComponentMesh,
        offset_a: int,
        offset_b: int,
    ) -> np.ndarray:
        """Find matching node pairs at the interface between two components.

        For each node in comp_a's ``top`` interface, find the closest
        node in comp_b's ``bottom`` interface by X-Z distance.

        Parameters
        ----------
        comp_a : ComponentMesh
            The lower component (its ``top`` face is the interface).
        comp_b : ComponentMesh
            The upper component (its ``bottom`` face is the interface).
        offset_a : int
            Global node index offset for comp_a.
        offset_b : int
            Global node index offset for comp_b.

        Returns
        -------
        np.ndarray, shape (n_pairs, 2)
            Pairs of global node indices ``[global_node_a, global_node_b]``.
            Empty ``(0, 2)`` array if no matching pairs found.
        """
        # Get interface node local indices
        top_local = comp_a.interface_nodes.get("top", np.array([], dtype=int))
        bottom_local = comp_b.interface_nodes.get("bottom", np.array([], dtype=int))

        if len(top_local) == 0 or len(bottom_local) == 0:
            return np.empty((0, 2), dtype=np.int64)

        # Get coordinates
        coords_a = comp_a.mesh.nodes[top_local]     # (n_a, 3)
        coords_b = comp_b.mesh.nodes[bottom_local]   # (n_b, 3)

        # Compute tolerance based on characteristic mesh spacing
        # Use the median nearest-neighbour distance in the X-Z plane
        tol = self._interface_tolerance(coords_a, coords_b)

        pairs = []
        used_b = set()

        for i, local_a in enumerate(top_local):
            xa, za = coords_a[i, 0], coords_a[i, 2]

            best_dist = float("inf")
            best_j = -1

            for j, local_b in enumerate(bottom_local):
                if j in used_b:
                    continue
                xb, zb = coords_b[j, 0], coords_b[j, 2]
                dist = math.sqrt((xa - xb) ** 2 + (za - zb) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0 and best_dist < tol:
                global_a = local_a + offset_a
                global_b = bottom_local[best_j] + offset_b
                pairs.append([global_a, global_b])
                used_b.add(best_j)

        if len(pairs) == 0:
            return np.empty((0, 2), dtype=np.int64)

        result = np.array(pairs, dtype=np.int64)

        logger.debug(
            "Found %d interface pairs between %r (top) and %r (bottom), "
            "tolerance=%.6e m",
            len(result),
            comp_a.name,
            comp_b.name,
            tol,
        )

        return result

    @staticmethod
    def _interface_tolerance(
        coords_a: np.ndarray, coords_b: np.ndarray
    ) -> float:
        """Compute a matching tolerance from interface node coordinates.

        Uses the bounding box diagonal of the combined interface nodes
        in the X-Z plane, scaled down.  Falls back to a generous default
        for very small meshes.

        Parameters
        ----------
        coords_a : np.ndarray, shape (n_a, 3)
        coords_b : np.ndarray, shape (n_b, 3)

        Returns
        -------
        float
            Distance tolerance in meters.
        """
        all_xz = np.vstack([
            coords_a[:, [0, 2]],
            coords_b[:, [0, 2]],
        ])
        bbox_diag = np.linalg.norm(all_xz.max(axis=0) - all_xz.min(axis=0))

        if bbox_diag > 0:
            # Tolerance is half the bbox diagonal -- generous for matching
            return bbox_diag * 0.5
        else:
            # Degenerate case: all nodes at same X-Z location
            return 1e-6

    # ------------------------------------------------------------------
    # Private: coupling methods
    # ------------------------------------------------------------------

    def _apply_bonded_coupling(
        self,
        K: sp.lil_matrix,
        M: sp.lil_matrix,
        pairs: np.ndarray,
        penalty: float,
    ) -> None:
        """Enforce bonded coupling via very stiff penalty springs.

        For each node pair ``(i, j)`` and each DOF direction ``d``:

        .. code-block:: text

            K[3*i+d, 3*i+d] += penalty
            K[3*j+d, 3*j+d] += penalty
            K[3*i+d, 3*j+d] -= penalty
            K[3*j+d, 3*i+d] -= penalty

        A small mass coupling is also applied to M using
        ``penalty / 1e6`` to maintain matrix conditioning without
        significantly altering the mass distribution.

        Parameters
        ----------
        K : scipy.sparse.lil_matrix
            Global stiffness matrix (modified in place).
        M : scipy.sparse.lil_matrix
            Global mass matrix (modified in place).
        pairs : np.ndarray, shape (n_pairs, 2)
            Global node index pairs.
        penalty : float
            Penalty stiffness value.
        """
        mass_penalty = penalty / 1e6

        for pair in pairs:
            node_i, node_j = int(pair[0]), int(pair[1])
            for d in range(3):
                dof_i = 3 * node_i + d
                dof_j = 3 * node_j + d

                # Stiffness coupling
                K[dof_i, dof_i] += penalty
                K[dof_j, dof_j] += penalty
                K[dof_i, dof_j] -= penalty
                K[dof_j, dof_i] -= penalty

                # Mass coupling (small)
                M[dof_i, dof_i] += mass_penalty
                M[dof_j, dof_j] += mass_penalty
                M[dof_i, dof_j] -= mass_penalty
                M[dof_j, dof_i] -= mass_penalty

    def _apply_penalty_coupling(
        self,
        K: sp.lil_matrix,
        M: sp.lil_matrix,
        pairs: np.ndarray,
        penalty: float,
    ) -> None:
        """Enforce tied contact via penalty springs.

        Identical structure to bonded coupling but the penalty stiffness
        is based on the configured ``penalty_factor`` which may be lower,
        modelling joint compliance.

        Parameters
        ----------
        K : scipy.sparse.lil_matrix
            Global stiffness matrix (modified in place).
        M : scipy.sparse.lil_matrix
            Global mass matrix (modified in place).
        pairs : np.ndarray, shape (n_pairs, 2)
            Global node index pairs.
        penalty : float
            Penalty stiffness value.
        """
        # Same implementation; the distinction is in the penalty magnitude
        # which is computed in build() based on config.penalty_factor.
        self._apply_bonded_coupling(K, M, pairs, penalty)

    # ------------------------------------------------------------------
    # Private: acoustic impedance
    # ------------------------------------------------------------------

    def _compute_impedance(self, comp: ComponentMesh) -> float:
        """Compute acoustic impedance Z = rho * c * A for a component.

        Parameters
        ----------
        comp : ComponentMesh
            The component whose impedance to compute.

        Returns
        -------
        float
            Acoustic impedance [Pa*s/m].  Returns 0.0 if material is
            not found (should not happen after validation).
        """
        mat = get_material(comp.material_name)
        if mat is None:
            return 0.0

        rho = mat["rho_kg_m3"]
        E = mat["E_pa"]

        # Longitudinal bar wave speed
        c = math.sqrt(E / rho)

        # Estimate interface cross-section area from interface nodes.
        # Use the "top" interface nodes; fall back to "bottom" if top is empty.
        interface_key = "top" if len(comp.interface_nodes.get("top", [])) > 0 else "bottom"
        iface_nodes = comp.interface_nodes.get(interface_key, np.array([], dtype=int))

        if len(iface_nodes) == 0:
            # No interface nodes; estimate from all nodes projected onto X-Z
            coords_xz = comp.mesh.nodes[:, [0, 2]]
        else:
            coords_xz = comp.mesh.nodes[iface_nodes][:, [0, 2]]

        A = self._estimate_cross_section_area(coords_xz)

        Z = rho * c * A

        logger.debug(
            "Component %r: rho=%.0f, c=%.1f m/s, A=%.6e m^2, Z=%.3e",
            comp.name,
            rho,
            c,
            A,
            Z,
        )

        return Z

    @staticmethod
    def _estimate_cross_section_area(coords_xz: np.ndarray) -> float:
        """Estimate cross-section area from X-Z coordinates.

        Computes the area of the minimum bounding circle (using the max
        distance from the centroid) as ``pi * r^2``.

        Parameters
        ----------
        coords_xz : np.ndarray, shape (N, 2)
            X-Z coordinates of interface nodes.

        Returns
        -------
        float
            Estimated area in m^2.
        """
        if len(coords_xz) < 2:
            # Cannot estimate area from fewer than 2 points
            # Return a small default area
            return 1e-6

        centroid = coords_xz.mean(axis=0)
        distances = np.linalg.norm(coords_xz - centroid, axis=1)
        r_max = distances.max()

        if r_max < 1e-12:
            # All points at the same location; use convex hull fallback
            return 1e-6

        return math.pi * r_max ** 2

    # ------------------------------------------------------------------
    # Private: transmission coefficient
    # ------------------------------------------------------------------

    def _transmission_coefficient(self, Z_A: float, Z_B: float) -> float:
        """Compute power transmission coefficient between two media.

        Parameters
        ----------
        Z_A : float
            Acoustic impedance of medium A [Pa*s/m].
        Z_B : float
            Acoustic impedance of medium B [Pa*s/m].

        Returns
        -------
        float
            Transmission coefficient T in [0, 1].
            ``T = 4 * Z_A * Z_B / (Z_A + Z_B)^2``
        """
        if Z_A <= 0 or Z_B <= 0:
            return 0.0

        denominator = (Z_A + Z_B) ** 2
        if denominator == 0:
            return 0.0

        return 4.0 * Z_A * Z_B / denominator

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names = [c.name for c in self._ordered]
        return (
            f"AssemblyBuilder(components={names}, "
            f"coupling={self._config.coupling_method!r})"
        )
