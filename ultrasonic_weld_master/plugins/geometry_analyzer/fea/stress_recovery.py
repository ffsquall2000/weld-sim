"""Stress recovery for FEA post-processing.

Provides Gauss-point stress recovery, Von Mises and principal stress
computation, Superconvergent Patch Recovery (SPR) nodal extrapolation,
and hotspot identification for ultrasonic welding FEA analysis.

Stress convention (Voigt notation)
-----------------------------------
[sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]

Von Mises formula
-----------------
sigma_vm = sqrt(0.5 * ((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2
                        + 6*(txy^2 + tyz^2 + txz^2)))

SPR approach
------------
For each node, collect the Gauss-point stresses from all elements
sharing that node.  Compute the physical coordinates of each Gauss
point, then fit the value at the node as the weighted average of
Gauss-point values, weighted by inverse distance from the Gauss
point to the node.  For uniform stress fields, this reduces to a
simple average and preserves the exact solution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StressField:
    """Container for full stress recovery results.

    Attributes
    ----------
    gauss_stresses : (n_elements, 4, 6)
        Voigt stress components at each Gauss point of each element.
    nodal_stresses : (n_nodes, 6)
        SPR-smoothed nodal stress field in Voigt notation.
    von_mises_gauss : (n_elements, 4)
        Von Mises stress at each Gauss point.
    von_mises_nodal : (n_nodes,)
        Von Mises stress at each node (from nodal stresses).
    principal_values : (n_nodes, 3)
        Principal stresses at each node, sorted descending.
    principal_directions : (n_nodes, 3, 3)
        Principal stress directions at each node.  Column i is the
        direction corresponding to principal_values[..., i].
    hotspots : list[dict]
        Top-N critical stress locations sorted by Von Mises descending.
        Each dict has keys: element_id, gauss_point, x, y, z, stress_mpa.
    """

    gauss_stresses: NDArray[np.float64]
    nodal_stresses: NDArray[np.float64]
    von_mises_gauss: NDArray[np.float64]
    von_mises_nodal: NDArray[np.float64]
    principal_values: NDArray[np.float64]
    principal_directions: NDArray[np.float64]
    hotspots: list[dict]


# ---------------------------------------------------------------------------
# StressRecovery class
# ---------------------------------------------------------------------------


class StressRecovery:
    """Stress recovery and post-processing for TET10 FEA results.

    Provides methods for:
    - Gauss-point stress recovery from displacement field
    - Vectorized Von Mises stress computation
    - Principal stress/direction computation via eigendecomposition
    - SPR-style nodal stress extrapolation with inverse-distance weighting
    - Critical hotspot identification

    Examples
    --------
    >>> sr = StressRecovery()
    >>> gauss = sr.recover_gauss_stresses(mesh, displacement, E, nu)
    >>> vm = sr.von_mises(gauss)
    >>> nodal = sr.extrapolate_to_nodes(mesh, gauss)
    """

    def __init__(self) -> None:
        self._element = TET10Element()

    # ------------------------------------------------------------------
    # 1. Gauss-point stress recovery
    # ------------------------------------------------------------------

    def recover_gauss_stresses(
        self,
        mesh: FEAMesh,
        displacement: NDArray[np.float64],
        E: float,
        nu: float,
    ) -> NDArray[np.float64]:
        """Compute stress at all 4 Gauss points for every element.

        sigma = D * B * u_e  at each Gauss point.

        Parameters
        ----------
        mesh : FEAMesh
            Mesh with nodes (N, 3) and elements (E, 10).
        displacement : (n_dof,)
            Global displacement vector.
        E : float
            Young's modulus [Pa].
        nu : float
            Poisson's ratio [-].

        Returns
        -------
        NDArray[np.float64]
            (n_elements, 4, 6) Voigt stress at each Gauss point.
        """
        D = TET10Element.isotropic_elasticity_matrix(E, nu)
        n_elements = mesh.elements.shape[0]
        gauss_stresses = np.zeros((n_elements, 4, 6), dtype=np.float64)

        gauss_pts = self._element.GAUSS_POINTS  # (4, 4)

        for e in range(n_elements):
            node_indices = mesh.elements[e]
            coords = mesh.nodes[node_indices]  # (10, 3)

            # Gather element displacement vector
            dof_map = np.empty(30, dtype=np.int64)
            for i in range(10):
                base = 3 * node_indices[i]
                dof_map[3 * i] = base
                dof_map[3 * i + 1] = base + 1
                dof_map[3 * i + 2] = base + 2
            u_e = displacement[dof_map]

            for gp_idx in range(4):
                xi, eta, zeta = gauss_pts[gp_idx, :3]
                try:
                    sigma = self._element.stress_at_point(
                        coords, u_e, D, xi, eta, zeta
                    )
                    gauss_stresses[e, gp_idx] = sigma
                except ValueError:
                    # Skip degenerate elements
                    logger.warning(
                        "Degenerate element %d at Gauss point %d, "
                        "setting stress to zero.",
                        e, gp_idx,
                    )
                    gauss_stresses[e, gp_idx] = 0.0

        return gauss_stresses

    # ------------------------------------------------------------------
    # 2. Von Mises stress
    # ------------------------------------------------------------------

    @staticmethod
    def von_mises(stress_voigt: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Von Mises equivalent stress from Voigt stress tensor.

        Fully vectorized: works on any (..., 6) shaped input.

        sigma_vm = sqrt(0.5 * ((sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2
                                + 6*(txy^2 + tyz^2 + txz^2)))

        Parameters
        ----------
        stress_voigt : (..., 6)
            Stress in Voigt notation:
            [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz].

        Returns
        -------
        NDArray[np.float64]
            (...) Von Mises stress (scalar per input point).
        """
        s = np.asarray(stress_voigt, dtype=np.float64)
        sxx = s[..., 0]
        syy = s[..., 1]
        szz = s[..., 2]
        txy = s[..., 3]
        tyz = s[..., 4]
        txz = s[..., 5]

        vm_sq = 0.5 * (
            (sxx - syy) ** 2
            + (syy - szz) ** 2
            + (szz - sxx) ** 2
            + 6.0 * (txy ** 2 + tyz ** 2 + txz ** 2)
        )
        return np.sqrt(np.maximum(vm_sq, 0.0))

    # ------------------------------------------------------------------
    # 3. Principal stresses
    # ------------------------------------------------------------------

    @staticmethod
    def principal_stresses(
        stress_voigt: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute principal stresses and directions via eigendecomposition.

        Constructs the symmetric 3x3 stress tensor from Voigt notation,
        then computes eigenvalues (principal stresses) and eigenvectors
        (principal directions).  Results are sorted in descending order
        of principal stress.

        Parameters
        ----------
        stress_voigt : (..., 6)
            Stress in Voigt notation:
            [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz].

        Returns
        -------
        principal_values : (..., 3)
            Principal stresses sorted descending (sigma_1 >= sigma_2 >= sigma_3).
        principal_directions : (..., 3, 3)
            Orthogonal eigenvector matrix.  Column i corresponds to
            principal_values[..., i].
        """
        s = np.asarray(stress_voigt, dtype=np.float64)
        original_shape = s.shape[:-1]

        # Flatten to (M, 6) for batch processing
        flat = s.reshape(-1, 6)
        n = flat.shape[0]

        vals_out = np.empty((n, 3), dtype=np.float64)
        dirs_out = np.empty((n, 3, 3), dtype=np.float64)

        for i in range(n):
            sxx, syy, szz, txy, tyz, txz = flat[i]
            tensor = np.array([
                [sxx, txy, txz],
                [txy, syy, tyz],
                [txz, tyz, szz],
            ], dtype=np.float64)

            eigenvalues, eigenvectors = np.linalg.eigh(tensor)

            # eigh returns ascending order; reverse for descending
            idx = np.argsort(eigenvalues)[::-1]
            vals_out[i] = eigenvalues[idx]
            dirs_out[i] = eigenvectors[:, idx]

        # Reshape to match input batch dimensions
        vals_out = vals_out.reshape(original_shape + (3,))
        dirs_out = dirs_out.reshape(original_shape + (3, 3))

        return vals_out, dirs_out

    # ------------------------------------------------------------------
    # 4. SPR nodal extrapolation
    # ------------------------------------------------------------------

    def extrapolate_to_nodes(
        self,
        mesh: FEAMesh,
        gauss_stresses: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Extrapolate Gauss-point stresses to nodes using SPR averaging.

        For each node, collects Gauss-point stresses from all elements
        sharing that node, computes the physical coordinates of each
        Gauss point, and averages using inverse-distance weighting.

        For uniform stress fields, the result is exact.

        Parameters
        ----------
        mesh : FEAMesh
            Mesh with nodes (N, 3) and elements (E, 10).
        gauss_stresses : (n_elements, 4, 6)
            Voigt stress at each Gauss point (from recover_gauss_stresses).

        Returns
        -------
        NDArray[np.float64]
            (n_nodes, 6) smoothed nodal stress field.
        """
        n_nodes = mesh.nodes.shape[0]
        n_elements = mesh.elements.shape[0]
        gauss_pts = self._element.GAUSS_POINTS  # (4, 4)

        # Accumulate weighted stress and total weight for each node
        nodal_stress_accum = np.zeros((n_nodes, 6), dtype=np.float64)
        nodal_weight_accum = np.zeros(n_nodes, dtype=np.float64)

        for e in range(n_elements):
            node_indices = mesh.elements[e]
            elem_coords = mesh.nodes[node_indices]  # (10, 3)

            # Compute physical coordinates of each Gauss point
            gp_phys_coords = np.empty((4, 3), dtype=np.float64)
            for gp_idx in range(4):
                xi, eta, zeta = gauss_pts[gp_idx, :3]
                N = TET10Element.shape_functions(xi, eta, zeta)  # (10,)
                gp_phys_coords[gp_idx] = N @ elem_coords  # (3,)

            # For each node in this element, accumulate inverse-distance
            # weighted Gauss-point stresses
            for local_node, global_node in enumerate(node_indices):
                node_coord = mesh.nodes[global_node]  # (3,)

                for gp_idx in range(4):
                    dist = np.linalg.norm(
                        gp_phys_coords[gp_idx] - node_coord
                    )
                    # Use inverse distance; avoid division by zero
                    # with a small epsilon
                    weight = 1.0 / (dist + 1e-30)
                    nodal_stress_accum[global_node] += (
                        weight * gauss_stresses[e, gp_idx]
                    )
                    nodal_weight_accum[global_node] += weight

        # Normalize by total weight
        nonzero_mask = nodal_weight_accum > 0.0
        nodal_stress_accum[nonzero_mask] /= nodal_weight_accum[
            nonzero_mask, np.newaxis
        ]

        return nodal_stress_accum

    # ------------------------------------------------------------------
    # 5. Hotspot identification
    # ------------------------------------------------------------------

    def find_hotspots(
        self,
        mesh: FEAMesh,
        gauss_vm: NDArray[np.float64],
        n_top: int = 10,
        yield_strength_mpa: Optional[float] = None,
    ) -> list[dict]:
        """Find the top-N critical stress locations (highest Von Mises).

        Parameters
        ----------
        mesh : FEAMesh
            Mesh with nodes and elements.
        gauss_vm : (n_elements, 4)
            Von Mises stress at each Gauss point.
        n_top : int
            Number of hotspots to return (default: 10).
        yield_strength_mpa : float or None
            Material yield strength in MPa.  If provided, safety_factor
            is computed as yield_strength / stress.

        Returns
        -------
        list[dict]
            List of hotspot dicts sorted by stress descending.  Each dict
            has keys: element_id, gauss_point, x, y, z, stress_mpa,
            safety_factor.
        """
        gauss_pts = self._element.GAUSS_POINTS  # (4, 4)

        # Flatten all Gauss-point VM stresses to find top-N
        flat_vm = gauss_vm.ravel()
        n_total = len(flat_vm)
        n_return = min(n_top, n_total)

        if n_return == 0:
            return []

        # Indices of the top-N largest values
        top_flat_indices = np.argpartition(flat_vm, -n_return)[-n_return:]
        # Sort them by value descending
        top_flat_indices = top_flat_indices[
            np.argsort(flat_vm[top_flat_indices])[::-1]
        ]

        hotspots = []
        for flat_idx in top_flat_indices:
            elem_id = int(flat_idx // 4)
            gp_id = int(flat_idx % 4)

            # Compute physical coordinate of this Gauss point
            node_indices = mesh.elements[elem_id]
            elem_coords = mesh.nodes[node_indices]  # (10, 3)
            xi, eta, zeta = gauss_pts[gp_id, :3]
            N = TET10Element.shape_functions(xi, eta, zeta)
            phys_coord = N @ elem_coords

            stress_mpa = float(flat_vm[flat_idx]) / 1e6

            # Compute safety factor if yield strength is provided
            if yield_strength_mpa is not None and stress_mpa > 0.0:
                safety_factor = yield_strength_mpa / stress_mpa
            else:
                safety_factor = float("inf")

            hotspots.append({
                "element_id": elem_id,
                "gauss_point": gp_id,
                "x": float(phys_coord[0]),
                "y": float(phys_coord[1]),
                "z": float(phys_coord[2]),
                "stress_mpa": stress_mpa,
                "safety_factor": safety_factor,
            })

        return hotspots


# ---------------------------------------------------------------------------
# Convenience: full pipeline
# ---------------------------------------------------------------------------


def full_stress_recovery(
    mesh: FEAMesh,
    displacement: NDArray[np.float64],
    E: float,
    nu: float,
    n_hotspots: int = 10,
    yield_strength_mpa: Optional[float] = None,
) -> StressField:
    """Run the full stress recovery pipeline.

    1. Recover Gauss-point stresses
    2. Compute Von Mises at Gauss points
    3. Extrapolate to nodes (SPR averaging)
    4. Compute Von Mises at nodes
    5. Compute principal stresses at nodes
    6. Identify hotspots

    Parameters
    ----------
    mesh : FEAMesh
        Mesh with nodes (N, 3) and elements (E, 10).
    displacement : (n_dof,)
        Global displacement vector from static or harmonic analysis.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].
    n_hotspots : int
        Number of critical hotspots to identify (default: 10).
    yield_strength_mpa : float or None
        Material yield strength in MPa for safety factor computation.

    Returns
    -------
    StressField
        Complete stress recovery results.
    """
    sr = StressRecovery()

    # 1. Gauss-point stresses
    gauss_stresses = sr.recover_gauss_stresses(mesh, displacement, E, nu)

    # 2. Von Mises at Gauss points
    von_mises_gauss = sr.von_mises(gauss_stresses)

    # 3. SPR nodal extrapolation
    nodal_stresses = sr.extrapolate_to_nodes(mesh, gauss_stresses)

    # 4. Von Mises at nodes
    von_mises_nodal = sr.von_mises(nodal_stresses)

    # 5. Principal stresses at nodes
    principal_values, principal_directions = sr.principal_stresses(
        nodal_stresses
    )

    # 6. Hotspots
    hotspots = sr.find_hotspots(
        mesh, von_mises_gauss,
        n_top=n_hotspots,
        yield_strength_mpa=yield_strength_mpa,
    )

    return StressField(
        gauss_stresses=gauss_stresses,
        nodal_stresses=nodal_stresses,
        von_mises_gauss=von_mises_gauss,
        von_mises_nodal=von_mises_nodal,
        principal_values=principal_values,
        principal_directions=principal_directions,
        hotspots=hotspots,
    )
