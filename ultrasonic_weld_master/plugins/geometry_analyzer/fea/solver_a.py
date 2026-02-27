"""Self-developed FEA solver using numpy/scipy.

Implements modal analysis via the shift-invert Lanczos method
(``scipy.sparse.linalg.eigsh``) on global stiffness and mass matrices
assembled from TET10 quadratic tetrahedral elements.

Algorithm overview
------------------
1. Look up material properties from the material database.
2. Build global K, M using ``GlobalAssembler``.
3. Apply boundary conditions:
   - **free-free**: no constraints; request extra eigenvalues to account
     for rigid body modes, then filter out modes below 100 Hz.
   - **clamped**: penalty method on constrained DOFs.
4. Solve the generalised eigenvalue problem K * phi = omega^2 * M * phi
   using shift-invert around the target frequency.
5. Filter rigid body modes (f < 100 Hz) for free-free analysis.
6. Sort modes by ascending frequency.
7. Mass-normalise eigenvectors: phi_i / sqrt(phi_i^T * M * phi_i).
8. Classify modes (basic: longitudinal / flexural / compound).
9. Compute effective mass ratios per mode.
10. Return a ``ModalResult`` dataclass.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
    GlobalAssembler,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    HarmonicConfig,
    ModalConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    HarmonicResult,
    ModalResult,
    StaticResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_interface import (
    SolverInterface,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * np.pi
_RIGID_BODY_CUTOFF_HZ = 100.0  # Modes below this are rigid body for free-free
_PENALTY_FACTOR = 1e20  # Penalty multiplier for clamped BC
_EXTRA_RIGID_BODY_MODES = 6  # 3 translations + 3 rotations


class SolverA(SolverInterface):
    """Self-developed FEA solver using numpy/scipy.

    Implements modal analysis using the shift-invert Lanczos method via
    ``scipy.sparse.linalg.eigsh``.  Harmonic and static analyses are
    placeholders for future phases.

    Examples
    --------
    >>> from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
    >>> solver = SolverA()
    >>> result = solver.modal_analysis(config)
    """

    # ------------------------------------------------------------------
    # SolverInterface: modal_analysis
    # ------------------------------------------------------------------
    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Run eigenvalue analysis.

        Steps
        -----
        1. Look up material properties from ``config.material_name``.
        2. Build global K, M using ``GlobalAssembler``.
        3. Apply boundary conditions (free-free: skip; clamped: penalty method).
        4. Solve eigenvalue problem using shift-invert ``eigsh``.
        5. Filter: discard modes where f < 100 Hz (rigid body modes for free-free).
        6. Sort modes by frequency ascending.
        7. Mass-normalise eigenvectors: phi_i / sqrt(phi_i^T * M * phi_i).
        8. Return ``ModalResult`` with frequencies, mode_shapes, mode_types.

        Parameters
        ----------
        config : ModalConfig
            Configuration for the modal analysis (mesh, material, n_modes, etc.).

        Returns
        -------
        ModalResult
            Result container with frequencies, mode shapes, types and metadata.

        Raises
        ------
        ValueError
            If material is unknown or boundary_conditions is unsupported.
        RuntimeError
            If the eigensolver fails to converge.
        """
        t_start = time.perf_counter()

        # 1. Validate material
        mat = get_material(config.material_name)
        if mat is None:
            raise ValueError(
                f"Unknown material {config.material_name!r}. "
                "Use material_properties.list_materials() for available names."
            )

        # 2. Assemble global stiffness and mass matrices
        logger.info(
            "Assembling global matrices for %d-node mesh with material %r",
            config.mesh.nodes.shape[0],
            config.material_name,
        )
        assembler = GlobalAssembler(config.mesh, config.material_name)
        K, M = assembler.assemble()
        n_dof = config.mesh.n_dof

        logger.info(
            "Assembly complete: %d DOFs, K nnz=%d, M nnz=%d",
            n_dof,
            K.nnz,
            M.nnz,
        )

        # 3. Apply boundary conditions
        bc_type = config.boundary_conditions.lower().strip()
        if bc_type == "free-free":
            K_bc, M_bc = K, M
            n_extra = _EXTRA_RIGID_BODY_MODES
        elif bc_type == "clamped":
            K_bc, M_bc = self._apply_clamped_bc(
                K, M, config.mesh, config.fixed_node_sets
            )
            n_extra = 0
        else:
            raise ValueError(
                f"Unsupported boundary_conditions: {config.boundary_conditions!r}. "
                "Must be 'free-free' or 'clamped'."
            )

        # 4. Solve eigenvalue problem
        n_request = config.n_modes + n_extra
        # Ensure we don't request more eigenvalues than DOFs - 2
        # (eigsh limitation: k must be < n)
        max_k = n_dof - 2
        if n_request > max_k:
            logger.warning(
                "Requested %d eigenvalues but system has only %d DOFs. "
                "Reducing to %d.",
                n_request,
                n_dof,
                max_k,
            )
            n_request = max_k

        sigma = (_TWO_PI * config.target_frequency_hz) ** 2

        logger.info(
            "Solving eigenvalue problem: n_request=%d, sigma=%.6e (f_target=%.1f Hz)",
            n_request,
            sigma,
            config.target_frequency_hz,
        )

        try:
            eigenvalues, eigenvectors = spla.eigsh(
                K_bc,
                k=n_request,
                M=M_bc,
                sigma=sigma,
                which="LM",
            )
        except spla.ArpackNoConvergence as exc:
            # Partial results may be available
            n_converged = len(exc.eigenvalues)
            if n_converged > 0:
                logger.warning(
                    "ARPACK did not fully converge. "
                    "Got %d of %d eigenvalues. Using partial results.",
                    n_converged,
                    n_request,
                )
                eigenvalues = exc.eigenvalues
                eigenvectors = exc.eigenvectors
            else:
                raise RuntimeError(
                    f"Eigenvalue solver failed to converge. "
                    f"Requested {n_request} modes with sigma={sigma:.6e}."
                ) from exc

        # eigenvalues are omega^2; convert to frequencies in Hz
        # Handle potential negative eigenvalues from numerical noise
        omega_sq = np.real(eigenvalues)
        omega = np.sqrt(np.abs(omega_sq))
        frequencies_hz = omega / _TWO_PI

        # Sign correction: eigenvalues near zero can be slightly negative
        # We use abs() above to handle this gracefully.

        # 5. Filter rigid body modes for free-free
        if bc_type == "free-free":
            keep_mask = frequencies_hz >= _RIGID_BODY_CUTOFF_HZ
            frequencies_hz = frequencies_hz[keep_mask]
            eigenvectors = eigenvectors[:, keep_mask]

        # 6. Sort by ascending frequency
        sort_idx = np.argsort(frequencies_hz)
        frequencies_hz = frequencies_hz[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Trim to requested number of modes
        n_actual = min(config.n_modes, len(frequencies_hz))
        frequencies_hz = frequencies_hz[:n_actual]
        eigenvectors = eigenvectors[:, :n_actual]

        # 7. Mass-normalise eigenvectors
        # Use the ORIGINAL M (not M_bc with penalties) for normalisation
        # so that the physical mass-normalisation is correct.
        eigenvectors = self._mass_normalize(eigenvectors, M)

        # 8. Classify modes and compute effective mass ratios
        mode_types = []
        effective_mass_ratios = np.zeros((n_actual, 3), dtype=np.float64)

        # Build unit direction vectors (expanded to full DOF space)
        e_x = np.zeros(n_dof, dtype=np.float64)
        e_x[0::3] = 1.0
        e_y = np.zeros(n_dof, dtype=np.float64)
        e_y[1::3] = 1.0
        e_z = np.zeros(n_dof, dtype=np.float64)
        e_z[2::3] = 1.0

        # Pre-compute M @ e_dir for efficiency
        M_ex = M @ e_x
        M_ey = M @ e_y
        M_ez = M @ e_z

        for i in range(n_actual):
            phi = eigenvectors[:, i]

            # Mode classification (basic)
            mode_types.append(self._classify_mode(phi))

            # Effective mass ratios
            gamma_x = phi @ M_ex
            gamma_y = phi @ M_ey
            gamma_z = phi @ M_ez
            effective_mass_ratios[i, 0] = gamma_x ** 2
            effective_mass_ratios[i, 1] = gamma_y ** 2
            effective_mass_ratios[i, 2] = gamma_z ** 2

        t_elapsed = time.perf_counter() - t_start

        logger.info(
            "Modal analysis complete: %d modes found in %.2f s. "
            "Frequency range: %.1f - %.1f Hz",
            n_actual,
            t_elapsed,
            frequencies_hz[0] if n_actual > 0 else 0.0,
            frequencies_hz[-1] if n_actual > 0 else 0.0,
        )

        return ModalResult(
            frequencies_hz=frequencies_hz,
            mode_shapes=eigenvectors.T,  # (n_modes, n_dof)
            mode_types=mode_types,
            effective_mass_ratios=effective_mass_ratios,
            mesh=config.mesh,
            solve_time_s=t_elapsed,
            solver_name="SolverA",
        )

    # ------------------------------------------------------------------
    # SolverInterface: harmonic_analysis (Phase 3 placeholder)
    # ------------------------------------------------------------------
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Placeholder for Phase 3.

        Raises
        ------
        NotImplementedError
            Always. Harmonic analysis will be implemented in Phase 3.
        """
        raise NotImplementedError(
            "Harmonic analysis will be implemented in Phase 3"
        )

    # ------------------------------------------------------------------
    # SolverInterface: static_analysis (Phase 4 placeholder)
    # ------------------------------------------------------------------
    def static_analysis(self, config: StaticConfig) -> StaticResult:
        """Placeholder for Phase 4.

        Raises
        ------
        NotImplementedError
            Always. Static analysis will be implemented in Phase 4.
        """
        raise NotImplementedError(
            "Static analysis will be implemented in Phase 4"
        )

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_clamped_bc(
        K: sp.csr_matrix,
        M: sp.csr_matrix,
        mesh: object,
        fixed_node_sets: list[str],
    ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Apply clamped boundary conditions using the penalty method.

        For each DOF in the constrained node sets:
        - K[dof, dof] += penalty * max(K.diagonal())
        - M[dof, dof] = 0  (effectively removes inertia for constrained DOFs)

        Parameters
        ----------
        K : sp.csr_matrix
            Global stiffness matrix (will be copied).
        M : sp.csr_matrix
            Global mass matrix (will be copied).
        mesh :
            FEAMesh with node_sets attribute.
        fixed_node_sets : list[str]
            Names of node sets to constrain.

        Returns
        -------
        K_bc : sp.csr_matrix
            Stiffness matrix with penalty applied.
        M_bc : sp.csr_matrix
            Mass matrix with constrained DOFs zeroed.
        """
        K_bc = K.copy().tolil()
        M_bc = M.copy().tolil()

        K_diag_max = K.diagonal().max()
        penalty = _PENALTY_FACTOR * K_diag_max

        constrained_dofs = set()
        for set_name in fixed_node_sets:
            if set_name not in mesh.node_sets:
                logger.warning(
                    "Node set %r not found in mesh. "
                    "Available sets: %s. Skipping.",
                    set_name,
                    list(mesh.node_sets.keys()),
                )
                continue

            node_indices = mesh.node_sets[set_name]
            for node_idx in node_indices:
                for dof_offset in range(3):
                    constrained_dofs.add(3 * int(node_idx) + dof_offset)

        for dof in constrained_dofs:
            K_bc[dof, dof] += penalty
            M_bc[dof, dof] = 0.0

        logger.info(
            "Applied clamped BC: %d constrained DOFs from sets %s",
            len(constrained_dofs),
            fixed_node_sets,
        )

        return K_bc.tocsr(), M_bc.tocsr()

    # ------------------------------------------------------------------
    # Mass normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _mass_normalize(
        eigenvectors: NDArray[np.float64],
        M: sp.csr_matrix,
    ) -> NDArray[np.float64]:
        """Mass-normalise eigenvectors so that phi_i^T * M * phi_i = 1.

        Parameters
        ----------
        eigenvectors : NDArray[np.float64]
            (n_dof, n_modes) matrix of column eigenvectors.
        M : sp.csr_matrix
            Global mass matrix.

        Returns
        -------
        NDArray[np.float64]
            Mass-normalised eigenvectors, same shape.
        """
        n_modes = eigenvectors.shape[1]
        result = eigenvectors.copy()

        for i in range(n_modes):
            phi = result[:, i]
            m_gen = phi @ (M @ phi)

            if m_gen <= 0.0:
                # If generalised mass is not positive, just use L2 norm
                logger.warning(
                    "Mode %d has non-positive generalised mass (%.6e). "
                    "Falling back to L2 normalisation.",
                    i,
                    m_gen,
                )
                norm = np.linalg.norm(phi)
                if norm > 0.0:
                    result[:, i] = phi / norm
            else:
                result[:, i] = phi / np.sqrt(m_gen)

        return result

    # ------------------------------------------------------------------
    # Mode classification (basic)
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_mode(phi: NDArray[np.float64]) -> str:
        """Classify a mode shape based on displacement component ratios.

        The horn's longitudinal axis is Y (GmshMesher creates cylinders
        along Y).

        Classification rules (basic):
        - R_y > 0.70 -> "longitudinal"
        - max(R_x, R_z) > 0.60 -> "flexural"
        - else -> "compound"

        Parameters
        ----------
        phi : NDArray[np.float64]
            (n_dof,) eigenvector.

        Returns
        -------
        str
            One of "longitudinal", "flexural", "compound".
        """
        u_x = phi[0::3]
        u_y = phi[1::3]
        u_z = phi[2::3]

        sum_x2 = np.sum(u_x ** 2)
        sum_y2 = np.sum(u_y ** 2)
        sum_z2 = np.sum(u_z ** 2)
        total = sum_x2 + sum_y2 + sum_z2

        if total == 0.0:
            return "compound"

        R_y = sum_y2 / total

        if R_y > 0.70:
            return "longitudinal"

        if max(sum_x2, sum_z2) / total > 0.60:
            return "flexural"

        return "compound"

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return "SolverA(backend='scipy.sparse.linalg.eigsh')"
