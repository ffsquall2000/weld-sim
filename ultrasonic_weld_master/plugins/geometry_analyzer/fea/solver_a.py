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
   - **pre-stressed**: static preload -> geometric stiffness K_sigma ->
     modified eigenvalue problem (K + K_sigma) phi = omega^2 M phi.
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
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import (
    TET10Element,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.damping import (
    HystereticDamping,
    ModalDamping,
    RayleighDamping,
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
        elif bc_type == "pre-stressed":
            # Run static analysis first to get stress state under preload
            static_config = StaticConfig(
                mesh=config.mesh,
                material_name=config.material_name,
                loads=config.pre_stress_loads,
                boundary_conditions=config.pre_stress_bcs,
            )
            static_result = self.static_analysis(static_config)

            # Compute geometric stiffness K_sigma from preload stress
            K_sigma = self._compute_geometric_stiffness(
                config.mesh, static_result.stress_tensor, mat
            )

            # Modified stiffness: K_eff = K + K_sigma
            K_eff = K + K_sigma

            # Apply clamped BCs to the modified system
            K_bc, M_bc = self._apply_clamped_bc(
                K_eff, M, config.mesh, config.fixed_node_sets
            )
            n_extra = 0  # Pre-stressed is constrained, no rigid body modes
        else:
            raise ValueError(
                f"Unsupported boundary_conditions: {config.boundary_conditions!r}. "
                "Must be 'free-free', 'clamped', or 'pre-stressed'."
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
    # SolverInterface: harmonic_analysis
    # ------------------------------------------------------------------
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Run harmonic response analysis.

        Computes the steady-state displacement response to harmonic
        force excitation over a sweep of frequencies.  The force is
        distributed uniformly over the excitation node set in the
        Y (longitudinal) direction.

        Supports direct and modal superposition solvers, and three
        damping formulations: hysteretic, Rayleigh, and modal.

        The gain is defined as the ratio of mean |U_y| at the response
        face to the mean |U_y| at the excitation face.

        Parameters
        ----------
        config : HarmonicConfig
            Configuration specifying mesh, material, frequency range,
            damping model, and excitation / response node sets.

        Returns
        -------
        HarmonicResult
            Result container with FRF data, gain, Q-factor, and
            contact face uniformity.

        Raises
        ------
        ValueError
            If material is unknown or damping_model is unsupported.
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
            "Harmonic analysis: assembling matrices for %d-node mesh, material %r",
            config.mesh.nodes.shape[0],
            config.material_name,
        )
        assembler = GlobalAssembler(config.mesh, config.material_name)
        K, M = assembler.assemble()
        n_dof = config.mesh.n_dof

        # 3. Build frequency sweep
        frequencies_hz = np.linspace(
            config.freq_min_hz, config.freq_max_hz, config.n_freq_points
        )
        omegas = _TWO_PI * frequencies_hz

        # 4. Identify excitation and response DOFs
        excit_node_indices = self._get_node_set(
            config.mesh, config.excitation_node_set
        )
        resp_node_indices = self._get_node_set(
            config.mesh, config.response_node_set
        )

        # Y-direction DOFs (index 1 within each 3-DOF node)
        excit_y_dofs = np.array([3 * int(n) + 1 for n in excit_node_indices])
        resp_y_dofs = np.array([3 * int(n) + 1 for n in resp_node_indices])

        # Build force vector: unit total force distributed over excitation
        # nodes in the Y (longitudinal) direction.
        F = np.zeros(n_dof, dtype=np.float64)
        F[excit_y_dofs] = 1.0 / len(excit_y_dofs)

        # 5. Create damping model
        damping_model_name = config.damping_model.lower().strip()

        # 6. Solve depending on damping model
        if damping_model_name == "modal":
            displacement_amplitudes = self._harmonic_modal_superposition(
                K, M, n_dof, omegas, excit_y_dofs, resp_y_dofs,
                F, config,
            )
        elif damping_model_name in ("hysteretic", "rayleigh"):
            damping = self._create_damping_model(
                damping_model_name, config.damping_ratio,
                config.freq_min_hz, config.freq_max_hz,
            )
            displacement_amplitudes = self._harmonic_direct_force(
                K, M, n_dof, omegas, damping, F,
            )
        else:
            raise ValueError(
                f"Unsupported damping_model: {config.damping_model!r}. "
                "Must be 'hysteretic', 'rayleigh', or 'modal'."
            )

        # 7. Post-process: extract response amplitudes and compute metrics
        # displacement_amplitudes: (n_freq, n_dof) complex
        # Compute Y-direction amplitude at response and excitation nodes
        resp_amps = np.abs(displacement_amplitudes[:, resp_y_dofs])  # (n_freq, n_resp)
        excit_amps = np.abs(displacement_amplitudes[:, excit_y_dofs])

        mean_resp_per_freq = np.mean(resp_amps, axis=1)  # (n_freq,)
        mean_excit_per_freq = np.mean(excit_amps, axis=1)

        # Gain = mean |U_y| at response / mean |U_y| at excitation
        with np.errstate(divide="ignore", invalid="ignore"):
            gain_per_freq = np.where(
                mean_excit_per_freq > 0.0,
                mean_resp_per_freq / mean_excit_per_freq,
                0.0,
            )

        # Find resonance (peak amplitude frequency)
        idx_res = int(np.argmax(mean_resp_per_freq))
        f_res = frequencies_hz[idx_res]
        peak_gain = float(gain_per_freq[idx_res])

        logger.info(
            "Harmonic analysis: resonance at %.1f Hz, gain=%.2f",
            f_res, peak_gain,
        )

        # Uniformity at resonance: min / mean of |U_y| at response nodes
        resp_amps_at_res = resp_amps[idx_res]  # (n_resp,)
        mean_resp_at_res = np.mean(resp_amps_at_res)
        if mean_resp_at_res > 0.0:
            uniformity = float(np.min(resp_amps_at_res) / mean_resp_at_res)
        else:
            uniformity = 0.0

        # Q-factor from 3dB bandwidth of the response FRF
        q_factor = self._compute_q_factor(frequencies_hz, mean_resp_per_freq)

        t_elapsed = time.perf_counter() - t_start

        logger.info(
            "Harmonic analysis complete in %.2f s: gain=%.2f, Q=%.1f, "
            "uniformity=%.3f",
            t_elapsed, peak_gain, q_factor, uniformity,
        )

        return HarmonicResult(
            frequencies_hz=frequencies_hz,
            displacement_amplitudes=displacement_amplitudes,
            contact_face_uniformity=uniformity,
            gain=peak_gain,
            q_factor=q_factor,
            mesh=config.mesh,
            solve_time_s=t_elapsed,
            solver_name="SolverA",
        )

    # ------------------------------------------------------------------
    # Harmonic analysis: direct frequency sweep (force excitation)
    # ------------------------------------------------------------------
    def _harmonic_direct_force(
        self,
        K: sp.csr_matrix,
        M: sp.csr_matrix,
        n_dof: int,
        omegas: NDArray[np.float64],
        damping: object,
        F: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Solve harmonic response using the direct method with force excitation.

        At each frequency, build the dynamic stiffness D(omega) and solve:

            D(omega) * U = F

        Parameters
        ----------
        K : sp.csr_matrix
            Global stiffness matrix.
        M : sp.csr_matrix
            Global mass matrix.
        n_dof : int
            Total number of DOFs.
        omegas : ndarray
            Circular frequencies [rad/s] for the sweep.
        damping : DampingModel
            Damping model with ``build_dynamic_stiffness(K, M, omega)``.
        F : ndarray, shape (n_dof,)
            Force vector (real, applied in physical coordinates).

        Returns
        -------
        ndarray, shape (n_freq, n_dof), complex
            Displacement amplitudes at each frequency.
        """
        n_freq = len(omegas)
        F_complex = np.asarray(F, dtype=np.complex128)
        result = np.zeros((n_freq, n_dof), dtype=np.complex128)

        for i, omega in enumerate(omegas):
            D = damping.build_dynamic_stiffness(K, M, omega)
            result[i] = spla.spsolve(D, F_complex)

        return result

    # ------------------------------------------------------------------
    # Harmonic analysis: modal superposition (force excitation)
    # ------------------------------------------------------------------
    def _harmonic_modal_superposition(
        self,
        K: sp.csr_matrix,
        M: sp.csr_matrix,
        n_dof: int,
        omegas: NDArray[np.float64],
        excit_y_dofs: NDArray[np.int64],
        resp_y_dofs: NDArray[np.int64],
        F: NDArray[np.float64],
        config: HarmonicConfig,
    ) -> NDArray[np.complex128]:
        """Solve harmonic response using modal superposition.

        Uses ModalDamping.modal_frf() with mode shapes from a preliminary
        modal analysis.

        Parameters
        ----------
        K, M : sp.csr_matrix
            Global stiffness and mass matrices.
        n_dof : int
            Total DOFs.
        omegas : ndarray
            Circular frequency sweep [rad/s].
        excit_y_dofs : ndarray
            Excitation DOF indices.
        resp_y_dofs : ndarray
            Response DOF indices.
        F : ndarray, shape (n_dof,)
            Force vector.
        config : HarmonicConfig
            Harmonic analysis configuration.

        Returns
        -------
        ndarray, shape (n_freq, n_dof), complex
        """
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
            ModalConfig,
        )

        # Run a preliminary modal analysis to get natural frequencies and modes
        f_center = (config.freq_min_hz + config.freq_max_hz) / 2.0
        modal_config = ModalConfig(
            mesh=config.mesh,
            material_name=config.material_name,
            n_modes=30,
            target_frequency_hz=f_center,
            boundary_conditions="free-free",
        )
        modal_result = self.modal_analysis(modal_config)

        # Mode shapes: (n_modes, n_dof) -> transpose to (n_dof, n_modes)
        phi_n = modal_result.mode_shapes.T  # (n_dof, n_modes)
        omega_n = _TWO_PI * modal_result.frequencies_hz  # (n_modes,)
        n_modes = len(omega_n)

        # Generalised masses (should be ~1.0 for mass-normalised modes)
        M_phi = np.ones(n_modes, dtype=np.float64)

        # Create modal damping model
        damping = ModalDamping(zeta=config.damping_ratio)

        n_freq = len(omegas)
        result = np.zeros((n_freq, n_dof), dtype=np.complex128)

        for i, omega in enumerate(omegas):
            U = damping.modal_frf(omega_n, phi_n, M_phi, F, omega)
            result[i] = U

        return result

    # ------------------------------------------------------------------
    # Damping model factory
    # ------------------------------------------------------------------
    @staticmethod
    def _create_damping_model(
        model_name: str,
        damping_ratio: float,
        freq_min_hz: float,
        freq_max_hz: float,
    ) -> object:
        """Create an appropriate DampingModel instance.

        Parameters
        ----------
        model_name : str
            One of 'hysteretic', 'rayleigh'.
        damping_ratio : float
            Damping ratio or loss factor.
        freq_min_hz, freq_max_hz : float
            Frequency range (used for Rayleigh fitting).

        Returns
        -------
        DampingModel instance.
        """
        if model_name == "hysteretic":
            # For hysteretic damping, eta ~ 2 * zeta for small damping
            eta = 2.0 * damping_ratio
            return HystereticDamping(eta=eta)

        if model_name == "rayleigh":
            f_center = (freq_min_hz + freq_max_hz) / 2.0
            f1 = 0.8 * f_center
            f2 = 1.2 * f_center
            zeta = damping_ratio
            return RayleighDamping.from_frequencies(f1, f2, zeta, zeta)

        raise ValueError(f"Unknown damping model: {model_name!r}")

    # ------------------------------------------------------------------
    # Q-factor computation from 3dB bandwidth
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_q_factor(
        frequencies_hz: NDArray[np.float64],
        gain: NDArray[np.float64],
    ) -> float:
        """Compute Q-factor from the 3dB bandwidth of a gain curve.

        Q = f_res / (f_hi - f_lo), where f_lo and f_hi are the
        frequencies at which gain = peak_gain / sqrt(2).

        Parameters
        ----------
        frequencies_hz : ndarray
            Frequency sweep array [Hz].
        gain : ndarray
            Gain (amplitude) at each frequency.

        Returns
        -------
        float
            Estimated Q-factor. Returns 0.0 if bandwidth cannot be
            determined (e.g., peak at sweep boundary).
        """
        idx_peak = int(np.argmax(gain))
        peak_gain = gain[idx_peak]
        f_res = frequencies_hz[idx_peak]

        threshold = peak_gain / np.sqrt(2.0)

        # Find f_lo: search left of peak for crossing below threshold
        f_lo = None
        for i in range(idx_peak - 1, -1, -1):
            if gain[i] <= threshold:
                # Linear interpolation between i and i+1
                g_low, g_high = gain[i], gain[i + 1]
                f_low, f_high = frequencies_hz[i], frequencies_hz[i + 1]
                if g_high - g_low != 0.0:
                    frac = (threshold - g_low) / (g_high - g_low)
                    f_lo = f_low + frac * (f_high - f_low)
                else:
                    f_lo = f_low
                break

        # Find f_hi: search right of peak for crossing below threshold
        f_hi = None
        for i in range(idx_peak + 1, len(gain)):
            if gain[i] <= threshold:
                # Linear interpolation between i-1 and i
                g_high, g_low = gain[i - 1], gain[i]
                f_high_prev, f_low_curr = frequencies_hz[i - 1], frequencies_hz[i]
                if g_high - g_low != 0.0:
                    frac = (g_high - threshold) / (g_high - g_low)
                    f_hi = f_high_prev + frac * (f_low_curr - f_high_prev)
                else:
                    f_hi = f_low_curr
                break

        if f_lo is not None and f_hi is not None:
            bandwidth = f_hi - f_lo
            if bandwidth > 0.0:
                return f_res / bandwidth

        # If we cannot determine the bandwidth, estimate from peak shape
        # using half-width at half-maximum approximation
        return 0.0

    # ------------------------------------------------------------------
    # Node set helper
    # ------------------------------------------------------------------
    @staticmethod
    def _get_node_set(mesh: object, set_name: str) -> NDArray[np.int64]:
        """Retrieve a node set from the mesh by name.

        Parameters
        ----------
        mesh : FEAMesh
            Mesh with ``node_sets`` attribute.
        set_name : str
            Name of the node set.

        Returns
        -------
        ndarray
            0-based node indices in the set.

        Raises
        ------
        ValueError
            If the node set is not found.
        """
        if set_name not in mesh.node_sets:
            raise ValueError(
                f"Node set {set_name!r} not found in mesh. "
                f"Available sets: {list(mesh.node_sets.keys())}"
            )
        return np.asarray(mesh.node_sets[set_name], dtype=np.int64)

    # ------------------------------------------------------------------
    # SolverInterface: static_analysis
    # ------------------------------------------------------------------
    def static_analysis(self, config: StaticConfig) -> StaticResult:
        """Run linear static stress analysis.

        Solves the linear static equilibrium equation K * u = f for
        nodal displacements, then recovers element stresses at centroids.

        Steps
        -----
        1. Look up material properties from ``config.material_name``.
        2. Assemble global K using ``GlobalAssembler``.
        3. Build load vector f based on ``config.loads``.
        4. Apply boundary conditions based on ``config.boundary_conditions``
           using the penalty method.
        5. Solve K_mod * u = f_mod via ``scipy.sparse.linalg.spsolve``.
        6. Recover element stresses at centroids using
           ``TET10Element.stress_at_point()``.
        7. Compute Von Mises stress at element centroids.
        8. Return ``StaticResult``.

        Parameters
        ----------
        config : StaticConfig
            Configuration for the static analysis (mesh, material, loads,
            boundary conditions).

        Returns
        -------
        StaticResult
            Result container with displacement, stress fields, and metadata.

        Raises
        ------
        ValueError
            If material is unknown or a node set is not found.
        """
        t_start = time.perf_counter()

        # 1. Validate material
        mat = get_material(config.material_name)
        if mat is None:
            raise ValueError(
                f"Unknown material {config.material_name!r}. "
                "Use material_properties.list_materials() for available names."
            )

        E = mat["E_pa"]
        nu = mat["nu"]
        rho = mat["rho_kg_m3"]

        # 2. Assemble global stiffness matrix
        logger.info(
            "Static analysis: assembling matrices for %d-node mesh, material %r",
            config.mesh.nodes.shape[0],
            config.material_name,
        )
        assembler = GlobalAssembler(config.mesh, config.material_name)
        K, _M = assembler.assemble()
        n_dof = config.mesh.n_dof

        # 3. Build load vector
        f = np.zeros(n_dof, dtype=np.float64)
        elem_calculator = TET10Element()

        for load in config.loads:
            load_type = load["type"].lower().strip()

            if load_type == "force":
                f = self._apply_force_load(f, config.mesh, load)
            elif load_type == "pressure":
                f = self._apply_pressure_load(f, config.mesh, load)
            elif load_type == "gravity":
                f = self._apply_gravity_load(
                    f, config.mesh, load, rho, elem_calculator
                )
            else:
                raise ValueError(
                    f"Unsupported load type: {load['type']!r}. "
                    "Must be 'force', 'pressure', or 'gravity'."
                )

        # 4. Apply boundary conditions (penalty method on K and f)
        K_mod = K.copy().tolil()
        f_mod = f.copy()

        K_diag_max = K.diagonal().max()
        penalty = _PENALTY_FACTOR * K_diag_max

        for bc in config.boundary_conditions:
            bc_type = bc["type"].lower().strip()

            if bc_type == "fixed":
                node_indices = self._get_node_set(config.mesh, bc["node_set"])
                for node_idx in node_indices:
                    for dof_offset in range(3):
                        dof = 3 * int(node_idx) + dof_offset
                        K_mod[dof, dof] += penalty
                        f_mod[dof] = 0.0

            elif bc_type == "symmetry":
                node_indices = self._get_node_set(config.mesh, bc["node_set"])
                normal = np.array(bc["normal"], dtype=np.float64)
                normal = normal / np.linalg.norm(normal)
                for node_idx in node_indices:
                    # Constrain DOF in normal direction using penalty
                    # The constrained DOF is n^T * u_node = 0
                    # For penalty method on a single direction:
                    # Add penalty * n_i * n_j to K[3*node+i, 3*node+j]
                    base = 3 * int(node_idx)
                    for i in range(3):
                        for j in range(3):
                            K_mod[base + i, base + j] += (
                                penalty * normal[i] * normal[j]
                            )

            else:
                raise ValueError(
                    f"Unsupported boundary condition type: {bc['type']!r}. "
                    "Must be 'fixed' or 'symmetry'."
                )

        K_mod = K_mod.tocsr()

        # 5. Solve K_mod * u = f_mod
        logger.info("Solving static system: %d DOFs", n_dof)
        u = spla.spsolve(K_mod, f_mod)

        # 6. Recover element stresses at centroids
        D = TET10Element.isotropic_elasticity_matrix(E, nu)
        n_elements = config.mesh.elements.shape[0]
        stress_tensor = np.zeros((n_elements, 6), dtype=np.float64)

        # Centroid of TET10 in natural coordinates
        xi_c, eta_c, zeta_c = 0.25, 0.25, 0.25

        for e in range(n_elements):
            node_indices = config.mesh.elements[e]
            coords = config.mesh.nodes[node_indices]

            # Gather element displacement vector
            dof_map = np.empty(30, dtype=np.int64)
            for i in range(10):
                base = 3 * node_indices[i]
                dof_map[3 * i] = base
                dof_map[3 * i + 1] = base + 1
                dof_map[3 * i + 2] = base + 2
            u_e = u[dof_map]

            # Stress at centroid
            try:
                sigma = elem_calculator.stress_at_point(
                    coords, u_e, D, xi_c, eta_c, zeta_c
                )
                stress_tensor[e] = sigma
            except ValueError:
                # Skip degenerate elements
                stress_tensor[e] = 0.0

        # 7. Compute Von Mises stress
        # Voigt: [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz]
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

        max_stress_pa = float(np.max(stress_vm)) if n_elements > 0 else 0.0
        max_stress_mpa = max_stress_pa / 1e6

        t_elapsed = time.perf_counter() - t_start

        logger.info(
            "Static analysis complete in %.2f s: max VM stress = %.2f MPa",
            t_elapsed,
            max_stress_mpa,
        )

        return StaticResult(
            displacement=u,
            stress_vm=stress_vm,
            stress_tensor=stress_tensor,
            max_stress_mpa=max_stress_mpa,
            mesh=config.mesh,
            solve_time_s=t_elapsed,
            solver_name="SolverA",
        )

    # ------------------------------------------------------------------
    # Static load helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_force_load(
        f: NDArray[np.float64],
        mesh: object,
        load: dict,
    ) -> NDArray[np.float64]:
        """Apply a concentrated force load distributed over a node set.

        Parameters
        ----------
        f : ndarray
            Global force vector (modified in-place).
        mesh : FEAMesh
            Mesh with node_sets.
        load : dict
            Load specification with keys: node_set, direction, magnitude.

        Returns
        -------
        ndarray
            Updated force vector.
        """
        node_set_name = load["node_set"]
        if node_set_name not in mesh.node_sets:
            raise ValueError(
                f"Node set {node_set_name!r} not found in mesh. "
                f"Available sets: {list(mesh.node_sets.keys())}"
            )
        node_indices = np.asarray(mesh.node_sets[node_set_name], dtype=np.int64)

        direction = np.array(load["direction"], dtype=np.float64)
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm

        magnitude = float(load["magnitude"])
        n_nodes = len(node_indices)
        if n_nodes == 0:
            return f

        force_per_node = magnitude / n_nodes

        for node_idx in node_indices:
            for d in range(3):
                f[3 * int(node_idx) + d] += force_per_node * direction[d]

        return f

    @staticmethod
    def _apply_pressure_load(
        f: NDArray[np.float64],
        mesh: object,
        load: dict,
    ) -> NDArray[np.float64]:
        """Apply a pressure load over a node set (simplified uniform).

        Distributes pressure in the Y (longitudinal) direction as a
        simplified uniform distribution: f_i = pressure * area / n_nodes.

        Parameters
        ----------
        f : ndarray
            Global force vector (modified in-place).
        mesh : FEAMesh
            Mesh with node_sets and nodes.
        load : dict
            Load specification with keys: node_set, magnitude.

        Returns
        -------
        ndarray
            Updated force vector.
        """
        node_set_name = load["node_set"]
        if node_set_name not in mesh.node_sets:
            raise ValueError(
                f"Node set {node_set_name!r} not found in mesh. "
                f"Available sets: {list(mesh.node_sets.keys())}"
            )
        node_indices = np.asarray(mesh.node_sets[node_set_name], dtype=np.int64)
        n_nodes = len(node_indices)
        if n_nodes == 0:
            return f

        pressure = float(load["magnitude"])

        # Estimate the face area from the node positions (convex hull area
        # approximation): use the bounding circle of the face nodes.
        face_coords = mesh.nodes[node_indices]
        # Compute area from the range of X and Z coordinates (cross-section)
        x_range = face_coords[:, 0].max() - face_coords[:, 0].min()
        z_range = face_coords[:, 2].max() - face_coords[:, 2].min()
        # For a circular face, approximate area as pi * (D/2)^2
        diameter = max(x_range, z_range)
        area = np.pi * (diameter / 2.0) ** 2

        force_per_node = pressure * area / n_nodes

        # Apply in Y direction (longitudinal)
        for node_idx in node_indices:
            f[3 * int(node_idx) + 1] += force_per_node

        return f

    @staticmethod
    def _apply_gravity_load(
        f: NDArray[np.float64],
        mesh: object,
        load: dict,
        rho: float,
        elem: TET10Element,
    ) -> NDArray[np.float64]:
        """Apply gravity body force to all elements.

        For each element: f_i = rho * V_element * g / n_nodes_per_elem
        distributed to each node's DOFs in the gravity direction.

        Parameters
        ----------
        f : ndarray
            Global force vector (modified in-place).
        mesh : FEAMesh
            Mesh with elements and nodes.
        load : dict
            Load specification with keys: direction, magnitude.
        rho : float
            Material density [kg/m^3].
        elem : TET10Element
            Element calculator for volume computation.

        Returns
        -------
        ndarray
            Updated force vector.
        """
        direction = np.array(load["direction"], dtype=np.float64)
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction = direction / dir_norm

        g_mag = float(load["magnitude"])
        n_elements = mesh.elements.shape[0]

        for e in range(n_elements):
            node_indices = mesh.elements[e]
            coords = mesh.nodes[node_indices]

            try:
                vol = elem.volume(coords)
            except ValueError:
                continue  # skip degenerate elements

            # Body force per node = rho * V * g / 10 (for TET10)
            force_per_node = rho * vol * g_mag / 10.0

            for node_idx in node_indices:
                for d in range(3):
                    f[3 * int(node_idx) + d] += (
                        force_per_node * direction[d]
                    )

        return f

    # ------------------------------------------------------------------
    # Geometric stiffness for pre-stressed modal analysis
    # ------------------------------------------------------------------
    def _compute_geometric_stiffness(
        self,
        mesh: object,
        stress_tensor: NDArray[np.float64],
        mat: dict,
    ) -> sp.csr_matrix:
        """Compute the geometric stiffness matrix from an initial stress state.

        For each element, at each Gauss point:
        1. Build the G matrix (9 x 30) from shape function derivatives
        2. Build the S matrix (9 x 9) from the initial stress tensor
        3. K_sigma_e += G^T * S * G * det(J) * w

        Then assemble into global K_sigma using COO format.

        Parameters
        ----------
        mesh : FEAMesh
            The finite element mesh.
        stress_tensor : NDArray[np.float64]
            (n_elements, 6) element stress tensors in Voigt notation
            [sxx, syy, szz, txy, tyz, txz] at element centroids.
        mat : dict
            Material properties dictionary (unused but kept for API consistency).

        Returns
        -------
        sp.csr_matrix
            Global geometric stiffness matrix (n_dof x n_dof).
        """
        elem = TET10Element()
        n_dof = mesh.n_dof
        n_elements = mesh.elements.shape[0]

        # Pre-allocate COO arrays
        entries_per_elem = 30 * 30  # TET10: 30 DOFs
        nnz_estimate = n_elements * entries_per_elem
        rows = np.empty(nnz_estimate, dtype=np.int64)
        cols = np.empty(nnz_estimate, dtype=np.int64)
        vals = np.empty(nnz_estimate, dtype=np.float64)

        # Pre-compute local index arrays for COO scatter
        local_i = np.empty(entries_per_elem, dtype=np.int64)
        local_j = np.empty(entries_per_elem, dtype=np.int64)
        for a in range(30):
            for b in range(30):
                idx = a * 30 + b
                local_i[idx] = a
                local_j[idx] = b

        offset = 0

        for e in range(n_elements):
            node_indices = mesh.elements[e]
            coords = mesh.nodes[node_indices]  # (10, 3)

            # Element stress in Voigt: [sxx, syy, szz, txy, tyz, txz]
            s = stress_tensor[e]
            sxx, syy, szz, txy, tyz, txz = s[0], s[1], s[2], s[3], s[4], s[5]

            # Build 3x3 Cauchy stress tensor
            sigma_3x3 = np.array([
                [sxx, txy, txz],
                [txy, syy, tyz],
                [txz, tyz, szz],
            ], dtype=np.float64)

            # Build 9x9 block-diagonal S matrix
            S = np.zeros((9, 9), dtype=np.float64)
            S[0:3, 0:3] = sigma_3x3
            S[3:6, 3:6] = sigma_3x3
            S[6:9, 6:9] = sigma_3x3

            # Element geometric stiffness via Gauss quadrature
            Kg_e = np.zeros((30, 30), dtype=np.float64)

            for gp in elem.GAUSS_POINTS:
                xi_g, eta_g, zeta_g, w_g = gp

                # Shape function derivatives in natural coordinates
                dN_nat = elem.shape_derivatives(xi_g, eta_g, zeta_g)  # (3, 10)

                # Jacobian
                J = dN_nat @ coords  # (3, 3)
                det_J = np.linalg.det(J)
                if det_J <= 0.0:
                    continue

                J_inv = np.linalg.inv(J)

                # Physical derivatives: dN_phys = J_inv @ dN_nat  -> (3, 10)
                dN_phys = J_inv @ dN_nat

                # Build G matrix (9 x 30)
                G = np.zeros((9, 30), dtype=np.float64)
                for i in range(10):
                    col = 3 * i
                    dNx = dN_phys[0, i]
                    dNy = dN_phys[1, i]
                    dNz = dN_phys[2, i]

                    # Block for x-component derivatives
                    G[0, col] = dNx
                    G[1, col] = dNy
                    G[2, col] = dNz
                    # Block for y-component derivatives
                    G[3, col + 1] = dNx
                    G[4, col + 1] = dNy
                    G[5, col + 1] = dNz
                    # Block for z-component derivatives
                    G[6, col + 2] = dNx
                    G[7, col + 2] = dNy
                    G[8, col + 2] = dNz

                # K_sigma_e += G^T * S * G * det(J) * w
                SG = S @ G  # (9, 30)
                Kg_e += (G.T @ SG) * det_J * w_g

            # Build global DOF mapping
            dof_map = np.empty(30, dtype=np.int64)
            for i in range(10):
                base = 3 * node_indices[i]
                dof_map[3 * i] = base
                dof_map[3 * i + 1] = base + 1
                dof_map[3 * i + 2] = base + 2

            # Scatter into COO arrays
            start = offset
            end = offset + entries_per_elem
            rows[start:end] = dof_map[local_i]
            cols[start:end] = dof_map[local_j]
            vals[start:end] = Kg_e.ravel()
            offset = end

        # Trim if needed
        if offset < nnz_estimate:
            rows = rows[:offset]
            cols = cols[:offset]
            vals = vals[:offset]

        # Build COO and convert to CSR
        K_sigma = sp.coo_matrix(
            (vals, (rows, cols)), shape=(n_dof, n_dof)
        ).tocsr()

        # Symmetrize for numerical cleanup
        K_sigma = (K_sigma + K_sigma.T) / 2.0
        K_sigma.eliminate_zeros()

        logger.info(
            "Geometric stiffness assembled: nnz=%d, max |K_sigma|=%.6e",
            K_sigma.nnz,
            abs(K_sigma).max() if K_sigma.nnz > 0 else 0.0,
        )

        return K_sigma

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
