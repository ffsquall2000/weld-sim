"""Damping models for harmonic response analysis.

Provides three damping formulations used in ultrasonic welding FEA:

1. **HystereticDamping** -- constant loss factor (structural damping).
2. **RayleighDamping** -- proportional damping (alpha*M + beta*K).
3. **ModalDamping** -- per-mode damping ratios for modal superposition.

Each model builds a complex dynamic stiffness matrix D(omega) or operates
in modal space to compute frequency response functions (FRFs).

Typical Q-factors for ultrasonic horn materials:

=============  ==============  =======================
Material       Q-factor        eta = 1/Q
=============  ==============  =======================
Ti-6Al-4V      8000--15000     6.7e-5 -- 1.25e-4
Al 7075-T6     5000--10000     1.0e-4 -- 2.0e-4
Steel D2       10000--20000    5.0e-5 -- 1.0e-4
=============  ==============  =======================
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DampingModel(ABC):
    """Abstract base for damping models.

    Every concrete damping model must declare its ``damping_type`` and
    implement ``build_dynamic_stiffness`` (for direct frequency-response)
    or an alternative modal interface.
    """

    @abstractmethod
    def build_dynamic_stiffness(
        self,
        K: sp.spmatrix,
        M: sp.spmatrix,
        omega: float,
    ) -> sp.spmatrix:
        """Build the complex dynamic stiffness matrix D(omega).

        Parameters
        ----------
        K : scipy.sparse matrix
            Global stiffness matrix (real, symmetric, positive semi-definite).
        M : scipy.sparse matrix
            Global mass matrix (real, symmetric, positive definite).
        omega : float
            Excitation circular frequency [rad/s].

        Returns
        -------
        D : scipy.sparse matrix (complex)
            Complex dynamic stiffness:
            ``D(omega) = K_eff - omega**2 * M_eff + damping_terms``
        """
        ...

    @property
    @abstractmethod
    def damping_type(self) -> str:
        """Return the damping type identifier.

        One of ``'hysteretic'``, ``'rayleigh'``, or ``'modal'``.
        """
        ...


# ---------------------------------------------------------------------------
# Hysteretic (structural) damping
# ---------------------------------------------------------------------------

class HystereticDamping(DampingModel):
    """Constant loss-factor (structural / hysteretic) damping.

    The complex stiffness replacement is:

    .. math::

        K_{\\text{complex}} = K \\, (1 + j \\, \\eta)

    so the dynamic stiffness becomes:

    .. math::

        D(\\omega) = K (1 + j \\eta) - \\omega^2 M

    Parameters
    ----------
    eta : float
        Material loss factor.  For ultrasonic horn metals this is
        typically in the range 5e-5 to 1e-3.  It equals the reciprocal
        of the Q-factor: ``eta = 1 / Q``.
    """

    def __init__(self, eta: float) -> None:
        if eta < 0.0:
            raise ValueError(f"Loss factor eta must be >= 0, got {eta}")
        self._eta = eta

    # -- public properties -------------------------------------------------

    @property
    def eta(self) -> float:
        """Material loss factor."""
        return self._eta

    @property
    def damping_type(self) -> str:  # noqa: D401
        return "hysteretic"

    # -- interface ---------------------------------------------------------

    def build_dynamic_stiffness(
        self,
        K: sp.spmatrix,
        M: sp.spmatrix,
        omega: float,
    ) -> sp.spmatrix:
        """Build D(omega) = K*(1 + j*eta) - omega^2 * M.

        The result is a sparse complex matrix in CSR format.
        """
        # Convert to CSR for efficient arithmetic
        K_csr = sp.csr_matrix(K, dtype=np.complex128)
        M_csr = sp.csr_matrix(M, dtype=np.complex128)

        D = K_csr * (1.0 + 1j * self._eta) - (omega ** 2) * M_csr
        return D

    def __repr__(self) -> str:
        return f"HystereticDamping(eta={self._eta:.6e})"


# ---------------------------------------------------------------------------
# Rayleigh (proportional) damping
# ---------------------------------------------------------------------------

class RayleighDamping(DampingModel):
    """Rayleigh (proportional) damping: C = alpha * M + beta * K.

    The dynamic stiffness in the frequency domain is:

    .. math::

        D(\\omega) = K + j \\omega (\\alpha M + \\beta K) - \\omega^2 M

    Parameters
    ----------
    alpha : float
        Mass-proportional damping coefficient [1/s].
    beta : float
        Stiffness-proportional damping coefficient [s].
    """

    def __init__(self, alpha: float, beta: float) -> None:
        if alpha < 0.0:
            raise ValueError(f"alpha must be >= 0, got {alpha}")
        if beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self._alpha = alpha
        self._beta = beta

    # -- class methods -----------------------------------------------------

    @classmethod
    def from_frequencies(
        cls,
        f1: float,
        f2: float,
        zeta1: float,
        zeta2: float,
    ) -> RayleighDamping:
        """Compute alpha, beta from two target frequencies and damping ratios.

        Given two frequencies f1, f2 (Hz) and their associated damping
        ratios zeta1, zeta2, the Rayleigh coefficients are:

        .. math::

            \\omega_1 = 2\\pi f_1, \\quad \\omega_2 = 2\\pi f_2

            \\alpha = \\frac{2 (\\zeta_1 \\omega_2 - \\zeta_2 \\omega_1)}
                           {\\omega_2^2 - \\omega_1^2} \\, \\omega_1 \\omega_2

            \\beta  = \\frac{2 (\\zeta_2 \\omega_2 - \\zeta_1 \\omega_1)}
                           {\\omega_2^2 - \\omega_1^2}

        Parameters
        ----------
        f1, f2 : float
            Target frequencies in Hz (f1 < f2).
        zeta1, zeta2 : float
            Damping ratios at f1 and f2 respectively.

        Returns
        -------
        RayleighDamping
            Instance with computed alpha and beta.

        Raises
        ------
        ValueError
            If f1 >= f2 or frequencies are non-positive.
        """
        if f1 <= 0.0 or f2 <= 0.0:
            raise ValueError(
                f"Frequencies must be positive, got f1={f1}, f2={f2}"
            )
        if f1 >= f2:
            raise ValueError(
                f"f1 must be less than f2, got f1={f1}, f2={f2}"
            )
        if zeta1 < 0.0 or zeta2 < 0.0:
            raise ValueError(
                f"Damping ratios must be >= 0, got zeta1={zeta1}, zeta2={zeta2}"
            )

        omega1 = 2.0 * np.pi * f1
        omega2 = 2.0 * np.pi * f2

        denom = omega2 ** 2 - omega1 ** 2

        alpha = (
            2.0
            * (zeta1 * omega2 - zeta2 * omega1)
            / denom
            * omega1
            * omega2
        )
        beta = (
            2.0
            * (zeta2 * omega2 - zeta1 * omega1)
            / denom
        )

        if alpha < 0.0:
            raise ValueError(
                f"Computed alpha={alpha:.4e} is negative. This occurs when "
                f"zeta2/zeta1 ({zeta2 / zeta1:.2f}) exceeds f2/f1 "
                f"({f2 / f1:.2f}). Consider using equal damping ratios at "
                f"both frequencies."
            )

        return cls(alpha=alpha, beta=beta)

    # -- public properties -------------------------------------------------

    @property
    def alpha(self) -> float:
        """Mass-proportional coefficient."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Stiffness-proportional coefficient."""
        return self._beta

    @property
    def damping_type(self) -> str:  # noqa: D401
        return "rayleigh"

    # -- interface ---------------------------------------------------------

    def effective_damping_ratio(self, freq_hz: float) -> float:
        """Compute the effective damping ratio at a given frequency.

        The Rayleigh damping ratio at circular frequency omega is:

        .. math::

            \\zeta(\\omega) = \\frac{\\alpha}{2 \\omega}
                            + \\frac{\\beta \\omega}{2}

        Parameters
        ----------
        freq_hz : float
            Frequency in Hz.

        Returns
        -------
        float
            Effective damping ratio at the given frequency.
        """
        omega = 2.0 * np.pi * freq_hz
        if omega == 0.0:
            return float("inf") if self._alpha > 0.0 else 0.0
        return self._alpha / (2.0 * omega) + self._beta * omega / 2.0

    def build_dynamic_stiffness(
        self,
        K: sp.spmatrix,
        M: sp.spmatrix,
        omega: float,
    ) -> sp.spmatrix:
        """Build D(omega) = K + j*omega*(alpha*M + beta*K) - omega^2*M.

        The result is a sparse complex matrix in CSR format.
        """
        K_csr = sp.csr_matrix(K, dtype=np.complex128)
        M_csr = sp.csr_matrix(M, dtype=np.complex128)

        # C = alpha * M + beta * K  (the viscous damping matrix)
        # D(omega) = K + j * omega * C - omega^2 * M
        D = (
            K_csr
            + 1j * omega * (self._alpha * M_csr + self._beta * K_csr)
            - (omega ** 2) * M_csr
        )
        return D

    def __repr__(self) -> str:
        return (
            f"RayleighDamping(alpha={self._alpha:.6e}, beta={self._beta:.6e})"
        )


# ---------------------------------------------------------------------------
# Modal damping
# ---------------------------------------------------------------------------

class ModalDamping(DampingModel):
    """Per-mode damping ratios for modal superposition.

    This model works in modal (reduced) space rather than building
    a full-size dynamic stiffness matrix.  Each mode *i* with natural
    frequency omega_i and damping ratio zeta_i satisfies:

    .. math::

        \\ddot{q}_i + 2 \\zeta_i \\omega_i \\dot{q}_i + \\omega_i^2 q_i
            = \\phi_i^T F / m_i

    In the frequency domain at excitation frequency omega:

    .. math::

        q_i(\\omega) = \\frac{\\phi_i^T F}
            {\\omega_i^2 - \\omega^2 + 2j \\zeta_i \\omega_i \\omega}

    Parameters
    ----------
    zeta : float or array-like
        Damping ratio(s).  If a scalar is given, the same ratio is applied
        to every mode.  If array-like, its length must match the number of
        modes passed to :meth:`modal_frf`.
    """

    def __init__(self, zeta: float | NDArray[np.float64]) -> None:
        self._zeta = np.atleast_1d(np.asarray(zeta, dtype=np.float64))
        if np.any(self._zeta < 0.0):
            raise ValueError("All damping ratios must be >= 0")

    # -- public properties -------------------------------------------------

    @property
    def zeta(self) -> NDArray[np.float64]:
        """Damping ratio array (may be length-1 for uniform damping)."""
        return self._zeta

    @property
    def damping_type(self) -> str:  # noqa: D401
        return "modal"

    # -- interface ---------------------------------------------------------

    def build_dynamic_stiffness(
        self,
        K: sp.spmatrix,
        M: sp.spmatrix,
        omega: float,
    ) -> sp.spmatrix:
        """Not the primary interface for ModalDamping.

        ModalDamping works in modal space via :meth:`modal_frf`.
        This method raises ``NotImplementedError`` to signal callers
        to use the modal interface instead.
        """
        raise NotImplementedError(
            "ModalDamping operates in modal space. "
            "Use modal_frf() instead of build_dynamic_stiffness()."
        )

    def modal_frf(
        self,
        omega_n: NDArray[np.float64],
        phi_n: NDArray[np.float64],
        M_phi: NDArray[np.float64],
        F: NDArray[np.float64],
        omega: float,
    ) -> NDArray[np.complex128]:
        """Compute displacement via modal superposition.

        Parameters
        ----------
        omega_n : ndarray, shape (n_modes,)
            Natural frequencies in rad/s.
        phi_n : ndarray, shape (n_dof, n_modes)
            Mode shape matrix (columns are mode shapes).
        M_phi : ndarray, shape (n_modes,)
            Generalised mass per mode.  For mass-normalised modes this
            is an array of ones.
        F : ndarray, shape (n_dof,)
            Applied force vector (physical coordinates).
        omega : float
            Excitation circular frequency [rad/s].

        Returns
        -------
        U : ndarray, shape (n_dof,), complex
            Complex displacement vector in physical coordinates.
        """
        n_modes = len(omega_n)
        n_dof = phi_n.shape[0]

        # Expand zeta to match modes
        if self._zeta.size == 1:
            zeta_arr = np.full(n_modes, self._zeta[0])
        else:
            if self._zeta.size != n_modes:
                raise ValueError(
                    f"zeta has {self._zeta.size} entries but "
                    f"{n_modes} modes were provided."
                )
            zeta_arr = self._zeta

        # Modal participation factors: phi_i^T * F
        modal_forces = phi_n.T @ F  # (n_modes,)

        # Modal coordinates in frequency domain
        # q_i = (phi_i^T * F) / (omega_i^2 - omega^2 + 2j*zeta_i*omega_i*omega) / m_i
        denominators = (
            omega_n ** 2
            - omega ** 2
            + 2j * zeta_arr * omega_n * omega
        )

        q = modal_forces / (denominators * M_phi)  # (n_modes,) complex

        # Transform back to physical coordinates: U = sum_i phi_i * q_i
        U = phi_n @ q  # (n_dof,) complex

        return U

    def __repr__(self) -> str:
        if self._zeta.size == 1:
            return f"ModalDamping(zeta={self._zeta[0]:.6e})"
        return f"ModalDamping(zeta=[{self._zeta.size} modes])"
