"""Tests for FEA damping models.

Covers hysteretic, Rayleigh, and modal damping formulations used in
harmonic response analysis for ultrasonic welding.

Uses small synthetic K, M matrices (2-DOF spring-mass and SDOF systems)
so that analytical results are available for verification.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.damping import (
    DampingModel,
    HystereticDamping,
    ModalDamping,
    RayleighDamping,
)

# ---------------------------------------------------------------------------
# Helpers -- small synthetic systems
# ---------------------------------------------------------------------------

_TWO_PI = 2.0 * np.pi


def _make_2dof_spring_mass():
    """Create a symmetric 2-DOF spring-mass system.

    System layout::

        |---k1---[m1]---k2---[m2]---k3---|

    With k1 = k2 = k3 = 1e6, m1 = m2 = 1.0.

    K = [[ k1+k2,  -k2 ],
         [ -k2,  k2+k3 ]]

    M = [[ m1, 0 ],
         [ 0, m2 ]]

    Returns (K_sparse, M_sparse) in CSR format.
    """
    k = 1.0e6
    K = np.array([
        [2.0 * k, -k],
        [-k, 2.0 * k],
    ])
    M = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    return sp.csr_matrix(K), sp.csr_matrix(M)


def _make_6dof_chain():
    """Create a 6-DOF chain of springs and masses.

    k_i = 1e8, m_i = 0.1 (i = 1..6).
    Returns (K_sparse, M_sparse) in CSR format.
    """
    n = 6
    k = 1.0e8
    m = 0.1
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k
        if i > 0:
            K[i, i] += k
            K[i, i - 1] -= k
            K[i - 1, i] -= k
    # Fixed wall on left -> k already added to (0,0)
    # Fixed wall on right -> add k to (n-1, n-1)
    K[n - 1, n - 1] += k

    M = np.diag(np.full(n, m))
    return sp.csr_matrix(K), sp.csr_matrix(M)


# ---------------------------------------------------------------------------
# SDOF parameters for modal damping tests
# ---------------------------------------------------------------------------
# m = 1 kg, f_n = 20 kHz, zeta = 0.003
_SDOF_M = 1.0
_SDOF_FN = 20_000.0  # Hz
_SDOF_OMEGA_N = _TWO_PI * _SDOF_FN
_SDOF_K = _SDOF_OMEGA_N ** 2 * _SDOF_M
_SDOF_ZETA = 0.003


# ===========================================================================
# Tests -- HystereticDamping
# ===========================================================================


class TestHystereticDamping:
    """Tests for HystereticDamping (constant loss factor)."""

    def test_hysteretic_dynamic_stiffness_type(self):
        """D(omega) should be a complex sparse matrix."""
        K, M = _make_2dof_spring_mass()
        model = HystereticDamping(eta=0.005)
        omega = _TWO_PI * 20_000.0

        D = model.build_dynamic_stiffness(K, M, omega)

        assert sp.issparse(D), "D must be a sparse matrix"
        assert np.iscomplexobj(D.toarray()), "D must be complex"

    def test_hysteretic_symmetry(self):
        """D(omega) should be symmetric."""
        K, M = _make_6dof_chain()
        model = HystereticDamping(eta=1e-4)
        omega = _TWO_PI * 15_000.0

        D = model.build_dynamic_stiffness(K, M, omega)
        D_dense = D.toarray()

        np.testing.assert_allclose(
            D_dense, D_dense.T, atol=1e-10,
            err_msg="Dynamic stiffness matrix should be symmetric",
        )

    def test_hysteretic_reduces_to_K_at_zero_freq(self):
        """At omega=0, D(0) should equal K * (1 + j*eta)."""
        K, M = _make_2dof_spring_mass()
        eta = 0.002
        model = HystereticDamping(eta=eta)

        D = model.build_dynamic_stiffness(K, M, omega=0.0)
        D_expected = K.toarray() * (1.0 + 1j * eta)

        np.testing.assert_allclose(
            D.toarray(), D_expected, rtol=1e-12,
            err_msg="At zero frequency D should be K*(1+j*eta)",
        )

    def test_hysteretic_at_resonance(self):
        """At resonance the imaginary part of D provides damping.

        For the 2-DOF system, the first natural frequency is
        omega_1 = sqrt(k/m).  At that frequency, D should have a
        non-trivial imaginary part that limits the response.
        """
        K, M = _make_2dof_spring_mass()
        eta = 0.01

        # First natural freq: omega^2 = k/m = 1e6 for the lower mode
        omega_1 = np.sqrt(1.0e6)  # sqrt of eigenvalue of K M^{-1}
        model = HystereticDamping(eta=eta)

        D = model.build_dynamic_stiffness(K, M, omega_1)
        D_dense = D.toarray()

        # Imaginary part should be eta * K
        imag_expected = eta * K.toarray()
        np.testing.assert_allclose(
            D_dense.imag, imag_expected, rtol=1e-12,
            err_msg="Imaginary part should be eta*K at any frequency",
        )

        # Real part: K - omega^2 * M
        real_expected = K.toarray() - omega_1 ** 2 * M.toarray()
        np.testing.assert_allclose(
            D_dense.real, real_expected, rtol=1e-10,
            err_msg="Real part should be K - omega^2 M",
        )

    def test_hysteretic_negative_eta_raises(self):
        """Negative eta should raise ValueError."""
        with pytest.raises(ValueError, match="eta must be >= 0"):
            HystereticDamping(eta=-0.001)

    def test_hysteretic_damping_type(self):
        """damping_type property should return 'hysteretic'."""
        model = HystereticDamping(eta=0.001)
        assert model.damping_type == "hysteretic"


# ===========================================================================
# Tests -- RayleighDamping
# ===========================================================================


class TestRayleighDamping:
    """Tests for RayleighDamping (proportional damping)."""

    def test_rayleigh_alpha_beta_from_frequencies(self):
        """Verify alpha, beta computation for known f1, f2, zeta values.

        If zeta1 = zeta2 = zeta (equal damping at both frequencies),
        the formulas simplify and we can verify the results analytically.
        """
        f1 = 16_000.0
        f2 = 24_000.0
        zeta = 0.003

        model = RayleighDamping.from_frequencies(f1, f2, zeta, zeta)

        omega1 = _TWO_PI * f1
        omega2 = _TWO_PI * f2

        # For equal zeta at both frequencies:
        # alpha = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
        # beta  = 2 * zeta / (omega1 + omega2)
        alpha_expected = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
        beta_expected = 2.0 * zeta / (omega1 + omega2)

        np.testing.assert_allclose(
            model.alpha, alpha_expected, rtol=1e-10,
            err_msg="alpha should match analytical formula for equal zeta",
        )
        np.testing.assert_allclose(
            model.beta, beta_expected, rtol=1e-10,
            err_msg="beta should match analytical formula for equal zeta",
        )

    def test_rayleigh_damping_ratio_at_target(self):
        """At f_target, the effective damping ratio should match input.

        For zeta1 = zeta2 = zeta with f1 = 0.8*f_t, f2 = 1.2*f_t,
        we verify that effective_damping_ratio(f_t) is close to zeta.
        The Rayleigh curve passes exactly through (f1, zeta1) and
        (f2, zeta2), so at intermediate f_t it should be close but
        not exact.
        """
        f_target = 20_000.0
        f1 = 0.8 * f_target
        f2 = 1.2 * f_target
        zeta = 0.003

        model = RayleighDamping.from_frequencies(f1, f2, zeta, zeta)

        # Check damping ratio at the two fitting frequencies
        zeta_at_f1 = model.effective_damping_ratio(f1)
        zeta_at_f2 = model.effective_damping_ratio(f2)

        np.testing.assert_allclose(
            zeta_at_f1, zeta, rtol=1e-10,
            err_msg="Damping ratio at f1 should match input zeta1",
        )
        np.testing.assert_allclose(
            zeta_at_f2, zeta, rtol=1e-10,
            err_msg="Damping ratio at f2 should match input zeta2",
        )

        # At f_target (midpoint), damping should be close to zeta
        # (slightly below for equal-zeta Rayleigh fitting)
        zeta_at_target = model.effective_damping_ratio(f_target)
        assert abs(zeta_at_target - zeta) < 0.1 * zeta, (
            f"Effective zeta at f_target should be within 10% of input: "
            f"got {zeta_at_target:.6e} vs {zeta:.6e}"
        )

    def test_rayleigh_dynamic_stiffness_type(self):
        """D(omega) should be a complex sparse matrix."""
        K, M = _make_2dof_spring_mass()
        model = RayleighDamping(alpha=10.0, beta=1e-6)
        omega = _TWO_PI * 20_000.0

        D = model.build_dynamic_stiffness(K, M, omega)

        assert sp.issparse(D), "D must be a sparse matrix"
        assert np.iscomplexobj(D.toarray()), "D must be complex"

    def test_rayleigh_symmetry(self):
        """D(omega) should be symmetric for symmetric K, M."""
        K, M = _make_6dof_chain()
        model = RayleighDamping(alpha=5.0, beta=2e-7)
        omega = _TWO_PI * 18_000.0

        D = model.build_dynamic_stiffness(K, M, omega)
        D_dense = D.toarray()

        np.testing.assert_allclose(
            D_dense, D_dense.T, atol=1e-6,
            err_msg="Dynamic stiffness should be symmetric",
        )

    def test_rayleigh_dynamic_stiffness_values(self):
        """Verify the dynamic stiffness formula K + j*w*C - w^2*M."""
        K, M = _make_2dof_spring_mass()
        alpha = 15.0
        beta = 5e-7
        model = RayleighDamping(alpha=alpha, beta=beta)
        omega = _TWO_PI * 20_000.0

        D = model.build_dynamic_stiffness(K, M, omega)
        D_dense = D.toarray()

        C = alpha * M.toarray() + beta * K.toarray()
        D_expected = (
            K.toarray()
            + 1j * omega * C
            - omega ** 2 * M.toarray()
        )

        np.testing.assert_allclose(
            D_dense, D_expected, rtol=1e-10,
            err_msg="D should equal K + j*omega*C - omega^2*M",
        )

    def test_rayleigh_invalid_frequencies(self):
        """from_frequencies should reject f1 >= f2 and negative frequencies."""
        with pytest.raises(ValueError, match="f1 must be less than f2"):
            RayleighDamping.from_frequencies(20000, 16000, 0.003, 0.003)

        with pytest.raises(ValueError, match="positive"):
            RayleighDamping.from_frequencies(-1000, 20000, 0.003, 0.003)

    def test_rayleigh_damping_type(self):
        """damping_type property should return 'rayleigh'."""
        model = RayleighDamping(alpha=1.0, beta=1e-6)
        assert model.damping_type == "rayleigh"


# ===========================================================================
# Tests -- ModalDamping
# ===========================================================================


class TestModalDamping:
    """Tests for ModalDamping (per-mode damping ratios)."""

    def test_modal_damping_single_mode_frf(self):
        """Known SDOF FRF: H(omega) = 1/(k - w^2*m + j*2*zeta*w*wn).

        For a single mode, the modal FRF should reproduce the exact SDOF
        transfer function.
        """
        m = _SDOF_M
        k = _SDOF_K
        omega_n = _SDOF_OMEGA_N
        zeta = _SDOF_ZETA

        model = ModalDamping(zeta=zeta)

        # Mode shape for SDOF: phi = [1], mass-normalised: phi = 1/sqrt(m)
        phi_norm = 1.0 / np.sqrt(m)
        phi_n = np.array([[phi_norm]])  # (1, 1) mode shape matrix
        omega_n_arr = np.array([omega_n])
        M_phi = np.array([1.0])  # mass-normalised
        F = np.array([1.0])  # unit force

        # Evaluate at some test frequency
        omega_test = 0.95 * omega_n
        U = model.modal_frf(omega_n_arr, phi_n, M_phi, F, omega_test)

        # Analytical SDOF response:
        # u = (phi * phi^T * F) / (omega_n^2 - omega^2 + 2j*zeta*omega_n*omega) / M_phi
        # With phi = 1/sqrt(m), phi^T F = 1/sqrt(m), phi * q = (1/sqrt(m)) * q
        # So u = (1/m) / (omega_n^2 - omega^2 + 2j*zeta*omega_n*omega)
        denom = omega_n ** 2 - omega_test ** 2 + 2j * zeta * omega_n * omega_test
        u_expected = (phi_norm * phi_norm * F[0]) / denom

        np.testing.assert_allclose(
            U[0], u_expected, rtol=1e-10,
            err_msg="Modal FRF should match analytical SDOF result",
        )

    def test_modal_damping_peak_at_resonance(self):
        """FRF peak should occur near omega_n.

        Sweep through frequencies near omega_n and verify the maximum
        response amplitude is at or very close to the natural frequency.
        """
        omega_n = _SDOF_OMEGA_N
        zeta = _SDOF_ZETA
        m = _SDOF_M

        model = ModalDamping(zeta=zeta)

        phi_norm = 1.0 / np.sqrt(m)
        phi_n = np.array([[phi_norm]])
        omega_n_arr = np.array([omega_n])
        M_phi = np.array([1.0])
        F = np.array([1.0])

        # Sweep: 0.95 * omega_n to 1.05 * omega_n, 1001 points
        omegas = np.linspace(0.95 * omega_n, 1.05 * omega_n, 1001)
        amps = np.zeros(len(omegas))

        for i, w in enumerate(omegas):
            U = model.modal_frf(omega_n_arr, phi_n, M_phi, F, w)
            amps[i] = np.abs(U[0])

        peak_omega = omegas[np.argmax(amps)]

        # For light damping, the damped natural frequency is
        # omega_d = omega_n * sqrt(1 - zeta^2) ~ omega_n
        # The peak of |H(omega)| occurs at omega_peak = omega_n * sqrt(1 - 2*zeta^2)
        omega_peak_expected = omega_n * np.sqrt(1 - 2 * zeta ** 2)

        # Should be within 0.1% of the expected peak location
        np.testing.assert_allclose(
            peak_omega, omega_peak_expected, rtol=1e-3,
            err_msg="FRF peak should occur near omega_n for light damping",
        )

    def test_modal_damping_q_factor(self):
        """Peak amplitude should be approximately 1/(2*zeta*k) for SDOF.

        For a unit-force SDOF system: |H(omega_n)| ~ 1/(2*zeta*omega_n^2*m)
        = 1/(2*zeta*k).
        """
        omega_n = _SDOF_OMEGA_N
        zeta = _SDOF_ZETA
        m = _SDOF_M
        k = _SDOF_K

        model = ModalDamping(zeta=zeta)

        phi_norm = 1.0 / np.sqrt(m)
        phi_n = np.array([[phi_norm]])
        omega_n_arr = np.array([omega_n])
        M_phi = np.array([1.0])
        F = np.array([1.0])

        # Evaluate exactly at omega_n
        U_at_resonance = model.modal_frf(omega_n_arr, phi_n, M_phi, F, omega_n)
        amp_at_resonance = np.abs(U_at_resonance[0])

        # For mass-normalised SDOF:
        # phi^T F / M_phi = phi_norm * 1 / 1 = 1/sqrt(m)
        # At omega = omega_n: denominator = 2j*zeta*omega_n^2
        # |q_n| = phi_norm / (2*zeta*omega_n^2)
        # |u| = phi_norm * |q_n| = (1/m) / (2*zeta*omega_n^2) = 1/(2*zeta*k)
        expected_amp = 1.0 / (2.0 * zeta * k)

        np.testing.assert_allclose(
            amp_at_resonance, expected_amp, rtol=1e-6,
            err_msg="Peak amplitude should be ~1/(2*zeta*k) for SDOF",
        )

    def test_modal_damping_multi_mode(self):
        """Two-mode system with different damping ratios per mode."""
        # Two modes at 15 kHz and 20 kHz
        omega_n = np.array([_TWO_PI * 15_000.0, _TWO_PI * 20_000.0])
        # Simplified 2-DOF mode shapes (mass-normalised for m=1)
        phi_n = np.array([
            [1.0, 0.5],
            [0.5, -1.0],
        ]) / np.sqrt(1.0)
        M_phi = np.array([1.0, 1.0])
        F = np.array([1.0, 0.0])

        zetas = np.array([0.002, 0.005])
        model = ModalDamping(zeta=zetas)

        # Evaluate far from both resonances
        omega_test = _TWO_PI * 10_000.0
        U = model.modal_frf(omega_n, phi_n, M_phi, F, omega_test)

        assert U.shape == (2,), "Output should have n_dof entries"
        assert np.all(np.isfinite(U)), "FRF should be finite away from resonance"

    def test_modal_damping_zeta_mismatch_raises(self):
        """zeta array length must match number of modes."""
        model = ModalDamping(zeta=np.array([0.001, 0.002]))

        omega_n = np.array([1000.0, 2000.0, 3000.0])  # 3 modes
        phi_n = np.zeros((6, 3))
        M_phi = np.ones(3)
        F = np.zeros(6)

        with pytest.raises(ValueError, match="zeta has 2 entries but 3 modes"):
            model.modal_frf(omega_n, phi_n, M_phi, F, 1500.0)

    def test_modal_damping_negative_zeta_raises(self):
        """Negative zeta should raise ValueError."""
        with pytest.raises(ValueError, match="must be >= 0"):
            ModalDamping(zeta=-0.001)

    def test_modal_damping_type(self):
        """damping_type property should return 'modal'."""
        model = ModalDamping(zeta=0.003)
        assert model.damping_type == "modal"

    def test_modal_build_dynamic_stiffness_raises(self):
        """build_dynamic_stiffness should raise NotImplementedError."""
        model = ModalDamping(zeta=0.003)
        K, M = _make_2dof_spring_mass()
        with pytest.raises(NotImplementedError, match="modal space"):
            model.build_dynamic_stiffness(K, M, 1000.0)


# ===========================================================================
# Tests -- DampingModel ABC
# ===========================================================================


class TestDampingModelABC:
    """Tests for DampingModel abstract base class."""

    def test_damping_model_is_abstract(self):
        """Cannot instantiate DampingModel directly."""
        with pytest.raises(TypeError):
            DampingModel()

    def test_damping_model_incomplete_subclass(self):
        """A subclass that does not implement all abstract methods
        should not be instantiable."""

        class IncompleteDamping(DampingModel):
            @property
            def damping_type(self) -> str:
                return "incomplete"

        with pytest.raises(TypeError):
            IncompleteDamping()
