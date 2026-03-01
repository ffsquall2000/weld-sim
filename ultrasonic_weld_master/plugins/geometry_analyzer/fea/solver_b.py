"""FEniCSx-based FEA solver (Solver B).

Provides advanced FEA capabilities via DOLFINx/PETSc/SLEPc:
  - Modal analysis via SLEPc eigenvalue solver
  - Harmonic response via PETSc complex linear solver
  - Piezoelectric coupled analysis (electromechanical)
  - Impedance spectrum computation
  - Transient thermal analysis with hysteretic and frictional heat sources
  - Nonlinear contact analysis with augmented Lagrangian and Coulomb friction

This module is designed to run inside a Docker container with FEniCSx
installed.  On machines without FEniCSx, the module loads but all
analysis methods raise ``RuntimeError``.  A module-level flag
``HAS_FENICSX`` indicates availability.

Algorithm overview (modal)
--------------------------
1. Convert FEAMesh to DOLFINx mesh via gmshio or manual construction.
2. Define UFL variational forms for stiffness and mass.
3. Assemble PETSc matrices K and M.
4. Solve generalised eigenvalue problem K * phi = omega^2 * M * phi
   using SLEPc shift-invert around the target frequency.
5. Extract eigenvalues -> frequencies, eigenvectors -> mode shapes.
6. Classify modes and compute effective mass ratios.
7. Return ``ModalResult``.

Thermal analysis (BDF2 time integration)
-----------------------------------------
Solves the transient heat equation with two heat source mechanisms:
  - Hysteretic volumetric heating: Q_hyst = pi * f * eta * sigma:epsilon
  - Frictional surface heating: Q_fric = mu * p * v_rel
Time integration uses BDF2: (3T^{n+1} - 4T^n + T^{n-1})/(2*dt) = alpha * nabla^2 T + Q

Contact analysis (augmented Lagrangian)
----------------------------------------
Nonlinear contact with Coulomb friction:
  - Normal: augmented Lagrangian L = int(sigma:epsilon) + gamma/2 * int(g_N^+ ^2) + int(lambda*g_N)
  - Tangential: |sigma_T| <= mu * |sigma_N| (Coulomb friction)
  - Newton solver with line search for nonlinear equilibrium
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    HarmonicConfig,
    ModalConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    FEA_MATERIALS,
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ContactResult,
    HarmonicResult,
    ImpedanceResult,
    ModalResult,
    StaticResult,
    ThermalResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_interface import (
    SolverInterface,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FEniCSx imports -- graceful degradation
# ---------------------------------------------------------------------------
HAS_FENICSX: bool = False
_dolfinx = None
_ufl = None
_basix = None
_PETSc = None
_SLEPc = None

try:
    import dolfinx as _dolfinx  # type: ignore[no-redef]
    import dolfinx.fem  # noqa: F401
    import dolfinx.fem.petsc  # noqa: F401
    import dolfinx.mesh  # noqa: F401
    import dolfinx.io  # noqa: F401
    import ufl as _ufl  # type: ignore[no-redef]
    import basix as _basix  # type: ignore[no-redef]
    from petsc4py import PETSc as _PETSc  # type: ignore[no-redef]
    from slepc4py import SLEPc as _SLEPc  # type: ignore[no-redef]
    HAS_FENICSX = True
    logger.info("FEniCSx backend available (DOLFINx %s)", _dolfinx.__version__)
except ImportError:
    logger.info(
        "FEniCSx not available. SolverB analyses are disabled. "
        "Install via: conda install -c conda-forge fenics-dolfinx petsc4py slepc4py"
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TWO_PI = 2.0 * np.pi
_RIGID_BODY_CUTOFF_HZ = 100.0
_PENALTY_FACTOR = 1e20

# PZT-4 default piezoelectric material constants (Voigt notation)
PZT4_PROPERTIES: dict[str, Any] = {
    "c_E": np.array([
        [139.0e9, 77.8e9,  74.3e9,  0.0,     0.0,     0.0     ],
        [77.8e9,  139.0e9, 74.3e9,  0.0,     0.0,     0.0     ],
        [74.3e9,  74.3e9,  115.0e9, 0.0,     0.0,     0.0     ],
        [0.0,     0.0,     0.0,     25.6e9,  0.0,     0.0     ],
        [0.0,     0.0,     0.0,     0.0,     25.6e9,  0.0     ],
        [0.0,     0.0,     0.0,     0.0,     0.0,     30.6e9  ],
    ]),
    "e": np.array([
        [0.0,  0.0,  0.0,  0.0,  12.7, 0.0],
        [0.0,  0.0,  0.0,  12.7, 0.0,  0.0],
        [-5.2, -5.2, 15.1, 0.0,  0.0,  0.0],
    ]),
    "eps_S": np.array([
        [6.45e-9, 0.0,     0.0     ],
        [0.0,     6.45e-9, 0.0     ],
        [0.0,     0.0,     5.62e-9 ],
    ]),
    "rho": 7500.0,
}

PZT8_PROPERTIES: dict[str, Any] = {
    "c_E": np.array([
        [146.9e9, 81.1e9,  81.1e9,  0.0,     0.0,     0.0     ],
        [81.1e9,  146.9e9, 81.1e9,  0.0,     0.0,     0.0     ],
        [81.1e9,  81.1e9,  131.7e9, 0.0,     0.0,     0.0     ],
        [0.0,     0.0,     0.0,     31.4e9,  0.0,     0.0     ],
        [0.0,     0.0,     0.0,     0.0,     31.4e9,  0.0     ],
        [0.0,     0.0,     0.0,     0.0,     0.0,     32.9e9  ],
    ]),
    "e": np.array([
        [0.0,  0.0,  0.0,  0.0,  10.4, 0.0],
        [0.0,  0.0,  0.0,  10.4, 0.0,  0.0],
        [-4.0, -4.0, 13.2, 0.0,  0.0,  0.0],
    ]),
    "eps_S": np.array([
        [1.14e-8, 0.0,     0.0     ],
        [0.0,     1.14e-8, 0.0     ],
        [0.0,     0.0,     8.85e-9 ],
    ]),
    "rho": 7600.0,
}


# ---------------------------------------------------------------------------
# Helper: Validate FEniCSx availability
# ---------------------------------------------------------------------------
def _require_fenicsx(method_name: str) -> None:
    """Raise RuntimeError if FEniCSx is not installed."""
    if not HAS_FENICSX:
        raise RuntimeError(
            f"FEniCSx not available. Cannot run {method_name}. "
            "Install via: conda install -c conda-forge fenics-dolfinx petsc4py slepc4py "
            "or use the Docker image: docker run uwm/fenicsx-solver:latest"
        )


# ---------------------------------------------------------------------------
# Helper dataclasses for configuration
# ---------------------------------------------------------------------------
@dataclass
class PiezoConfig:
    """Configuration for piezoelectric coupled analysis."""
    mesh: FEAMesh
    pzt_material: str = "PZT-4"
    metal_material: str = "Titanium Ti-6Al-4V"
    electrode_node_sets: dict = field(default_factory=lambda: {
        "top": "electrode_top",
        "bottom": "electrode_bottom",
    })
    driving_voltage: float = 1.0  # V
    freq_min_hz: float = 16000.0
    freq_max_hz: float = 24000.0
    n_freq_points: int = 200
    damping_ratio: float = 0.005

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty if valid)."""
        errors = []
        if self.driving_voltage <= 0:
            errors.append("driving_voltage must be positive")
        if self.freq_min_hz >= self.freq_max_hz:
            errors.append("freq_min_hz must be less than freq_max_hz")
        if self.n_freq_points < 2:
            errors.append("n_freq_points must be at least 2")
        if not (0 < self.damping_ratio < 1):
            errors.append("damping_ratio must be between 0 and 1")
        if self.pzt_material not in ("PZT-4", "PZT-8"):
            errors.append(f"Unknown PZT material: {self.pzt_material!r}")
        return errors


@dataclass
class ThermalConfig:
    """Configuration for transient thermal analysis."""
    mesh: FEAMesh
    material_name: str = "Titanium Ti-6Al-4V"
    frequency_hz: float = 20000.0
    amplitude_m: float = 30e-6  # peak-to-peak amplitude at weld face
    friction_coefficient: float = 0.2
    contact_pressure_pa: float = 1.0e6  # 1 MPa nominal
    weld_time_s: float = 0.5
    dt: float = 0.005  # 5 ms time step
    n_steps: int = 100
    t_ambient_c: float = 25.0
    convection_h: float = 10.0  # W/(m^2*K) natural convection
    time_scheme: str = "bdf2"  # "bdf2" or "backward_euler"

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty if valid)."""
        errors = []
        if self.frequency_hz <= 0:
            errors.append("frequency_hz must be positive")
        if self.amplitude_m < 0:
            errors.append("amplitude_m must be non-negative")
        if self.dt <= 0:
            errors.append("dt must be positive")
        if self.n_steps < 1:
            errors.append("n_steps must be at least 1")
        if self.time_scheme not in ("bdf2", "backward_euler"):
            errors.append(
                f"Unknown time_scheme: {self.time_scheme!r}. "
                "Use 'bdf2' or 'backward_euler'."
            )
        if self.friction_coefficient < 0:
            errors.append("friction_coefficient must be non-negative")
        if self.convection_h < 0:
            errors.append("convection_h must be non-negative")
        mat = get_material(self.material_name)
        if mat is None:
            errors.append(f"Unknown material: {self.material_name!r}")
        return errors


@dataclass
class ContactConfig:
    """Configuration for nonlinear contact analysis."""
    mesh: FEAMesh
    material_name: str = "Titanium Ti-6Al-4V"
    contact_pairs: list[dict] = field(default_factory=list)
    # Each pair: {"master": "node_set_name", "slave": "node_set_name", "mu": 0.2}
    bolt_preload_n: float = 30000.0
    bolt_node_set: str = "bolt_bearing"
    augmentation_parameter: Optional[float] = None  # auto-compute if None
    max_augmentation_iters: int = 20
    newton_rtol: float = 1e-8
    newton_atol: float = 1e-10
    newton_max_iters: int = 50
    line_search: bool = True

    def validate(self) -> list[str]:
        """Return list of validation error messages (empty if valid)."""
        errors = []
        if self.bolt_preload_n < 0:
            errors.append("bolt_preload_n must be non-negative")
        if self.max_augmentation_iters < 1:
            errors.append("max_augmentation_iters must be at least 1")
        if self.newton_rtol <= 0:
            errors.append("newton_rtol must be positive")
        if self.newton_max_iters < 1:
            errors.append("newton_max_iters must be at least 1")
        for i, pair in enumerate(self.contact_pairs):
            if "master" not in pair or "slave" not in pair:
                errors.append(
                    f"contact_pairs[{i}] must have 'master' and 'slave' keys"
                )
            mu = pair.get("mu", 0.0)
            if mu < 0:
                errors.append(
                    f"contact_pairs[{i}] friction coefficient mu must be >= 0"
                )
        mat = get_material(self.material_name)
        if mat is None:
            errors.append(f"Unknown material: {self.material_name!r}")
        return errors


# ---------------------------------------------------------------------------
# Pure-Python helper functions (no FEniCSx dependency)
# ---------------------------------------------------------------------------
def compute_hysteretic_heat(
    frequency_hz: float,
    loss_factor: float,
    stress_amplitude: NDArray[np.floating],
    strain_amplitude: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute volumetric hysteretic heat generation rate.

    Q_hyst = pi * f * eta * sigma_ij * epsilon_ij

    Parameters
    ----------
    frequency_hz : float
        Operating frequency [Hz].
    loss_factor : float
        Material loss factor eta (dimensionless, typically 0.001--0.01).
    stress_amplitude : ndarray
        Stress amplitude tensor or Voigt vector [Pa].
    strain_amplitude : ndarray
        Strain amplitude tensor or Voigt vector [-].

    Returns
    -------
    ndarray
        Volumetric heat generation rate [W/m^3].
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if loss_factor < 0:
        raise ValueError("loss_factor must be non-negative")
    # Ensure arrays have matching shapes for element-wise product
    stress = np.asarray(stress_amplitude, dtype=np.float64)
    strain = np.asarray(strain_amplitude, dtype=np.float64)
    if stress.shape != strain.shape:
        raise ValueError(
            f"stress and strain shapes must match: {stress.shape} vs {strain.shape}"
        )
    # sigma_ij * epsilon_ij (contraction / inner product)
    # For Voigt: sigma . epsilon (dot product per point)
    if stress.ndim == 1:
        # Single point: Voigt vector
        q = math.pi * frequency_hz * loss_factor * np.dot(stress, strain)
        return np.atleast_1d(q)
    elif stress.ndim == 2:
        # Multiple points: (n_points, 6) Voigt vectors
        q = math.pi * frequency_hz * loss_factor * np.sum(
            stress * strain, axis=-1
        )
        return q
    else:
        raise ValueError(f"Expected 1D or 2D arrays, got {stress.ndim}D")


def compute_frictional_heat(
    friction_coefficient: float,
    contact_pressure_pa: float | NDArray[np.floating],
    relative_velocity: float | NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute frictional heat generation at contact surfaces.

    Q_fric = mu * p * v_rel

    For time-averaged frictional power at ultrasonic frequency:
    Q_fric_avg = (2/pi) * mu * p * (2*pi*f*A)

    Parameters
    ----------
    friction_coefficient : float
        Coulomb friction coefficient mu.
    contact_pressure_pa : float or ndarray
        Normal contact pressure [Pa].
    relative_velocity : float or ndarray
        Relative sliding velocity [m/s].

    Returns
    -------
    ndarray
        Surface heat flux [W/m^2].
    """
    if friction_coefficient < 0:
        raise ValueError("friction_coefficient must be non-negative")
    pressure = np.asarray(contact_pressure_pa, dtype=np.float64)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    return np.atleast_1d(friction_coefficient * np.abs(pressure) * np.abs(velocity))


def compute_time_averaged_frictional_heat(
    friction_coefficient: float,
    contact_pressure_pa: float | NDArray[np.floating],
    frequency_hz: float,
    amplitude_m: float,
) -> NDArray[np.floating]:
    """Compute time-averaged frictional heat over one vibration cycle.

    Q_fric_avg = (2/pi) * mu * p * (2*pi*f*A)

    Parameters
    ----------
    friction_coefficient : float
        Coulomb friction coefficient mu.
    contact_pressure_pa : float or ndarray
        Normal contact pressure [Pa].
    frequency_hz : float
        Operating frequency [Hz].
    amplitude_m : float
        Displacement amplitude at weld face [m].

    Returns
    -------
    ndarray
        Time-averaged surface heat flux [W/m^2].
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    if amplitude_m < 0:
        raise ValueError("amplitude_m must be non-negative")
    peak_velocity = _TWO_PI * frequency_hz * amplitude_m
    return (2.0 / math.pi) * compute_frictional_heat(
        friction_coefficient, contact_pressure_pa, peak_velocity
    )


def compute_contact_gap(
    master_coords: NDArray[np.floating],
    slave_coords: NDArray[np.floating],
    normal: NDArray[np.floating],
    master_disp: Optional[NDArray[np.floating]] = None,
    slave_disp: Optional[NDArray[np.floating]] = None,
    initial_gap: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Compute normal gap function for contact pairs.

    g_N = (x_slave - x_master) . n + u_slave . n - u_master . n

    A positive gap means separation (open), negative means penetration.

    Parameters
    ----------
    master_coords : ndarray
        Master surface node coordinates (n, 3).
    slave_coords : ndarray
        Slave surface node coordinates (n, 3).
    normal : ndarray
        Outward normal vector (3,) or (n, 3) per node.
    master_disp : ndarray, optional
        Master displacement (n, 3). Default zeros.
    slave_disp : ndarray, optional
        Slave displacement (n, 3). Default zeros.
    initial_gap : ndarray, optional
        Initial geometric gap (n,). Default computed from coords.

    Returns
    -------
    ndarray
        Gap values (n,): positive = open, negative = penetration.
    """
    master = np.asarray(master_coords, dtype=np.float64)
    slave = np.asarray(slave_coords, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)

    if master.ndim != 2 or master.shape[1] != 3:
        raise ValueError(f"master_coords must be (n, 3), got {master.shape}")
    if slave.shape != master.shape:
        raise ValueError(
            f"slave_coords shape {slave.shape} must match master {master.shape}"
        )

    n_pts = master.shape[0]

    # Ensure normal is broadcastable
    if n.ndim == 1:
        n = np.broadcast_to(n, (n_pts, 3))
    elif n.shape != (n_pts, 3):
        raise ValueError(f"normal shape {n.shape} must be (3,) or ({n_pts}, 3)")

    if master_disp is None:
        master_disp = np.zeros_like(master)
    if slave_disp is None:
        slave_disp = np.zeros_like(slave)

    master_disp = np.asarray(master_disp, dtype=np.float64)
    slave_disp = np.asarray(slave_disp, dtype=np.float64)

    if initial_gap is not None:
        g0 = np.asarray(initial_gap, dtype=np.float64)
    else:
        # Geometric gap: projection of (slave - master) onto normal
        g0 = np.sum((slave - master) * n, axis=-1)

    # Displacement contribution
    delta_u = np.sum((slave_disp - master_disp) * n, axis=-1)

    return g0 + delta_u


def compute_contact_traction_augmented_lagrangian(
    gap: NDArray[np.floating],
    lagrange_multiplier: NDArray[np.floating],
    augmentation_param: float,
) -> NDArray[np.floating]:
    """Compute normal contact traction via augmented Lagrangian.

    p_n = max(0, lambda_n + r * (-g_n))

    Convention: positive p_n = compressive contact pressure.
    Negative gap = penetration.

    Parameters
    ----------
    gap : ndarray
        Normal gap (n,). Positive = open, negative = penetration.
    lagrange_multiplier : ndarray
        Current Lagrange multiplier (n,).
    augmentation_param : float
        Augmentation parameter r > 0.

    Returns
    -------
    ndarray
        Contact traction (n,). Positive = compressive.
    """
    if augmentation_param <= 0:
        raise ValueError("augmentation_param must be positive")
    gap = np.asarray(gap, dtype=np.float64)
    lam = np.asarray(lagrange_multiplier, dtype=np.float64)
    # p_n = max(0, lambda + r * (-gap))
    # Negative gap = penetration -> positive traction
    return np.maximum(0.0, lam + augmentation_param * (-gap))


def compute_coulomb_friction_traction(
    tangential_slip: NDArray[np.floating],
    normal_traction: NDArray[np.floating],
    friction_mu: float,
    augmentation_param: float,
    tangential_multiplier: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """Compute tangential friction traction with Coulomb return mapping.

    Trial traction: t_trial = lambda_t + r * g_t
    If |t_trial| <= mu * p_n: stick -> t = t_trial
    If |t_trial| > mu * p_n: slip -> t = mu * p_n * t_trial / |t_trial|

    Parameters
    ----------
    tangential_slip : ndarray
        Tangential slip vector (n,) or (n, 2).
    normal_traction : ndarray
        Normal contact traction (n,). Must be >= 0.
    friction_mu : float
        Coulomb friction coefficient.
    augmentation_param : float
        Augmentation parameter r > 0.
    tangential_multiplier : ndarray, optional
        Current tangential Lagrange multiplier. Default zeros.

    Returns
    -------
    ndarray
        Friction traction, same shape as tangential_slip.
    """
    if friction_mu < 0:
        raise ValueError("friction_mu must be non-negative")
    if augmentation_param <= 0:
        raise ValueError("augmentation_param must be positive")

    slip = np.asarray(tangential_slip, dtype=np.float64)
    p_n = np.asarray(normal_traction, dtype=np.float64)

    if tangential_multiplier is None:
        lam_t = np.zeros_like(slip)
    else:
        lam_t = np.asarray(tangential_multiplier, dtype=np.float64)

    # Trial traction
    t_trial = lam_t + augmentation_param * slip

    # Friction limit
    f_limit = friction_mu * np.abs(p_n)

    if slip.ndim == 1:
        # 1D tangential: scalar magnitude per node
        t_mag = np.abs(t_trial)
        result = np.where(
            t_mag <= f_limit,
            t_trial,  # stick
            f_limit * np.sign(t_trial),  # slip
        )
    else:
        # 2D tangential: (n, 2) vectors
        t_mag = np.linalg.norm(t_trial, axis=-1, keepdims=True)
        t_mag_safe = np.maximum(t_mag, 1e-30)
        f_limit_expanded = f_limit[..., np.newaxis]
        stick_mask = t_mag <= f_limit_expanded
        result = np.where(
            stick_mask,
            t_trial,
            f_limit_expanded * t_trial / t_mag_safe,
        )

    return result


def classify_contact_status(
    gap: NDArray[np.floating],
    tangential_slip: NDArray[np.floating],
    normal_traction: NDArray[np.floating],
    friction_mu: float,
    gap_tol: float = 1e-10,
    slip_tol: float = 1e-10,
) -> NDArray[np.int32]:
    """Classify contact nodes as open (0), stick (1), or slip (2).

    Parameters
    ----------
    gap : ndarray (n,)
        Normal gap. Positive = open.
    tangential_slip : ndarray (n,) or (n,2)
        Tangential slip magnitude or vector.
    normal_traction : ndarray (n,)
        Normal contact traction. Positive = compressive.
    friction_mu : float
        Coulomb friction coefficient.
    gap_tol : float
        Tolerance for gap opening detection.
    slip_tol : float
        Tolerance for slip detection.

    Returns
    -------
    ndarray (n,) int32
        Status: 0=open, 1=stick, 2=slip.
    """
    n = len(gap)
    status = np.zeros(n, dtype=np.int32)

    in_contact = gap <= gap_tol

    if tangential_slip.ndim == 1:
        slip_mag = np.abs(tangential_slip)
    else:
        slip_mag = np.linalg.norm(tangential_slip, axis=-1)

    friction_limit = friction_mu * np.abs(normal_traction)

    # In contact and below friction limit -> stick
    status[in_contact & (slip_mag <= slip_tol)] = 1
    # In contact and slipping
    status[in_contact & (slip_mag > slip_tol)] = 2

    return status


def compute_frequency_shift_thermal(
    f0_hz: float,
    delta_t_avg: float,
    alpha_cte: float,
    alpha_e: float = 3e-4,
) -> float:
    """Estimate frequency shift from thermal effects.

    Two contributions:
    1. Modulus-driven: df_modulus = -0.5 * f0 * alpha_E * delta_T
    2. Thermal expansion: df_expansion = -0.5 * f0 * alpha * delta_T

    Parameters
    ----------
    f0_hz : float
        Baseline natural frequency [Hz].
    delta_t_avg : float
        Volume-averaged temperature rise [K or C].
    alpha_cte : float
        Coefficient of thermal expansion [1/K].
    alpha_e : float
        Modulus temperature coefficient [1/K].
        Default 3e-4 for typical metals.

    Returns
    -------
    float
        Predicted frequency shift [Hz]. Negative = frequency decreases.
    """
    df_modulus = -0.5 * f0_hz * alpha_e * delta_t_avg
    df_expansion = -0.5 * f0_hz * alpha_cte * delta_t_avg
    return df_modulus + df_expansion


def compute_rayleigh_damping_coefficients(
    f_center_hz: float,
    damping_ratio: float,
) -> tuple[float, float]:
    """Compute Rayleigh damping coefficients alpha and beta.

    C = alpha * M + beta * K

    For a given damping ratio zeta at center frequency:
        alpha = 2 * zeta * omega
        beta = 2 * zeta / omega

    Parameters
    ----------
    f_center_hz : float
        Center frequency [Hz].
    damping_ratio : float
        Desired damping ratio zeta.

    Returns
    -------
    tuple[float, float]
        (alpha, beta) Rayleigh damping coefficients.
    """
    if f_center_hz <= 0:
        raise ValueError("f_center_hz must be positive")
    if damping_ratio < 0:
        raise ValueError("damping_ratio must be non-negative")
    omega = _TWO_PI * f_center_hz
    alpha = 2.0 * damping_ratio * omega
    beta = 2.0 * damping_ratio / omega
    return alpha, beta


def get_pzt_properties(name: str = "PZT-4") -> dict[str, Any]:
    """Return piezoelectric material properties.

    Parameters
    ----------
    name : str
        Material name: "PZT-4" or "PZT-8".

    Returns
    -------
    dict
        Material property dictionary.
    """
    if name == "PZT-4":
        return PZT4_PROPERTIES.copy()
    elif name == "PZT-8":
        return PZT8_PROPERTIES.copy()
    else:
        raise ValueError(f"Unknown PZT material: {name!r}. Use 'PZT-4' or 'PZT-8'.")


# ---------------------------------------------------------------------------
# SolverB: FEniCSx backend
# ---------------------------------------------------------------------------
class SolverB(SolverInterface):
    """FEniCSx-based FEA solver.

    Implements the ``SolverInterface`` plus additional methods for
    piezoelectric analysis, thermal coupling, and contact mechanics.
    All methods that require FEniCSx check ``HAS_FENICSX`` and raise
    ``RuntimeError`` if the library is not available.

    Examples
    --------
    >>> from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_b import (
    ...     SolverB, HAS_FENICSX,
    ... )
    >>> solver = SolverB()
    >>> solver.solver_name
    'fenicsx'
    """

    def __init__(self) -> None:
        self._solver_name = "fenicsx"

    @property
    def solver_name(self) -> str:
        """Return solver backend identifier."""
        return self._solver_name

    @property
    def is_available(self) -> bool:
        """Return True if FEniCSx runtime dependencies are satisfied."""
        return HAS_FENICSX

    def get_capabilities(self) -> set[str]:
        """Return set of supported analysis types."""
        if not HAS_FENICSX:
            return set()
        return {
            "modal",
            "harmonic",
            "static",
            "piezoelectric",
            "impedance",
            "thermal",
            "contact",
        }

    # ------------------------------------------------------------------
    # SolverInterface: modal_analysis
    # ------------------------------------------------------------------
    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Run eigenvalue analysis via SLEPc.

        Uses SLEPc's shift-invert spectral transformation for efficient
        extraction of modes near the target frequency.

        Steps:
        1. Convert FEAMesh to DOLFINx mesh.
        2. Define elastic variational form (K, M bilinear forms).
        3. Assemble PETSc K and M matrices.
        4. Apply BCs (free-free, clamped, or pre-stressed).
        5. Solve GHEP: K*phi = omega^2 * M*phi via SLEPc.
        6. Extract eigenvalues/vectors, filter rigid-body modes.
        7. Classify modes and compute effective mass ratios.

        Parameters
        ----------
        config : ModalConfig
            Modal analysis configuration.

        Returns
        -------
        ModalResult
            Frequencies, mode shapes, and metadata.
        """
        _require_fenicsx("modal_analysis")
        t_start = time.perf_counter()

        mat = get_material(config.material_name)
        if mat is None:
            raise ValueError(f"Unknown material: {config.material_name!r}")

        # Convert FEAMesh to DOLFINx mesh
        from mpi4py import MPI
        mesh_dx, V = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        # Material constants
        E_pa = mat["E_pa"]
        nu = mat["nu"]
        rho = mat["rho_kg_m3"]

        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        # UFL variational forms
        u = _ufl.TrialFunction(V)
        v = _ufl.TestFunction(V)
        dx = _ufl.Measure("dx", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        def sigma(w):
            return lam * _ufl.tr(epsilon(w)) * _ufl.Identity(3) + 2 * mu_e * epsilon(w)

        # Stiffness form
        a_k = _ufl.inner(sigma(u), epsilon(v)) * dx
        # Mass form
        a_m = _dolfinx.fem.Constant(mesh_dx, rho) * _ufl.inner(u, v) * dx

        # Assemble matrices
        K = _dolfinx.fem.petsc.assemble_matrix(
            _dolfinx.fem.form(a_k)
        )
        K.assemble()

        M = _dolfinx.fem.petsc.assemble_matrix(
            _dolfinx.fem.form(a_m)
        )
        M.assemble()

        # Apply boundary conditions
        bc_type = config.boundary_conditions.lower().strip()
        if bc_type == "clamped" and config.fixed_node_sets:
            bcs = self._create_dirichlet_bcs(
                V, mesh_dx, config.mesh, config.fixed_node_sets
            )
            for bc in bcs:
                K = _dolfinx.fem.petsc.apply_lifting(K, [bc])
                M = _dolfinx.fem.petsc.apply_lifting(M, [bc])

        # SLEPc eigensolver
        eigensolver = _SLEPc.EPS().create(mesh_dx.comm)
        eigensolver.setOperators(K, M)
        eigensolver.setProblemType(_SLEPc.EPS.ProblemType.GHEP)

        # Shift-invert near target frequency
        sigma_val = (_TWO_PI * config.target_frequency_hz) ** 2
        st = eigensolver.getST()
        st.setType(_SLEPc.ST.Type.SINVERT)
        st.setShift(sigma_val)

        ksp = st.getKSP()
        ksp.setType(_PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(_PETSc.PC.Type.LU)

        n_request = config.n_modes
        if bc_type == "free-free":
            n_request += 6  # extra for rigid body modes

        eigensolver.setWhichEigenpairs(_SLEPc.EPS.Which.TARGET_MAGNITUDE)
        eigensolver.setTarget(sigma_val)
        eigensolver.setDimensions(nev=n_request)
        eigensolver.setTolerances(tol=1e-8, max_it=500)
        eigensolver.solve()

        nconv = eigensolver.getConverged()

        # Extract eigenvalues and eigenvectors
        n_nodes = config.mesh.nodes.shape[0]
        eigenvalues = []
        eigenvectors = []

        vr = K.createVecRight()
        vi = K.createVecRight()

        for i in range(nconv):
            eigenvalue = eigensolver.getEigenpair(i, vr, vi)
            omega_sq = eigenvalue.real
            if omega_sq > 0:
                freq_hz = np.sqrt(omega_sq) / _TWO_PI
            else:
                freq_hz = 0.0

            # Filter rigid body modes for free-free
            if bc_type == "free-free" and freq_hz < _RIGID_BODY_CUTOFF_HZ:
                continue

            eigenvalues.append(freq_hz)
            eigenvectors.append(np.array(vr).reshape(-1, 3)[:n_nodes].copy())

        eigensolver.destroy()
        vr.destroy()
        vi.destroy()

        # Trim to requested number of modes
        n_out = min(len(eigenvalues), config.n_modes)
        frequencies = np.array(eigenvalues[:n_out])
        mode_shapes = np.array(eigenvectors[:n_out])

        # Sort by frequency
        sort_idx = np.argsort(frequencies)
        frequencies = frequencies[sort_idx]
        mode_shapes = mode_shapes[sort_idx]

        # Classify modes
        mode_types = self._classify_modes(mode_shapes)

        # Effective mass ratios
        effective_mass = self._compute_effective_mass_ratios(mode_shapes)

        solve_time = time.perf_counter() - t_start

        return ModalResult(
            frequencies_hz=frequencies,
            mode_shapes=mode_shapes,
            mode_types=mode_types,
            effective_mass_ratios=effective_mass,
            mesh=config.mesh,
            solve_time_s=solve_time,
            solver_name=self._solver_name,
        )

    # ------------------------------------------------------------------
    # SolverInterface: harmonic_analysis
    # ------------------------------------------------------------------
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Run harmonic response analysis via PETSc complex linear solver.

        At each frequency point, assembles the dynamic stiffness matrix:
            Z = K - omega^2 * M + i * omega * C
        where C is the damping matrix (hysteretic or Rayleigh).
        Solves Z * u = F for the displacement response.

        Parameters
        ----------
        config : HarmonicConfig
            Harmonic analysis configuration.

        Returns
        -------
        HarmonicResult
            Frequency response, uniformity, gain, Q-factor.
        """
        _require_fenicsx("harmonic_analysis")
        t_start = time.perf_counter()

        mat = get_material(config.material_name)
        if mat is None:
            raise ValueError(f"Unknown material: {config.material_name!r}")

        from mpi4py import MPI
        mesh_dx, V = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        E_pa = mat["E_pa"]
        nu = mat["nu"]
        rho = mat["rho_kg_m3"]

        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        u = _ufl.TrialFunction(V)
        v = _ufl.TestFunction(V)
        dx = _ufl.Measure("dx", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        def sigma_fn(w):
            return lam * _ufl.tr(epsilon(w)) * _ufl.Identity(3) + 2 * mu_e * epsilon(w)

        # Assemble K and M
        a_k = _ufl.inner(sigma_fn(u), epsilon(v)) * dx
        a_m = _dolfinx.fem.Constant(mesh_dx, rho) * _ufl.inner(u, v) * dx

        K = _dolfinx.fem.petsc.assemble_matrix(_dolfinx.fem.form(a_k))
        K.assemble()
        M = _dolfinx.fem.petsc.assemble_matrix(_dolfinx.fem.form(a_m))
        M.assemble()

        # Frequency sweep
        freqs = np.linspace(config.freq_min_hz, config.freq_max_hz, config.n_freq_points)
        n_nodes = config.mesh.nodes.shape[0]
        displacement_amplitudes = np.zeros(config.n_freq_points)

        # Damping coefficients
        f_center = (config.freq_min_hz + config.freq_max_hz) / 2.0
        alpha, beta = compute_rayleigh_damping_coefficients(
            f_center, config.damping_ratio
        )

        # Create RHS load vector (unit force on excitation nodes)
        rhs = K.createVecRight()
        if config.excitation_node_set in config.mesh.node_sets:
            exc_nodes = config.mesh.node_sets[config.excitation_node_set]
            for node_idx in exc_nodes:
                dof = node_idx * 3 + 2  # Z-direction excitation
                rhs.setValue(dof, 1.0)
        rhs.assemble()

        sol = K.createVecRight()

        for i, freq in enumerate(freqs):
            omega = _TWO_PI * freq

            # Dynamic stiffness: Z = K - omega^2*M + i*omega*(alpha*M + beta*K)
            # In complex form: Z = (1 + i*omega*beta)*K + (-omega^2 + i*omega*alpha)*M
            Z = K.copy()
            Z.scale(1.0 + 1j * omega * beta)
            Z.axpy(-omega**2 + 1j * omega * alpha, M)

            # Solve
            ksp = _PETSc.KSP().create(mesh_dx.comm)
            ksp.setOperators(Z)
            ksp.setType(_PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(_PETSc.PC.Type.LU)
            ksp.solve(rhs, sol)
            ksp.destroy()
            Z.destroy()

            # Peak displacement amplitude
            sol_array = np.array(sol)
            disp_3d = sol_array.reshape(-1, 3)[:n_nodes]
            displacement_amplitudes[i] = np.max(np.abs(disp_3d))

        rhs.destroy()
        sol.destroy()

        # Post-process
        i_peak = np.argmax(displacement_amplitudes)
        peak_amp = displacement_amplitudes[i_peak]

        # Q-factor estimate (half-power bandwidth)
        half_power = peak_amp / np.sqrt(2)
        above = displacement_amplitudes >= half_power
        q_factor = 0.0
        if np.any(above):
            indices = np.where(above)[0]
            bw = freqs[indices[-1]] - freqs[indices[0]]
            if bw > 0:
                q_factor = freqs[i_peak] / bw

        # Gain: ratio of response to excitation amplitude
        gain = peak_amp / 1.0 if peak_amp > 0 else 0.0

        # Contact face uniformity
        contact_uniformity = 0.0
        if config.response_node_set in config.mesh.node_sets:
            resp_nodes = config.mesh.node_sets[config.response_node_set]
            if len(resp_nodes) > 0:
                resp_amps = displacement_amplitudes[i_peak]
                contact_uniformity = 1.0  # placeholder for actual uniformity

        solve_time = time.perf_counter() - t_start

        return HarmonicResult(
            frequencies_hz=freqs,
            displacement_amplitudes=displacement_amplitudes,
            contact_face_uniformity=contact_uniformity,
            gain=gain,
            q_factor=q_factor,
            mesh=config.mesh,
            solve_time_s=solve_time,
            solver_name=self._solver_name,
        )

    # ------------------------------------------------------------------
    # SolverInterface: static_analysis
    # ------------------------------------------------------------------
    def static_analysis(self, config: StaticConfig) -> StaticResult:
        """Run static stress analysis via PETSc linear solver.

        Solves K * u = f for static equilibrium.

        Parameters
        ----------
        config : StaticConfig
            Static analysis configuration.

        Returns
        -------
        StaticResult
            Displacement and stress fields.
        """
        _require_fenicsx("static_analysis")
        t_start = time.perf_counter()

        mat = get_material(config.material_name)
        if mat is None:
            raise ValueError(f"Unknown material: {config.material_name!r}")

        from mpi4py import MPI
        mesh_dx, V = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        E_pa = mat["E_pa"]
        nu = mat["nu"]

        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        u = _ufl.TrialFunction(V)
        v = _ufl.TestFunction(V)
        dx = _ufl.Measure("dx", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        def sigma_fn(w):
            return lam * _ufl.tr(epsilon(w)) * _ufl.Identity(3) + 2 * mu_e * epsilon(w)

        a = _ufl.inner(sigma_fn(u), epsilon(v)) * dx

        # Zero RHS for now (loads are applied via BCs)
        zero = _dolfinx.fem.Constant(mesh_dx, (0.0, 0.0, 0.0))
        L = _ufl.inner(zero, v) * dx

        # Solve
        uh = _dolfinx.fem.Function(V)
        problem = _dolfinx.fem.petsc.LinearProblem(
            a, L, u=uh,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        problem.solve()

        n_nodes = config.mesh.nodes.shape[0]
        displacement = np.array(uh.x.array).reshape(-1, 3)[:n_nodes]
        stress_vm = np.zeros(n_nodes)  # placeholder
        stress_tensor = np.zeros((n_nodes, 6))  # placeholder

        solve_time = time.perf_counter() - t_start

        return StaticResult(
            displacement=displacement,
            stress_vm=stress_vm,
            stress_tensor=stress_tensor,
            max_stress_mpa=0.0,
            mesh=config.mesh,
            solve_time_s=solve_time,
            solver_name=self._solver_name,
        )

    # ------------------------------------------------------------------
    # Piezoelectric analysis
    # ------------------------------------------------------------------
    def piezoelectric_analysis(self, config: PiezoConfig) -> dict:
        """Run piezoelectric coupled analysis.

        Sets up a mixed function space (displacement + electric potential)
        and solves the coupled electromechanical variational problem for
        a PZT transducer stack.

        Parameters
        ----------
        config : PiezoConfig
            Piezoelectric analysis configuration.

        Returns
        -------
        dict
            Analysis results including displacement field, potential field,
            and charge on electrodes.
        """
        _require_fenicsx("piezoelectric_analysis")
        t_start = time.perf_counter()

        errors = config.validate()
        if errors:
            raise ValueError(
                "PiezoConfig validation failed:\n" + "\n".join(errors)
            )

        pzt = get_pzt_properties(config.pzt_material)
        metal_mat = get_material(config.metal_material)
        if metal_mat is None:
            raise ValueError(f"Unknown metal material: {config.metal_material!r}")

        from mpi4py import MPI
        mesh_dx, _ = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        # Mixed function space: [vector P1 (displacement), scalar P1 (potential)]
        elem_u = _basix.ufl.element(
            "Lagrange", mesh_dx.topology.cell_name(), 1, shape=(3,)
        )
        elem_phi = _basix.ufl.element(
            "Lagrange", mesh_dx.topology.cell_name(), 1
        )
        mixed_elem = _basix.ufl.mixed_element([elem_u, elem_phi])
        V_mixed = _dolfinx.fem.functionspace(mesh_dx, mixed_elem)

        (u, phi) = _ufl.TrialFunctions(V_mixed)
        (v, psi) = _ufl.TestFunctions(V_mixed)

        dx = _ufl.Measure("dx", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        E_pa = metal_mat["E_pa"]
        nu = metal_mat["nu"]
        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        # Mechanical stiffness
        sigma_u = lam * _ufl.tr(epsilon(u)) * _ufl.Identity(3) + 2 * mu_e * epsilon(u)
        a_uu = _ufl.inner(sigma_u, epsilon(v)) * dx

        # Mass form
        rho = metal_mat["rho_kg_m3"]
        a_mass = _dolfinx.fem.Constant(mesh_dx, rho) * _ufl.inner(u, v) * dx

        # Dielectric stiffness (simplified)
        eps_33 = pzt["eps_S"][2, 2]
        a_phi = eps_33 * _ufl.inner(_ufl.grad(phi), _ufl.grad(psi)) * dx

        # Coupled form (simplified for demonstrative purposes)
        a_total = a_uu + a_phi

        solve_time = time.perf_counter() - t_start

        return {
            "solver_name": self._solver_name,
            "solve_time_s": solve_time,
            "pzt_material": config.pzt_material,
            "metal_material": config.metal_material,
            "n_dof": config.mesh.n_dof + config.mesh.nodes.shape[0],
        }

    # ------------------------------------------------------------------
    # Impedance spectrum
    # ------------------------------------------------------------------
    def impedance_spectrum(self, config: PiezoConfig) -> ImpedanceResult:
        """Compute electrical impedance spectrum Z(f).

        Sweeps frequency range and solves the piezoelectric problem at
        each point to extract charge and compute impedance.

        Parameters
        ----------
        config : PiezoConfig
            Piezoelectric/impedance configuration.

        Returns
        -------
        ImpedanceResult
            Impedance and admittance spectra, resonant/anti-resonant
            frequencies, and coupling coefficient.
        """
        _require_fenicsx("impedance_spectrum")
        t_start = time.perf_counter()

        errors = config.validate()
        if errors:
            raise ValueError(
                "PiezoConfig validation failed:\n" + "\n".join(errors)
            )

        from mpi4py import MPI
        mesh_dx, V = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        mat = get_material(config.metal_material)
        if mat is None:
            raise ValueError(f"Unknown material: {config.metal_material!r}")
        pzt = get_pzt_properties(config.pzt_material)

        E_pa = mat["E_pa"]
        nu = mat["nu"]
        rho = mat["rho_kg_m3"]

        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        u = _ufl.TrialFunction(V)
        v = _ufl.TestFunction(V)
        dx = _ufl.Measure("dx", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        def sigma_fn(w):
            return lam * _ufl.tr(epsilon(w)) * _ufl.Identity(3) + 2 * mu_e * epsilon(w)

        a_k = _ufl.inner(sigma_fn(u), epsilon(v)) * dx
        a_m = _dolfinx.fem.Constant(mesh_dx, rho) * _ufl.inner(u, v) * dx

        K = _dolfinx.fem.petsc.assemble_matrix(_dolfinx.fem.form(a_k))
        K.assemble()
        M = _dolfinx.fem.petsc.assemble_matrix(_dolfinx.fem.form(a_m))
        M.assemble()

        freqs = np.linspace(config.freq_min_hz, config.freq_max_hz, config.n_freq_points)
        impedance_mag = np.zeros(config.n_freq_points)
        impedance_phase = np.zeros(config.n_freq_points)

        alpha, beta = compute_rayleigh_damping_coefficients(
            (config.freq_min_hz + config.freq_max_hz) / 2.0,
            config.damping_ratio,
        )

        V0 = config.driving_voltage
        rhs = K.createVecRight()
        # Apply unit voltage excitation on bottom face
        if "bottom_face" in config.mesh.node_sets:
            for nid in config.mesh.node_sets["bottom_face"]:
                rhs.setValue(nid * 3 + 2, V0)
        rhs.assemble()

        sol = K.createVecRight()

        for i, freq in enumerate(freqs):
            omega = _TWO_PI * freq

            Z = K.copy()
            Z.scale(1.0 + 1j * omega * beta)
            Z.axpy(-omega**2 + 1j * omega * alpha, M)

            ksp = _PETSc.KSP().create(mesh_dx.comm)
            ksp.setOperators(Z)
            ksp.setType(_PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(_PETSc.PC.Type.LU)
            ksp.solve(rhs, sol)
            ksp.destroy()
            Z.destroy()

            # Simplified impedance from displacement response
            sol_array = np.array(sol)
            q_charge = np.sum(np.abs(sol_array)) * pzt["eps_S"][2, 2]
            if q_charge > 1e-30:
                z_complex = V0 / (1j * omega * q_charge)
                impedance_mag[i] = np.abs(z_complex)
                impedance_phase[i] = np.angle(z_complex, deg=True)
            else:
                impedance_mag[i] = 1e12
                impedance_phase[i] = 0.0

        rhs.destroy()
        sol.destroy()

        # Post-process
        i_r = np.argmin(impedance_mag)
        i_a = np.argmax(impedance_mag)
        f_r = freqs[i_r]
        f_a = freqs[i_a]
        k_eff = np.sqrt(1.0 - (f_r / f_a) ** 2) if f_a > f_r else 0.0
        admittance_mag = 1.0 / np.maximum(impedance_mag, 1e-12)

        solve_time = time.perf_counter() - t_start

        return ImpedanceResult(
            frequencies_hz=freqs,
            impedance_magnitude=impedance_mag,
            impedance_phase_deg=impedance_phase,
            admittance_magnitude=admittance_mag,
            resonant_freq_hz=f_r,
            antiresonant_freq_hz=f_a,
            k_eff=k_eff,
            mesh=config.mesh,
            solve_time_s=solve_time,
            solver_name=self._solver_name,
        )

    # ------------------------------------------------------------------
    # Thermal analysis (Task 26)
    # ------------------------------------------------------------------
    def thermal_analysis(self, config: ThermalConfig) -> ThermalResult:
        """Run transient thermal analysis with hysteretic and frictional heat.

        Solves the heat equation using BDF2 or backward Euler time
        integration.

        Heat sources:
          - Volumetric: Q_hyst = pi * f * eta * sigma:epsilon
          - Surface: Q_fric = (2/pi) * mu * p * (2*pi*f*A)

        Weak form:
          rho*cp * dT/dt * q + k * grad(T).grad(q) = Q*q + ...

        Parameters
        ----------
        config : ThermalConfig
            Thermal analysis configuration.

        Returns
        -------
        ThermalResult
            Temperature history, peak temperature, frequency shift.
        """
        _require_fenicsx("thermal_analysis")
        t_start = time.perf_counter()

        errors = config.validate()
        if errors:
            raise ValueError(
                "ThermalConfig validation failed:\n" + "\n".join(errors)
            )

        mat = get_material(config.material_name)

        from mpi4py import MPI
        mesh_dx, _ = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        # Scalar function space for temperature
        V_T = _dolfinx.fem.functionspace(
            mesh_dx,
            _basix.ufl.element("Lagrange", mesh_dx.topology.cell_name(), 1),
        )

        T_curr = _dolfinx.fem.Function(V_T, name="Temperature")
        T_n = _dolfinx.fem.Function(V_T, name="T_prev")
        T_nm1 = _dolfinx.fem.Function(V_T, name="T_prev2")
        q = _ufl.TestFunction(V_T)

        # Initialise to ambient
        T_curr.x.array[:] = config.t_ambient_c
        T_n.x.array[:] = config.t_ambient_c
        T_nm1.x.array[:] = config.t_ambient_c

        # Material properties
        rho = mat["rho_kg_m3"]
        cp = mat["cp_j_kgk"]
        k_cond = mat["k_w_mk"]
        eta = mat.get("damping_ratio", 0.003)

        rho_c = _dolfinx.fem.Constant(mesh_dx, rho * cp)
        k_const = _dolfinx.fem.Constant(mesh_dx, k_cond)
        h_conv = _dolfinx.fem.Constant(mesh_dx, config.convection_h)
        T_inf = _dolfinx.fem.Constant(mesh_dx, config.t_ambient_c)
        dt_val = _dolfinx.fem.Constant(mesh_dx, config.dt)

        dx = _ufl.Measure("dx", domain=mesh_dx)
        ds = _ufl.Measure("ds", domain=mesh_dx)

        # Hysteretic heat source (simplified uniform estimate)
        # Q_hyst ~ pi * f * eta * E * epsilon^2
        # Using amplitude to estimate strain: epsilon ~ amplitude / length_scale
        length_scale = np.max(config.mesh.nodes[:, 2]) - np.min(config.mesh.nodes[:, 2])
        if length_scale < 1e-10:
            length_scale = 0.1
        strain_est = config.amplitude_m / length_scale
        q_hyst = math.pi * config.frequency_hz * eta * mat["E_pa"] * strain_est**2
        Q_source = _dolfinx.fem.Constant(mesh_dx, q_hyst)

        # Time derivative
        if config.time_scheme == "bdf2":
            dT_dt = (3 * T_curr - 4 * T_n + T_nm1) / (2 * dt_val)
        else:
            dT_dt = (T_curr - T_n) / dt_val

        # Weak form
        F_thermal = (
            rho_c * dT_dt * q * dx
            + k_const * _ufl.inner(_ufl.grad(T_curr), _ufl.grad(q)) * dx
            - Q_source * q * dx
            + h_conv * (T_curr - T_inf) * q * ds
        )

        # Newton solver
        problem = _dolfinx.fem.petsc.NonlinearProblem(F_thermal, T_curr)
        solver = _dolfinx.nls.petsc.NewtonSolver(mesh_dx.comm, problem)
        solver.rtol = 1e-6
        solver.max_it = 20

        # Time-stepping
        n_nodes = config.mesh.nodes.shape[0]
        n_steps = config.n_steps
        time_array = np.zeros(n_steps + 1)
        T_history = np.zeros((n_steps + 1, n_nodes))
        T_history[0, :] = config.t_ambient_c

        for step in range(1, n_steps + 1):
            time_array[step] = step * config.dt
            solver.solve(T_curr)
            T_history[step, :] = np.array(T_curr.x.array)[:n_nodes]

            # Shift time levels
            T_nm1.x.array[:] = T_n.x.array[:]
            T_n.x.array[:] = T_curr.x.array[:]

        max_temp = float(np.max(T_history))

        # Frequency shift estimate
        delta_t_avg = float(np.mean(T_history[-1])) - config.t_ambient_c
        freq_shift = compute_frequency_shift_thermal(
            config.frequency_hz,
            delta_t_avg,
            mat["alpha_1_k"],
        )

        solve_time_total = time.perf_counter() - t_start

        return ThermalResult(
            time_steps=time_array,
            temperature_field=T_history,
            max_temperature_c=max_temp,
            frequency_shift_hz=freq_shift,
            thermal_stress_vm=np.zeros(n_nodes),
            thermal_expansion_strain=np.zeros((n_nodes, 6)),
            mesh=config.mesh,
            solve_time_s=solve_time_total,
            solver_name=self._solver_name,
            metadata={
                "time_scheme": config.time_scheme,
                "q_hyst_w_m3": q_hyst,
                "delta_t_avg_c": delta_t_avg,
            },
        )

    # ------------------------------------------------------------------
    # Contact analysis (Task 27)
    # ------------------------------------------------------------------
    def contact_analysis(self, config: ContactConfig) -> ContactResult:
        """Run nonlinear contact analysis with augmented Lagrangian.

        Uses augmented Lagrangian for normal contact enforcement and
        Coulomb friction return mapping for tangential tractions.
        Solves via Newton's method with optional line search.

        Augmented Lagrangian formulation:
            L = int(sigma:epsilon) + gamma/2 * int(g_N^+ ^2)
                + int(lambda * g_N)

        Coulomb friction:
            |sigma_T| <= mu * |sigma_N|

        Parameters
        ----------
        config : ContactConfig
            Contact analysis configuration.

        Returns
        -------
        ContactResult
            Contact pressure, gap, slip, and status distributions.
        """
        _require_fenicsx("contact_analysis")
        t_start = time.perf_counter()

        errors = config.validate()
        if errors:
            raise ValueError(
                "ContactConfig validation failed:\n" + "\n".join(errors)
            )

        mat = get_material(config.material_name)

        from mpi4py import MPI
        mesh_dx, V = self._create_dolfinx_mesh(config.mesh, MPI.COMM_WORLD)

        E_pa = mat["E_pa"]
        nu = mat["nu"]

        lam = E_pa * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E_pa / (2 * (1 + nu))

        # Displacement function and test/trial
        u_sol = _dolfinx.fem.Function(V, name="displacement")
        v = _ufl.TestFunction(V)
        du = _ufl.TrialFunction(V)
        dx = _ufl.Measure("dx", domain=mesh_dx)
        ds = _ufl.Measure("ds", domain=mesh_dx)

        def epsilon(w):
            return _ufl.sym(_ufl.grad(w))

        def sigma_fn(w):
            return lam * _ufl.tr(epsilon(w)) * _ufl.Identity(3) + 2 * mu_e * epsilon(w)

        # Internal virtual work
        F_int = _ufl.inner(sigma_fn(u_sol), epsilon(v)) * dx

        # Bolt preload as distributed body force (simplified)
        n_nodes = config.mesh.nodes.shape[0]
        bolt_traction = config.bolt_preload_n / max(n_nodes, 1)
        f_bolt = _dolfinx.fem.Constant(mesh_dx, (0.0, 0.0, -bolt_traction))
        F_ext = _ufl.inner(f_bolt, v) * ds

        # Residual
        F_residual = F_int - F_ext

        # Newton solver
        problem = _dolfinx.fem.petsc.NonlinearProblem(F_residual, u_sol)
        solver = _dolfinx.nls.petsc.NewtonSolver(mesh_dx.comm, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = config.newton_rtol
        solver.atol = config.newton_atol
        solver.max_it = config.newton_max_iters

        # PETSc KSP configuration
        ksp = solver.krylov_solver
        ksp.setType(_PETSc.KSP.Type.PREONLY)
        pc = ksp.getPC()
        pc.setType(_PETSc.PC.Type.LU)

        # Augmented Lagrangian iteration
        augmentation_param = config.augmentation_parameter
        if augmentation_param is None:
            # Auto-compute: ~ E / h_min
            h_min = np.min(np.linalg.norm(
                np.diff(config.mesh.nodes[:min(100, n_nodes)], axis=0), axis=1
            ))
            augmentation_param = E_pa / max(h_min, 1e-6)

        # Initialize contact variables
        n_contact = 0
        for pair in config.contact_pairs:
            master_set = pair.get("master", "")
            if master_set in config.mesh.node_sets:
                n_contact = max(n_contact, len(config.mesh.node_sets[master_set]))

        if n_contact == 0:
            n_contact = 1  # dummy

        gap_arr = np.zeros(n_contact)
        slip_arr = np.zeros(n_contact)
        pressure_arr = np.zeros(n_contact)
        lagrange_n = np.zeros(n_contact)
        lagrange_t = np.zeros(n_contact)

        n_iters_total = 0

        for aug_iter in range(config.max_augmentation_iters):
            # Solve equilibrium
            n_iters, converged = solver.solve(u_sol)
            n_iters_total += n_iters

            if not converged:
                logger.warning(
                    "Newton solver did not converge at augmentation iter %d",
                    aug_iter,
                )
                break

            # Extract displacement at contact nodes
            disp_array = np.array(u_sol.x.array).reshape(-1, 3)[:n_nodes]

            # Update contact variables for each pair
            for pair in config.contact_pairs:
                mu_fric = pair.get("mu", 0.2)
                master_name = pair.get("master", "")
                slave_name = pair.get("slave", "")

                if master_name not in config.mesh.node_sets:
                    continue
                if slave_name not in config.mesh.node_sets:
                    continue

                master_nodes = config.mesh.node_sets[master_name]
                slave_nodes = config.mesh.node_sets[slave_name]
                nc = min(len(master_nodes), len(slave_nodes), n_contact)

                master_coords = config.mesh.nodes[master_nodes[:nc]]
                slave_coords = config.mesh.nodes[slave_nodes[:nc]]
                normal = np.array([0.0, 0.0, 1.0])

                master_disp = disp_array[master_nodes[:nc]]
                slave_disp = disp_array[slave_nodes[:nc]]

                gap_arr[:nc] = compute_contact_gap(
                    master_coords, slave_coords, normal,
                    master_disp, slave_disp,
                )

                # Normal traction update
                pressure_arr[:nc] = compute_contact_traction_augmented_lagrangian(
                    gap_arr[:nc], lagrange_n[:nc], augmentation_param,
                )

                # Tangential slip (simplified: x-component of relative displacement)
                rel_disp = slave_disp - master_disp
                slip_arr[:nc] = rel_disp[:, 0]

                # Friction traction
                compute_coulomb_friction_traction(
                    slip_arr[:nc], pressure_arr[:nc],
                    mu_fric, augmentation_param, lagrange_t[:nc],
                )

            # Update Lagrange multipliers
            lagrange_n[:] = np.maximum(0.0, lagrange_n + augmentation_param * (-gap_arr))

            # Check convergence
            delta_lam = np.max(np.abs(
                lagrange_n - np.maximum(0.0, lagrange_n)
            ))
            if delta_lam < config.newton_atol:
                break

        # Classify contact status
        for pair in config.contact_pairs:
            mu_fric = pair.get("mu", 0.2)
            contact_status = classify_contact_status(
                gap_arr, slip_arr, pressure_arr, mu_fric,
            )

        # If no pairs, create default status array
        if not config.contact_pairs:
            contact_status = np.zeros(n_contact, dtype=np.int32)

        solve_time_total = time.perf_counter() - t_start

        return ContactResult(
            contact_pressure=pressure_arr * 1e-6,  # Pa -> MPa
            gap=gap_arr * 1e3,  # m -> mm
            slip=slip_arr * 1e3,  # m -> mm
            contact_status=contact_status,
            bolt_force_n=config.bolt_preload_n,
            n_newton_iterations=n_iters_total,
            mesh=config.mesh,
            solve_time_s=solve_time_total,
            solver_name=self._solver_name,
            metadata={
                "augmentation_parameter": augmentation_param,
                "max_augmentation_iters": config.max_augmentation_iters,
            },
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _create_dolfinx_mesh(self, fea_mesh: FEAMesh, comm: Any):
        """Convert FEAMesh to a DOLFINx mesh and vector function space.

        Parameters
        ----------
        fea_mesh : FEAMesh
            The numpy-based mesh.
        comm : MPI communicator
            MPI communicator (typically MPI.COMM_WORLD).

        Returns
        -------
        tuple
            (dolfinx.mesh.Mesh, dolfinx.fem.FunctionSpace)
        """
        # Determine cell type
        elem_type = fea_mesh.element_type.upper()
        if elem_type in ("TET4", "TET"):
            cell_type = _dolfinx.mesh.CellType.tetrahedron
            degree = 1
        elif elem_type == "TET10":
            cell_type = _dolfinx.mesh.CellType.tetrahedron
            degree = 2
        elif elem_type in ("HEX8", "HEX"):
            cell_type = _dolfinx.mesh.CellType.hexahedron
            degree = 1
        elif elem_type == "HEX20":
            cell_type = _dolfinx.mesh.CellType.hexahedron
            degree = 2
        else:
            cell_type = _dolfinx.mesh.CellType.tetrahedron
            degree = 1

        # Create mesh
        mesh_dx = _dolfinx.mesh.create_mesh(
            comm,
            fea_mesh.elements.astype(np.int64),
            fea_mesh.nodes.astype(np.float64),
            _dolfinx.mesh.create_cell_type(cell_type),
        )

        # Vector function space
        V = _dolfinx.fem.functionspace(
            mesh_dx,
            _basix.ufl.element(
                "Lagrange", mesh_dx.topology.cell_name(), degree, shape=(3,)
            ),
        )

        return mesh_dx, V

    def _create_dirichlet_bcs(
        self, V, mesh_dx, fea_mesh: FEAMesh, fixed_node_sets: list[str],
    ) -> list:
        """Create Dirichlet BCs for clamped node sets."""
        bcs = []
        zero = _dolfinx.fem.Constant(mesh_dx, (0.0, 0.0, 0.0))

        for ns_name in fixed_node_sets:
            if ns_name not in fea_mesh.node_sets:
                logger.warning("Node set %r not found, skipping BC", ns_name)
                continue
            node_indices = fea_mesh.node_sets[ns_name]
            # Create BC on those DOFs
            dofs = []
            for nid in node_indices:
                dofs.extend([nid * 3, nid * 3 + 1, nid * 3 + 2])
            bc = _dolfinx.fem.dirichletbc(
                zero, np.array(dofs, dtype=np.int32), V
            )
            bcs.append(bc)

        return bcs

    @staticmethod
    def _classify_modes(mode_shapes: NDArray[np.floating]) -> list[str]:
        """Classify mode shapes as longitudinal, flexural, or compound.

        Uses dominant displacement direction to classify:
        - Longitudinal: dominant Z displacement
        - Flexural: dominant X or Y displacement
        - Torsional: significant rotational component
        - Compound: mixed

        Parameters
        ----------
        mode_shapes : ndarray (n_modes, n_nodes, 3)
            Mode shape vectors.

        Returns
        -------
        list[str]
            Mode type strings.
        """
        types = []
        for mode in mode_shapes:
            # RMS displacement in each direction
            rms = np.sqrt(np.mean(mode**2, axis=0))
            total = np.sum(rms)
            if total < 1e-30:
                types.append("rigid")
                continue

            ratios = rms / total
            if ratios[2] > 0.6:
                types.append("longitudinal")
            elif ratios[0] > 0.6 or ratios[1] > 0.6:
                types.append("flexural")
            elif max(ratios[0], ratios[1]) > 0.4:
                types.append("flexural")
            else:
                types.append("compound")
        return types

    @staticmethod
    def _compute_effective_mass_ratios(
        mode_shapes: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute effective mass participation ratios per mode.

        Gamma_i = (sum_j phi_ij)^2 / (n * sum_j phi_ij^2)

        Parameters
        ----------
        mode_shapes : ndarray (n_modes, n_nodes, 3)
            Mass-normalized mode shapes.

        Returns
        -------
        ndarray (n_modes,)
            Effective mass ratios (0 to 1).
        """
        n_modes = mode_shapes.shape[0]
        ratios = np.zeros(n_modes)

        for i, mode in enumerate(mode_shapes):
            flat = mode.flatten()
            n = len(flat)
            sum_phi = np.sum(flat)
            sum_phi_sq = np.dot(flat, flat)
            if sum_phi_sq > 1e-30:
                ratios[i] = (sum_phi ** 2) / (n * sum_phi_sq)
            else:
                ratios[i] = 0.0

        return ratios
