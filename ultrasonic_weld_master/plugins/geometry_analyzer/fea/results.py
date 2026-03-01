"""FEA result container dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModalResult:
    """Modal analysis result."""
    frequencies_hz: np.ndarray
    mode_shapes: np.ndarray
    mode_types: list[str]
    effective_mass_ratios: np.ndarray
    mesh: Optional[object]
    solve_time_s: float
    solver_name: str


@dataclass
class HarmonicResult:
    """Harmonic response result."""
    frequencies_hz: np.ndarray
    displacement_amplitudes: np.ndarray
    contact_face_uniformity: float
    gain: float
    q_factor: float
    mesh: Optional[object]
    solve_time_s: float
    solver_name: str


@dataclass
class StaticResult:
    """Static analysis result."""
    displacement: np.ndarray
    stress_vm: np.ndarray
    stress_tensor: np.ndarray
    max_stress_mpa: float
    mesh: Optional[object]
    solve_time_s: float
    solver_name: str


@dataclass
class FatigueResult:
    """Fatigue assessment result."""
    safety_factors: np.ndarray
    min_safety_factor: float
    critical_location: np.ndarray
    estimated_life_cycles: float
    sn_curve_name: str


@dataclass
class ThermalResult:
    """Transient thermal analysis result.

    Stores temperature history, peak temperature, thermal stress,
    and predicted frequency shift from thermal effects.
    """
    time_steps: np.ndarray                 # (n_steps,) time in seconds
    temperature_field: np.ndarray          # (n_steps, n_nodes) in Celsius
    max_temperature_c: float               # peak temperature anywhere
    frequency_shift_hz: float              # predicted shift from thermal effects
    thermal_stress_vm: np.ndarray          # (n_nodes,) Von Mises at final time
    thermal_expansion_strain: np.ndarray   # (n_nodes, 6) Voigt at final time
    mesh: Optional[object] = None
    solve_time_s: float = 0.0
    solver_name: str = "fenicsx"
    metadata: dict = field(default_factory=dict)


@dataclass
class ContactResult:
    """Nonlinear contact analysis result.

    Stores contact pressure, gap, and slip distributions, plus
    optionally the pre-stressed modal frequencies.
    """
    contact_pressure: np.ndarray           # (n_contact_nodes,) in MPa
    gap: np.ndarray                        # (n_contact_nodes,) in mm (>0 = open)
    slip: np.ndarray                       # (n_contact_nodes,) tangential slip in mm
    contact_status: np.ndarray             # (n_contact_nodes,) 0=open, 1=stick, 2=slip
    bolt_force_n: float                    # resultant bolt preload
    n_newton_iterations: int = 0           # Newton iterations to convergence
    stressed_frequencies_hz: Optional[np.ndarray] = None
    stressed_mode_shapes: Optional[np.ndarray] = None
    mesh: Optional[object] = None
    solve_time_s: float = 0.0
    solver_name: str = "fenicsx"
    metadata: dict = field(default_factory=dict)


@dataclass
class ImpedanceResult:
    """Electrical impedance spectrum from piezoelectric analysis."""
    frequencies_hz: np.ndarray             # (n_freq,) sweep frequencies
    impedance_magnitude: np.ndarray        # (n_freq,) |Z| in Ohms
    impedance_phase_deg: np.ndarray        # (n_freq,) phase in degrees
    admittance_magnitude: np.ndarray       # (n_freq,) |Y| in Siemens
    resonant_freq_hz: float                # f_r (minimum impedance)
    antiresonant_freq_hz: float            # f_a (maximum impedance)
    k_eff: float                           # electromechanical coupling coefficient
    mesh: Optional[object] = None
    solve_time_s: float = 0.0
    solver_name: str = "fenicsx"
    metadata: dict = field(default_factory=dict)
