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
