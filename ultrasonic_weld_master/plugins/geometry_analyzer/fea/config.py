"""FEA configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FEAMesh:
    """Container for a finite element mesh."""
    nodes: np.ndarray              # (N, 3) coordinates in meters
    elements: np.ndarray           # (E, nodes_per_elem) connectivity
    element_type: str              # "TET4", "TET10", "HEX8", "HEX20"
    node_sets: dict[str, np.ndarray]
    element_sets: dict[str, np.ndarray]
    surface_tris: np.ndarray       # (F, 3) for visualization
    mesh_stats: dict

    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (3 per node)."""
        return self.nodes.shape[0] * 3


@dataclass
class ModalConfig:
    """Configuration for modal (eigenvalue) analysis."""
    mesh: FEAMesh
    material_name: str
    n_modes: int = 20
    target_frequency_hz: float = 20000.0
    boundary_conditions: str = "free-free"
    fixed_node_sets: list[str] = field(default_factory=list)


@dataclass
class HarmonicConfig:
    """Configuration for harmonic response analysis."""
    mesh: FEAMesh
    material_name: str
    freq_min_hz: float = 16000.0
    freq_max_hz: float = 24000.0
    n_freq_points: int = 201
    damping_model: str = "hysteretic"
    damping_ratio: float = 0.005
    excitation_node_set: str = "bottom_face"
    response_node_set: str = "top_face"


@dataclass
class StaticConfig:
    """Configuration for static stress analysis."""
    mesh: FEAMesh
    material_name: str
    loads: list[dict] = field(default_factory=list)
    boundary_conditions: list[dict] = field(default_factory=list)
