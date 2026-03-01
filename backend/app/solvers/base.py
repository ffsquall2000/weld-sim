"""Solver abstraction layer -- uniform interface for all FEA backends.

Provides a common API that can be implemented by different solver engines:
  - PreviewSolver  (numpy/scipy, bundled)
  - FEniCSSolver   (FEniCS / dolfinx, optional)
  - ElmerSolver    (Elmer FEM, optional)
  - CalculiXSolver (CalculiX, optional)

Each backend converts a :class:`SolverConfig` into solver-specific input,
runs the analysis, and returns a uniform :class:`SolverResult` /
:class:`FieldData` that the frontend can consume directly.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Analysis type enumeration
# ---------------------------------------------------------------------------

class AnalysisType(str, Enum):
    """Supported FEA analysis types."""

    MODAL = "modal"
    HARMONIC = "harmonic"
    STATIC_STRUCTURAL = "static_structural"
    THERMAL_STEADY = "thermal_steady"
    THERMAL_TRANSIENT = "thermal_transient"
    PIEZOELECTRIC = "piezoelectric"
    ACOUSTIC = "acoustic"
    COUPLED_THERMO_STRUCTURAL = "coupled_thermo_structural"


# ---------------------------------------------------------------------------
# Input data classes
# ---------------------------------------------------------------------------

@dataclass
class MaterialAssignment:
    """Assign a material (with physical properties) to a mesh region."""

    region_id: str
    material_name: str
    properties: dict[str, float]
    # Expected keys: E_pa, nu, rho_kg_m3, k_w_mk, cp_j_kgk,
    #                yield_mpa, alpha_1_k  (see material_properties.py)


@dataclass
class BoundaryCondition:
    """A single boundary condition applied to a mesh region."""

    bc_type: str  # fixed, force, pressure, temperature, displacement, contact
    region: str   # face / node-set identifier
    values: dict[str, Any]  # direction-specific values
    # Examples:
    #   bc_type="fixed",  region="bottom_face", values={}
    #   bc_type="force",  region="top_face",    values={"Fy": 1.0}
    #   bc_type="temperature", region="all",     values={"T": 300.0}


@dataclass
class SolverConfig:
    """Complete specification for an FEA analysis job.

    The ``parameters`` dict carries analysis-specific settings:
      - modal:     target_frequency_hz, n_modes
      - harmonic:  freq_min, freq_max, n_sweep, damping_ratio
      - thermal:   heat_sources, ambient_temp
      - structural: loads
      - horn geometry (for preview solver): horn_type, width_mm, height_mm,
        length_mm, mesh_density
    """

    analysis_type: AnalysisType
    mesh_path: str
    material_assignments: list[MaterialAssignment]
    boundary_conditions: list[BoundaryCondition]
    parameters: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

@dataclass
class PreparedJob:
    """Solver-specific input files ready for execution."""

    job_id: str
    work_dir: str
    input_files: list[str]
    solver_config: SolverConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        """Generate a unique job identifier."""
        return uuid.uuid4().hex[:12]


# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------

@dataclass
class FieldData:
    """Uniform result container for all solvers.

    Holds the mesh geometry and associated nodal / element fields that the
    frontend can render with VTK.js or Three.js.
    """

    points: np.ndarray  # (N, 3) node coordinates
    cells: list[np.ndarray]  # connectivity arrays per cell type
    cell_types: list[str]  # "hex8", "tet4", "tri3", etc.
    point_data: dict[str, np.ndarray] = field(default_factory=dict)
    # e.g., "displacement": (N,3), "von_mises_stress": (N,), "temperature": (N,)
    cell_data: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    # e.g., solve_time_s, convergence_info, iteration_count

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        """Number of mesh nodes."""
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        """Total number of cells across all types."""
        return sum(len(c) for c in self.cells)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xyz, max_xyz) bounding box."""
        return self.points.min(axis=0), self.points.max(axis=0)


@dataclass
class SolverResult:
    """Outcome of a solver run."""

    success: bool
    job_id: str
    output_files: list[str] = field(default_factory=list)
    field_data: Optional[FieldData] = None
    metrics: dict[str, float] = field(default_factory=dict)
    # Pre-computed scalar metrics:
    #   natural_frequency_hz, amplitude_uniformity,
    #   max_stress_mpa, frequency_deviation_percent, ...
    error_message: str = ""
    solver_log: str = ""
    compute_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[float, str], None]
# Called as: callback(percent, message)
# percent in [0.0, 100.0]


# ---------------------------------------------------------------------------
# Abstract solver backend
# ---------------------------------------------------------------------------

class SolverBackend(ABC):
    """Base class for all FEA solver backends.

    Subclasses must implement:
      - ``name``              (property)
      - ``supported_analyses`` (property)
      - ``prepare``           (async)
      - ``run``               (async)
      - ``read_results``
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Solver backend name (e.g., 'preview', 'fenics', 'elmer', 'calculix')."""
        ...

    @property
    @abstractmethod
    def supported_analyses(self) -> list[AnalysisType]:
        """List of analysis types this solver supports."""
        ...

    @abstractmethod
    async def prepare(self, config: SolverConfig) -> PreparedJob:
        """Convert mesh + materials + BCs into solver-specific input files.

        For lightweight solvers (preview) this may be a no-op that simply
        validates the configuration.  For external solvers this writes the
        input deck to disk.
        """
        ...

    @abstractmethod
    async def run(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback] = None,
    ) -> SolverResult:
        """Execute the solver.

        Implementations should call ``progress(percent, message)`` at
        meaningful milestones so the frontend can display a progress bar.
        """
        ...

    @abstractmethod
    def read_results(self, result: SolverResult) -> FieldData:
        """Parse solver output into uniform :class:`FieldData`.

        For solvers that populate ``result.field_data`` during ``run()``,
        this may simply return that field.  For external solvers it reads
        the output files listed in ``result.output_files``.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def supports(self, analysis_type: AnalysisType) -> bool:
        """Check whether this backend supports *analysis_type*."""
        return analysis_type in self.supported_analyses

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} name={self.name!r} "
            f"analyses={[a.value for a in self.supported_analyses]}>"
        )
