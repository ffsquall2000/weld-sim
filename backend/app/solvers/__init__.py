"""Solver abstraction layer for the Ultrasonic Metal Welding Virtual Simulation Platform.

Provides a uniform interface for multiple FEA solver backends:

  - :class:`SolverBackend` -- abstract base class
  - :class:`PreviewSolver` -- fast numpy/scipy solver (bundled)
  - :func:`get_solver`     -- retrieve a registered solver by name
  - :func:`init_solvers`   -- register default solvers at startup

Data classes:
  - :class:`AnalysisType`  -- modal, harmonic, structural, thermal, ...
  - :class:`SolverConfig`  -- complete analysis specification
  - :class:`SolverResult`  -- analysis outcome (success/failure + metrics)
  - :class:`FieldData`     -- uniform mesh + field container for rendering

Utilities:
  - :class:`MeshConverter`  -- STEP/STL to gmsh/meshio conversion
  - :class:`ResultReader`   -- parse VTU/FRD/XDMF into FieldData
"""

from .base import (
    AnalysisType,
    BoundaryCondition,
    FieldData,
    MaterialAssignment,
    PreparedJob,
    ProgressCallback,
    SolverBackend,
    SolverConfig,
    SolverResult,
)
from .preview_solver import PreviewSolver
from .registry import (
    get_solver,
    get_solver_for_analysis,
    init_solvers,
    list_solvers,
    register_solver,
)

__all__ = [
    # Abstract base
    "SolverBackend",
    # Concrete solver
    "PreviewSolver",
    # Registry
    "get_solver",
    "get_solver_for_analysis",
    "init_solvers",
    "list_solvers",
    "register_solver",
    # Data classes
    "AnalysisType",
    "BoundaryCondition",
    "FieldData",
    "MaterialAssignment",
    "PreparedJob",
    "ProgressCallback",
    "SolverConfig",
    "SolverResult",
]
