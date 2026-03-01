"""Solver registry -- discover and retrieve solver backends by name.

Usage::

    from backend.app.solvers.registry import get_solver, list_solvers, init_solvers

    # At application startup
    init_solvers()

    # Later
    solver = get_solver("preview")
    result = await solver.run(job)
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import AnalysisType, SolverBackend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal registry
# ---------------------------------------------------------------------------

_solvers: dict[str, SolverBackend] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_solver(solver: SolverBackend) -> None:
    """Register a solver backend instance.

    If a solver with the same name already exists it is replaced (with a
    warning).
    """
    name = solver.name
    if name in _solvers:
        logger.warning(
            "Replacing already-registered solver %r (%s -> %s)",
            name,
            type(_solvers[name]).__name__,
            type(solver).__name__,
        )
    _solvers[name] = solver
    logger.info(
        "Registered solver %r  analyses=%s",
        name,
        [a.value for a in solver.supported_analyses],
    )


def get_solver(name: str) -> SolverBackend:
    """Retrieve a registered solver by name.

    Raises :class:`KeyError` if no solver with that name exists.
    """
    try:
        return _solvers[name]
    except KeyError:
        available = ", ".join(sorted(_solvers.keys())) or "(none)"
        raise KeyError(
            f"No solver registered with name {name!r}.  "
            f"Available solvers: {available}"
        ) from None


def get_solver_for_analysis(analysis_type: AnalysisType) -> Optional[SolverBackend]:
    """Return the first registered solver that supports *analysis_type*.

    Prefers solvers in registration order.  Returns *None* if no solver
    supports the requested analysis.
    """
    for solver in _solvers.values():
        if solver.supports(analysis_type):
            return solver
    return None


def list_solvers() -> list[dict]:
    """Return metadata for all registered solvers.

    Each entry is a dict with keys:
        ``name``, ``class``, ``supported_analyses``
    """
    return [
        {
            "name": solver.name,
            "class": type(solver).__qualname__,
            "supported_analyses": [a.value for a in solver.supported_analyses],
        }
        for solver in _solvers.values()
    ]


def is_registered(name: str) -> bool:
    """Check whether a solver with *name* is registered."""
    return name in _solvers


def unregister_solver(name: str) -> bool:
    """Remove a solver from the registry.  Returns True if it existed."""
    if name in _solvers:
        del _solvers[name]
        logger.info("Unregistered solver %r", name)
        return True
    return False


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_solvers() -> None:
    """Register the default set of solver backends.

    Called at application startup.  Currently registers:

    - **preview** -- fast numpy/scipy solver for modal and harmonic analysis

    Future backends (FEniCS, Elmer, CalculiX) will be registered here once
    their wrappers are implemented.
    """
    # Always register the preview solver (no external deps beyond numpy/scipy)
    from .preview_solver import PreviewSolver

    if not is_registered("preview"):
        register_solver(PreviewSolver())

    # FEniCS solver (thermal + structural; falls back to numpy/scipy)
    if not is_registered("fenics"):
        try:
            from backend.app.solvers.fenics_solver import FEniCSSolver

            register_solver(FEniCSSolver())
        except Exception:
            logger.debug("FEniCS solver registration failed; skipping")

    # --- Future solvers ---
    # try:
    #     from .elmer_solver import ElmerSolver
    #     register_solver(ElmerSolver())
    # except ImportError:
    #     logger.debug("Elmer not available; skipping elmer solver")
    #
    # try:
    #     from .calculix_solver import CalculiXSolver
    #     register_solver(CalculiXSolver())
    # except ImportError:
    #     logger.debug("CalculiX not available; skipping calculix solver")

    logger.info(
        "Solver registry initialized: %d backend(s) available",
        len(_solvers),
    )
