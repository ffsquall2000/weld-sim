"""Analysis chain orchestrator with dependency resolution.

Resolves module dependencies so that requesting e.g. ``["fatigue"]`` will
automatically schedule ``["modal", "harmonic", "stress", "fatigue"]`` in the
correct topological order.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency graph: module -> list of modules it depends on
# ---------------------------------------------------------------------------
DEPENDENCY_GRAPH: dict[str, list[str]] = {
    "modal": [],
    "harmonic": ["modal"],
    "stress": ["harmonic"],
    "uniformity": ["harmonic"],
    "fatigue": ["stress"],
    "static": [],
}

# Canonical topological ordering used to sort the resolved set.
_TOPO_ORDER: list[str] = [
    "modal", "static", "harmonic", "uniformity", "stress", "fatigue",
]


def resolve_dependencies(requested: list[str]) -> list[str]:
    """Return modules in execution order, including all required dependencies.

    Example::

        >>> resolve_dependencies(["fatigue"])
        ['modal', 'harmonic', 'stress', 'fatigue']

        >>> resolve_dependencies(["modal"])
        ['modal']

        >>> resolve_dependencies(["harmonic"])
        ['modal', 'harmonic']

        >>> resolve_dependencies(["stress", "modal"])
        ['modal', 'harmonic', 'stress']
    """
    needed: set[str] = set()

    def _add(mod: str) -> None:
        if mod in needed:
            return
        for dep in DEPENDENCY_GRAPH.get(mod, []):
            _add(dep)
        needed.add(mod)

    for m in requested:
        _add(m)

    # Return in canonical topological order
    return [m for m in _TOPO_ORDER if m in needed]


def build_chain_steps(modules: list[str]) -> list[str]:
    """Build the flat list of progress steps for the chain worker.

    Each module contributes a ``<module>_run`` step.  The chain is bookended
    by ``init`` and ``packaging``.
    """
    steps = ["init"]
    for mod in modules:
        steps.append(f"{mod}_run")
    steps.append("packaging")
    return steps
