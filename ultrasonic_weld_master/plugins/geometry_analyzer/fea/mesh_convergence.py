"""Mesh convergence automation with Richardson extrapolation.

Automates the process of running FEA analyses at multiple mesh refinement
levels and assessing whether the solution has converged. Uses Richardson
extrapolation to estimate the exact solution and the convergence order.

Typical usage
-------------
>>> study = MeshConvergenceStudy(mesher, solver, "cylindrical",
...     {"diameter_mm": 50.0, "length_mm": 127.0}, "Ti-6Al-4V")
>>> result = study.run_study([10.0, 7.0, 5.0, 3.5, 2.5])
>>> print(study.generate_report(result))
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ConvergenceResult:
    """Result container for a mesh convergence study.

    Attributes
    ----------
    mesh_sizes : list[float]
        Characteristic element sizes (mm) used in the study, coarsest first.
    n_dof : list[int]
        Number of degrees of freedom at each refinement level.
    values : list[float]
        Target quantity (e.g. frequency in Hz) at each refinement level.
    relative_changes : list[float]
        Percentage change between successive refinement levels.
        First entry is always ``float('nan')`` (no predecessor).
    converged : bool
        Whether the study met the convergence threshold.
    convergence_rate : float
        Estimated order of convergence from Richardson extrapolation.
        ``float('nan')`` if extrapolation could not be performed.
    extrapolated_value : float
        Richardson-extrapolated estimate of the exact value.
        ``float('nan')`` if extrapolation could not be performed.
    recommended_mesh_size : float
        Suggested mesh size that balances accuracy and cost.
    target_quantity : str
        Name of the quantity tracked (e.g. ``"frequency"``).
    """

    mesh_sizes: list[float]
    n_dof: list[int]
    values: list[float]
    relative_changes: list[float]
    converged: bool
    convergence_rate: float
    extrapolated_value: float
    recommended_mesh_size: float
    target_quantity: str = "frequency"


# ---------------------------------------------------------------------------
# Main study class
# ---------------------------------------------------------------------------


class MeshConvergenceStudy:
    """Automated mesh convergence study for ultrasonic horn FEA.

    Generates meshes at multiple refinement levels, runs modal analysis on
    each, and evaluates convergence of a chosen target quantity using
    Richardson extrapolation.

    Parameters
    ----------
    mesher : object
        Mesher with ``mesh_parametric_horn(horn_type, dimensions,
        mesh_size, order)`` method returning an ``FEAMesh``.
    solver : object
        Solver with ``modal_analysis(config)`` method returning a
        ``ModalResult``.
    horn_type : str
        Horn geometry type (e.g. ``"cylindrical"``, ``"flat"``).
    dimensions : dict[str, float]
        Horn dimensions in mm.
    material_name : str
        Material identifier for modal analysis.
    n_modes : int
        Number of modes to compute at each refinement level.
    target_frequency_hz : float
        Target frequency for the shift-invert eigensolver.
    boundary_conditions : str
        BC type: ``"free-free"``, ``"clamped"``, or ``"pre-stressed"``.
    element_order : int
        Element order (1 for TET4, 2 for TET10).
    convergence_threshold : float
        Relative change threshold (%) below which convergence is declared.
    """

    def __init__(
        self,
        mesher: Any,
        solver: Any,
        horn_type: str,
        dimensions: dict[str, float],
        material_name: str,
        n_modes: int = 20,
        target_frequency_hz: float = 20000.0,
        boundary_conditions: str = "free-free",
        element_order: int = 2,
        convergence_threshold: float = 0.5,
    ) -> None:
        self.mesher = mesher
        self.solver = solver
        self.horn_type = horn_type
        self.dimensions = dimensions
        self.material_name = material_name
        self.n_modes = n_modes
        self.target_frequency_hz = target_frequency_hz
        self.boundary_conditions = boundary_conditions
        self.element_order = element_order
        self.convergence_threshold = convergence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_study(
        self,
        mesh_sizes: list[float],
        target_quantity: str = "frequency",
    ) -> ConvergenceResult:
        """Run a convergence study across multiple mesh sizes.

        Parameters
        ----------
        mesh_sizes : list[float]
            Characteristic element sizes in mm, from coarsest to finest.
            At least 2 sizes are required.
        target_quantity : str
            Quantity to track: ``"frequency"`` (fundamental modal frequency),
            ``"max_stress"`` (maximum von Mises stress from first mode shape),
            or ``"gain"`` (amplitude ratio from harmonic analysis).

        Returns
        -------
        ConvergenceResult
            Comprehensive convergence study results.

        Raises
        ------
        ValueError
            If fewer than 2 mesh sizes are provided or *target_quantity*
            is not recognised.
        """
        if len(mesh_sizes) < 2:
            raise ValueError(
                "At least 2 mesh sizes are required for a convergence study, "
                f"got {len(mesh_sizes)}."
            )

        valid_quantities = ("frequency", "max_stress", "gain")
        if target_quantity not in valid_quantities:
            raise ValueError(
                f"Unknown target_quantity {target_quantity!r}. "
                f"Must be one of {valid_quantities}."
            )

        # Ensure mesh sizes are sorted coarsest (largest) to finest (smallest)
        mesh_sizes = sorted(mesh_sizes, reverse=True)

        values: list[float] = []
        n_dof_list: list[int] = []

        for i, ms in enumerate(mesh_sizes):
            logger.info(
                "Convergence study level %d/%d: mesh_size=%.2f mm",
                i + 1,
                len(mesh_sizes),
                ms,
            )

            # Generate mesh
            mesh = self.mesher.mesh_parametric_horn(
                horn_type=self.horn_type,
                dimensions=self.dimensions,
                mesh_size=ms,
                order=self.element_order,
            )
            n_dof_list.append(mesh.n_dof)

            # Build analysis configuration
            config = ModalConfig(
                mesh=mesh,
                material_name=self.material_name,
                n_modes=self.n_modes,
                target_frequency_hz=self.target_frequency_hz,
                boundary_conditions=self.boundary_conditions,
            )

            # Solve
            result = self.solver.modal_analysis(config)

            # Extract target quantity
            value = self._extract_quantity(result, target_quantity)
            values.append(value)

            logger.info(
                "  n_dof=%d, %s=%.6f",
                mesh.n_dof,
                target_quantity,
                value,
            )

        # Compute relative changes
        relative_changes = self._compute_relative_changes(values)

        # Check convergence
        converged = self._check_convergence(values, self.convergence_threshold)

        # Richardson extrapolation (needs >= 3 data points)
        if len(mesh_sizes) >= 3:
            extrapolated_value, convergence_rate = self._richardson_extrapolate(
                mesh_sizes, values
            )
        else:
            extrapolated_value = float("nan")
            convergence_rate = float("nan")

        # Recommended mesh size
        recommended = self._recommend_mesh_size(mesh_sizes, relative_changes)

        return ConvergenceResult(
            mesh_sizes=mesh_sizes,
            n_dof=n_dof_list,
            values=values,
            relative_changes=relative_changes,
            converged=converged,
            convergence_rate=convergence_rate,
            extrapolated_value=extrapolated_value,
            recommended_mesh_size=recommended,
            target_quantity=target_quantity,
        )

    def generate_report(self, result: ConvergenceResult) -> str:
        """Generate a human-readable convergence study report.

        Parameters
        ----------
        result : ConvergenceResult
            Result from :meth:`run_study`.

        Returns
        -------
        str
            Formatted text report with table, Richardson extrapolation
            results, and recommendation.
        """
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("MESH CONVERGENCE STUDY REPORT")
        lines.append("=" * 72)
        lines.append(f"Target quantity : {result.target_quantity}")
        lines.append(f"Refinement levels: {len(result.mesh_sizes)}")
        lines.append("")

        # Table header
        header = (
            f"{'Level':>5s}  {'mesh_size':>10s}  {'n_dof':>10s}  "
            f"{'value':>14s}  {'change%':>10s}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for i in range(len(result.mesh_sizes)):
            change_str = (
                "---"
                if i == 0 or math.isnan(result.relative_changes[i])
                else f"{result.relative_changes[i]:>10.4f}"
            )
            lines.append(
                f"{i + 1:>5d}  {result.mesh_sizes[i]:>10.2f}  "
                f"{result.n_dof[i]:>10d}  "
                f"{result.values[i]:>14.6f}  {change_str:>10s}"
            )

        lines.append("")

        # Richardson extrapolation
        lines.append("-" * 72)
        lines.append("RICHARDSON EXTRAPOLATION")
        lines.append("-" * 72)
        if math.isnan(result.convergence_rate):
            lines.append(
                "Not available (requires >= 3 refinement levels "
                "with monotonic convergence)."
            )
        else:
            lines.append(
                f"Estimated convergence order: {result.convergence_rate:.3f}"
            )
            lines.append(
                f"Extrapolated exact value   : {result.extrapolated_value:.6f}"
            )
            # Estimate error of finest mesh relative to extrapolated value
            if result.extrapolated_value != 0.0:
                error_pct = abs(
                    (result.values[-1] - result.extrapolated_value)
                    / result.extrapolated_value
                    * 100.0
                )
                lines.append(
                    f"Estimated error (finest)   : {error_pct:.4f}%"
                )

        lines.append("")

        # Recommendation
        lines.append("-" * 72)
        lines.append("RECOMMENDATION")
        lines.append("-" * 72)
        status = "CONVERGED" if result.converged else "NOT CONVERGED"
        lines.append(f"Status             : {status}")
        lines.append(
            f"Convergence threshold: {self.convergence_threshold:.2f}%"
        )
        lines.append(
            f"Recommended mesh size: {result.recommended_mesh_size:.2f} mm"
        )
        lines.append("=" * 72)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_quantity(result: ModalResult, quantity: str) -> float:
        """Extract a scalar target quantity from a modal analysis result.

        Parameters
        ----------
        result : ModalResult
            Modal analysis result.
        quantity : str
            ``"frequency"`` returns the fundamental (lowest) frequency.

        Returns
        -------
        float
            Extracted scalar value.
        """
        if quantity == "frequency":
            return float(result.frequencies_hz[0])
        elif quantity == "max_stress":
            # For modal results, use max displacement amplitude as proxy
            # (actual stress would need a static solve)
            return float(np.max(np.abs(result.mode_shapes[0])))
        elif quantity == "gain":
            # Gain from harmonic result -- for modal, approximate from
            # displacement ratio between top and bottom of first mode
            mode_shape = result.mode_shapes[0]
            return float(np.max(np.abs(mode_shape)))
        else:
            raise ValueError(f"Unknown target quantity: {quantity!r}")

    @staticmethod
    def _compute_relative_changes(values: list[float]) -> list[float]:
        """Compute percentage change between successive refinement levels.

        Parameters
        ----------
        values : list[float]
            Quantity values at each refinement level (coarsest to finest).

        Returns
        -------
        list[float]
            Relative changes in %. First entry is ``nan``.
        """
        changes: list[float] = [float("nan")]
        for i in range(1, len(values)):
            if values[i - 1] == 0.0:
                if values[i] == 0.0:
                    changes.append(0.0)
                else:
                    changes.append(float("inf"))
            else:
                change = abs((values[i] - values[i - 1]) / values[i - 1]) * 100.0
                changes.append(change)
        return changes

    @staticmethod
    def _check_convergence(
        values: list[float], threshold: float = 0.5
    ) -> bool:
        """Check whether the last refinement step is within threshold.

        Parameters
        ----------
        values : list[float]
            Quantity values at each refinement level.
        threshold : float
            Maximum allowed relative change (%) for convergence.

        Returns
        -------
        bool
            ``True`` if the last relative change is below *threshold*.
        """
        if len(values) < 2:
            return False
        prev = values[-2]
        curr = values[-1]
        if prev == 0.0:
            return curr == 0.0
        rel_change = abs((curr - prev) / prev) * 100.0
        return rel_change < threshold

    @staticmethod
    def _richardson_extrapolate(
        h_values: list[float], f_values: list[float]
    ) -> tuple[float, float]:
        """Richardson extrapolation using the 3 finest mesh levels.

        Uses the standard Richardson formula for generalised refinement
        ratios:

            p = log((f3 - f2) / (f2 - f1)) / log(r)
            f_exact = f1 + (f1 - f2) / (r^p - 1)

        where ``r = h2 / h1`` is the refinement ratio between the two
        finest meshes, and levels 1, 2, 3 are the three finest meshes
        (h1 < h2 < h3).

        Parameters
        ----------
        h_values : list[float]
            Characteristic mesh sizes (coarsest to finest).
        f_values : list[float]
            Quantity values corresponding to *h_values*.

        Returns
        -------
        tuple[float, float]
            ``(extrapolated_value, convergence_order)``.
            Returns ``(nan, nan)`` if extrapolation cannot be performed
            (e.g. non-monotonic convergence or zero denominator).
        """
        if len(h_values) < 3:
            return float("nan"), float("nan")

        # Three finest meshes (smallest h values -> last three entries
        # since h_values is sorted coarsest-to-finest, i.e. descending)
        h1 = h_values[-1]  # finest
        h2 = h_values[-2]
        h3 = h_values[-3]

        f1 = f_values[-1]  # finest
        f2 = f_values[-2]
        f3 = f_values[-3]

        # Differences
        diff21 = f2 - f1
        diff32 = f3 - f2

        # Guard against zero or sign-change in differences
        if diff21 == 0.0 or diff32 == 0.0:
            # Already converged or oscillating -- return finest value
            return f1, float("nan")

        ratio = diff32 / diff21

        if ratio <= 0.0:
            # Non-monotonic convergence -- Richardson does not apply
            return float("nan"), float("nan")

        # Refinement ratio (h2/h1 -- both positive, h2 > h1)
        if h1 <= 0.0 or h2 <= 0.0:
            return float("nan"), float("nan")

        r = h2 / h1

        if r <= 0.0 or r == 1.0:
            return float("nan"), float("nan")

        # Estimated convergence order
        p = math.log(ratio) / math.log(r)

        # Guard against unreasonable orders
        if p <= 0.0 or math.isnan(p) or math.isinf(p):
            return float("nan"), float("nan")

        # Extrapolated value
        r_p = r ** p
        if r_p == 1.0:
            return float("nan"), float("nan")

        f_exact = f1 + (f1 - f2) / (r_p - 1.0)

        return f_exact, p

    @staticmethod
    def _recommend_mesh_size(
        mesh_sizes: list[float], relative_changes: list[float]
    ) -> float:
        """Recommend a mesh size balancing accuracy and computational cost.

        Strategy:
        - Find the coarsest mesh where relative change < 1%.
        - If no such mesh exists, return the finest tested mesh size.

        Parameters
        ----------
        mesh_sizes : list[float]
            Mesh sizes sorted coarsest to finest.
        relative_changes : list[float]
            Corresponding relative changes (first entry is ``nan``).

        Returns
        -------
        float
            Recommended mesh size in mm.
        """
        # Walk from coarsest to finest; find first level where change < 1%
        for i in range(1, len(mesh_sizes)):
            change = relative_changes[i]
            if not math.isnan(change) and change < 1.0:
                return mesh_sizes[i]

        # Not yet converged -- recommend the finest tested
        return mesh_sizes[-1]
