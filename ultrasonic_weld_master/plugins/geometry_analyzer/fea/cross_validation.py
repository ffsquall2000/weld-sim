"""Cross-validation harness for comparing FEA solver results.

Compares results from SolverA (numpy/scipy) and SolverB (FEniCSx) using:
- Modal Assurance Criterion (MAC) matrix
- Frequency deviation percentages
- Stress peak comparison
- Displacement comparison (RMS, max, correlation)
- Overall validation with configurable PASS/WARNING/FAIL thresholds
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import ModalResult


# ---------------------------------------------------------------------------
# Default thresholds
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "mac_pass": 0.95,
    "mac_warn": 0.80,
    "freq_pass_pct": 1.0,
    "freq_warn_pct": 5.0,
}


# ---------------------------------------------------------------------------
# ValidationReport dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Result of a cross-validation between two solver outputs."""

    status: str  # "PASS" | "WARNING" | "FAIL"
    mac_matrix: np.ndarray
    freq_deviations: np.ndarray
    paired_indices: list[tuple[int, int]]
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        n = self.mac_matrix.shape[0] if self.mac_matrix.ndim == 2 else 0
        return (
            f"ValidationReport(status={self.status!r}, "
            f"n_modes={n}, "
            f"n_paired={len(self.paired_indices)})"
        )


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------


class CrossValidator:
    """Compare FEA results from two independent solvers.

    All static methods can be used standalone; the :meth:`validate` method
    orchestrates a full comparison and returns a :class:`ValidationReport`.
    """

    # ------------------------------------------------------------------
    # 1. MAC matrix
    # ------------------------------------------------------------------

    @staticmethod
    def compute_mac(modes_a: np.ndarray, modes_b: np.ndarray) -> np.ndarray:
        """Compute the Modal Assurance Criterion (MAC) matrix.

        Parameters
        ----------
        modes_a : np.ndarray, shape (n_dof, n_modes_a)
            Mode shape matrix from solver A.  Each column is a mode.
        modes_b : np.ndarray, shape (n_dof, n_modes_b)
            Mode shape matrix from solver B.

        Returns
        -------
        np.ndarray, shape (n_modes_a, n_modes_b)
            MAC(i, j) = |phi_a_i^T phi_b_j|^2 / (|phi_a_i|^2 * |phi_b_j|^2)
        """
        if modes_a.size == 0 or modes_b.size == 0:
            return np.empty((0, 0))

        # Ensure 2-D
        if modes_a.ndim == 1:
            modes_a = modes_a.reshape(-1, 1)
        if modes_b.ndim == 1:
            modes_b = modes_b.reshape(-1, 1)

        # Norms squared for each column
        norm_a = np.sum(modes_a ** 2, axis=0)  # (n_modes_a,)
        norm_b = np.sum(modes_b ** 2, axis=0)  # (n_modes_b,)

        # Cross dot products
        cross = modes_a.T @ modes_b  # (n_modes_a, n_modes_b)

        # MAC matrix
        denom = np.outer(norm_a, norm_b)
        # Avoid division by zero for zero-norm modes
        denom = np.where(denom == 0.0, np.inf, denom)
        mac = cross ** 2 / denom
        return mac

    # ------------------------------------------------------------------
    # 2. Frequency deviation
    # ------------------------------------------------------------------

    @staticmethod
    def frequency_deviation(freq_a: np.ndarray, freq_b: np.ndarray) -> np.ndarray:
        """Percentage frequency deviation between two sets.

        Parameters
        ----------
        freq_a, freq_b : np.ndarray, shape (n,)
            Frequency arrays (must have the same length).

        Returns
        -------
        np.ndarray, shape (n,)
            dev_i = |f_a_i - f_b_i| / f_a_i * 100  (percent).
            If f_a_i == 0, deviation is 0 when f_b_i == 0, else inf.
        """
        freq_a = np.asarray(freq_a, dtype=float)
        freq_b = np.asarray(freq_b, dtype=float)
        if freq_a.size == 0:
            return np.empty(0)

        with np.errstate(divide="ignore", invalid="ignore"):
            dev = np.abs(freq_a - freq_b) / np.abs(freq_a) * 100.0
        # Fix 0/0 -> nan to 0
        both_zero = (freq_a == 0.0) & (freq_b == 0.0)
        dev[both_zero] = 0.0
        # 0 denominator with nonzero numerator -> inf (already handled by numpy)
        return dev

    # ------------------------------------------------------------------
    # 3. Stress peak comparison
    # ------------------------------------------------------------------

    @staticmethod
    def stress_comparison(stress_a: np.ndarray, stress_b: np.ndarray) -> dict:
        """Compare stress fields from two solvers.

        Parameters
        ----------
        stress_a, stress_b : np.ndarray
            Element-wise von Mises (or equivalent) stress arrays.

        Returns
        -------
        dict with keys:
            peak_a, peak_b          - maximum stress values
            peak_location_a/b       - index of peak element
            peak_relative_error_pct - |peak_a - peak_b| / peak_a * 100
            rms_error               - RMS of (stress_a - stress_b)
            max_abs_error           - max element-wise absolute difference
        """
        stress_a = np.asarray(stress_a, dtype=float)
        stress_b = np.asarray(stress_b, dtype=float)

        if stress_a.size == 0:
            return {
                "peak_a": 0.0,
                "peak_b": 0.0,
                "peak_location_a": -1,
                "peak_location_b": -1,
                "peak_relative_error_pct": 0.0,
                "rms_error": 0.0,
                "max_abs_error": 0.0,
            }

        peak_a = float(np.max(stress_a))
        peak_b = float(np.max(stress_b))
        loc_a = int(np.argmax(stress_a))
        loc_b = int(np.argmax(stress_b))

        if peak_a == 0.0:
            rel_err = 0.0 if peak_b == 0.0 else float("inf")
        else:
            rel_err = abs(peak_a - peak_b) / abs(peak_a) * 100.0

        diff = stress_a - stress_b
        rms = float(np.sqrt(np.mean(diff ** 2)))
        max_abs = float(np.max(np.abs(diff)))

        return {
            "peak_a": peak_a,
            "peak_b": peak_b,
            "peak_location_a": loc_a,
            "peak_location_b": loc_b,
            "peak_relative_error_pct": rel_err,
            "rms_error": rms,
            "max_abs_error": max_abs,
        }

    # ------------------------------------------------------------------
    # 4. Displacement comparison
    # ------------------------------------------------------------------

    @staticmethod
    def displacement_comparison(disp_a: np.ndarray, disp_b: np.ndarray) -> dict:
        """Compare displacement fields.

        Parameters
        ----------
        disp_a, disp_b : np.ndarray
            Displacement vectors (flattened or any shape; will be ravelled).

        Returns
        -------
        dict with keys:
            rms_error              - root-mean-square of difference
            max_error              - maximum absolute element difference
            correlation_coefficient - Pearson r between the two fields
        """
        disp_a = np.asarray(disp_a, dtype=float).ravel()
        disp_b = np.asarray(disp_b, dtype=float).ravel()

        if disp_a.size == 0:
            return {
                "rms_error": 0.0,
                "max_error": 0.0,
                "correlation_coefficient": 1.0,
            }

        diff = disp_a - disp_b
        rms = float(np.sqrt(np.mean(diff ** 2)))
        max_err = float(np.max(np.abs(diff)))

        # Pearson correlation coefficient
        std_a = np.std(disp_a)
        std_b = np.std(disp_b)
        if std_a == 0.0 or std_b == 0.0:
            # Constant arrays: correlation is 1 if identical, 0 otherwise
            corr = 1.0 if np.allclose(disp_a, disp_b) else 0.0
        else:
            corr = float(np.corrcoef(disp_a, disp_b)[0, 1])

        return {
            "rms_error": rms,
            "max_error": max_err,
            "correlation_coefficient": corr,
        }

    # ------------------------------------------------------------------
    # 5. Mode pairing
    # ------------------------------------------------------------------

    @staticmethod
    def pair_modes(
        freq_a: np.ndarray,
        freq_b: np.ndarray,
        modes_a: np.ndarray,
        modes_b: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Pair modes from solver A to solver B by highest MAC correlation.

        Uses a greedy algorithm: repeatedly pick the (i, j) pair with the
        highest MAC value that has not yet been assigned.

        Parameters
        ----------
        freq_a, freq_b : np.ndarray, shape (n_a,), (n_b,)
            Frequencies (used only for sizing; pairing is MAC-based).
        modes_a : np.ndarray, shape (n_dof, n_a)
        modes_b : np.ndarray, shape (n_dof, n_b)

        Returns
        -------
        list of (int, int)
            Index pairs (i_a, i_b) sorted by i_a ascending.
        """
        mac = CrossValidator.compute_mac(modes_a, modes_b)
        if mac.size == 0:
            return []

        n_a, n_b = mac.shape
        n_pairs = min(n_a, n_b)
        used_a: set[int] = set()
        used_b: set[int] = set()
        pairs: list[tuple[int, int]] = []

        # Work on a copy so we can mask out used rows/columns
        mac_work = mac.copy()

        for _ in range(n_pairs):
            idx = int(np.argmax(mac_work))
            i, j = divmod(idx, n_b)
            pairs.append((int(i), int(j)))
            used_a.add(i)
            used_b.add(j)
            # Zero out used row and column
            mac_work[i, :] = -1.0
            mac_work[:, j] = -1.0

        pairs.sort(key=lambda p: p[0])
        return pairs

    # ------------------------------------------------------------------
    # 6. Overall validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(
        result_a: ModalResult,
        result_b: ModalResult,
        thresholds: Optional[dict] = None,
    ) -> ValidationReport:
        """Run full cross-validation between two modal results.

        Parameters
        ----------
        result_a, result_b : ModalResult
        thresholds : dict, optional
            Override default thresholds.  Keys:
            ``mac_pass``, ``mac_warn``, ``freq_pass_pct``, ``freq_warn_pct``.

        Returns
        -------
        ValidationReport
        """
        th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

        # Pair modes
        pairs = CrossValidator.pair_modes(
            result_a.frequencies_hz,
            result_b.frequencies_hz,
            result_a.mode_shapes,
            result_b.mode_shapes,
        )

        # Full MAC matrix
        mac = CrossValidator.compute_mac(
            result_a.mode_shapes, result_b.mode_shapes
        )

        if len(pairs) == 0:
            return ValidationReport(
                status="FAIL",
                mac_matrix=mac,
                freq_deviations=np.empty(0),
                paired_indices=pairs,
                details={"reason": "No modes to compare"},
            )

        # Extract paired frequencies
        idx_a = np.array([p[0] for p in pairs])
        idx_b = np.array([p[1] for p in pairs])

        freq_a_paired = result_a.frequencies_hz[idx_a]
        freq_b_paired = result_b.frequencies_hz[idx_b]
        freq_dev = CrossValidator.frequency_deviation(freq_a_paired, freq_b_paired)

        # Diagonal MAC values for paired modes
        mac_paired = np.array([mac[ia, ib] for ia, ib in pairs])

        # Determine status
        min_mac = float(np.min(mac_paired)) if mac_paired.size > 0 else 0.0
        max_freq_dev = float(np.max(freq_dev)) if freq_dev.size > 0 else 0.0

        if min_mac >= th["mac_pass"] and max_freq_dev <= th["freq_pass_pct"]:
            status = "PASS"
        elif min_mac < th["mac_warn"] or max_freq_dev > th["freq_warn_pct"]:
            status = "FAIL"
        else:
            status = "WARNING"

        details = {
            "min_mac": min_mac,
            "max_mac": float(np.max(mac_paired)) if mac_paired.size > 0 else 0.0,
            "mean_mac": float(np.mean(mac_paired)) if mac_paired.size > 0 else 0.0,
            "max_freq_dev_pct": max_freq_dev,
            "mean_freq_dev_pct": float(np.mean(freq_dev)) if freq_dev.size > 0 else 0.0,
            "n_paired_modes": len(pairs),
            "solver_a": result_a.solver_name,
            "solver_b": result_b.solver_name,
            "thresholds": th,
        }

        return ValidationReport(
            status=status,
            mac_matrix=mac,
            freq_deviations=freq_dev,
            paired_indices=pairs,
            details=details,
        )

    # ------------------------------------------------------------------
    # 7. Report generation
    # ------------------------------------------------------------------

    @staticmethod
    def generate_report(validation: ValidationReport) -> str:
        """Generate a human-readable text report from a ValidationReport.

        Parameters
        ----------
        validation : ValidationReport

        Returns
        -------
        str
            Multi-line formatted report.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  FEA Cross-Validation Report")
        lines.append("=" * 60)
        lines.append("")

        # Status banner
        lines.append(f"  Overall Status : {validation.status}")
        lines.append("")

        # Details
        d = validation.details
        if "solver_a" in d:
            lines.append(f"  Solver A       : {d['solver_a']}")
            lines.append(f"  Solver B       : {d['solver_b']}")
            lines.append("")

        if "n_paired_modes" in d:
            lines.append(f"  Paired modes   : {d['n_paired_modes']}")
            lines.append("")

        # MAC summary
        if "min_mac" in d:
            lines.append("  MAC Summary")
            lines.append("  -----------")
            lines.append(f"    Min MAC      : {d['min_mac']:.6f}")
            lines.append(f"    Max MAC      : {d['max_mac']:.6f}")
            lines.append(f"    Mean MAC     : {d['mean_mac']:.6f}")
            lines.append("")

        # Frequency deviation summary
        if "max_freq_dev_pct" in d:
            lines.append("  Frequency Deviation Summary")
            lines.append("  ---------------------------")
            lines.append(f"    Max dev      : {d['max_freq_dev_pct']:.4f} %")
            lines.append(f"    Mean dev     : {d['mean_freq_dev_pct']:.4f} %")
            lines.append("")

        # Per-mode table
        if len(validation.paired_indices) > 0:
            lines.append("  Mode Pairing Detail")
            lines.append("  -------------------")
            header = f"  {'A':>4s}  {'B':>4s}  {'MAC':>10s}  {'Freq Dev %':>12s}"
            lines.append(header)

            mac = validation.mac_matrix
            for k, (ia, ib) in enumerate(validation.paired_indices):
                mac_val = mac[ia, ib]
                fd = validation.freq_deviations[k] if k < len(validation.freq_deviations) else float("nan")
                lines.append(f"  {ia:4d}  {ib:4d}  {mac_val:10.6f}  {fd:12.4f}")
            lines.append("")

        # Thresholds
        if "thresholds" in d:
            th = d["thresholds"]
            lines.append("  Thresholds")
            lines.append("  ----------")
            lines.append(f"    PASS  : MAC >= {th['mac_pass']}, freq dev <= {th['freq_pass_pct']}%")
            lines.append(f"    WARN  : MAC >= {th['mac_warn']}, freq dev <= {th['freq_warn_pct']}%")
            lines.append(f"    FAIL  : MAC <  {th['mac_warn']} or freq dev > {th['freq_warn_pct']}%")
            lines.append("")

        # Reason (if any)
        if "reason" in d:
            lines.append(f"  Note: {d['reason']}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
