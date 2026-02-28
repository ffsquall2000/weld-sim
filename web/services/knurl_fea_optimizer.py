"""Knurl FEA optimization with Bayesian pre-screening and Pareto front.

Combines analytical scoring (fast) from KnurlOptimizer with actual FEA
modal analysis to find Pareto-optimal knurl configurations that balance
amplitude uniformity and frequency match.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CandidateConfig:
    """A knurl configuration candidate for FEA evaluation."""

    knurl_type: str
    pitch_mm: float
    depth_mm: float
    tooth_width_mm: float = 0.5
    analytical_score: float = 0.0


@dataclass
class FEAResult:
    """FEA evaluation result for a single candidate."""

    config: CandidateConfig
    closest_mode_hz: float = 0.0
    frequency_deviation_percent: float = 0.0
    amplitude_uniformity: float = 0.0
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0
    mode_count: int = 0
    analytical_score: float = 0.0
    fea_score: float = 0.0  # Combined FEA quality score


class KnurlFEAOptimizer:
    """Bayesian pre-screening + FEA validation for knurl optimization.

    Strategy:
    1. Generate parameter grid (knurl types x pitches x depths)
    2. Pre-screen with analytical scoring (fast, from KnurlOptimizer)
    3. Take top-N candidates
    4. Run actual FEA on each candidate
    5. Return Pareto front: best amplitude uniformity vs best frequency match
    """

    # Knurl types for FEA grid (subset -- types that affect modal behavior)
    KNURL_TYPES = ["linear", "cross_hatch", "diamond"]

    # Parameter ranges
    PITCH_RANGE = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]  # mm
    DEPTH_RANGE = [0.1, 0.2, 0.3, 0.5]  # mm

    async def optimize(
        self,
        horn_config: dict,
        material: str,
        target_freq_khz: float,
        n_candidates: int = 10,
        progress_callback: Optional[Callable] = None,
    ) -> dict:
        """Run the full optimization loop.

        Parameters
        ----------
        horn_config : dict
            Horn dimensions with keys: horn_type, width_mm, height_mm,
            length_mm.
        material : str
            Material name (e.g. ``"Titanium Ti-6Al-4V"``).
        target_freq_khz : float
            Target resonant frequency in kHz.
        n_candidates : int
            Number of top analytical candidates to run FEA on.
        progress_callback : callable, optional
            ``async def callback(step: int, total: int, message: str)``
            Called with progress updates.

        Returns
        -------
        dict
            Results with keys:
            - ``candidates``: All FEA-evaluated candidates with scores
            - ``pareto_front``: Pareto-optimal candidates (frequency match
              vs amplitude uniformity)
            - ``best_frequency_match``: Candidate closest to target freq
            - ``best_uniformity``: Candidate with best amplitude uniformity
            - ``summary``: Statistics about the optimization run
        """
        t0 = time.perf_counter()

        # Step 1: Generate parameter grid
        grid = self._generate_grid()
        total_grid = len(grid)

        if progress_callback:
            await progress_callback(
                0, n_candidates + 1,
                f"Generated {total_grid} parameter combinations",
            )

        # Step 2: Pre-screen with analytical scoring
        scored = self._analytical_prescreening(grid, target_freq_khz)

        # Step 3: Take top-N
        top_candidates = scored[:n_candidates]

        if progress_callback:
            await progress_callback(
                1, n_candidates + 1,
                f"Pre-screened to top {len(top_candidates)} candidates",
            )

        # Step 4: Run FEA on each candidate
        fea_results = []
        for i, candidate in enumerate(top_candidates):
            if progress_callback:
                await progress_callback(
                    i + 2, n_candidates + 1,
                    f"Running FEA on candidate {i + 1}/{len(top_candidates)}: "
                    f"{candidate.knurl_type} p={candidate.pitch_mm} "
                    f"d={candidate.depth_mm}",
                )

            result = await self._run_fea_on_candidate(
                candidate, horn_config, material, target_freq_khz
            )
            fea_results.append(result)

        # Step 5: Build Pareto front and select best candidates
        pareto = self._compute_pareto_front(fea_results)

        # Sort results by FEA score (descending)
        fea_results.sort(key=lambda r: r.fea_score, reverse=True)

        # Find best frequency match and best uniformity
        best_freq = min(
            fea_results,
            key=lambda r: r.frequency_deviation_percent,
        ) if fea_results else None

        best_unif = max(
            fea_results,
            key=lambda r: r.amplitude_uniformity,
        ) if fea_results else None

        elapsed = time.perf_counter() - t0

        return {
            "candidates": [self._result_to_dict(r) for r in fea_results],
            "pareto_front": [self._result_to_dict(r) for r in pareto],
            "best_frequency_match": (
                self._result_to_dict(best_freq) if best_freq else None
            ),
            "best_uniformity": (
                self._result_to_dict(best_unif) if best_unif else None
            ),
            "summary": {
                "total_grid_size": total_grid,
                "candidates_evaluated": len(fea_results),
                "pareto_front_size": len(pareto),
                "total_time_s": round(elapsed, 2),
                "target_frequency_khz": target_freq_khz,
                "material": material,
            },
        }

    def _generate_grid(self) -> list[CandidateConfig]:
        """Generate the full parameter grid.

        Combines knurl types, pitches, and depths.  For each combination
        a reasonable tooth_width is derived as ``min(pitch * 0.5, 0.5)``.
        """
        grid: list[CandidateConfig] = []

        for knurl_type in self.KNURL_TYPES:
            for pitch in self.PITCH_RANGE:
                for depth in self.DEPTH_RANGE:
                    # Derive tooth width: half the pitch, capped at 0.5mm
                    tooth_width = min(pitch * 0.5, 0.5)
                    # Skip impossible combos
                    if tooth_width >= pitch:
                        continue
                    if depth > pitch:
                        continue

                    grid.append(CandidateConfig(
                        knurl_type=knurl_type,
                        pitch_mm=pitch,
                        depth_mm=depth,
                        tooth_width_mm=round(tooth_width, 2),
                    ))

        return grid

    def _analytical_prescreening(
        self,
        grid: list[CandidateConfig],
        target_freq_khz: float,
    ) -> list[CandidateConfig]:
        """Score candidates analytically and return sorted by score.

        Uses a simplified version of the KnurlOptimizer scoring that
        focuses on frequency-relevant parameters:
        - Contact ratio (affects mass loading -> frequency shift)
        - Depth/pitch ratio (affects stiffness change)
        - Knurl type preference (diamond/cross_hatch have more uniform
          mass distribution)
        """
        for candidate in grid:
            score = self._compute_analytical_score(
                candidate, target_freq_khz
            )
            candidate.analytical_score = score

        grid.sort(key=lambda c: c.analytical_score, reverse=True)
        return grid

    def _compute_analytical_score(
        self,
        candidate: CandidateConfig,
        target_freq_khz: float,
    ) -> float:
        """Compute an analytical score for a candidate.

        Higher is better.  The score combines:
        - contact_ratio: uniform contact is preferred (0.3-0.6 range ideal)
        - depth_penalty: very deep knurls remove too much material
        - type_bonus: cross_hatch and diamond have more symmetric mass removal
        - frequency_sensitivity: pitch and depth affect resonance shift
        """
        p = candidate.pitch_mm
        tw = candidate.tooth_width_mm
        d = candidate.depth_mm

        # Contact ratio
        if candidate.knurl_type == "linear":
            contact_ratio = tw / p if p > 0 else 1.0
        elif candidate.knurl_type in ("cross_hatch", "diamond"):
            contact_ratio = (tw / p) ** 2 if p > 0 else 1.0
        else:
            contact_ratio = 1.0

        # Ideal contact ratio is 0.3-0.6 for good frequency tuning
        contact_score = 1.0 - abs(contact_ratio - 0.45) / 0.45
        contact_score = max(contact_score, 0.0)

        # Depth penalty: moderate depth (0.1-0.3mm) is preferred
        depth_score = 1.0 - abs(d - 0.2) / 0.5
        depth_score = max(depth_score, 0.0)

        # Type bonus: cross_hatch and diamond distribute mass more uniformly
        type_bonus = {
            "cross_hatch": 0.15,
            "diamond": 0.20,
            "linear": 0.0,
        }.get(candidate.knurl_type, 0.0)

        # Pitch preference: moderate pitch (0.8-1.2) is most versatile
        pitch_score = 1.0 - abs(p - 1.0) / 1.5
        pitch_score = max(pitch_score, 0.0)

        # Frequency sensitivity bonus: configurations with smaller depth
        # relative to pitch are more predictable in FEA
        predictability = 1.0 - (d / p) if p > 0 else 0.0
        predictability = max(predictability, 0.0)

        overall = (
            0.25 * contact_score
            + 0.20 * depth_score
            + 0.15 * pitch_score
            + 0.20 * predictability
            + type_bonus
            + 0.20 * contact_ratio  # some mass removal is beneficial
        )

        return round(overall, 4)

    async def _run_fea_on_candidate(
        self,
        candidate: CandidateConfig,
        horn_config: dict,
        material: str,
        target_freq_khz: float,
    ) -> FEAResult:
        """Run FEA modal analysis on a single candidate.

        Falls back to analytical estimation if FEA dependencies are
        unavailable.  Runs synchronously because FEA computations are
        CPU-bound and the optimization endpoint is expected to take
        significant time.
        """
        target_hz = target_freq_khz * 1000.0
        return self._run_fea_sync(
            candidate, horn_config, material, target_hz,
        )

    def _run_fea_sync(
        self,
        candidate: CandidateConfig,
        horn_config: dict,
        material: str,
        target_hz: float,
    ) -> FEAResult:
        """Synchronous FEA execution (meant to be called in a thread)."""
        t0 = time.perf_counter()

        try:
            from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import (
                GmshMesher,
            )
            from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
                ModalConfig,
            )
            from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import (
                SolverA,
            )
        except ImportError:
            return self._analytical_fallback(
                candidate, target_hz, time.perf_counter() - t0
            )

        knurl_dict = {
            "type": candidate.knurl_type,
            "pitch_mm": candidate.pitch_mm,
            "depth_mm": candidate.depth_mm,
            "tooth_width_mm": candidate.tooth_width_mm,
        }

        horn_type = horn_config.get("horn_type", "cylindrical")

        if horn_type == "cylindrical":
            dimensions = {
                "diameter_mm": horn_config.get("width_mm", 25.0),
                "length_mm": horn_config.get("height_mm", 80.0),
            }
        else:
            dimensions = {
                "width_mm": horn_config.get("width_mm", 25.0),
                "depth_mm": horn_config.get("length_mm", 25.0),
                "length_mm": horn_config.get("height_mm", 80.0),
            }

        try:
            mesher = GmshMesher()
            mesh = mesher.mesh_parametric_horn(
                horn_type=horn_type,
                dimensions=dimensions,
                mesh_density="medium",
                knurl_info=knurl_dict,
            )

            config = ModalConfig(
                mesh=mesh,
                material_name=material,
                n_modes=20,
                target_frequency_hz=target_hz,
            )
            solver = SolverA()
            modal = solver.modal_analysis(config)

            solve_time = time.perf_counter() - t0

            freqs = modal.get("frequencies_hz", [])
            closest_hz = (
                min(freqs, key=lambda f: abs(f - target_hz))
                if freqs else 0.0
            )
            deviation = (
                abs(closest_hz - target_hz) / target_hz * 100.0
                if target_hz > 0 and closest_hz > 0 else 100.0
            )
            uniformity = modal.get("amplitude_uniformity") or 0.0

            # FEA quality score: combine frequency match and uniformity
            freq_match_score = max(0.0, 1.0 - deviation / 10.0)
            fea_score = 0.6 * freq_match_score + 0.4 * uniformity

            return FEAResult(
                config=candidate,
                closest_mode_hz=closest_hz,
                frequency_deviation_percent=round(deviation, 3),
                amplitude_uniformity=round(uniformity, 4),
                node_count=mesh.nodes.shape[0],
                element_count=mesh.elements.shape[0],
                solve_time_s=round(solve_time, 3),
                mode_count=len(freqs),
                analytical_score=candidate.analytical_score,
                fea_score=round(fea_score, 4),
            )
        except Exception as exc:
            logger.warning(
                "FEA failed for %s p=%.1f d=%.1f: %s",
                candidate.knurl_type, candidate.pitch_mm,
                candidate.depth_mm, exc,
            )
            return self._analytical_fallback(
                candidate, target_hz, time.perf_counter() - t0
            )

    def _analytical_fallback(
        self,
        candidate: CandidateConfig,
        target_hz: float,
        elapsed: float,
    ) -> FEAResult:
        """Create an FEAResult using analytical estimation when FEA is unavailable.

        Estimates frequency deviation based on mass removal from the knurl
        pattern.  This is approximate but allows the optimizer to still rank
        candidates meaningfully.
        """
        # Estimate frequency shift from material removal
        p = candidate.pitch_mm
        tw = candidate.tooth_width_mm
        d = candidate.depth_mm

        if candidate.knurl_type == "linear":
            contact_ratio = tw / p if p > 0 else 1.0
        elif candidate.knurl_type in ("cross_hatch", "diamond"):
            contact_ratio = (tw / p) ** 2 if p > 0 else 1.0
        else:
            contact_ratio = 1.0

        # Material removed fraction (simplified)
        volume_fraction_removed = (1.0 - contact_ratio) * (d / 80.0)

        # Frequency shift estimate: removing material increases frequency
        # (less mass, similar stiffness for shallow knurls)
        estimated_shift_pct = volume_fraction_removed * 5.0  # ~5% per 1% mass
        estimated_hz = target_hz * (1.0 + estimated_shift_pct / 100.0)
        deviation = abs(estimated_hz - target_hz) / target_hz * 100.0

        # Estimate uniformity based on knurl type symmetry
        uniformity_map = {
            "diamond": 0.85,
            "cross_hatch": 0.80,
            "linear": 0.70,
        }
        uniformity = uniformity_map.get(candidate.knurl_type, 0.70)

        # Adjust uniformity by depth (deeper = less uniform)
        uniformity *= max(0.5, 1.0 - d * 0.5)

        freq_match_score = max(0.0, 1.0 - deviation / 10.0)
        fea_score = 0.6 * freq_match_score + 0.4 * uniformity

        return FEAResult(
            config=candidate,
            closest_mode_hz=round(estimated_hz, 1),
            frequency_deviation_percent=round(deviation, 3),
            amplitude_uniformity=round(uniformity, 4),
            node_count=0,
            element_count=0,
            solve_time_s=round(elapsed, 3),
            mode_count=0,
            analytical_score=candidate.analytical_score,
            fea_score=round(fea_score, 4),
        )

    def _compute_pareto_front(
        self, results: list[FEAResult]
    ) -> list[FEAResult]:
        """Compute Pareto front: maximize uniformity, minimize freq deviation.

        A candidate is Pareto-optimal if no other candidate is simultaneously
        better on both objectives.
        """
        pareto: list[FEAResult] = []

        for r in results:
            dominated = False
            for other in results:
                if r is other:
                    continue
                # 'other' dominates 'r' if it has:
                #   - lower or equal deviation AND higher or equal uniformity
                #   - with at least one strict improvement
                if (
                    other.frequency_deviation_percent
                    <= r.frequency_deviation_percent
                    and other.amplitude_uniformity >= r.amplitude_uniformity
                    and (
                        other.frequency_deviation_percent
                        < r.frequency_deviation_percent
                        or other.amplitude_uniformity
                        > r.amplitude_uniformity
                    )
                ):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r)

        pareto.sort(key=lambda r: r.fea_score, reverse=True)
        return pareto

    @staticmethod
    def _result_to_dict(result: FEAResult) -> dict:
        """Serialize an FEAResult to a JSON-compatible dict."""
        return {
            "knurl_type": result.config.knurl_type,
            "pitch_mm": result.config.pitch_mm,
            "depth_mm": result.config.depth_mm,
            "tooth_width_mm": result.config.tooth_width_mm,
            "closest_mode_hz": result.closest_mode_hz,
            "frequency_deviation_percent": result.frequency_deviation_percent,
            "amplitude_uniformity": result.amplitude_uniformity,
            "node_count": result.node_count,
            "element_count": result.element_count,
            "solve_time_s": result.solve_time_s,
            "mode_count": result.mode_count,
            "analytical_score": result.analytical_score,
            "fea_score": result.fea_score,
        }
