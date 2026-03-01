"""Preview solver -- wraps the existing numpy/scipy FEA service.

This backend provides fast, approximate modal and harmonic analysis for
ultrasonic welding horns using the pure-Python FEA engine that ships with
the project.  It requires no external solver installation.

Wrapped service: ``web/services/fea_service.py`` (FEAService)
Material data:   ``ultrasonic_weld_master/plugins/geometry_analyzer/fea/material_properties.py``
"""

from __future__ import annotations

import asyncio
import logging
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure the project root is importable so we can reach both
# ``web.services.fea_service`` and ``ultrasonic_weld_master.*``.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Lazy imports -- avoid hard failure if numpy/scipy are not available in
# certain deployment scenarios (e.g. thin API gateway).
_FEAService = None
_get_material = None


def _ensure_imports() -> None:
    """Lazy-load FEAService and material helpers."""
    global _FEAService, _get_material  # noqa: PLW0603
    if _FEAService is not None:
        return
    try:
        from web.services.fea_service import FEAService as _Cls
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
            get_material as _gm,
        )
        _FEAService = _Cls
        _get_material = _gm
    except ImportError as exc:
        raise ImportError(
            "PreviewSolver requires the project-level FEAService and material "
            "database.  Make sure the project root is on sys.path and that "
            "numpy / scipy are installed."
        ) from exc


# ---------------------------------------------------------------------------
# Helper: extract horn parameters from SolverConfig
# ---------------------------------------------------------------------------

def _extract_horn_params(config: SolverConfig) -> dict[str, Any]:
    """Pull horn-specific parameters out of a SolverConfig.

    The preview solver does not use an external mesh file; instead it
    generates a structured hex mesh internally.  All geometry and material
    information is carried in ``config.parameters`` and
    ``config.material_assignments``.

    Expected keys in ``config.parameters``:
        horn_type     -- "rectangular", "exponential", "cylindrical" (default "rectangular")
        width_mm      -- horn width (default 40)
        height_mm     -- horn height / length along vibration axis (default 100)
        length_mm     -- horn depth (default 30)
        frequency_khz -- operating frequency (default 20)
        mesh_density  -- "coarse", "medium", "fine" (default "medium")
        n_modes       -- number of modes to extract (default 10)
    """
    p = config.parameters

    # Material: prefer first MaterialAssignment, then config.parameters
    material = "Titanium Ti-6Al-4V"
    if config.material_assignments:
        material = config.material_assignments[0].material_name
    elif "material" in p:
        material = p["material"]

    return {
        "horn_type": p.get("horn_type", "rectangular"),
        "width_mm": float(p.get("width_mm", 40.0)),
        "height_mm": float(p.get("height_mm", 100.0)),
        "length_mm": float(p.get("length_mm", 30.0)),
        "material": material,
        "frequency_khz": float(p.get("frequency_khz", 20.0)),
        "mesh_density": p.get("mesh_density", "medium"),
        "n_modes": int(p.get("n_modes", 10)),
    }


# ---------------------------------------------------------------------------
# Result conversion helpers
# ---------------------------------------------------------------------------

def _modal_dict_to_field_data(
    raw: dict[str, Any],
    params: dict[str, Any],
) -> FieldData:
    """Convert the dict returned by ``FEAService.run_modal_analysis`` to a
    :class:`FieldData` instance."""
    mesh = raw.get("mesh", {})
    verts = np.array(mesh.get("vertices", []), dtype=np.float64)
    faces = np.array(mesh.get("faces", []), dtype=np.int32)

    point_data: dict[str, np.ndarray] = {}
    cell_data: dict[str, np.ndarray] = {}

    # Attach mode-shape frequencies as metadata
    metadata: dict[str, Any] = {
        "analysis_type": "modal",
        "solve_time_s": raw.get("solve_time_s", 0.0),
        "mode_shapes": raw.get("mode_shapes", []),
        "closest_mode_hz": raw.get("closest_mode_hz"),
        "frequency_deviation_percent": raw.get("frequency_deviation_percent"),
        "horn_params": params,
    }

    cells: list[np.ndarray] = []
    cell_types: list[str] = []
    if len(faces) > 0:
        cells.append(faces)
        cell_types.append("tri3")

    return FieldData(
        points=verts if len(verts) > 0 else np.zeros((0, 3)),
        cells=cells,
        cell_types=cell_types,
        point_data=point_data,
        cell_data=cell_data,
        metadata=metadata,
    )


def _acoustic_dict_to_field_data(
    raw: dict[str, Any],
    params: dict[str, Any],
) -> FieldData:
    """Convert the dict returned by ``FEAService.run_acoustic_analysis`` to a
    :class:`FieldData` instance."""
    mesh = raw.get("mesh", {})
    verts = np.array(mesh.get("vertices", []), dtype=np.float64)
    faces = np.array(mesh.get("faces", []), dtype=np.int32)

    point_data: dict[str, np.ndarray] = {}

    # Amplitude distribution per contact-face node
    amp_dist = raw.get("amplitude_distribution", {})
    if amp_dist:
        amp_values = np.array(amp_dist.get("amplitudes", []), dtype=np.float64)
        amp_positions = np.array(amp_dist.get("node_positions", []), dtype=np.float64)
        if len(amp_values) > 0:
            # Map amplitudes onto full vertex array (zero for non-contact nodes)
            full_amps = np.zeros(len(verts), dtype=np.float64)
            # Try to map by position matching
            if len(amp_positions) > 0 and len(verts) > 0:
                for idx, pos in enumerate(amp_positions):
                    dists = np.linalg.norm(verts - pos, axis=1)
                    nearest = int(np.argmin(dists))
                    if dists[nearest] < 1.0:  # within 1 mm tolerance
                        full_amps[nearest] = amp_values[idx]
            point_data["amplitude"] = full_amps

    metadata: dict[str, Any] = {
        "analysis_type": "harmonic",
        "solve_time_s": raw.get("solve_time_s", 0.0),
        "modes": raw.get("modes", []),
        "closest_mode_hz": raw.get("closest_mode_hz"),
        "frequency_deviation_percent": raw.get("frequency_deviation_percent"),
        "harmonic_response": raw.get("harmonic_response"),
        "amplitude_uniformity": raw.get("amplitude_uniformity"),
        "stress_hotspots": raw.get("stress_hotspots", []),
        "stress_max_mpa": raw.get("stress_max_mpa"),
        "horn_params": params,
    }

    cells: list[np.ndarray] = []
    cell_types: list[str] = []
    if len(faces) > 0:
        cells.append(faces)
        cell_types.append("tri3")

    return FieldData(
        points=verts if len(verts) > 0 else np.zeros((0, 3)),
        cells=cells,
        cell_types=cell_types,
        point_data=point_data,
        cell_data={},
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# PreviewSolver
# ---------------------------------------------------------------------------

class PreviewSolver(SolverBackend):
    """Fast preview solver using the bundled numpy/scipy FEA engine.

    This solver does *not* read an external mesh file.  Instead it generates
    a structured hexahedral mesh internally from the horn geometry parameters
    supplied in :pyattr:`SolverConfig.parameters`.
    """

    # ------------------------------------------------------------------
    # SolverBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "preview"

    @property
    def supported_analyses(self) -> list[AnalysisType]:
        return [AnalysisType.MODAL, AnalysisType.HARMONIC]

    async def prepare(self, config: SolverConfig) -> PreparedJob:
        """Validate configuration and create a :class:`PreparedJob`.

        The preview solver does not write input files to disk; instead the
        ``PreparedJob.metadata`` carries the extracted horn parameters.
        """
        if config.analysis_type not in self.supported_analyses:
            raise ValueError(
                f"PreviewSolver does not support {config.analysis_type.value!r}. "
                f"Supported: {[a.value for a in self.supported_analyses]}"
            )

        params = _extract_horn_params(config)

        # Validate material is known
        _ensure_imports()
        mat = _get_material(params["material"])
        if mat is None:
            raise ValueError(
                f"Unknown material {params['material']!r}.  "
                "Check ultrasonic_weld_master material database."
            )

        work_dir = tempfile.mkdtemp(prefix="preview_solver_")
        job_id = PreparedJob.new_id()

        return PreparedJob(
            job_id=job_id,
            work_dir=work_dir,
            input_files=[],  # no disk-based input
            solver_config=config,
            metadata={"horn_params": params, "material_props": mat},
        )

    async def run(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback] = None,
    ) -> SolverResult:
        """Run the FEA analysis via the bundled FEAService."""
        _ensure_imports()
        params: dict[str, Any] = job.metadata["horn_params"]
        analysis = job.solver_config.analysis_type

        def _report(pct: float, msg: str) -> None:
            if progress is not None:
                progress(pct, msg)

        _report(0.0, "Initializing FEA service")
        fea = _FEAService()
        t0 = time.perf_counter()

        try:
            if analysis == AnalysisType.MODAL:
                _report(10.0, "Building mesh and assembling matrices")
                # Run in a thread so we don't block the event loop
                raw = await asyncio.to_thread(
                    fea.run_modal_analysis,
                    horn_type=params["horn_type"],
                    width_mm=params["width_mm"],
                    height_mm=params["height_mm"],
                    length_mm=params["length_mm"],
                    material=params["material"],
                    frequency_khz=params["frequency_khz"],
                    mesh_density=params["mesh_density"],
                )
                _report(90.0, "Converting results")
                field_data = _modal_dict_to_field_data(raw, params)

                metrics = {
                    "natural_frequency_hz": raw.get("closest_mode_hz", 0.0),
                    "target_frequency_hz": raw.get("target_frequency_hz", 0.0),
                    "frequency_deviation_percent": raw.get(
                        "frequency_deviation_percent", 0.0
                    ),
                    "max_stress_mpa": raw.get("stress_max_mpa", 0.0),
                    "node_count": float(raw.get("node_count", 0)),
                    "element_count": float(raw.get("element_count", 0)),
                }

            elif analysis == AnalysisType.HARMONIC:
                _report(10.0, "Building mesh and assembling matrices")
                raw = await asyncio.to_thread(
                    fea.run_acoustic_analysis,
                    horn_type=params["horn_type"],
                    width_mm=params["width_mm"],
                    height_mm=params["height_mm"],
                    length_mm=params["length_mm"],
                    material=params["material"],
                    frequency_khz=params["frequency_khz"],
                    mesh_density=params["mesh_density"],
                )
                _report(90.0, "Converting results")
                field_data = _acoustic_dict_to_field_data(raw, params)

                metrics = {
                    "natural_frequency_hz": raw.get("closest_mode_hz", 0.0),
                    "target_frequency_hz": raw.get("target_frequency_hz", 0.0),
                    "frequency_deviation_percent": raw.get(
                        "frequency_deviation_percent", 0.0
                    ),
                    "amplitude_uniformity": raw.get("amplitude_uniformity", 0.0),
                    "max_stress_mpa": raw.get("stress_max_mpa", 0.0),
                    "node_count": float(raw.get("node_count", 0)),
                    "element_count": float(raw.get("element_count", 0)),
                }

            else:
                raise ValueError(f"Unsupported analysis type: {analysis.value}")

            elapsed = time.perf_counter() - t0
            _report(100.0, "Analysis complete")

            return SolverResult(
                success=True,
                job_id=job.job_id,
                output_files=[],
                field_data=field_data,
                metrics=metrics,
                compute_time_s=round(elapsed, 3),
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.exception("PreviewSolver run failed: %s", exc)
            return SolverResult(
                success=False,
                job_id=job.job_id,
                error_message=str(exc),
                compute_time_s=round(elapsed, 3),
            )

    def read_results(self, result: SolverResult) -> FieldData:
        """Return the :class:`FieldData` already attached to the result.

        The preview solver populates ``result.field_data`` during ``run()``,
        so this method simply returns it (or raises if it is missing).
        """
        if result.field_data is not None:
            return result.field_data
        raise RuntimeError(
            f"No field_data in SolverResult for job {result.job_id}.  "
            "Was the analysis run successfully?"
        )
