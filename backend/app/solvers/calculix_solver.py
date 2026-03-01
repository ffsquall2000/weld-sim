"""CalculiX solver integration with numpy/scipy fallback.

This backend provides modal, harmonic, and static structural analysis
capabilities for ultrasonic welding simulations.  When the ``ccx`` CLI is
available on PATH it uses the real CalculiX engine; otherwise it falls back
to simplified numpy/scipy analytical models that produce physically
reasonable approximate results.

Supported analysis types:
  - modal
  - harmonic
  - static_structural
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from backend.app.solvers.base import (
    AnalysisType,
    FieldData,
    PreparedJob,
    ProgressCallback,
    SolverBackend,
    SolverConfig,
    SolverResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _check_ccx_available() -> bool:
    """Check if CalculiX (ccx) CLI is available on PATH."""
    return shutil.which("ccx") is not None


def _extract_material_props(config: SolverConfig) -> dict[str, float]:
    """Extract material properties from the first MaterialAssignment."""
    defaults: dict[str, float] = {
        "E_pa": 113.8e9,        # Ti-6Al-4V Young's modulus
        "nu": 0.342,            # Poisson's ratio
        "rho_kg_m3": 4430.0,    # Density
        "yield_mpa": 880.0,     # Yield strength
        "ultimate_mpa": 950.0,  # Ultimate tensile strength
        "fatigue_limit_mpa": 510.0,  # Endurance limit for Ti-6Al-4V
    }
    if config.material_assignments:
        props = dict(config.material_assignments[0].properties)
        for key, default_val in defaults.items():
            props.setdefault(key, default_val)
        return props
    return dict(defaults)


def _generate_hex_mesh(
    width: float, height: float, length: float,
    nx: int = 10, ny: int = 25, nz: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a structured hexahedral mesh for the horn geometry.

    Parameters
    ----------
    width, height, length : float
        Horn dimensions in mm.  *height* is along the vibration axis (Y).
    nx, ny, nz : int
        Number of nodes along each axis.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
    cells  : np.ndarray, shape (M, 8)
    """
    x = np.linspace(0.0, width, nx)
    y = np.linspace(0.0, height, ny)
    z = np.linspace(0.0, length, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    cell_list: list[np.ndarray] = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            for k in range(nz - 1):
                n0 = i * ny * nz + j * nz + k
                n1 = n0 + 1
                n2 = n0 + nz
                n3 = n2 + 1
                n4 = n0 + ny * nz
                n5 = n4 + 1
                n6 = n4 + nz
                n7 = n6 + 1
                cell_list.append(
                    np.array([n0, n4, n6, n2, n1, n5, n7, n3], dtype=np.int32)
                )

    cells = np.array(cell_list, dtype=np.int32) if cell_list else np.zeros((0, 8), dtype=np.int32)
    return points, cells


# ---------------------------------------------------------------------------
# Beam theory helpers for modal analysis
# ---------------------------------------------------------------------------

def _euler_bernoulli_frequencies(
    E: float,
    rho: float,
    width: float,
    height: float,
    length: float,
    n_modes: int = 6,
) -> list[float]:
    """Compute natural frequencies using Euler-Bernoulli beam theory.

    For a free-free beam the characteristic equation gives:
        beta_n * L = 4.7300, 7.8532, 10.9956, 14.1372, ...

    f_n = (beta_n^2 / (2 * pi * L^2)) * sqrt(E * I / (rho * A))

    Also includes longitudinal modes:
        f_long_n = (n / (2 * L)) * sqrt(E / rho)

    Parameters
    ----------
    E : float
        Young's modulus in Pa.
    rho : float
        Density in kg/m^3.
    width, height, length : float
        Dimensions in mm.  *height* is the beam length along vibration axis.
    n_modes : int
        Number of modes to compute.

    Returns
    -------
    list of float : sorted natural frequencies in Hz.
    """
    L_m = height / 1000.0
    W_m = width / 1000.0
    D_m = length / 1000.0

    A = W_m * D_m  # cross-section area (m^2)
    I_yy = W_m * D_m ** 3 / 12.0  # second moment of area (m^4)
    I_zz = D_m * W_m ** 3 / 12.0

    # Free-free beam eigenvalues (beta_n * L)
    beta_L_values = [4.7300, 7.8532, 10.9956, 14.1372, 17.2788, 20.4204]

    freqs: list[float] = []

    # Bending modes (about both axes)
    for i, bL in enumerate(beta_L_values):
        if len(freqs) >= n_modes * 2:
            break
        beta_n = bL / L_m
        # Bending about y-axis (width direction)
        f_yy = (beta_n ** 2 / (2.0 * np.pi)) * np.sqrt(E * I_yy / (rho * A))
        freqs.append(float(f_yy))
        # Bending about z-axis (length direction)
        f_zz = (beta_n ** 2 / (2.0 * np.pi)) * np.sqrt(E * I_zz / (rho * A))
        freqs.append(float(f_zz))

    # Longitudinal modes
    for n in range(1, n_modes + 1):
        f_long = (n / (2.0 * L_m)) * np.sqrt(E / rho)
        freqs.append(float(f_long))

    # Torsional modes (approximate for rectangular cross-section)
    # f_torsion_n = (n / (2L)) * sqrt(G * J / (rho * I_p))
    G = E / (2.0 * (1.0 + 0.342))  # shear modulus (approximate)
    a = max(W_m, D_m) / 2.0
    b = min(W_m, D_m) / 2.0
    # Saint-Venant torsional constant for rectangle
    J = a * b ** 3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b ** 4 / (12.0 * a ** 4)))
    I_p = rho * A * (W_m ** 2 + D_m ** 2) / 12.0  # polar mass moment per length
    for n in range(1, n_modes // 2 + 1):
        f_tor = (n / (2.0 * L_m)) * np.sqrt(G * J / (I_p / rho))
        freqs.append(float(f_tor))

    freqs.sort()
    return freqs[:n_modes]


# ---------------------------------------------------------------------------
# CalculiXSolver
# ---------------------------------------------------------------------------

class CalculiXSolver(SolverBackend):
    """CalculiX solver for modal, harmonic, and static structural analysis.

    Falls back to numpy/scipy simplified models when the ``ccx`` CLI
    is not available on PATH.
    """

    # ------------------------------------------------------------------
    # SolverBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "calculix"

    @property
    def supported_analyses(self) -> list[AnalysisType]:
        return [
            AnalysisType.MODAL,
            AnalysisType.HARMONIC,
            AnalysisType.STATIC_STRUCTURAL,
        ]

    async def prepare(self, config: SolverConfig) -> PreparedJob:
        """Validate configuration and create a :class:`PreparedJob`.

        For the fallback solver no external input files are written; all
        information is carried in the job metadata.  When CalculiX is
        available, this writes the ``.inp`` input deck to disk.
        """
        if config.analysis_type not in self.supported_analyses:
            raise ValueError(
                f"CalculiXSolver does not support {config.analysis_type.value!r}. "
                f"Supported: {[a.value for a in self.supported_analyses]}"
            )

        work_dir = Path(tempfile.mkdtemp(prefix="calculix_"))
        material_props = _extract_material_props(config)
        has_ccx = _check_ccx_available()

        input_files: list[str] = []
        if config.mesh_path and Path(config.mesh_path).exists():
            input_files.append(config.mesh_path)

        # When ccx is available, write the .inp input deck
        if has_ccx:
            inp_path = work_dir / "analysis.inp"
            inp_content = self._generate_inp(config, material_props)
            inp_path.write_text(inp_content, encoding="utf-8")
            input_files.append(str(inp_path))

        return PreparedJob(
            job_id=PreparedJob.new_id(),
            work_dir=str(work_dir),
            input_files=input_files,
            solver_config=config,
            metadata={
                "material_props": material_props,
                "analysis_type": config.analysis_type.value,
                "has_ccx": has_ccx,
            },
        )

    async def run(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback] = None,
    ) -> SolverResult:
        """Execute the solver, dispatching to ccx CLI or the fallback."""
        if job.metadata.get("has_ccx"):
            return await asyncio.to_thread(self._run_ccx, job, progress)
        else:
            return await asyncio.to_thread(self._run_fallback, job, progress)

    def read_results(self, result: SolverResult) -> FieldData:
        """Return the :class:`FieldData` already attached to the result.

        Both the CalculiX and fallback solvers populate ``result.field_data``
        during ``run()``.
        """
        if result.field_data is not None:
            return result.field_data
        raise RuntimeError(
            f"No field_data in SolverResult for job {result.job_id}.  "
            "Was the analysis run successfully?"
        )

    # ------------------------------------------------------------------
    # CalculiX .inp generation
    # ------------------------------------------------------------------

    def _generate_inp(
        self,
        config: SolverConfig,
        material_props: dict[str, float],
    ) -> str:
        """Generate a minimal CalculiX input deck (.inp).

        This creates a template that ``ccx`` can parse.  A complete
        production setup would include a full mesh definition.
        """
        params = config.parameters or {}
        E = material_props.get("E_pa", 113.8e9)
        nu = material_props.get("nu", 0.342)
        rho = material_props.get("rho_kg_m3", 4430.0)
        n_modes = int(params.get("n_modes", 10))
        freq_min = float(params.get("freq_min", 100.0))
        freq_max = float(params.get("freq_max", 50000.0))

        # Header
        inp_lines = [
            "** CalculiX input deck -- auto-generated by CalculiXSolver backend",
            f"** Analysis: {config.analysis_type.value}",
            "**",
        ]

        # Material definition
        inp_lines.extend([
            "*MATERIAL, NAME=HORN_MATERIAL",
            "*ELASTIC",
            f"{E:.6e}, {nu}",
            "*DENSITY",
            f"{rho}",
            "",
        ])

        # Analysis type-specific step
        if config.analysis_type == AnalysisType.MODAL:
            inp_lines.extend([
                "*STEP",
                "*FREQUENCY",
                f"{n_modes}",
                "*NODE FILE",
                "U",
                "*END STEP",
            ])
        elif config.analysis_type == AnalysisType.HARMONIC:
            n_sweep = int(params.get("n_sweep", 100))
            damping = float(params.get("damping_ratio", 0.001))
            inp_lines.extend([
                "*STEP",
                f"*STEADY STATE DYNAMICS",
                f"{freq_min}, {freq_max}, {n_sweep}, 1.0",
                f"*MODAL DAMPING",
                f"1, {n_modes}, {damping}",
                "*NODE FILE",
                "U",
                "*END STEP",
            ])
        elif config.analysis_type == AnalysisType.STATIC_STRUCTURAL:
            force_n = float(params.get("force_n", 1000.0))
            inp_lines.extend([
                "*STEP",
                "*STATIC",
                "*CLOAD",
                f"** Applied force: {force_n} N",
                "*NODE FILE",
                "U, S",
                "*EL FILE",
                "S, E",
                "*END STEP",
            ])

        return "\n".join(inp_lines) + "\n"

    # ------------------------------------------------------------------
    # Real CalculiX execution path
    # ------------------------------------------------------------------

    def _run_ccx(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback],
    ) -> SolverResult:
        """Execute real CalculiX solver via ccx CLI."""
        import subprocess

        start = time.perf_counter()
        try:
            if progress:
                progress(5.0, "Starting CalculiX solver...")

            inp_files = [f for f in job.input_files if f.endswith(".inp")]
            if not inp_files:
                raise FileNotFoundError("No .inp input file found in prepared job")

            # ccx expects the input file without the .inp extension
            inp_stem = str(Path(inp_files[0]).with_suffix(""))

            if progress:
                progress(10.0, "Running ccx...")

            result = subprocess.run(
                ["ccx", inp_stem],
                cwd=job.work_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if progress:
                progress(80.0, "Parsing CalculiX output...")

            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                return SolverResult(
                    success=False,
                    job_id=job.job_id,
                    output_files=[],
                    error_message=f"ccx exited with code {result.returncode}",
                    solver_log=result.stdout + "\n" + result.stderr,
                    compute_time_s=round(elapsed, 3),
                )

            # Collect output files (.frd, .dat, .sta)
            work_path = Path(job.work_dir)
            output_files = [
                str(f)
                for f in work_path.iterdir()
                if f.suffix in (".frd", ".dat", ".sta", ".cvg")
            ]

            if progress:
                progress(100.0, "CalculiX analysis complete")

            return SolverResult(
                success=True,
                job_id=job.job_id,
                output_files=output_files,
                metrics={"solver": 1.0},
                solver_log=result.stdout,
                compute_time_s=round(elapsed, 3),
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.exception("CalculiX solver failed: %s", exc)
            return SolverResult(
                success=False,
                job_id=job.job_id,
                output_files=[],
                error_message=str(exc),
                solver_log=f"CalculiX solver failed: {exc}",
                compute_time_s=round(elapsed, 3),
            )

    # ------------------------------------------------------------------
    # numpy/scipy fallback execution path
    # ------------------------------------------------------------------

    def _run_fallback(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback],
    ) -> SolverResult:
        """Numpy/scipy fallback for modal, harmonic, and structural analysis.

        Implements simplified analytical models based on real ultrasonic
        welding physics for rapid prototyping and preview.
        """
        start = time.perf_counter()
        config = job.solver_config
        material = job.metadata.get("material_props", {})
        params = config.parameters or {}

        log_lines: list[str] = [
            "ccx not available, using numpy/scipy fallback"
        ]

        # ---- Material properties (defaults: Ti-6Al-4V) ----
        E = float(material.get("E_pa", 113.8e9))
        nu = float(material.get("nu", 0.342))
        rho = float(material.get("rho_kg_m3", 4430.0))
        yield_mpa = float(material.get("yield_mpa", 880.0))
        ultimate_mpa = float(material.get("ultimate_mpa", 950.0))
        fatigue_limit_mpa = float(material.get("fatigue_limit_mpa", 510.0))

        # ---- Geometry parameters ----
        width = float(params.get("width_mm", 40.0))
        height = float(params.get("height_mm", 100.0))  # vibration axis
        length = float(params.get("length_mm", 30.0))
        horn_type = str(params.get("horn_type", "rectangular"))

        # ---- Target frequency ----
        target_freq = float(params.get("target_frequency_hz", 20000.0))
        if target_freq == 0.0:
            freq_khz = float(params.get("frequency_khz", 20.0))
            target_freq = freq_khz * 1000.0

        n_modes = int(params.get("n_modes", 10))

        log_lines.append(f"Analysis type: {config.analysis_type.value}")
        log_lines.append(f"Material: E={E:.2e} Pa, nu={nu}, rho={rho}")
        log_lines.append(f"Geometry: {width}x{height}x{length} mm ({horn_type})")
        log_lines.append(f"Target frequency: {target_freq:.0f} Hz")

        metrics: dict[str, float] = {}

        if config.analysis_type == AnalysisType.MODAL:
            if progress:
                progress(10.0, "Computing modal analysis...")
            metrics, field_data = self._solve_modal(
                E=E, nu=nu, rho=rho,
                yield_mpa=yield_mpa,
                width=width, height=height, length=length,
                horn_type=horn_type, target_freq=target_freq,
                n_modes=n_modes,
                log_lines=log_lines, progress=progress,
            )
        elif config.analysis_type == AnalysisType.HARMONIC:
            if progress:
                progress(10.0, "Computing harmonic analysis...")
            metrics, field_data = self._solve_harmonic(
                E=E, nu=nu, rho=rho,
                width=width, height=height, length=length,
                horn_type=horn_type, target_freq=target_freq,
                params=params,
                log_lines=log_lines, progress=progress,
            )
        elif config.analysis_type == AnalysisType.STATIC_STRUCTURAL:
            if progress:
                progress(10.0, "Computing static structural analysis...")
            metrics, field_data = self._solve_static_structural(
                E=E, nu=nu, rho=rho,
                yield_mpa=yield_mpa,
                ultimate_mpa=ultimate_mpa,
                fatigue_limit_mpa=fatigue_limit_mpa,
                width=width, height=height, length=length,
                horn_type=horn_type, params=params,
                log_lines=log_lines, progress=progress,
            )
        else:
            raise ValueError(f"Unsupported analysis type: {config.analysis_type.value}")

        elapsed = time.perf_counter() - start
        metrics["node_count"] = float(field_data.n_points)
        metrics["element_count"] = float(field_data.n_cells)
        log_lines.append(
            f"Mesh: {field_data.n_points} nodes, {field_data.n_cells} elements"
        )
        log_lines.append(f"Compute time: {elapsed:.3f} s")

        if progress:
            progress(100.0, "Analysis complete")

        return SolverResult(
            success=True,
            job_id=job.job_id,
            output_files=[],
            field_data=field_data,
            metrics=metrics,
            solver_log="\n".join(log_lines),
            compute_time_s=round(elapsed, 3),
        )

    # ------------------------------------------------------------------
    # Modal analysis (fallback)
    # ------------------------------------------------------------------

    def _solve_modal(
        self,
        *,
        E: float,
        nu: float,
        rho: float,
        yield_mpa: float,
        width: float,
        height: float,
        length: float,
        horn_type: str,
        target_freq: float,
        n_modes: int,
        log_lines: list[str],
        progress: Optional[ProgressCallback],
    ) -> tuple[dict[str, float], FieldData]:
        """Modal analysis using Euler-Bernoulli beam theory.

        Computes natural frequencies of the horn treating it as a free-free
        beam with longitudinal, bending, and torsional modes.

        Physics:
          f_n = (beta_n^2 / (2*pi*L^2)) * sqrt(E*I / (rho*A))
        """
        if progress:
            progress(25.0, "Computing natural frequencies...")

        # Compute natural frequencies
        frequencies = _euler_bernoulli_frequencies(
            E=E, rho=rho,
            width=width, height=height, length=length,
            n_modes=n_modes,
        )

        # Find the mode closest to target frequency
        if frequencies:
            freq_array = np.array(frequencies)
            closest_idx = int(np.argmin(np.abs(freq_array - target_freq)))
            natural_freq = frequencies[closest_idx]
        else:
            natural_freq = target_freq
            closest_idx = 0

        freq_deviation_pct = ((natural_freq - target_freq) / target_freq) * 100.0

        if progress:
            progress(45.0, "Computing modal separation...")

        # Modal separation: gap to nearest mode that is NOT the target
        modal_sep = float("inf")
        for i, f in enumerate(frequencies):
            if i != closest_idx:
                gap = abs(natural_freq - f)
                if gap < modal_sep and gap > 0:
                    modal_sep = gap
        if modal_sep == float("inf"):
            modal_sep = 0.0

        if progress:
            progress(60.0, "Computing stress distribution...")

        # Max von Mises stress at antinodes
        # For longitudinal vibration: sigma_max = E * epsilon_max
        # epsilon_max ~ A * omega^2 / c^2, where A is amplitude
        # Typical vibration amplitude for ultrasonic welding: 10-50 um peak-to-peak
        amplitude_um = float(30.0)  # typical amplitude
        omega = 2.0 * np.pi * natural_freq
        L_m = height / 1000.0
        c_sound = np.sqrt(E / rho)
        # Strain at antinode: epsilon = omega * A / c
        epsilon_max = omega * (amplitude_um * 1e-6) / c_sound
        sigma_max_pa = E * epsilon_max
        max_von_mises_mpa = float(sigma_max_pa / 1e6)

        # Amplitude uniformity on contact face
        # Depends on horn type and mode shape
        horn_type_lower = horn_type.lower()
        if horn_type_lower == "catenoidal":
            amplitude_uniformity = 0.95
        elif horn_type_lower == "exponential":
            amplitude_uniformity = 0.92
        elif horn_type_lower == "cylindrical":
            amplitude_uniformity = 0.90
        elif horn_type_lower == "stepped":
            amplitude_uniformity = 0.85
        else:  # rectangular
            amplitude_uniformity = 0.88

        if progress:
            progress(75.0, "Generating mode shape field data...")

        # --- Metrics ---
        metrics: dict[str, float] = {
            "natural_frequency_hz": float(round(natural_freq, 2)),
            "frequency_deviation_pct": float(round(freq_deviation_pct, 3)),
            "modal_separation_hz": float(round(modal_sep, 2)),
            "max_von_mises_stress_mpa": float(round(max_von_mises_mpa, 2)),
            "amplitude_uniformity": float(round(amplitude_uniformity, 4)),
            "target_frequency_hz": float(target_freq),
            "n_modes_computed": float(len(frequencies)),
        }

        # Add individual mode frequencies to metrics
        for i, f in enumerate(frequencies):
            metrics[f"mode_{i + 1}_frequency_hz"] = float(round(f, 2))

        log_lines.append(f"Closest mode: {natural_freq:.1f} Hz (#{closest_idx + 1})")
        log_lines.append(f"Deviation: {freq_deviation_pct:.3f} %")
        log_lines.append(f"Modal separation: {modal_sep:.1f} Hz")
        log_lines.append(f"Max stress: {max_von_mises_mpa:.1f} MPa")
        log_lines.append(f"Modes computed: {[f'{f:.0f}' for f in frequencies]}")

        # --- Generate mode shape field data ---
        nx, ny, nz = 10, 25, 10
        points, cells = _generate_hex_mesh(width, height, length, nx, ny, nz)

        y_norm = points[:, 1] / height  # normalised position [0, 1]

        # Mode shape: sinusoidal displacement pattern
        # For the closest mode (longitudinal)
        n_half_waves = closest_idx + 1
        mode_shape_y = np.sin(n_half_waves * np.pi * y_norm)

        # 3D mode shape (primarily Y-direction for longitudinal mode)
        displacement_y = mode_shape_y * amplitude_um  # um
        displacement_x = np.zeros_like(mode_shape_y)
        displacement_z = np.zeros_like(mode_shape_y)

        # Von Mises stress distribution (proportional to strain = d(displacement)/dy)
        # Strain is proportional to cos(n * pi * y / L)
        strain_pattern = np.abs(np.cos(n_half_waves * np.pi * y_norm))
        stress_field = max_von_mises_mpa * strain_pattern

        field_data = FieldData(
            points=points,
            cells=[cells],
            cell_types=["hex8"],
            point_data={
                "displacement_y": displacement_y,
                "displacement_x": displacement_x,
                "displacement_z": displacement_z,
                "von_mises_stress": stress_field,
            },
            metadata={
                "analysis_type": "modal",
                "horn_type": horn_type,
                "mode_frequencies_hz": frequencies,
                "closest_mode_index": closest_idx,
            },
        )

        return metrics, field_data

    # ------------------------------------------------------------------
    # Harmonic analysis (fallback)
    # ------------------------------------------------------------------

    def _solve_harmonic(
        self,
        *,
        E: float,
        nu: float,
        rho: float,
        width: float,
        height: float,
        length: float,
        horn_type: str,
        target_freq: float,
        params: dict[str, Any],
        log_lines: list[str],
        progress: Optional[ProgressCallback],
    ) -> tuple[dict[str, float], FieldData]:
        """Harmonic frequency response analysis.

        Computes the forced response of the horn at the excitation frequency,
        including amplitude amplification (horn gain) and energy coupling.

        Physics:
          - Horn gain: area ratio * dynamic magnification factor
          - Amplitude uniformity: frequency-dependent
          - Energy coupling: based on impedance matching at resonance
        """
        if progress:
            progress(25.0, "Computing frequency response function...")

        L_m = height / 1000.0
        W_m = width / 1000.0
        D_m = length / 1000.0

        # Natural frequency of the horn (longitudinal, first mode)
        f_natural = (1.0 / (2.0 * L_m)) * np.sqrt(E / rho)

        # Find the harmonic closest to target
        n_harmonic = max(1, round(target_freq / f_natural))
        f_resonance = n_harmonic * f_natural

        # Damping ratio
        damping_ratio = float(params.get("damping_ratio", 0.001))
        # Typical Q factor for titanium horns: 500-5000
        Q_factor = 1.0 / (2.0 * damping_ratio) if damping_ratio > 0 else 2000.0

        # Frequency ratio
        r = target_freq / f_resonance if f_resonance > 0 else 1.0

        # Dynamic magnification factor (single-DOF forced vibration)
        # H(r) = 1 / sqrt((1 - r^2)^2 + (2*zeta*r)^2)
        denominator = np.sqrt((1.0 - r ** 2) ** 2 + (2.0 * damping_ratio * r) ** 2)
        dynamic_magnification = 1.0 / denominator if denominator > 0 else Q_factor

        if progress:
            progress(45.0, "Computing horn gain...")

        # --- Horn gain ---
        # Geometric gain from cross-section ratio
        horn_type_lower = horn_type.lower()
        if horn_type_lower == "exponential":
            taper_ratio = 0.33
            geometric_gain = float(np.sqrt(1.0 / taper_ratio))
        elif horn_type_lower == "stepped":
            step_ratio = 0.25
            geometric_gain = 1.0 / step_ratio
        elif horn_type_lower == "catenoidal":
            taper_ratio = 0.30
            geometric_gain = float(np.sqrt(1.0 / taper_ratio)) * 1.15
        elif horn_type_lower == "cylindrical":
            geometric_gain = 1.0
        else:  # rectangular
            geometric_gain = 1.0

        # Effective horn gain includes dynamic effects near resonance
        # Near resonance, the gain can exceed the geometric ratio
        horn_gain = geometric_gain * min(dynamic_magnification, Q_factor * 0.1)
        # Clamp to physically reasonable range
        horn_gain = float(np.clip(horn_gain, 0.5, 20.0))

        if progress:
            progress(60.0, "Computing amplitude uniformity and coupling...")

        # --- Amplitude uniformity at target frequency ---
        # Off-resonance degrades uniformity
        freq_deviation = abs(target_freq - f_resonance) / f_resonance
        base_uniformity = 0.92 if horn_type_lower in ("catenoidal", "exponential") else 0.88
        # Degrade uniformity off-resonance
        uniformity = base_uniformity * (1.0 - 0.5 * min(freq_deviation, 0.1))
        amplitude_uniformity = float(np.clip(uniformity, 0.70, 0.98))

        # --- Energy coupling efficiency ---
        # Based on impedance matching between transducer and horn at resonance
        # eta = 1 / (1 + (r_mech / (omega * m))^2) at resonance => near 1
        # Off-resonance: decreases rapidly
        # Simplified: eta ~ H(r)^2 / (H(r)^2 + loss_factor)
        loss_factor = 2.0 * damping_ratio
        H_sq = dynamic_magnification ** 2
        energy_coupling = H_sq / (H_sq + loss_factor) if (H_sq + loss_factor) > 0 else 0.0
        energy_coupling = float(np.clip(energy_coupling, 0.0, 1.0))

        if progress:
            progress(75.0, "Computing frequency sweep and generating field data...")

        # --- Frequency sweep (for FRF visualization) ---
        freq_min = float(params.get("freq_min", target_freq * 0.8))
        freq_max = float(params.get("freq_max", target_freq * 1.2))
        n_sweep = int(params.get("n_sweep", 100))
        sweep_freqs = np.linspace(freq_min, freq_max, n_sweep)
        r_sweep = sweep_freqs / f_resonance
        H_sweep = 1.0 / np.sqrt(
            (1.0 - r_sweep ** 2) ** 2 + (2.0 * damping_ratio * r_sweep) ** 2
        )

        # --- Assemble metrics ---
        metrics: dict[str, float] = {
            "horn_gain": float(round(horn_gain, 3)),
            "amplitude_uniformity": float(round(amplitude_uniformity, 4)),
            "energy_coupling_efficiency": float(round(energy_coupling, 4)),
            "resonance_frequency_hz": float(round(f_resonance, 2)),
            "target_frequency_hz": float(target_freq),
            "dynamic_magnification": float(round(dynamic_magnification, 3)),
            "q_factor": float(round(Q_factor, 1)),
            "geometric_gain": float(round(geometric_gain, 3)),
        }

        log_lines.append(f"Resonance: {f_resonance:.1f} Hz")
        log_lines.append(f"Horn gain: {horn_gain:.3f}")
        log_lines.append(f"Uniformity: {amplitude_uniformity:.4f}")
        log_lines.append(f"Energy coupling: {energy_coupling:.4f}")
        log_lines.append(f"Q factor: {Q_factor:.0f}")

        # --- Generate field data (displacement at target frequency) ---
        nx, ny, nz = 10, 25, 10
        points, cells = _generate_hex_mesh(width, height, length, nx, ny, nz)
        y_norm = points[:, 1] / height

        # Displacement magnitude at target frequency
        displacement_magnitude = horn_gain * np.abs(
            np.sin(n_harmonic * np.pi * y_norm)
        )

        # Phase field (for animated display)
        phase_field = np.angle(
            np.sin(n_harmonic * np.pi * y_norm)
            * np.exp(1j * np.arctan2(
                2.0 * damping_ratio * r,
                1.0 - r ** 2,
            ))
        )

        field_data = FieldData(
            points=points,
            cells=[cells],
            cell_types=["hex8"],
            point_data={
                "displacement_magnitude": displacement_magnitude,
                "phase": phase_field,
            },
            metadata={
                "analysis_type": "harmonic",
                "horn_type": horn_type,
                "sweep_frequencies_hz": sweep_freqs.tolist(),
                "sweep_response": H_sweep.tolist(),
            },
        )

        return metrics, field_data

    # ------------------------------------------------------------------
    # Static structural analysis (fallback)
    # ------------------------------------------------------------------

    def _solve_static_structural(
        self,
        *,
        E: float,
        nu: float,
        rho: float,
        yield_mpa: float,
        ultimate_mpa: float,
        fatigue_limit_mpa: float,
        width: float,
        height: float,
        length: float,
        horn_type: str,
        params: dict[str, Any],
        log_lines: list[str],
        progress: Optional[ProgressCallback],
    ) -> tuple[dict[str, float], FieldData]:
        """Static structural analysis (linear elasticity).

        Computes stress, deflection, safety factor, and fatigue life
        estimate under static loading conditions.

        Physics:
          - Stress: sigma = F/A with stress concentration factors
          - Deflection: beam theory (Euler-Bernoulli)
          - Fatigue: S-N curve approximation (Basquin equation)
        """
        if progress:
            progress(25.0, "Computing stress distribution...")

        # Applied loads
        force_n = float(params.get("force_n", 1000.0))
        # Additional dynamic load from ultrasonic vibration
        amplitude_um = float(params.get("amplitude_um", 30.0))
        target_freq = float(params.get("target_frequency_hz", 20000.0))

        # Cross-section properties
        W_m = width / 1000.0
        D_m = length / 1000.0
        L_m = height / 1000.0
        A_cross = W_m * D_m  # m^2
        I_yy = W_m * D_m ** 3 / 12.0  # m^4
        I_zz = D_m * W_m ** 3 / 12.0  # m^4
        I_min = min(I_yy, I_zz)

        # --- Direct stress from applied force ---
        sigma_direct_pa = force_n / A_cross  # Pa
        sigma_direct_mpa = sigma_direct_pa / 1e6

        # --- Bending stress (if force is offset or eccentric) ---
        # Assume worst-case eccentricity of 5% of width
        eccentricity = 0.05 * max(W_m, D_m)
        M_bending = force_n * eccentricity  # N*m
        c_max = max(W_m, D_m) / 2.0
        sigma_bending_pa = M_bending * c_max / I_min if I_min > 0 else 0.0
        sigma_bending_mpa = sigma_bending_pa / 1e6

        # --- Stress concentration factor ---
        # Depends on horn geometry transitions
        horn_type_lower = horn_type.lower()
        if horn_type_lower == "stepped":
            # Stepped horns have high stress concentration at the step
            K_t = 2.5
        elif horn_type_lower == "exponential":
            K_t = 1.3
        elif horn_type_lower == "catenoidal":
            K_t = 1.2
        else:
            # Rectangular / cylindrical: fillet radius dependent
            K_t = 1.5

        # --- Von Mises stress ---
        # Combined loading: sigma_vm = K_t * sqrt(sigma_x^2 + 3*tau^2)
        # For primarily axial loading with bending:
        sigma_max_mpa = K_t * (sigma_direct_mpa + sigma_bending_mpa)

        # Add dynamic stress from vibration
        omega = 2.0 * np.pi * target_freq
        c_sound = np.sqrt(E / rho)
        epsilon_dynamic = omega * (amplitude_um * 1e-6) / c_sound
        sigma_dynamic_mpa = (E * epsilon_dynamic) / 1e6
        max_von_mises_mpa = float(np.sqrt(
            sigma_max_mpa ** 2 + sigma_dynamic_mpa ** 2
        ))

        if progress:
            progress(45.0, "Computing safety factor...")

        # --- Safety factor ---
        stress_safety_factor = yield_mpa / max_von_mises_mpa if max_von_mises_mpa > 0 else 999.0

        if progress:
            progress(55.0, "Computing deflection...")

        # --- Maximum deflection ---
        # Cantilever beam: delta = F * L^3 / (3 * E * I)
        # Simply-supported beam: delta = F * L^3 / (48 * E * I)
        # For horn clamped at transducer end:
        deflection_m = force_n * L_m ** 3 / (3.0 * E * I_min) if I_min > 0 else 0.0
        max_deflection_mm = deflection_m * 1000.0

        if progress:
            progress(70.0, "Computing fatigue life estimate...")

        # --- Fatigue life estimate (S-N curve / Basquin equation) ---
        # N = (S_a / sigma_f')^(1/b)
        # For Ti-6Al-4V:
        #   sigma_f' ~ 1.5 * ultimate (fatigue strength coefficient)
        #   b ~ -0.095 (fatigue strength exponent)
        # Alternating stress from vibration:
        S_a = sigma_dynamic_mpa  # alternating stress amplitude (MPa)
        # Mean stress from static loading:
        S_m = sigma_max_mpa

        # Goodman correction for mean stress:
        # S_a_eff = S_a / (1 - S_m / S_ult)
        if S_m < ultimate_mpa:
            S_a_eff = S_a / (1.0 - S_m / ultimate_mpa)
        else:
            S_a_eff = S_a * 10.0  # very high mean stress

        # Basquin equation: S_a = sigma_f' * (2N)^b
        sigma_f_prime = 1.5 * ultimate_mpa  # fatigue strength coefficient
        b_basquin = -0.095  # fatigue strength exponent for Ti alloys

        if S_a_eff > 0 and S_a_eff < sigma_f_prime:
            # 2N = (S_a / sigma_f')^(1/b)
            two_N = (S_a_eff / sigma_f_prime) ** (1.0 / b_basquin)
            N_cycles = two_N / 2.0
            # Clamp to reasonable range
            N_cycles = float(np.clip(N_cycles, 1.0, 1e15))
        elif S_a_eff <= 0:
            N_cycles = 1e15  # essentially infinite life
        else:
            N_cycles = 1.0  # immediate failure

        # At 20 kHz, number of seconds of operation:
        if target_freq > 0:
            life_seconds = N_cycles / target_freq
        else:
            life_seconds = float("inf")

        if progress:
            progress(85.0, "Generating stress field data...")

        # --- Metrics ---
        metrics: dict[str, float] = {
            "max_von_mises_stress_mpa": float(round(max_von_mises_mpa, 2)),
            "stress_safety_factor": float(round(stress_safety_factor, 3)),
            "max_deflection_mm": float(round(max_deflection_mm, 6)),
            "fatigue_cycle_estimate": float(round(N_cycles, 0)),
            "fatigue_life_seconds": float(round(life_seconds, 1)),
            "static_stress_mpa": float(round(sigma_max_mpa, 2)),
            "dynamic_stress_mpa": float(round(sigma_dynamic_mpa, 2)),
            "stress_concentration_factor": float(K_t),
            "applied_force_n": float(force_n),
        }

        log_lines.append(f"Max von Mises: {max_von_mises_mpa:.1f} MPa")
        log_lines.append(f"Safety factor: {stress_safety_factor:.3f}")
        log_lines.append(f"Max deflection: {max_deflection_mm:.6f} mm")
        log_lines.append(f"Fatigue cycles: {N_cycles:.2e}")
        log_lines.append(f"Fatigue life: {life_seconds:.1f} s at {target_freq:.0f} Hz")

        # --- Generate stress distribution field data ---
        nx, ny, nz = 10, 25, 10
        points, cells = _generate_hex_mesh(width, height, length, nx, ny, nz)
        y_norm = points[:, 1] / height

        # Stress distribution: higher at base (clamped end), decreases along length
        # Combined static + dynamic pattern
        # Static: linear decrease from base
        static_stress = sigma_max_mpa * (1.0 - 0.7 * y_norm)

        # Dynamic: sinusoidal pattern (stress at antinodes of strain)
        dynamic_stress = sigma_dynamic_mpa * np.abs(
            np.cos(np.pi * y_norm)
        )

        # Combined von Mises
        combined_stress = np.sqrt(static_stress ** 2 + dynamic_stress ** 2)

        # Displacement field
        # Static displacement (cantilever beam shape)
        displacement_static = max_deflection_mm * (
            3.0 * y_norm ** 2 - 2.0 * y_norm ** 3
        )

        # Safety factor field (local)
        safety_factor_field = np.where(
            combined_stress > 0,
            yield_mpa / combined_stress,
            999.0,
        )
        safety_factor_field = np.clip(safety_factor_field, 0.0, 100.0)

        field_data = FieldData(
            points=points,
            cells=[cells],
            cell_types=["hex8"],
            point_data={
                "von_mises_stress": combined_stress,
                "displacement": displacement_static,
                "safety_factor": safety_factor_field,
            },
            metadata={
                "analysis_type": "static_structural",
                "horn_type": horn_type,
            },
        )

        return metrics, field_data
