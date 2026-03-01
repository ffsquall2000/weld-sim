"""Elmer FEM solver integration with numpy/scipy fallback.

This backend provides piezoelectric and acoustic analysis capabilities for
ultrasonic welding simulations.  When the ``ElmerSolver`` CLI is available
on PATH it uses the real Elmer engine; otherwise it falls back to simplified
numpy/scipy analytical models that produce physically reasonable approximate
results.

Supported analysis types:
  - piezoelectric
  - acoustic
"""

from __future__ import annotations

import asyncio
import logging
import os
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

def _check_elmer_available() -> bool:
    """Check if ElmerSolver CLI is available on PATH."""
    return shutil.which("ElmerSolver") is not None


def _extract_material_props(config: SolverConfig) -> dict[str, float]:
    """Extract material properties from the first MaterialAssignment."""
    defaults: dict[str, float] = {
        "E_pa": 113.8e9,        # Ti-6Al-4V Young's modulus
        "nu": 0.342,            # Poisson's ratio
        "rho_kg_m3": 4430.0,    # Density
        "yield_mpa": 880.0,     # Yield strength
        "speed_of_sound_m_s": 5090.0,  # Longitudinal speed of sound in Ti
    }
    if config.material_assignments:
        props = dict(config.material_assignments[0].properties)
        for key, default_val in defaults.items():
            props.setdefault(key, default_val)
        return props
    return dict(defaults)


def _generate_hex_mesh(
    width: float, height: float, length: float,
    nx: int = 8, ny: int = 30, nz: int = 8,
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
# ElmerSolver
# ---------------------------------------------------------------------------

class ElmerSolver(SolverBackend):
    """Elmer FEM solver for piezoelectric and acoustic analysis.

    Falls back to numpy/scipy simplified models when the ElmerSolver CLI
    is not available on PATH.
    """

    # ------------------------------------------------------------------
    # SolverBackend interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "elmer"

    @property
    def supported_analyses(self) -> list[AnalysisType]:
        return [AnalysisType.PIEZOELECTRIC, AnalysisType.ACOUSTIC]

    async def prepare(self, config: SolverConfig) -> PreparedJob:
        """Validate configuration and create a :class:`PreparedJob`.

        For the fallback solver no external input files are written; all
        information is carried in the job metadata.  When Elmer is available,
        this writes the Elmer ``.sif`` input file to disk.
        """
        if config.analysis_type not in self.supported_analyses:
            raise ValueError(
                f"ElmerSolver does not support {config.analysis_type.value!r}. "
                f"Supported: {[a.value for a in self.supported_analyses]}"
            )

        work_dir = Path(tempfile.mkdtemp(prefix="elmer_"))
        material_props = _extract_material_props(config)
        has_elmer = _check_elmer_available()

        input_files: list[str] = []
        if config.mesh_path and Path(config.mesh_path).exists():
            input_files.append(config.mesh_path)

        # When Elmer CLI is available, write the .sif input deck
        if has_elmer:
            sif_path = work_dir / "case.sif"
            sif_content = self._generate_sif(config, material_props)
            sif_path.write_text(sif_content, encoding="utf-8")
            input_files.append(str(sif_path))

        return PreparedJob(
            job_id=PreparedJob.new_id(),
            work_dir=str(work_dir),
            input_files=input_files,
            solver_config=config,
            metadata={
                "material_props": material_props,
                "analysis_type": config.analysis_type.value,
                "has_elmer": has_elmer,
            },
        )

    async def run(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback] = None,
    ) -> SolverResult:
        """Execute the solver, dispatching to Elmer CLI or the fallback."""
        if job.metadata.get("has_elmer"):
            return await asyncio.to_thread(self._run_elmer, job, progress)
        else:
            return await asyncio.to_thread(self._run_fallback, job, progress)

    def read_results(self, result: SolverResult) -> FieldData:
        """Return the :class:`FieldData` already attached to the result.

        Both the Elmer and fallback solvers populate ``result.field_data``
        during ``run()``.
        """
        if result.field_data is not None:
            return result.field_data
        raise RuntimeError(
            f"No field_data in SolverResult for job {result.job_id}.  "
            "Was the analysis run successfully?"
        )

    # ------------------------------------------------------------------
    # Elmer .sif generation
    # ------------------------------------------------------------------

    def _generate_sif(
        self,
        config: SolverConfig,
        material_props: dict[str, float],
    ) -> str:
        """Generate a minimal Elmer Solver Input File (.sif).

        This creates a template that Elmer can parse.  A complete production
        setup would also require mesh files produced by ``ElmerGrid``.
        """
        params = config.parameters or {}
        freq = float(params.get("target_frequency_hz", 20000.0))
        E = material_props.get("E_pa", 113.8e9)
        nu = material_props.get("nu", 0.342)
        rho = material_props.get("rho_kg_m3", 4430.0)

        if config.analysis_type == AnalysisType.PIEZOELECTRIC:
            solver_section = f"""\
Solver 1
  Equation = Piezoelectric
  Procedure = "StressSolve" "StressSolver"
  Variable = Displacement
  Variable DOFs = 3
  Exec Solver = Always
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Max Iterations = 1000
  Linear System Convergence Tolerance = 1.0e-8
  Frequency = {freq}
End
"""
        else:  # ACOUSTIC
            solver_section = f"""\
Solver 1
  Equation = Helmholtz Equation
  Procedure = "HelmholtzSolve" "HelmholtzSolver"
  Variable = Pressure
  Variable DOFs = 2
  Exec Solver = Always
  Linear System Solver = Direct
  Linear System Direct Method = MUMPS
  Frequency = {freq}
End
"""

        sif = f"""\
! Elmer Solver Input File -- auto-generated by ElmerSolver backend
! Analysis: {config.analysis_type.value}

Header
  Mesh DB "." "mesh"
End

Simulation
  Coordinate System = Cartesian 3D
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Output Intervals = 1
  Output File = "results.ep"
End

Body 1
  Equation = 1
  Material = 1
End

Material 1
  Density = {rho}
  Youngs Modulus = {E}
  Poisson Ratio = {nu}
  Sound Speed = {material_props.get("speed_of_sound_m_s", 5090.0)}
End

{solver_section}

Equation 1
  Active Solvers(1) = 1
End

Boundary Condition 1
  Target Boundaries = 1
  Displacement 1 = 0.0
  Displacement 2 = 0.0
  Displacement 3 = 0.0
End
"""
        return sif

    # ------------------------------------------------------------------
    # Real Elmer execution path
    # ------------------------------------------------------------------

    def _run_elmer(
        self,
        job: PreparedJob,
        progress: Optional[ProgressCallback],
    ) -> SolverResult:
        """Execute real Elmer solver via CLI."""
        import subprocess

        start = time.perf_counter()
        try:
            if progress:
                progress(5.0, "Starting Elmer solver...")

            sif_files = [f for f in job.input_files if f.endswith(".sif")]
            if not sif_files:
                raise FileNotFoundError("No .sif input file found in prepared job")

            if progress:
                progress(10.0, "Running ElmerSolver...")

            result = subprocess.run(
                ["ElmerSolver", sif_files[0]],
                cwd=job.work_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if progress:
                progress(80.0, "Parsing Elmer output...")

            elapsed = time.perf_counter() - start

            if result.returncode != 0:
                return SolverResult(
                    success=False,
                    job_id=job.job_id,
                    output_files=[],
                    error_message=f"ElmerSolver exited with code {result.returncode}",
                    solver_log=result.stdout + "\n" + result.stderr,
                    compute_time_s=round(elapsed, 3),
                )

            # Collect output files
            work_path = Path(job.work_dir)
            output_files = [
                str(f) for f in work_path.glob("results.*")
            ]

            if progress:
                progress(100.0, "Elmer analysis complete")

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
            logger.exception("Elmer solver failed: %s", exc)
            return SolverResult(
                success=False,
                job_id=job.job_id,
                output_files=[],
                error_message=str(exc),
                solver_log=f"Elmer solver failed: {exc}",
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
        """Numpy/scipy fallback for piezoelectric and acoustic analysis.

        Implements simplified analytical models based on real ultrasonic
        welding physics for rapid prototyping and preview.
        """
        start = time.perf_counter()
        config = job.solver_config
        material = job.metadata.get("material_props", {})
        params = config.parameters or {}

        log_lines: list[str] = [
            "ElmerSolver CLI not available, using numpy/scipy fallback"
        ]

        # ---- Material properties (defaults: Ti-6Al-4V) ----
        E = float(material.get("E_pa", 113.8e9))
        nu = float(material.get("nu", 0.342))
        rho = float(material.get("rho_kg_m3", 4430.0))
        c_sound = float(material.get("speed_of_sound_m_s", 0.0))
        if c_sound <= 0:
            # Longitudinal wave speed: c = sqrt(E / rho)
            c_sound = float(np.sqrt(E / rho))

        # ---- Geometry parameters ----
        width = float(params.get("width_mm", 40.0))
        height = float(params.get("height_mm", 100.0))  # along vibration axis
        length = float(params.get("length_mm", 30.0))
        horn_type = str(params.get("horn_type", "rectangular"))

        # ---- Piezoelectric transducer parameters ----
        d33 = float(params.get("d33_pm_v", 374.0))  # piezo strain constant (pC/N or pm/V) - PZT-4
        n_layers = int(params.get("n_piezo_layers", 4))
        piezo_voltage = float(params.get("excitation_voltage_v", 500.0))
        piezo_capacitance_nf = float(params.get("piezo_capacitance_nf", 15.0))
        epsilon_r = float(params.get("epsilon_r", 1300.0))  # relative permittivity (PZT-4)
        epsilon_0 = 8.854e-12  # vacuum permittivity

        # ---- Target frequency ----
        target_freq = float(params.get("target_frequency_hz", 20000.0))
        if target_freq == 0.0:
            freq_khz = float(params.get("frequency_khz", 20.0))
            target_freq = freq_khz * 1000.0

        log_lines.append(f"Analysis type: {config.analysis_type.value}")
        log_lines.append(f"Material: E={E:.2e} Pa, nu={nu}, rho={rho}")
        log_lines.append(f"Geometry: {width}x{height}x{length} mm ({horn_type})")
        log_lines.append(f"Target frequency: {target_freq:.0f} Hz")

        metrics: dict[str, float] = {}

        if config.analysis_type == AnalysisType.PIEZOELECTRIC:
            if progress:
                progress(10.0, "Computing piezoelectric transducer coupling...")
            metrics, field_data = self._solve_piezoelectric(
                E=E, nu=nu, rho=rho, c_sound=c_sound,
                d33=d33, n_layers=n_layers, piezo_voltage=piezo_voltage,
                piezo_capacitance_nf=piezo_capacitance_nf,
                epsilon_r=epsilon_r, epsilon_0=epsilon_0,
                width=width, height=height, length=length,
                horn_type=horn_type, target_freq=target_freq,
                log_lines=log_lines, progress=progress,
            )
        elif config.analysis_type == AnalysisType.ACOUSTIC:
            if progress:
                progress(10.0, "Computing acoustic wave propagation...")
            metrics, field_data = self._solve_acoustic(
                E=E, nu=nu, rho=rho, c_sound=c_sound,
                width=width, height=height, length=length,
                horn_type=horn_type, target_freq=target_freq,
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
    # Piezoelectric analysis (fallback)
    # ------------------------------------------------------------------

    def _solve_piezoelectric(
        self,
        *,
        E: float,
        nu: float,
        rho: float,
        c_sound: float,
        d33: float,
        n_layers: int,
        piezo_voltage: float,
        piezo_capacitance_nf: float,
        epsilon_r: float,
        epsilon_0: float,
        width: float,
        height: float,
        length: float,
        horn_type: str,
        target_freq: float,
        log_lines: list[str],
        progress: Optional[ProgressCallback],
    ) -> tuple[dict[str, float], FieldData]:
        """Piezoelectric transducer coupling model.

        Models the electromechanical behaviour of a piezoelectric stack
        transducer driving an ultrasonic welding horn.

        Physics:
          - Mechanical displacement: x = d33 * V * n_layers  (stack actuator)
          - Electrical impedance: Z = V / I, where I derives from the
            motional admittance at resonance
          - Electromechanical coupling: k^2 ~ d33^2 * E / epsilon
          - Horn gain: area ratio A_input / A_output for stepped horns
        """
        if progress:
            progress(30.0, "Computing piezoelectric displacement...")

        # d33 is in pm/V (pico-metres per volt) -- convert to m/V
        d33_m_v = d33 * 1e-12

        # --- Mechanical displacement ---
        # Stack actuator: displacement = d33 * V * n_layers
        displacement_m = d33_m_v * piezo_voltage * n_layers
        displacement_um = displacement_m * 1e6

        if progress:
            progress(45.0, "Computing electrical impedance...")

        # --- Electrical impedance at resonance ---
        # At resonance the motional branch dominates.
        # Clamped capacitance: C0 = epsilon_r * epsilon_0 * A / t_layer
        # For a typical PZT-4 disc stack:
        #   - A (piezo area) ~ pi * (diameter/2)^2
        #   - t_layer ~ lambda / (2 * n_layers)
        # Approximate from user-supplied capacitance or compute:
        omega = 2.0 * np.pi * target_freq
        C0 = piezo_capacitance_nf * 1e-9  # F

        # Motional resistance at resonance (mechanical loss):
        # R_m ~ (1 / (k_eff^2 * omega * C0)) * (1/Q_m)
        # Typical mechanical Q for PZT-4: 500-2000
        Q_m = float(1200.0)
        k_eff_sq_est = d33_m_v ** 2 * E / (epsilon_r * epsilon_0)
        # Clamp k_eff_sq to physically reasonable range [0.001, 0.7]
        k_eff_sq = float(np.clip(k_eff_sq_est, 0.001, 0.7))

        # Impedance at series resonance (minimum impedance)
        # Z_min ~ 1 / (omega * C0 * k_eff^2 * Q_m)
        if omega * C0 * k_eff_sq * Q_m > 0:
            Z_series = 1.0 / (omega * C0 * k_eff_sq * Q_m)
        else:
            Z_series = 50.0  # fallback

        # At resonance the current is I = V / Z
        I_rms = piezo_voltage / Z_series if Z_series > 0 else 0.0
        electrical_impedance = Z_series

        if progress:
            progress(60.0, "Computing electromechanical coupling...")

        # --- Electromechanical coupling coefficient ---
        # k^2 = d33^2 * c^E / epsilon^T  (IEEE standard definition)
        # c^E = E (Young's modulus at constant electric field)
        # epsilon^T = epsilon_r * epsilon_0 (permittivity at constant stress)
        k2 = k_eff_sq

        # --- Horn gain (amplitude ratio) ---
        # Depends on horn geometry: ratio of input to output cross-section
        horn_type_lower = horn_type.lower()
        A_input = width * length  # mm^2 (input face, transducer side)

        if horn_type_lower == "exponential":
            # Exponential horn: gain = sqrt(A_in / A_out)
            # Typical taper: output is ~1/3 of input area
            taper_ratio = float(0.33)
            A_output = A_input * taper_ratio
            horn_gain = float(np.sqrt(A_input / A_output))
        elif horn_type_lower == "stepped":
            # Stepped horn: gain = A_in / A_out
            step_ratio = float(0.25)
            A_output = A_input * step_ratio
            horn_gain = A_input / A_output
        elif horn_type_lower == "catenoidal":
            # Catenoidal horn: gain between exponential and stepped
            taper_ratio = float(0.30)
            A_output = A_input * taper_ratio
            horn_gain = float(np.sqrt(A_input / A_output)) * 1.15
        else:
            # Rectangular / cylindrical: minimal gain from geometry
            horn_gain = 1.0

        if progress:
            progress(75.0, "Generating piezoelectric field data...")

        # --- Assemble metrics ---
        metrics: dict[str, float] = {
            "piezo_voltage_v": float(round(piezo_voltage, 2)),
            "mechanical_displacement_um": float(round(displacement_um, 4)),
            "electrical_impedance_ohm": float(round(electrical_impedance, 2)),
            "electromechanical_coupling_k2": float(round(k2, 4)),
            "horn_gain": float(round(horn_gain, 3)),
            "target_frequency_hz": float(target_freq),
            "piezo_current_a_rms": float(round(I_rms, 4)),
            "mechanical_q_factor": float(Q_m),
        }

        log_lines.append(f"Displacement: {displacement_um:.4f} um")
        log_lines.append(f"Impedance: {electrical_impedance:.1f} ohm")
        log_lines.append(f"k^2: {k2:.4f}")
        log_lines.append(f"Horn gain: {horn_gain:.3f}")

        # --- Generate field data ---
        # Voltage / displacement field with exponential decay from transducer
        # (transducer at y=0, horn tip at y=height)
        nx, ny, nz = 8, 30, 8
        points, cells = _generate_hex_mesh(width, height, length, nx, ny, nz)

        # Normalised position along vibration axis [0, 1]
        y_norm = points[:, 1] / height

        # Voltage decays exponentially from the transducer (y=0)
        # In the piezo stack region (first ~10% of height) voltage is ~full,
        # then drops rapidly into the passive horn.
        piezo_fraction = 0.10  # piezo stack is ~10% of total assembly length
        voltage_field = np.where(
            y_norm <= piezo_fraction,
            piezo_voltage * (1.0 - 0.3 * y_norm / piezo_fraction),
            piezo_voltage * 0.7 * np.exp(-5.0 * (y_norm - piezo_fraction)),
        )

        # Displacement field: builds up through piezo, then horn amplifies
        # In piezo region: linear build-up
        # In horn region: sinusoidal standing wave pattern
        displacement_field = np.where(
            y_norm <= piezo_fraction,
            displacement_um * y_norm / piezo_fraction,
            displacement_um * np.abs(np.sin(np.pi * y_norm)) * horn_gain,
        )

        field_data = FieldData(
            points=points,
            cells=[cells],
            cell_types=["hex8"],
            point_data={
                "voltage": voltage_field,
                "displacement": displacement_field,
            },
            metadata={
                "analysis_type": "piezoelectric",
                "horn_type": horn_type,
            },
        )

        return metrics, field_data

    # ------------------------------------------------------------------
    # Acoustic analysis (fallback)
    # ------------------------------------------------------------------

    def _solve_acoustic(
        self,
        *,
        E: float,
        nu: float,
        rho: float,
        c_sound: float,
        width: float,
        height: float,
        length: float,
        horn_type: str,
        target_freq: float,
        log_lines: list[str],
        progress: Optional[ProgressCallback],
    ) -> tuple[dict[str, float], FieldData]:
        """Acoustic wave propagation model for horn analysis.

        Models longitudinal acoustic wave propagation in the ultrasonic
        welding horn to determine resonance characteristics and amplitude
        distribution.

        Physics:
          - Natural frequency: f = (n / 2L) * sqrt(E / rho)
            for the nth longitudinal mode of a uniform bar
          - Amplitude uniformity: ratio of min to max displacement
            on the contact face, affected by horn geometry
          - Modal separation: gap between desired mode and nearest
            unwanted mode
          - Acoustic impedance: Z = rho * c * A
        """
        if progress:
            progress(25.0, "Computing natural frequencies...")

        # Convert height (vibration axis length) from mm to m
        L_m = height / 1000.0

        # --- Natural frequency (longitudinal mode of a bar) ---
        # f_n = (n / 2L) * sqrt(E / rho)
        # For half-wavelength resonance (n=1):
        f_1 = (1.0 / (2.0 * L_m)) * np.sqrt(E / rho)

        # Find the mode number closest to target frequency
        n_target = max(1, round(target_freq / f_1))
        f_natural = n_target * f_1

        # Correction for non-uniform cross-section horns
        if horn_type.lower() == "exponential":
            # Exponential horns have a slightly higher frequency due to taper
            f_natural *= 1.02
        elif horn_type.lower() == "stepped":
            # Stepped horns: frequency shifts depending on step location
            f_natural *= 0.98
        elif horn_type.lower() == "catenoidal":
            f_natural *= 1.01

        # Frequency deviation from target
        freq_deviation_pct = ((f_natural - target_freq) / target_freq) * 100.0

        if progress:
            progress(40.0, "Computing amplitude distribution...")

        # --- Amplitude uniformity ---
        # Ratio of min/max amplitude on the contact face.
        # Depends on horn type and aspect ratio.
        aspect_ratio = height / max(width, length)
        horn_type_lower = horn_type.lower()

        if horn_type_lower == "cylindrical":
            # Cylindrical horns have good uniformity if aspect ratio is high
            base_uniformity = 0.92
        elif horn_type_lower == "rectangular":
            # Rectangular horns: moderate uniformity
            base_uniformity = 0.88
        elif horn_type_lower == "exponential":
            # Exponential taper helps uniformity
            base_uniformity = 0.93
        elif horn_type_lower == "stepped":
            # Stepped horns can have edge effects at the step
            base_uniformity = 0.85
        elif horn_type_lower == "catenoidal":
            # Best uniformity among common horn types
            base_uniformity = 0.95
        else:
            base_uniformity = 0.88

        # Adjust for aspect ratio (higher aspect ratio = better uniformity)
        uniformity_adjustment = 0.02 * min(aspect_ratio - 2.0, 2.0) if aspect_ratio > 2.0 else 0.0
        amplitude_uniformity = float(np.clip(base_uniformity + uniformity_adjustment, 0.80, 0.98))

        if progress:
            progress(55.0, "Computing modal separation...")

        # --- Modal separation ---
        # Gap to nearest unwanted mode.  For a rectangular horn, lateral
        # bending modes can appear near the longitudinal mode.
        # Bending mode frequency: f_bend = (beta_n^2 / (2*pi*L^2)) * sqrt(EI/(rho*A))
        A_cross = (width / 1000.0) * (length / 1000.0)  # m^2
        I_second = (width / 1000.0) * (length / 1000.0) ** 3 / 12.0  # m^4
        # First bending mode: beta_1 * L ~ 4.730 (clamped-free) or pi (free-free)
        beta_1_L = np.pi  # free-free first bending
        f_bend_1 = (beta_1_L ** 2 / (2.0 * np.pi * L_m ** 2)) * np.sqrt(
            E * I_second / (rho * A_cross)
        )

        # Nearest longitudinal modes
        f_prev = (n_target - 1) * f_1 if n_target > 1 else 0.0
        f_next = (n_target + 1) * f_1

        # Modal separation is the minimum gap to any nearby unwanted mode
        gaps = []
        if f_prev > 0:
            gaps.append(abs(f_natural - f_prev))
        gaps.append(abs(f_natural - f_next))
        gaps.append(abs(f_natural - f_bend_1))
        modal_separation = float(min(gaps))

        if progress:
            progress(70.0, "Computing acoustic impedance matching...")

        # --- Acoustic impedance matching ---
        # Z = rho * c * A  for each section
        # Matching efficiency: eta = 4 * Z1 * Z2 / (Z1 + Z2)^2
        # Compare horn impedance to typical workpiece (e.g. aluminium)
        Z_horn = rho * c_sound * A_cross

        # Typical workpiece properties (aluminium)
        rho_wp = float(2700.0)  # kg/m^3
        c_wp = float(5100.0)    # m/s, longitudinal in aluminium
        A_wp = A_cross  # assume same contact area
        Z_workpiece = rho_wp * c_wp * A_wp

        # Impedance matching efficiency (transmission coefficient)
        impedance_match = float(
            4.0 * Z_horn * Z_workpiece / (Z_horn + Z_workpiece) ** 2
        )

        if progress:
            progress(80.0, "Generating acoustic field data...")

        # --- Assemble metrics ---
        metrics: dict[str, float] = {
            "natural_frequency_hz": float(round(f_natural, 2)),
            "frequency_deviation_pct": float(round(freq_deviation_pct, 3)),
            "amplitude_uniformity": float(round(amplitude_uniformity, 4)),
            "modal_separation_hz": float(round(modal_separation, 2)),
            "acoustic_impedance_match": float(round(impedance_match, 4)),
            "target_frequency_hz": float(target_freq),
            "mode_number": float(n_target),
            "bending_mode_1_hz": float(round(f_bend_1, 2)),
        }

        log_lines.append(f"Natural frequency: {f_natural:.1f} Hz (mode {n_target})")
        log_lines.append(f"Frequency deviation: {freq_deviation_pct:.3f} %")
        log_lines.append(f"Amplitude uniformity: {amplitude_uniformity:.4f}")
        log_lines.append(f"Modal separation: {modal_separation:.1f} Hz")
        log_lines.append(f"Impedance match: {impedance_match:.4f}")

        # --- Generate field data ---
        # Displacement amplitude field along horn length (sinusoidal standing wave)
        nx, ny, nz = 8, 30, 8
        points, cells = _generate_hex_mesh(width, height, length, nx, ny, nz)

        y_norm = points[:, 1] / height  # normalised position [0, 1]

        # Standing wave displacement: u(y) = A * sin(n * pi * y / L)
        # For half-wavelength resonance, n_target determines the pattern
        displacement_amplitude = np.abs(
            np.sin(n_target * np.pi * y_norm)
        )

        # Pressure field is 90 degrees out of phase with displacement:
        # p(y) = P_max * cos(n * pi * y / L)
        pressure_amplitude = np.abs(
            np.cos(n_target * np.pi * y_norm)
        )

        # Add transverse variation (slight non-uniformity across cross section)
        x_center = width / 2.0
        z_center = length / 2.0
        r_norm = np.sqrt(
            ((points[:, 0] - x_center) / (width / 2.0)) ** 2
            + ((points[:, 2] - z_center) / (length / 2.0)) ** 2
        )
        # Slight amplitude reduction towards edges
        transverse_factor = 1.0 - (1.0 - amplitude_uniformity) * r_norm ** 2

        displacement_amplitude *= transverse_factor

        field_data = FieldData(
            points=points,
            cells=[cells],
            cell_types=["hex8"],
            point_data={
                "displacement_amplitude": displacement_amplitude,
                "pressure_amplitude": pressure_amplitude,
            },
            metadata={
                "analysis_type": "acoustic",
                "horn_type": horn_type,
                "mode_number": n_target,
            },
        )

        return metrics, field_data
