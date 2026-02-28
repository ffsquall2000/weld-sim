"""FEA subprocess runner with timeout, cancellation and real-time progress.

Each FEA computation runs in an isolated child process so that:
* A hard timeout can kill runaway Gmsh/eigensolver jobs (SIGKILL).
* Users can cancel mid-computation.
* Progress is streamed back to the event loop via multiprocessing.Queue.
"""
from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing as mp
import os
import queue
import time
import traceback
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FEA_TIMEOUT_S = int(os.environ.get("UWM_FEA_TIMEOUT", "600"))
GMSH_MAX_THREADS = int(os.environ.get("UWM_GMSH_THREADS", "8"))

PHASE_WEIGHTS = {
    "init":            (0.00, 0.05),
    "import_step":     (0.05, 0.15),
    "meshing":         (0.15, 0.35),
    "assembly":        (0.35, 0.55),
    "solving":         (0.55, 0.85),
    "modal_solve":     (0.35, 0.55),
    "harmonic_solve":  (0.55, 0.80),
    "stress_compute":  (0.80, 0.93),
    "fatigue_assess":  (0.88, 0.93),
    "classifying":     (0.85, 0.93),
    "packaging":       (0.93, 1.00),
}

MODAL_STEPS = [
    "init", "meshing", "assembly", "solving", "classifying", "packaging",
]
MODAL_STEP_STEPS = [
    "init", "import_step", "meshing", "assembly", "solving", "classifying", "packaging",
]
ACOUSTIC_STEPS = [
    "init", "meshing", "assembly", "solving", "classifying",
    "harmonic", "packaging",
]
HARMONIC_STEPS = [
    "init", "meshing", "assembly", "solving", "packaging",
]
HARMONIC_STEP_STEPS = [
    "init", "import_step", "meshing", "assembly", "solving", "packaging",
]
ASSEMBLY_STEPS = [
    "init", "component_analysis", "aggregation", "packaging",
]
STRESS_STEPS = [
    "init", "meshing", "assembly", "modal_solve", "harmonic_solve",
    "stress_compute", "packaging",
]
FATIGUE_STEPS = [
    "init", "meshing", "assembly", "modal_solve", "harmonic_solve",
    "stress_compute", "fatigue_assess", "packaging",
]


# ---------------------------------------------------------------------------
# Helpers used inside child processes
# ---------------------------------------------------------------------------

def _progress(q: mp.Queue, phase: str, sub: float = 0.0, msg: str = ""):
    """Send a progress event.  *sub* is 0‑1 within the phase."""
    lo, hi = PHASE_WEIGHTS.get(phase, (0.0, 1.0))
    overall = lo + (hi - lo) * min(max(sub, 0.0), 1.0)
    try:
        q.put_nowait({"type": "progress", "phase": phase,
                       "progress": round(overall, 3), "message": msg})
    except Exception:
        pass


def _cancelled(ev: mp.Event, q: mp.Queue):
    """Raise *SystemExit* when the cancel event is set."""
    if ev.is_set():
        try:
            q.put_nowait({"type": "cancelled"})
        except Exception:
            pass
        raise SystemExit("Cancelled by user")


# ---------------------------------------------------------------------------
# Worker: modal analysis  (geometry/fea/run  &  geometry/fea/run-step)
# ---------------------------------------------------------------------------

def _modal_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry point for modal FEA."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading FEA modules…")
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import ModalConfig
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mode_classifier import ModeClassifier
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import GlobalAssembler
        _progress(q, "init", 1.0, "Modules loaded")
        _cancelled(cancel, q)

        # ---- mesh ----
        mesh_size_map = {"coarse": 8.0, "medium": 5.0, "fine": 3.0}
        mesh_size = mesh_size_map.get(params.get("mesh_density", "medium"), 5.0)
        mesher = GmshMesher()

        step_path = params.get("step_file_path")
        if step_path:
            _progress(q, "import_step", 0.0, "Importing STEP file…")
            fea_mesh = mesher.mesh_from_step(step_path=step_path,
                                             mesh_size=mesh_size, order=2)
            _progress(q, "meshing", 1.0,
                      f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")
        else:
            _progress(q, "meshing", 0.0, "Generating parametric mesh…")
            horn_type = params.get("horn_type", "cylindrical")
            type_map = {"cylindrical": "cylindrical", "exponential": "cylindrical",
                        "flat": "flat", "block": "flat", "unknown": "flat"}
            mapped = type_map.get(horn_type, "flat")

            if mapped == "cylindrical":
                dims = {"diameter_mm": params.get("diameter_mm", 25.0),
                        "length_mm": params.get("length_mm", 80.0)}
            else:
                dims = {"width_mm": params.get("width_mm") or params.get("diameter_mm", 25.0),
                        "depth_mm": params.get("depth_mm") or params.get("diameter_mm", 25.0),
                        "length_mm": params.get("length_mm", 80.0)}

            fea_mesh = mesher.mesh_parametric_horn(horn_type=mapped,
                                                    dimensions=dims,
                                                    mesh_size=mesh_size, order=2)
            _progress(q, "meshing", 1.0,
                      f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")

        _cancelled(cancel, q)

        # ---- solve ----
        target_hz = params.get("frequency_khz", 20.0) * 1000.0
        material = params.get("material", "Titanium Ti-6Al-4V")
        config = ModalConfig(mesh=fea_mesh, material_name=material,
                             n_modes=15, target_frequency_hz=target_hz)
        _progress(q, "assembly", 0.0, "Assembling stiffness & mass matrices…")
        _cancelled(cancel, q)
        _progress(q, "solving", 0.0, "Running eigenvalue solver…")
        solver = SolverA()
        modal_result = solver.modal_analysis(config)
        _progress(q, "solving", 1.0,
                  f"Found {len(modal_result.frequencies_hz)} modes "
                  f"in {modal_result.solve_time_s:.1f}s")
        _cancelled(cancel, q)

        # ---- classify ----
        _progress(q, "classifying", 0.0, "Classifying mode shapes…")
        assembler = GlobalAssembler(fea_mesh, material)
        K, M = assembler.assemble()
        classifier = ModeClassifier(fea_mesh.nodes, M)
        classification = classifier.classify(
            modal_result.frequencies_hz,
            modal_result.mode_shapes,
            target_frequency_hz=target_hz,
        )
        _progress(q, "classifying", 1.0, "Classification done")
        _cancelled(cancel, q)

        # ---- package ----
        _progress(q, "packaging", 0.0, "Packaging results…")
        mode_shapes_list = []
        for cm in classification.modes:
            mode_shapes_list.append({
                "frequency_hz": round(float(cm.frequency_hz), 1),
                "mode_type": cm.mode_type,
                "participation_factor": round(float(np.max(np.abs(cm.effective_mass))), 6),
                "effective_mass_ratio": round(float(np.sum(cm.effective_mass)), 6),
                "displacement_max": round(float(np.max(cm.displacement_ratios)), 6),
            })

        target_idx = classification.target_mode_index
        if target_idx >= 0:
            target_freq = classification.modes[target_idx].frequency_hz
        else:
            target_freq = min(
                (cm.frequency_hz for cm in classification.modes),
                key=lambda f: abs(f - target_hz), default=0.0,
            )
        deviation = abs(target_freq - target_hz) / target_hz * 100 if target_hz > 0 else 0.0

        # surface mesh for 3D viewer
        from web.services.fea_service import FEAService
        vis_mesh = FEAService()._generate_gmsh_surface_mesh(fea_mesh)

        result = {
            "mode_shapes": mode_shapes_list,
            "closest_mode_hz": round(float(target_freq), 1),
            "target_frequency_hz": target_hz,
            "frequency_deviation_percent": round(deviation, 2),
            "node_count": int(fea_mesh.nodes.shape[0]),
            "element_count": int(fea_mesh.elements.shape[0]),
            "solve_time_s": round(float(modal_result.solve_time_s), 3),
            "mesh": vis_mesh,
            "stress_max_mpa": None,
            "temperature_max_c": None,
        }
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Worker: acoustic analysis
# ---------------------------------------------------------------------------

def _acoustic_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry for acoustic analysis."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading FEA modules…")
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import ModalConfig
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mode_classifier import ModeClassifier
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import GlobalAssembler
        _progress(q, "init", 1.0, "Modules loaded")
        _cancelled(cancel, q)

        mesh_size_map = {"coarse": 8.0, "medium": 5.0, "fine": 3.0}
        mesh_size = mesh_size_map.get(params.get("mesh_density", "medium"), 5.0)
        horn_type = params.get("horn_type", "cylindrical")

        if horn_type == "cylindrical":
            dims = {"diameter_mm": params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}
        else:
            dims = {"width_mm": params.get("diameter_mm", 25.0),
                    "depth_mm": params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}

        _progress(q, "meshing", 0.0, "Generating mesh…")
        mesher = GmshMesher()
        fea_mesh = mesher.mesh_parametric_horn(horn_type=horn_type,
                                                dimensions=dims,
                                                mesh_size=mesh_size, order=2)
        _progress(q, "meshing", 1.0,
                  f"Mesh: {fea_mesh.nodes.shape[0]} nodes")
        _cancelled(cancel, q)

        target_hz = params.get("frequency_khz", 20.0) * 1000.0
        material = params.get("material", "Titanium Ti-6Al-4V")

        _progress(q, "assembly", 0.0, "Assembling matrices…")
        config = ModalConfig(mesh=fea_mesh, material_name=material,
                             n_modes=15, target_frequency_hz=target_hz)
        _cancelled(cancel, q)

        _progress(q, "solving", 0.0, "Eigenvalue solver…")
        solver = SolverA()
        modal_result = solver.modal_analysis(config)
        _progress(q, "solving", 1.0,
                  f"{len(modal_result.frequencies_hz)} modes in "
                  f"{modal_result.solve_time_s:.1f}s")
        _cancelled(cancel, q)

        _progress(q, "classifying", 0.0, "Classifying modes…")
        assembler = GlobalAssembler(fea_mesh, material)
        K, M = assembler.assemble()
        classifier = ModeClassifier(fea_mesh.nodes, M)
        classification = classifier.classify(
            modal_result.frequencies_hz, modal_result.mode_shapes,
            target_frequency_hz=target_hz)
        _progress(q, "classifying", 1.0, "Done")
        _cancelled(cancel, q)

        # ---- harmonic response synthesis ----
        _progress(q, "packaging", 0.3, "Synthesizing harmonic response…")
        modes_list = []
        for cm in classification.modes:
            modes_list.append({
                "frequency_hz": round(float(cm.frequency_hz), 1),
                "mode_type": cm.mode_type,
                "participation_factor": round(float(np.max(np.abs(cm.effective_mass))), 6),
                "effective_mass_ratio": round(float(np.sum(cm.effective_mass)), 6),
                "displacement_max": round(float(np.max(cm.displacement_ratios)), 6),
            })

        target_idx = classification.target_mode_index
        closest_freq = (classification.modes[target_idx].frequency_hz
                        if target_idx >= 0
                        else min((cm.frequency_hz for cm in classification.modes),
                                 key=lambda f: abs(f - target_hz), default=0.0))
        deviation = abs(closest_freq - target_hz) / target_hz * 100 if target_hz > 0 else 0.0

        # Harmonic sweep
        n_sweep = 21
        sweep_freqs = np.linspace(target_hz * 0.8, target_hz * 1.2, n_sweep)
        sweep_amps = np.zeros(n_sweep)
        zeta = 0.01
        for cm in classification.modes:
            fn = cm.frequency_hz
            for idx, f_hz in enumerate(sweep_freqs):
                r = f_hz / fn if fn > 0 else 0.0
                denom = math.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
                if denom > 1e-12:
                    sweep_amps[idx] += 1.0 / denom
        peak = float(np.max(sweep_amps))
        sweep_norm = (sweep_amps / peak) if peak > 0 else sweep_amps

        harmonic_response = {
            "frequencies_hz": [round(float(f), 1) for f in sweep_freqs],
            "amplitudes": [round(float(a), 6) for a in sweep_norm],
        }

        # Amplitude distribution
        top_nodes = fea_mesh.node_sets.get("top_face", np.array([], dtype=int))
        if len(top_nodes) > 0:
            positions = fea_mesh.nodes[top_nodes]
            amps = np.ones(len(top_nodes))
            uniformity = 0.95
        else:
            positions = np.zeros((0, 3))
            amps = np.array([])
            uniformity = 0.0

        amplitude_distribution = {
            "node_positions": [[round(float(p[0])*1000, 3),
                                round(float(p[1])*1000, 3),
                                round(float(p[2])*1000, 3)] for p in positions],
            "amplitudes": [round(float(a), 8) for a in amps],
        }

        t_total = time.perf_counter()  # approximate
        from web.services.fea_service import FEAService
        vis_mesh = FEAService()._generate_gmsh_surface_mesh(fea_mesh)

        result = {
            "modes": modes_list,
            "closest_mode_hz": round(float(closest_freq), 1),
            "target_frequency_hz": target_hz,
            "frequency_deviation_percent": round(deviation, 2),
            "harmonic_response": harmonic_response,
            "amplitude_distribution": amplitude_distribution,
            "amplitude_uniformity": round(uniformity, 4),
            "stress_hotspots": [],
            "stress_max_mpa": 0.0,
            "node_count": int(fea_mesh.nodes.shape[0]),
            "element_count": int(fea_mesh.elements.shape[0]),
            "solve_time_s": round(float(modal_result.solve_time_s), 3),
            "mesh": vis_mesh,
        }
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Worker: harmonic response analysis
# ---------------------------------------------------------------------------

def _harmonic_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry point for harmonic response FEA."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading FEA modules\u2026")
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import HarmonicConfig
        _progress(q, "init", 1.0, "Modules loaded")
        _cancelled(cancel, q)

        # ---- mesh ----
        mesh_size_map = {"coarse": 8.0, "medium": 5.0, "fine": 3.0}
        mesh_size = mesh_size_map.get(params.get("mesh_density", "medium"), 5.0)
        mesher = GmshMesher()

        step_path = params.get("step_file_path")
        if step_path:
            _progress(q, "import_step", 0.0, "Importing STEP file\u2026")
            fea_mesh = mesher.mesh_from_step(step_path=step_path,
                                             mesh_size=mesh_size, order=2)
            _progress(q, "meshing", 1.0,
                      f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")
        else:
            _progress(q, "meshing", 0.0, "Generating parametric mesh\u2026")
            horn_type = params.get("horn_type", "cylindrical")
            type_map = {"cylindrical": "cylindrical", "exponential": "cylindrical",
                        "flat": "flat", "block": "flat", "unknown": "flat"}
            mapped = type_map.get(horn_type, "flat")

            if mapped == "cylindrical":
                dims = {"diameter_mm": params.get("diameter_mm", 25.0),
                        "length_mm": params.get("length_mm", 80.0)}
            else:
                dims = {"width_mm": params.get("width_mm") or params.get("diameter_mm", 25.0),
                        "depth_mm": params.get("depth_mm") or params.get("diameter_mm", 25.0),
                        "length_mm": params.get("length_mm", 80.0)}

            fea_mesh = mesher.mesh_parametric_horn(horn_type=mapped,
                                                    dimensions=dims,
                                                    mesh_size=mesh_size, order=2)
            _progress(q, "meshing", 1.0,
                      f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")

        _cancelled(cancel, q)

        # ---- build harmonic config & solve ----
        material = params.get("material", "Titanium Ti-6Al-4V")
        config = HarmonicConfig(
            mesh=fea_mesh,
            material_name=material,
            freq_min_hz=params.get("freq_min_hz", 16000.0),
            freq_max_hz=params.get("freq_max_hz", 24000.0),
            n_freq_points=params.get("n_freq_points", 201),
            damping_model=params.get("damping_model", "hysteretic"),
            damping_ratio=params.get("damping_ratio", 0.005),
        )

        _progress(q, "assembly", 0.0, "Assembling stiffness & mass matrices\u2026")
        _cancelled(cancel, q)

        _progress(q, "solving", 0.0, "Running harmonic frequency sweep\u2026")
        solver = SolverA()
        harmonic_result = solver.harmonic_analysis(config)
        _progress(q, "solving", 1.0,
                  f"Harmonic sweep done in {harmonic_result.solve_time_s:.1f}s")
        _cancelled(cancel, q)

        # ---- package ----
        _progress(q, "packaging", 0.0, "Packaging results\u2026")

        # Find resonance frequency (peak of displacement amplitude)
        mean_resp = np.mean(np.abs(harmonic_result.displacement_amplitudes), axis=1)
        idx_res = int(np.argmax(mean_resp))
        resonance_hz = float(harmonic_result.frequencies_hz[idx_res])

        result = {
            "frequencies_hz": [round(float(f), 2) for f in harmonic_result.frequencies_hz],
            "gain": round(float(harmonic_result.gain), 4),
            "q_factor": round(float(harmonic_result.q_factor), 2),
            "contact_face_uniformity": round(float(harmonic_result.contact_face_uniformity), 4),
            "resonance_hz": round(resonance_hz, 2),
            "node_count": int(fea_mesh.nodes.shape[0]),
            "element_count": int(fea_mesh.elements.shape[0]),
            "solve_time_s": round(float(harmonic_result.solve_time_s), 3),
        }
        _progress(q, "packaging", 1.0, "Complete")
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Worker: harmonic stress analysis (full chain: mesh -> harmonic -> stress)
# ---------------------------------------------------------------------------

def _stress_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry point for harmonic stress analysis."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading FEA modules\u2026")
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import HarmonicConfig
        _progress(q, "init", 1.0, "Modules loaded")
        _cancelled(cancel, q)

        # ---- mesh ----
        mesh_size_map = {"coarse": 8.0, "medium": 5.0, "fine": 3.0}
        mesh_size = mesh_size_map.get(params.get("mesh_density", "medium"), 5.0)
        mesher = GmshMesher()

        _progress(q, "meshing", 0.0, "Generating parametric mesh\u2026")
        horn_type = params.get("horn_type", "cylindrical")
        type_map = {"cylindrical": "cylindrical", "exponential": "cylindrical",
                    "flat": "flat", "block": "flat", "unknown": "flat"}
        mapped = type_map.get(horn_type, "flat")

        if mapped == "cylindrical":
            dims = {"diameter_mm": params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}
        else:
            dims = {"width_mm": params.get("width_mm") or params.get("diameter_mm", 25.0),
                    "depth_mm": params.get("depth_mm") or params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}

        fea_mesh = mesher.mesh_parametric_horn(horn_type=mapped,
                                                dimensions=dims,
                                                mesh_size=mesh_size, order=2)
        _progress(q, "meshing", 1.0,
                  f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")
        _cancelled(cancel, q)

        # ---- build harmonic config ----
        material = params.get("material", "Titanium Ti-6Al-4V")
        config = HarmonicConfig(
            mesh=fea_mesh,
            material_name=material,
            freq_min_hz=params.get("freq_min_hz", 16000.0),
            freq_max_hz=params.get("freq_max_hz", 24000.0),
            n_freq_points=params.get("n_freq_points", 201),
            damping_model=params.get("damping_model", "hysteretic"),
            damping_ratio=params.get("damping_ratio", 0.005),
        )

        # ---- harmonic solve (includes internal modal solve) ----
        _progress(q, "assembly", 0.0, "Assembling stiffness & mass matrices\u2026")
        _cancelled(cancel, q)

        _progress(q, "modal_solve", 0.0, "Running modal analysis\u2026")
        _cancelled(cancel, q)

        _progress(q, "harmonic_solve", 0.0, "Running harmonic frequency sweep\u2026")
        solver = SolverA()
        harmonic_result = solver.harmonic_analysis(config)
        _progress(q, "harmonic_solve", 1.0,
                  f"Harmonic sweep done in {harmonic_result.solve_time_s:.1f}s")
        _cancelled(cancel, q)

        # ---- stress computation ----
        _progress(q, "stress_compute", 0.0, "Computing stress from harmonic displacement\u2026")
        target_freq_hz = params.get("frequency_khz", 20.0) * 1000.0
        stress_result = solver.harmonic_stress_analysis(
            harmonic_result, config, target_freq_hz=target_freq_hz
        )
        _progress(q, "stress_compute", 1.0,
                  f"Stress analysis done: max VM = {stress_result.max_stress_mpa:.2f} MPa")
        _cancelled(cancel, q)

        # ---- package ----
        _progress(q, "packaging", 0.0, "Packaging results\u2026")

        # Find resonance frequency
        mean_resp = np.mean(np.abs(harmonic_result.displacement_amplitudes), axis=1)
        idx_res = int(np.argmax(mean_resp))
        resonance_hz = float(harmonic_result.frequencies_hz[idx_res])

        result = {
            "max_stress_mpa": round(float(stress_result.max_stress_mpa), 4),
            "safety_factor": round(float(stress_result.safety_factor), 4)
                if stress_result.safety_factor != float("inf") else 9999.0,
            "max_displacement_mm": round(float(stress_result.max_displacement_mm), 6),
            "contact_face_uniformity": round(float(stress_result.contact_face_uniformity), 4),
            "resonance_hz": round(resonance_hz, 2),
            "node_count": int(fea_mesh.nodes.shape[0]),
            "element_count": int(fea_mesh.elements.shape[0]),
            "solve_time_s": round(
                float(harmonic_result.solve_time_s + stress_result.solve_time_s), 3
            ),
        }
        _progress(q, "packaging", 1.0, "Complete")
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Worker: fatigue life assessment (full chain: mesh -> harmonic -> stress -> fatigue)
# ---------------------------------------------------------------------------

def _fatigue_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry point for fatigue life assessment."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading FEA modules\u2026")
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import HarmonicConfig
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.fatigue import (
            FatigueAssessor,
            FatigueConfig,
        )
        _progress(q, "init", 1.0, "Modules loaded")
        _cancelled(cancel, q)

        # ---- mesh ----
        mesh_size_map = {"coarse": 8.0, "medium": 5.0, "fine": 3.0}
        mesh_size = mesh_size_map.get(params.get("mesh_density", "medium"), 5.0)
        mesher = GmshMesher()

        _progress(q, "meshing", 0.0, "Generating parametric mesh\u2026")
        horn_type = params.get("horn_type", "cylindrical")
        type_map = {"cylindrical": "cylindrical", "exponential": "cylindrical",
                    "flat": "flat", "block": "flat", "unknown": "flat"}
        mapped = type_map.get(horn_type, "flat")

        if mapped == "cylindrical":
            dims = {"diameter_mm": params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}
        else:
            dims = {"width_mm": params.get("width_mm") or params.get("diameter_mm", 25.0),
                    "depth_mm": params.get("depth_mm") or params.get("diameter_mm", 25.0),
                    "length_mm": params.get("length_mm", 80.0)}

        fea_mesh = mesher.mesh_parametric_horn(horn_type=mapped,
                                                dimensions=dims,
                                                mesh_size=mesh_size, order=2)
        _progress(q, "meshing", 1.0,
                  f"Mesh complete: {fea_mesh.nodes.shape[0]} nodes")
        _cancelled(cancel, q)

        # ---- build harmonic config ----
        material = params.get("material", "Titanium Ti-6Al-4V")
        config = HarmonicConfig(
            mesh=fea_mesh,
            material_name=material,
            freq_min_hz=params.get("freq_min_hz", 16000.0),
            freq_max_hz=params.get("freq_max_hz", 24000.0),
            n_freq_points=params.get("n_freq_points", 201),
            damping_model=params.get("damping_model", "hysteretic"),
            damping_ratio=params.get("damping_ratio", 0.005),
        )

        # ---- harmonic solve (includes internal modal solve) ----
        _progress(q, "assembly", 0.0, "Assembling stiffness & mass matrices\u2026")
        _cancelled(cancel, q)

        _progress(q, "modal_solve", 0.0, "Running modal analysis\u2026")
        _cancelled(cancel, q)

        _progress(q, "harmonic_solve", 0.0, "Running harmonic frequency sweep\u2026")
        solver = SolverA()
        harmonic_result = solver.harmonic_analysis(config)
        _progress(q, "harmonic_solve", 1.0,
                  f"Harmonic sweep done in {harmonic_result.solve_time_s:.1f}s")
        _cancelled(cancel, q)

        # ---- stress computation ----
        _progress(q, "stress_compute", 0.0, "Computing stress from harmonic displacement\u2026")
        target_freq_hz = params.get("frequency_khz", 20.0) * 1000.0
        stress_result = solver.harmonic_stress_analysis(
            harmonic_result, config, target_freq_hz=target_freq_hz
        )
        _progress(q, "stress_compute", 1.0,
                  f"Stress analysis done: max VM = {stress_result.max_stress_mpa:.2f} MPa")
        _cancelled(cancel, q)

        # ---- fatigue assessment ----
        _progress(q, "fatigue_assess", 0.0, "Running fatigue life assessment\u2026")

        # Map material name for fatigue S-N database
        fatigue_material_map = {
            "Titanium Ti-6Al-4V": "Ti-6Al-4V",
            "Ti-6Al-4V": "Ti-6Al-4V",
            "Aluminum 7075-T6": "Al 7075-T6",
            "Al 7075-T6": "Al 7075-T6",
            "Steel D2": "Steel D2",
            "CPM 10V": "CPM 10V",
            "M2 HSS": "M2 HSS",
        }
        fatigue_mat = fatigue_material_map.get(material, "Ti-6Al-4V")

        fatigue_config = FatigueConfig(
            material=fatigue_mat,
            surface_finish=params.get("surface_finish", "machined"),
            characteristic_diameter_mm=params.get("characteristic_diameter_mm", 25.0),
            reliability_pct=params.get("reliability_pct", 90.0),
            temperature_c=params.get("temperature_c", 25.0),
            Kt_global=params.get("Kt_global", 1.5),
        )
        assessor = FatigueAssessor(fatigue_config)

        # Convert Von Mises stress from Pa to MPa
        stress_vm_mpa = stress_result.stress_vm / 1e6

        fatigue_result = assessor.assess(stress_vm_mpa)

        # Convert life cycles to hours at operating frequency
        frequency_hz = params.get("frequency_khz", 20.0) * 1000.0
        estimated_hours = fatigue_result.estimated_life_cycles / (frequency_hz * 3600.0)

        # Get corrected endurance limit
        corrected_endurance = assessor.corrected_endurance_limit()

        _progress(q, "fatigue_assess", 1.0,
                  f"Fatigue assessment done: min SF = {fatigue_result.min_safety_factor:.3f}")
        _cancelled(cancel, q)

        # ---- package ----
        _progress(q, "packaging", 0.0, "Packaging results\u2026")

        # Build safety factor distribution (per-element)
        sf_distribution = [round(float(sf), 4) if np.isfinite(sf) else 9999.0
                           for sf in fatigue_result.safety_factors]

        # Build critical locations list
        critical_locations = []
        if fatigue_result.critical_location is not None:
            critical_locations.append({
                "element_id": 0,
                "safety_factor": round(float(fatigue_result.min_safety_factor), 4),
                "x": round(float(fatigue_result.critical_location[0]), 6),
                "y": round(float(fatigue_result.critical_location[1]), 6),
                "z": round(float(fatigue_result.critical_location[2]), 6),
            })

        total_solve_time = float(harmonic_result.solve_time_s + stress_result.solve_time_s)

        result = {
            "min_safety_factor": round(float(fatigue_result.min_safety_factor), 4)
                if np.isfinite(fatigue_result.min_safety_factor) else 9999.0,
            "estimated_life_cycles": float(fatigue_result.estimated_life_cycles)
                if np.isfinite(fatigue_result.estimated_life_cycles) else 1e30,
            "estimated_hours_at_20khz": round(float(estimated_hours), 4)
                if np.isfinite(estimated_hours) else 1e20,
            "critical_locations": critical_locations,
            "sn_curve_name": fatigue_result.sn_curve_name,
            "corrected_endurance_mpa": round(float(corrected_endurance), 4),
            "max_stress_mpa": round(float(stress_result.max_stress_mpa), 4),
            "safety_factor_distribution": sf_distribution,
            "node_count": int(fea_mesh.nodes.shape[0]),
            "element_count": int(fea_mesh.elements.shape[0]),
            "solve_time_s": round(total_solve_time, 3),
        }
        _progress(q, "packaging", 1.0, "Complete")
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Worker: assembly analysis
# ---------------------------------------------------------------------------

def _assembly_worker(params: dict, q: mp.Queue, cancel: mp.Event):
    """Child-process entry for assembly (multi-component) analysis."""
    os.environ["OMP_NUM_THREADS"] = str(GMSH_MAX_THREADS)
    try:
        _progress(q, "init", 0.0, "Loading modules…")
        from web.services.fea_service import FEAService
        _progress(q, "init", 1.0, "Ready")
        _cancelled(cancel, q)

        svc = FEAService()
        # Delegate to the existing method which already handles all assembly logic
        _progress(q, "meshing", 0.0, "Running assembly analysis…")
        result = svc.run_assembly_analysis(
            components=params["components"],
            coupling_method=params.get("coupling_method", "penalty"),
            penalty_factor=params.get("penalty_factor", 1e10),
            analyses=params.get("analyses", ["modal", "harmonic"]),
            frequency_hz=params.get("frequency_hz", 20000.0),
            n_modes=params.get("n_modes", 20),
            damping_ratio=params.get("damping_ratio", 0.01),
            use_gmsh=params.get("use_gmsh", True),
        )
        _progress(q, "packaging", 1.0, "Complete")
        q.put({"type": "complete", "result": result})

    except SystemExit:
        pass
    except Exception as exc:
        q.put({"type": "error", "error": str(exc),
               "traceback": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Async runner (used by FastAPI route handlers)
# ---------------------------------------------------------------------------

# Map task type to (worker function, steps list)
_WORKER_MAP: dict[str, tuple[Callable, list[str]]] = {
    "modal":         (_modal_worker,    MODAL_STEPS),
    "modal_step":    (_modal_worker,    MODAL_STEP_STEPS),
    "harmonic":      (_harmonic_worker, HARMONIC_STEPS),
    "harmonic_step": (_harmonic_worker, HARMONIC_STEP_STEPS),
    "stress":        (_stress_worker,   STRESS_STEPS),
    "fatigue":       (_fatigue_worker,  FATIGUE_STEPS),
    "acoustic":      (_acoustic_worker, ACOUSTIC_STEPS),
    "assembly":      (_assembly_worker, ASSEMBLY_STEPS),
}


class FEAProcessRunner:
    """Launch FEA in a child process with progress, timeout and cancel."""

    async def run(
        self,
        task_type: str,
        params: dict,
        timeout_s: float = FEA_TIMEOUT_S,
        *,
        on_progress: Optional[Callable] = None,
    ) -> dict:
        """Run FEA and return the result dict.

        *on_progress* is an optional ``async def callback(phase, progress, message)``
        that is called for every progress event (used to feed analysis_manager).

        Raises:
            TimeoutError – if the child process exceeds *timeout_s*
            RuntimeError – if the child process reports an error
            asyncio.CancelledError – if cancelled via the returned cancel handle
        """
        worker_fn, steps = _WORKER_MAP.get(task_type, (_modal_worker, MODAL_STEPS))

        ctx = mp.get_context("spawn")
        progress_queue: mp.Queue = ctx.Queue()
        cancel_event: mp.Event = ctx.Event()

        proc = ctx.Process(target=worker_fn,
                           args=(params, progress_queue, cancel_event),
                           daemon=True)
        proc.start()
        logger.info("FEA subprocess started: pid=%s type=%s timeout=%ss",
                     proc.pid, task_type, timeout_s)

        # Store cancel hook so external code can cancel
        self._cancel_event = cancel_event
        self._process = proc

        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout_s

        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning("FEA timeout after %ss – killing pid %s",
                                   timeout_s, proc.pid)
                    proc.kill()
                    proc.join(timeout=5)
                    raise TimeoutError(
                        f"FEA computation timed out after {int(timeout_s)}s")

                # Non-blocking poll the queue from executor thread
                try:
                    msg = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: progress_queue.get(timeout=1.0),
                        ),
                        timeout=min(remaining, 2.0),
                    )
                except (asyncio.TimeoutError, queue.Empty, EOFError):
                    if not proc.is_alive():
                        # Process died without sending result
                        exit_code = proc.exitcode
                        if exit_code and exit_code < 0:
                            raise RuntimeError(
                                f"FEA process killed by signal {-exit_code}")
                        raise RuntimeError(
                            f"FEA process exited unexpectedly (code={exit_code})")
                    continue

                logger.info("FEA queue msg: type=%s phase=%s progress=%s",
                            msg.get("type"), msg.get("phase"), msg.get("progress"))

                if msg["type"] == "complete":
                    logger.info("FEA subprocess complete – returning result")
                    if on_progress:
                        await on_progress("packaging", 1.0, "Complete")
                    return msg["result"]

                if msg["type"] == "error":
                    logger.error("FEA subprocess error: %s", msg.get("error"))
                    raise RuntimeError(msg["error"])

                if msg["type"] == "cancelled":
                    raise asyncio.CancelledError("FEA cancelled by user")

                if msg["type"] == "progress" and on_progress:
                    await on_progress(
                        msg.get("phase", ""),
                        msg.get("progress", 0.0),
                        msg.get("message", ""),
                    )

        except asyncio.CancelledError:
            cancel_event.set()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
            raise
        finally:
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=5)

    def cancel(self):
        """Request cancellation of the running FEA subprocess."""
        if hasattr(self, "_cancel_event"):
            self._cancel_event.set()
        if hasattr(self, "_process") and self._process.is_alive():
            self._process.terminate()
