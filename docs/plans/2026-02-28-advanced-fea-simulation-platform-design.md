# Advanced FEA Simulation Platform Design

**Date**: 2026-02-28
**Status**: Approved
**Approach**: B (Hybrid: SolverA + FEniCSx)

## Overview

Extend the ultrasonic welding simulation platform from basic modal analysis to a comprehensive FEA workstation covering the full analysis chain, knurl simulation, contact mechanics, thermal analysis, and optimized 3D export. The platform uses a dual-solver architecture: SolverA (scipy/numpy) for linear analysis and FEniCSx (PETSc/MPI) for nonlinear contact and thermal analysis.

## Architecture

### Dual-Solver Router

```
Frontend (Vue 3) — Analysis Workbench (hybrid layout)
    |
    | HTTP + WebSocket
    v
FastAPI Backend — Unified API + AnalysisManager (subprocess) + WS Progress
    |
    v
Unified Mesh Pipeline: CadQuery -> Gmsh -> TET10 Mesh -> Solver Router
    |                           |
    v                           v
SolverA (scipy/numpy)      FEniCSx (dolfinx/PETSc/MPI)
  - Modal analysis           - Penalty/Nitsche contact
  - Harmonic response        - Horn-Workpiece-Anvil coupling
  - Static stress            - Transient thermal
  - Fatigue evaluation       - Friction heating
```

### Key Design Decisions

1. **Dual-solver routing**: SolverA handles linear analysis (modal, harmonic, stress, fatigue), FEniCSx handles nonlinear contact and thermal. Unified API, backend selects solver automatically.

2. **Unified mesh pipeline**: CadQuery generates/imports geometry -> Gmsh meshes with TET10 -> both solvers share the same mesh format. FEniCSx uses `dolfinx.io.gmshio` for conversion.

3. **Analysis chain**: Modal -> Harmonic -> Stress -> Fatigue all via SolverA (code exists). Contact analysis uses FEniCSx.

4. **Progress/cancel**: Reuse existing subprocess + WebSocket architecture. FEniCSx analyses also run in subprocess isolation.

5. **FEniCSx deployment**: Docker container (`dolfinx/dolfinx:stable`), invoked via `docker exec` or subprocess from backend.

## Analysis Chain (Flexible DAG)

### Component Auto-Detection

When a STEP file is uploaded:
1. Parse STEP assembly structure (OCP TopoDS_Compound)
2. Auto-classify by geometric features:
   - Horn: tapered/exponential/stepped main body
   - Booster: cylindrical transition piece
   - Transducer: ring/cylindrical piece
   - Anvil: bottom support piece
   - Workpiece: user-specified or auto-inferred
3. Display results for user confirmation/correction

### Module Dependency Graph

```
                Meshing (required)
                    |
         +----------+----------+
         v          v          v
    Modal       Static     Contact
   (SolverA)  (SolverA)  (FEniCSx)
     |                      |
     v                      v
   Harmonic              Thermal
   (SolverA)            (FEniCSx)
     |
   +-+--------+
   v          v
Stress    Amplitude
(SolverA) Uniformity
   |
   v
Fatigue
(SolverA)
```

### Module Dependencies

| Module | Standalone? | Requires | Solver |
|--------|------------|----------|--------|
| Modal analysis | Yes | Mesh only | SolverA |
| Static analysis | Yes | Mesh only | SolverA |
| Harmonic analysis | No | Modal results (frequency range) | SolverA |
| Stress analysis | No | Harmonic displacement field | SolverA |
| Amplitude uniformity | No | Harmonic displacement field | SolverA |
| Fatigue evaluation | No | Stress field | SolverA |
| Contact analysis | Yes | Multi-component meshes | FEniCSx |
| Thermal analysis | No | Contact results (friction work) | FEniCSx |

### User Interaction

1. Upload model -> auto-detect components -> user confirms/corrects
2. Select scope: single component / assembly / full system
3. Check analysis modules -> system auto-resolves dependency chain
4. One-click run -> execute in dependency order -> real-time progress + intermediate previews

## Detailed Analysis Modules

### Modal Analysis (existing, no changes needed)

- Input: K, M matrices from TET10 mesh
- Output: natural frequencies, mode shapes, mode type classification
- Key metrics: closest longitudinal mode, frequency deviation %

### Harmonic Response Analysis (SolverA code exists, needs API exposure)

- Input: K, M, damping model, excitation frequency range, excitation force
- Three damping models: hysteretic, Rayleigh, modal superposition
- Output: FRF (frequency-amplitude curve), full complex displacement field
- Key metrics:
  - Weld face amplitude uniformity (min/mean at contact face)
  - Maximum amplitude location and magnitude
  - Gain and Q-factor
  - 3D amplitude distribution contour
- New API: `POST /api/v1/analysis/harmonic`
- Auto-inherit frequency range from modal results (target +/- 5%)

### Harmonic Stress Analysis (new: derive from harmonic displacement)

- Not static loading — compute stress directly from harmonic displacement field U(omega_0)
- Reuse B-matrix and D-matrix code from static_analysis()
- New method: `harmonic_stress_analysis()`
- Output: stress tensor field, Von Mises contour
- Key metrics:
  - Max Von Mises stress vs material yield strength
  - Stress concentration location markers
  - Safety factor = sigma_yield / sigma_max

### Fatigue Evaluation (new module: fatigue.py)

- Method: High-cycle fatigue (S-N curve approach)
- Ultrasonic welding: 20kHz cycling, typically 10^9+ cycles
- S-N data added to material database (Ti-6Al-4V fatigue limit ~500 MPa)
- Goodman correction for mean stress effects
- Miner cumulative damage rule
- Output: fatigue life (cycles), damage factor distribution
- Key metrics:
  - Shortest fatigue life location
  - Equivalent operating hours (at 20kHz)
  - Safe/warning/danger zone markers

### Contact Analysis (FEniCSx)

Full system model:
```
Transducer (fixed at top)
  | bonded
Booster
  | bonded (threaded connection)
Horn (+ knurl) <- 20kHz excitation
  | CONTACT (friction)
Workpiece
  | CONTACT (friction)
Anvil (+ knurl, fixed at bottom)
```

- Contact type: Coulomb friction (mu = 0.3-0.6)
- Contact algorithm: Augmented Lagrangian / Nitsche
- Solve flow:
  1. Static contact: apply pressure -> solve contact state
  2. Harmonic excitation overlay: apply 20kHz vibration on contact state
  3. Friction work: q = mu * N * delta_u (per-cycle friction heating)

### Thermal Analysis (FEniCSx)

- Input: friction work from contact analysis
- Equation: rho*c * dT/dt = k * nabla^2(T) + q
- Boundary conditions: convective cooling
- Output: temperature field time evolution
- Key metrics:
  - Maximum temperature at weld interface
  - Time to reach melting point
  - Temperature distribution uniformity

## Knurl FEA Pipeline

### Geometry Generation

1. Define knurl parameters: type (linear/cross_hatch/diamond/conical/spherical), pitch, tooth_width, depth, angle
2. CadQuery generates 3D knurl features on horn weld face:
   - Extend existing `horn_generator._cq_apply_knurl()` to work on imported STEP horn bodies
   - Boolean operations to merge knurl features
3. Output: B-Rep geometry with knurl

### Adaptive Meshing

- Knurl region: fine mesh (element_size = 0.3-0.5mm)
- Away from knurl: coarse mesh (element_size = 5-8mm)
- Transition zone: gradual mesh refinement
- Total node count stays manageable (not exploding to millions)

### Optimization Loop

```
Initial params -> CadQuery generate -> FEA chain -> Evaluate metrics
     ^                                                    |
     |         Optimization (parameter sweep / Bayesian)  |
     +----------------------------------------------------+
```

Metrics: amplitude uniformity, max stress < limit, energy transfer efficiency, contact area coverage

### STEP Export

- CadQuery `exporters.export()` for CAD-grade STEP output
- New API: `POST /api/v1/knurl/export-step`
- Download via frontend

## Performance Optimization

| Bottleneck | Current | Solution | Expected |
|-----------|---------|----------|----------|
| Matrix assembly | Single-thread Python | Vectorized + numba JIT | 3-5x |
| Eigenvalue solve (eigsh) | Single sigma shift-invert | AMG preconditioner + LOBPCG | 2-3x |
| BLAS/LAPACK | Possibly unoptimized | Confirm MKL or OpenBLAS multithreaded | 1.5-2x |
| Mesh density | uniform medium=5mm | Adaptive mesh (fine at stress, coarse elsewhere) | 40-60% fewer nodes |
| Repeated analysis | Rebuild from scratch | Cache K/M matrices (same model+mesh) | Skip 60s assembly |

Target: 363s -> 60-120s for medium mesh modal analysis. Full chain 8-15 minutes.

## UI Design

### Page Architecture (restructured)

| Current Page | Action |
|-------------|--------|
| GeometryView | -> Merge into Analysis Workbench |
| SimulationView | -> Merge into Analysis Workbench |
| AcousticView | -> Keep (acoustic-specific) |
| ParameterView | -> Keep (parameter adjustment) |
| FatigueView | -> Merge into Analysis Workbench fatigue tab |
| AssemblyView | -> Merge into Analysis Workbench components panel |
| KnurlView | -> Migrate to Knurl Workbench |
| MaterialView | -> Keep (material database) |
| HistoryView | -> Keep |
| DashboardView | -> Keep (home page) |

### Navigation Sidebar

- Project management (new/open/recent)
- Analysis Workbench (core: merge Geometry+Simulation+Assembly+Fatigue)
- Knurl Workbench (new: knurl design + FEA + optimization + export)
- Parameter adjustment (keep)
- Acoustic analysis (keep)
- History
- Settings

### Analysis Workbench Layout (hybrid)

Left panel: 3D view (Three.js) with model rotation/zoom, component highlighting, result contour overlay, modal animation, cross-section view.

Right panel (top): Component panel (auto-detected, checkboxes for scope selection) + Analysis module checkboxes with dependency indicators.

Right panel (bottom): Material, frequency, mesh, damping configuration.

Bottom panel: Result tabs [Modal] [Harmonic] [Stress] [Fatigue] [Contact] [Thermal] [Report]

### Knurl Workbench Layout

Left: 3D preview (real-time knurl parameter update)
Right: Knurl parameters (type, pitch, width, depth, angle) + source (manual / optimizer / import) + target horn/anvil selection + action buttons (FEA / export STEP)
Bottom: [FEA Results] [Optimization History] [Comparison] tabs

## Implementation Phases

### Phase 1: Complete Analysis Chain + Performance + UI Foundation (1-2 weeks)

Backend:
- Expose `harmonic_analysis()` as API endpoint (code exists)
- New `harmonic_stress_analysis()` — compute stress from harmonic displacement
- New `fatigue.py` module — S-N curve fatigue evaluation
- Performance: vectorized assembly, AMG preconditioner, adaptive mesh
- K/M matrix caching for repeated analysis

Frontend:
- Create `AnalysisWorkbench.vue` — merge Geometry+Simulation
- Component detection panel + module checkboxes + dependency resolver
- Result tabs (modal/harmonic/stress/fatigue)
- 3D result contour rendering

Deliverable: Upload STEP -> one-click full chain modal->harmonic->stress->fatigue

### Phase 2: Knurl FEA + STEP Export (2-3 weeks)

Backend:
- Extend `horn_generator._cq_apply_knurl()` for imported STEP horn bodies
- Gmsh adaptive mesh strategy (fine at knurl, coarse elsewhere)
- Knurl FEA API endpoint
- CadQuery STEP export API endpoint
- Knurl optimization loop (parameter sweep + FEA evaluation)

Frontend:
- Create `KnurlWorkbench.vue` — parameter editor + 3D preview + FEA + export
- Real-time knurl 3D preview (update mesh on parameter change)
- Knurl vs no-knurl comparison panel

Deliverable: Design knurl -> FEA verify -> optimize -> export STEP

### Phase 3: FEniCSx Contact + Thermal + Anvil (3-4 weeks)

Infrastructure:
- Deploy FEniCSx Docker container on server
- SolverA <-> FEniCSx mesh conversion pipeline
- FEniCSx subprocess isolation + progress push

Backend:
- Contact analysis module: penalty/Nitsche contact formulation
- Anvil parametric generation (flat/groove/knurled/contour)
- Full system assembly: Horn + Workpiece + Anvil
- Friction heating calculation + transient thermal solve
- Thermal results (temperature field time evolution)

Frontend:
- Analysis workbench adds contact/thermal tabs
- Anvil component upload/generation UI
- Temperature field animation playback
- Full system 3D assembly view

Deliverable: Full system Horn-Workpiece-Anvil contact + thermal analysis
