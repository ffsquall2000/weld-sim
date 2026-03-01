# Ultrasonic Metal Welding Virtual Simulation Platform - Design Document

**Date:** 2026-03-01
**Status:** Approved
**Scope:** Complete platform restructure from parameter auto-adjuster to full virtual simulation platform

---

## Context

The current system is a welding parameter auto-adjuster with simplified numpy/scipy FEA, plugin-based engine, Vue 3 frontend with wizard-style interface, and SQLite storage. The goal is to transform it into a full CAE virtual simulation platform that:

- Integrates open-source multi-physics solvers (FEniCS, Elmer, CalculiX)
- Provides a professional CAE workbench UI with dockable panels, workflow canvas, and VTK.js 3D
- Supports single-body and assembly coupled simulation
- Implements geometry optimization closed-loop (automatic + manual)
- Uses PostgreSQL for robust data management
- Covers all simulation objects: horn, anvil, frame, full machine assembly

---

## A) Architecture Overview

### Code Reuse Strategy

**Keep & reuse:**
- `core/plugin_api.py`, `event_bus.py`, `plugin_manager.py` — Plugin system
- `plugins/material_db/` — Material database with `materials.yaml`
- `plugins/li_battery/physics.py` — Physics models (acoustic impedance, friction, thermal)
- `plugins/geometry_analyzer/horn_generator.py` — Parametric horn generator (CadQuery + numpy)
- `web/services/suggestion_service.py` — Parameter suggestion engine
- `web/services/knurl_service.py` — Knurl optimizer with Pareto computation

**Rewrite:**
- `web/services/fea_service.py` → Solver abstraction layer (keep as "preview solver")
- `core/database.py` → PostgreSQL + SQLAlchemy + Alembic
- `frontend/` → Complete restructure to CAE workbench
- `ThreeViewer.vue` → VTK.js professional visualization

### Directory Structure

```
project-root/
├── backend/
│   ├── alembic/                    # DB migrations
│   ├── app/
│   │   ├── main.py                 # FastAPI app factory
│   │   ├── config.py               # pydantic-settings
│   │   ├── dependencies.py         # DI: db session, task queue
│   │   ├── models/                 # SQLAlchemy ORM
│   │   │   ├── project.py
│   │   │   ├── geometry_version.py
│   │   │   ├── simulation_case.py
│   │   │   ├── run.py
│   │   │   ├── artifact.py
│   │   │   ├── metric.py
│   │   │   ├── comparison.py
│   │   │   ├── material.py
│   │   │   └── optimization_study.py
│   │   ├── schemas/                # Pydantic request/response
│   │   ├── routers/                # API endpoints
│   │   │   ├── projects.py
│   │   │   ├── geometries.py
│   │   │   ├── simulations.py
│   │   │   ├── runs.py
│   │   │   ├── artifacts.py
│   │   │   ├── metrics.py
│   │   │   ├── comparisons.py
│   │   │   ├── optimizations.py
│   │   │   ├── materials.py
│   │   │   ├── workflows.py
│   │   │   ├── ws.py               # WebSocket
│   │   │   └── health.py
│   │   ├── services/               # Business logic
│   │   │   ├── geometry_service.py
│   │   │   ├── simulation_service.py
│   │   │   ├── optimization_service.py
│   │   │   ├── comparison_service.py
│   │   │   ├── material_service.py
│   │   │   ├── artifact_service.py
│   │   │   ├── workflow_service.py
│   │   │   ├── suggestion_service.py
│   │   │   └── report_service.py
│   │   ├── solvers/                # Solver abstraction
│   │   │   ├── base.py             # SolverBackend ABC
│   │   │   ├── fenics_solver.py
│   │   │   ├── elmer_solver.py
│   │   │   ├── calculix_solver.py
│   │   │   ├── preview_solver.py   # Existing numpy/scipy
│   │   │   ├── mesh_converter.py   # gmsh + meshio
│   │   │   ├── result_reader.py
│   │   │   └── coupling.py         # Multi-physics orchestrator
│   │   ├── tasks/                  # Celery async tasks
│   │   │   ├── solver_tasks.py
│   │   │   ├── optimization_tasks.py
│   │   │   └── export_tasks.py
│   │   └── domain/                 # Migrated domain logic
│   │       ├── physics.py
│   │       ├── horn_generator.py
│   │       ├── knurl_optimizer.py
│   │       ├── material_properties.py
│   │       └── knowledge_rules.py
│   ├── storage/                    # Artifact file storage
│   ├── tests/
│   ├── pyproject.toml
│   ├── Dockerfile
│   └── docker-compose.yml          # PostgreSQL + Redis + backend
│
├── frontend/
│   ├── src/
│   │   ├── main.ts
│   │   ├── App.vue
│   │   ├── api/                    # Generated from OpenAPI
│   │   ├── stores/                 # Pinia stores
│   │   │   ├── project.ts
│   │   │   ├── simulation.ts
│   │   │   ├── geometry.ts
│   │   │   ├── workflow.ts
│   │   │   ├── viewer3d.ts
│   │   │   ├── optimization.ts
│   │   │   ├── layout.ts
│   │   │   └── settings.ts
│   │   ├── components/
│   │   │   ├── layout/             # AppShell, PanelSystem, DockPanel, StatusBar
│   │   │   ├── viewer/             # VtkViewport, ContourLegend, ViewportToolbar, SlicePlane
│   │   │   ├── workflow/           # WorkflowCanvas, SimNode, ConnectionEdge, NodePalette
│   │   │   ├── optimization/       # ParetoChart, VariableEditor, IterationTable
│   │   │   ├── comparison/         # MetricsTable, RadarChart, SideBySideViewer
│   │   │   ├── panels/             # ProjectExplorer, PropertyEditor, SolverConsole, etc.
│   │   │   └── common/             # Reused charts, wizard components
│   │   ├── views/
│   │   │   ├── ProjectView.vue
│   │   │   ├── WorkbenchView.vue   # Main simulation workbench
│   │   │   ├── OptimizationView.vue
│   │   │   ├── ComparisonView.vue
│   │   │   └── SettingsView.vue
│   │   ├── composables/
│   │   │   ├── useVtkViewer.ts
│   │   │   ├── useWorkflowGraph.ts
│   │   │   ├── useWebSocket.ts
│   │   │   └── useDockLayout.ts
│   │   ├── router/
│   │   └── i18n/
│   ├── package.json
│   └── vite.config.ts
│
├── contracts/
│   ├── openapi.yaml                # OpenAPI 3.1 spec
│   └── websocket-protocol.md
│
└── docs/
```

---

## B) Frontend Architecture

### Library Choices

| Concern | Library | License | Purpose |
|---------|---------|---------|---------|
| Panel system | golden-layout v2 | MIT | Dockable, draggable panel layout |
| Workflow canvas | @vue-flow/core | MIT | Node graph for simulation pipelines |
| 3D visualization | @kitware/vtk.js | BSD | FEA mesh, contours, iso-surfaces, slicing |
| State management | Pinia | MIT | (already in use) |
| Charts | ECharts | Apache 2.0 | (already in use) |

### Default Layout

```
AppShell
  ├── TopMenuBar (project name, file/edit/view menus)
  ├── PanelSystem (golden-layout)
  │   ├── Left: ProjectExplorer + GeometryTree
  │   ├── Center: VtkViewport / WorkflowCanvas (tabbed)
  │   ├── Right: PropertyEditor + MetricsPanel
  │   └── Bottom: SolverConsole + StatusBar
  └── StatusBar (solver status, progress, connection)
```

### Workflow Node Types

1. **GeometryNode** — Import STEP/STL or parametric generation
2. **MeshNode** — Configure mesh density/refinement
3. **MaterialNode** — Assign material properties
4. **BoundaryConditionNode** — Define loads, constraints, contacts
5. **SolverNode** — Select solver + analysis type
6. **PostProcessNode** — Extract metrics, generate contours
7. **CompareNode** — Multi-run comparison
8. **OptimizeNode** — Launch parametric optimization

### VTK.js Viewport Features

- WebGL rendering with GPU acceleration
- Unstructured mesh display (VTU format from solvers)
- Contour plots with color legends
- Iso-surfaces and vector field glyphs
- Slice planes with real-time interaction
- Wireframe/solid toggle
- Scalar field selector dropdown

---

## C) Solver Strategy

### Solver Abstraction

```python
class SolverBackend(ABC):
    def name(self) -> str: ...
    def supported_analyses(self) -> list[AnalysisType]: ...
    async def prepare(self, config: SolverConfig) -> PreparedJob: ...
    async def run(self, job: PreparedJob, progress_callback) -> SolverResult: ...
    def read_results(self, result: SolverResult) -> FieldData: ...
```

### Solver Mapping

| Solver | Capabilities | Mesh Format | Run Mode |
|--------|-------------|-------------|----------|
| FEniCS (dolfinx) | Thermal steady/transient, structural | XDMF/HDF5 via meshio | In-process Python API |
| Elmer | Piezoelectric, acoustic, impedance | Elmer format via ElmerGrid | subprocess CLI |
| CalculiX | Modal, harmonic, static structural | Abaqus .inp via meshio | subprocess CLI |
| Preview | Modal, harmonic (simplified) | In-memory numpy arrays | In-process (existing code) |

### Multi-Physics Coupling

```
Step 1: Piezoelectric (Elmer) → horn vibration amplitude
Step 2: Contact/structural (FEniCS/CalculiX) → pressure distribution + friction heat
Step 3: Thermal (FEniCS) → temperature field
Step 4: Post-processing → extract 12 standard metrics
```

### Mesh Conversion Pipeline

```
STEP → gmsh (Python API) → .msh → meshio → solver-specific format
```

### Assembly Simulation

- Each body is a separate GeometryVersion
- SimulationCase.assembly_components lists all body IDs
- Contact conditions defined in boundary_conditions
- Gmsh merges multiple STEP bodies into single mesh with tagged physical groups

---

## D) PostgreSQL Data Models

### Entity Relationships

```
Project 1──N GeometryVersion
Project 1──N SimulationCase
Project 1──N Comparison

SimulationCase 1──N Run
GeometryVersion 1──N Run

Run 1──N Artifact
Run 1──N Metric

SimulationCase 1──N OptimizationStudy
OptimizationStudy 1──N Run
```

### Table Definitions

**Project:** id(UUID), name, description, application_type, settings(JSONB), tags(VARCHAR[])

**GeometryVersion:** id, project_id(FK), version_number, source_type(parametric/imported_step/imported_stl), parametric_params(JSONB), file_path, mesh_config(JSONB), mesh_file_path, parent_version_id(FK self), metadata(JSONB)

**SimulationCase:** id, project_id(FK), name, analysis_type, solver_backend, configuration(JSONB), boundary_conditions(JSONB), material_assignments(JSONB), assembly_components(JSONB), workflow_dag(JSONB)

**Run:** id, simulation_case_id(FK), geometry_version_id(FK), optimization_study_id(FK nullable), iteration_number, status(queued/running/completed/failed/cancelled), solver_log, compute_time_s, input_snapshot(JSONB)

**Artifact:** id, run_id(FK), artifact_type(mesh_input/solver_input/result_vtu/screenshot/report_pdf/report_excel/contour_image), file_path, file_size_bytes, mime_type, metadata(JSONB)

**Metric:** id, run_id(FK), metric_name, value(FLOAT), unit, metadata(JSONB). UNIQUE(run_id, metric_name)

**Comparison:** id, project_id(FK), name, run_ids(UUID[]), metric_names(VARCHAR[]), baseline_run_id(FK nullable)
- Child: comparison_results(comparison_id, run_id, metric_name, value, delta_from_baseline, delta_percent)

**OptimizationStudy:** id, simulation_case_id(FK), name, strategy(parametric_sweep/bayesian/genetic/manual_guided), design_variables(JSONB), constraints(JSONB), objectives(JSONB), status, total_iterations, completed_iterations, best_run_id(FK), pareto_front_run_ids(UUID[])

---

## E) 12 Standardized Metrics

| # | Name | Unit | Description |
|---|------|------|-------------|
| 1 | natural_frequency_hz | Hz | Closest natural frequency to target |
| 2 | frequency_deviation_pct | % | Deviation from target frequency |
| 3 | amplitude_uniformity | 0-1 | Min/max amplitude ratio on contact face |
| 4 | max_von_mises_stress_mpa | MPa | Maximum Von Mises stress |
| 5 | stress_safety_factor | - | Yield strength / max stress |
| 6 | max_temperature_rise_c | C | Max temperature increase at interface |
| 7 | contact_pressure_uniformity | 0-1 | Pressure uniformity across weld area |
| 8 | effective_contact_area_mm2 | mm2 | Actual contact area (with knurl) |
| 9 | energy_coupling_efficiency | 0-1 | Useful energy / total input energy |
| 10 | horn_gain | - | Amplitude amplification factor |
| 11 | modal_separation_hz | Hz | Gap to nearest unwanted mode |
| 12 | fatigue_cycle_estimate | cycles | Estimated fatigue life |

---

## F) Optimization Strategy

### Automatic Parametric Optimization

**Design variables (horn):** width_mm(15-60), height_mm(40-150), length_mm(15-60), horn_type(categorical), knurl_pitch_mm(0.5-3.0), knurl_depth_mm(0.05-0.8), chamfer_radius_mm(0-2.0)

**Constraints:**
- natural_frequency_hz within 1% of target
- max_von_mises_stress_mpa <= material yield / safety_factor
- stress_safety_factor >= 2.0
- max_temperature_rise_c <= 200
- amplitude_uniformity >= 0.80

**Objectives (multi-objective Pareto):**
- Maximize amplitude_uniformity (weight 0.4)
- Maximize stress_safety_factor (weight 0.3)
- Maximize energy_coupling_efficiency (weight 0.2)
- Maximize horn_gain (weight 0.1)

**Strategies:**
- Parametric sweep (full factorial / Latin Hypercube)
- Bayesian optimization (scikit-optimize, GP surrogate)
- Genetic algorithm (pymoo, NSGA-II)

### Closed-Loop Flow

```
Define variables/constraints/objectives
  → Generate parameter set
  → horn_generator.generate(params) → new GeometryVersion
  → mesh_converter.step_to_mesh() → mesh file
  → Submit Run → Solver executes → Results
  → Extract metrics → Constraint check
  → Record (params, metrics, feasible)
  → Update Pareto front
  → Next iteration (if not converged)
  → WebSocket progress broadcast
```

### Rule-Guided Manual Optimization

- User modifies parameter in PropertyEditor
- SuggestionService evaluates against knowledge base rules + previous metrics
- Recommendations panel shows predicted impact + confidence
- User accepts/rejects → submits new Run
- All iterations tracked in OptimizationStudy with strategy="manual_guided"

---

## G) API Contract (Priority Endpoints)

### Core 7 Endpoints (Sprint 1)

1. `POST/GET /api/v2/projects` — Project CRUD
2. `POST /api/v2/projects/{pid}/geometries` — Create geometry version
3. `POST /api/v2/simulations/{sid}/runs` — Submit solver run
4. `GET /api/v2/runs/{id}` — Query run status
5. `WS /api/v2/ws/runs/{run_id}` — WebSocket progress stream
6. `GET /api/v2/runs/{id}/results/field/{name}` — VTK field data
7. `GET /api/v2/runs/{id}/metrics` — Get metrics

### WebSocket Protocol

```json
{"type": "progress", "run_id": "uuid", "percent": 45, "phase": "solving", "message": "...", "elapsed_s": 12.3}
{"type": "metric_update", "run_id": "uuid", "metric_name": "natural_frequency_hz", "value": 19850.3, "unit": "Hz"}
{"type": "completed", "run_id": "uuid", "status": "completed", "compute_time_s": 45.2, "metrics_summary": {...}}
{"type": "error", "run_id": "uuid", "error": "Solver failed to converge"}
```

### Full Endpoint List

**Projects:** POST/GET /projects, GET/PATCH/DELETE /projects/{id}
**Geometries:** POST /projects/{pid}/geometries, GET /geometries/{id}, POST /geometries/{id}/generate, POST /geometries/{id}/mesh, POST /geometries/upload, GET /geometries/{id}/preview
**Simulations:** POST /projects/{pid}/simulations, GET /simulations/{id}, PATCH /simulations/{id}, POST /simulations/{id}/validate
**Runs:** POST /simulations/{sid}/runs, GET /runs/{id}, POST /runs/{id}/cancel, GET /runs/{id}/metrics, GET /runs/{id}/artifacts, GET /runs/{id}/artifacts/{aid}/download, GET /runs/{id}/results/field/{name}
**Comparisons:** POST /projects/{pid}/comparisons, GET /comparisons/{id}, POST /comparisons/{id}/refresh
**Optimizations:** POST /simulations/{sid}/optimize, GET /optimizations/{id}, GET /optimizations/{id}/iterations, GET /optimizations/{id}/pareto, POST /optimizations/{id}/pause, POST /optimizations/{id}/resume
**Materials:** GET /materials, GET /materials/{id}, POST /materials, GET /materials/fea
**Workflows:** POST /workflows/validate, POST /workflows/execute, GET /workflows/{id}/status

---

## H) Implementation Phases

### Phase 1: Foundation (Weeks 1-3)
- PostgreSQL + Alembic migrations for all entities
- Solver base class + preview solver adapter
- VTK.js viewport component
- golden-layout panel system
- OpenAPI spec + auto-generated TypeScript types

### Phase 2: Core Simulation Loop (Weeks 4-6)
- FEniCS solver integration (thermal + structural)
- Mesh conversion pipeline (gmsh + meshio)
- Run management with Celery async tasks
- WebSocket progress streaming
- Workflow canvas with basic nodes (vue-flow)

### Phase 3: Advanced Solvers + Optimization (Weeks 7-9)
- Elmer solver integration (piezoelectric + acoustic)
- CalculiX solver integration (structural)
- Optimization engine (parametric sweep + Bayesian + genetic)
- Comparison view with standardized metrics
- Assembly simulation support

### Phase 4: Polish + Manual Optimization (Weeks 10-12)
- Rule-guided manual optimization with suggestion engine
- Full Pareto visualization
- Report generation from simulation results
- i18n migration for new UI strings
- Performance optimization and caching

---

## I) Deliverables

1. **Code:** Complete frontend + backend restructure
2. **Configuration:** docker-compose.yml (PostgreSQL + Redis + backend), .env templates
3. **API Contract:** openapi.yaml + websocket-protocol.md
4. **Example Project:** Pre-configured horn simulation with sample results
5. **Report Templates:** PDF/Excel with standardized metrics
6. **Database Migrations:** Alembic migration scripts for all entities
7. **Test Suite:** Unit tests for solvers, integration tests for API, E2E for workflow
8. **Documentation:** Architecture docs, solver integration guide, deployment guide
