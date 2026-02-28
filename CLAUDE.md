# CLAUDE.md — Ultrasonic Weld Master

## Project Overview
Ultrasonic welding parameter auto-adjuster with FEA simulation platform. Python backend (FastAPI) + Vue 3 frontend.

## Tech Stack
- **Backend:** Python 3.12, FastAPI, scipy, numpy, gmsh (optional), cadquery (optional), FEniCSx (Docker)
- **Frontend:** Vue 3 + TypeScript + Vite + Tailwind CSS + ECharts + Three.js
- **Testing:** pytest (backend), vue-tsc (frontend type check)
- **Deploy:** SSH to `squall@180.152.71.166`, PM2 id=104, port 8001

## Commands
```bash
# Backend tests
python3 -m pytest tests/ -v --tb=short

# Frontend type check
cd frontend && npx vue-tsc --noEmit

# Frontend build
cd frontend && npm run build

# Run dev server
python3 run_web.py
```

## Project Structure
```
ultrasonic_weld_master/          # Core calculation engine
  plugins/
    geometry_analyzer/
      fea/                       # FEA solver system
        solver_a.py              # SolverA (numpy/scipy modal, harmonic, stress)
        solver_fenicsx.py        # ContactSolver (FEniCSx via Docker)
        thermal_solver.py        # ThermalSolver (friction heating)
        mesher.py                # Gmsh TET10 meshing with adaptive/knurl modes
        mesh_converter.py        # Gmsh→XDMF format converter for dolfinx
        assembler.py             # K/M matrix assembly with caching
        fatigue.py               # S-N curve fatigue assessment
      horn_generator.py          # Parametric horn + knurl geometry
      anvil_generator.py         # Parametric anvil (flat/groove/knurled/contour)
      knurl_optimizer.py         # Analytical knurl optimization
web/                             # FastAPI web layer
  routers/
    geometry.py                  # FEA endpoints (modal, harmonic, stress, chain)
    knurl_fea.py                 # Knurl FEA (generate, analyze, compare, optimize, export)
    contact.py                   # Contact + thermal analysis endpoints
    assembly.py                  # Multi-body assembly analysis
  services/
    fea_process_runner.py        # Subprocess isolation for heavy FEA
    fenicsx_runner.py            # Docker execution wrapper for FEniCSx
    component_detector.py        # STEP assembly component classification
    step_export_service.py       # STEP file export and download
    knurl_fea_optimizer.py       # Bayesian + FEA knurl optimization
docker/
  scripts/
    contact_solver.py            # FEniCSx contact solver (runs in Docker)
    thermal_solver.py            # FEniCSx thermal solver (runs in Docker)
frontend/src/
  views/
    AnalysisWorkbench.vue        # Main FEA workbench (modal/harmonic/stress/fatigue)
    KnurlWorkbench.vue           # Knurl FEA workbench
    ContactWorkbench.vue         # Contact + thermal workbench
  api/                           # Typed API clients (analysis, contact, knurl-fea, etc.)
  i18n/                          # zh-CN.json and en.json
tests/
  test_fea/                      # 200+ FEA solver tests
  test_web/                      # 140+ API endpoint tests
```

## Key Patterns

### Backend
- **Router pattern:** FastAPI router with Pydantic request/response models in `web/routers/`
- **Service pattern:** Business logic in `web/services/`, imported by routers
- **FEA pipeline:** Gmsh mesh → TET10 assembly → SolverA (scipy eigsh) → post-processing
- **Optional deps:** cadquery/gmsh may not be installed. Always guard imports:
  ```python
  try:
      import cadquery as cq
      HAS_CADQUERY = True
  except ImportError:
      HAS_CADQUERY = False
  ```
- **Subprocess isolation:** Heavy FEA runs use `fea_process_runner.py` with multiprocessing
- **Docker FEA:** FEniCSx-based analysis (contact, thermal) runs inside Docker via `fenicsx_runner.py`
- **App registration:** New routers must be imported in `web/app.py` and registered with `application.include_router()`

### Frontend
- **Component style:** Vue 3 `<script setup lang="ts">` + Composition API
- **Styling:** Tailwind CSS utility classes
- **API clients:** Typed functions in `frontend/src/api/` using shared `apiClient`
- **i18n:** All user-facing text uses `$t('key')`, translations in `zh-CN.json` and `en.json`
- **Routing:** Add routes in `frontend/src/router/index.ts`, nav links in `Sidebar.vue`

### Testing
- **Backend tests:** Use `pytest`, mock optional deps with `unittest.mock`
- **Skip pattern:** `pytest.importorskip("cadquery")` for cadquery-dependent tests
- **API tests:** Use FastAPI `TestClient` from `web/app.py`'s `create_app()`
- **No pytest-asyncio:** Test async endpoints via TestClient, not `@pytest.mark.asyncio`

## Known Issues & Workarounds

### asyncio.to_thread in Tests
`asyncio.to_thread` can deadlock inside FastAPI TestClient (starlette/anyio).
**Fix:** In optimizer/heavy-compute services, call sync functions directly instead of wrapping with `asyncio.to_thread` when the endpoint already runs in a thread pool.

### No pytest-asyncio
pytest-asyncio is not installed. Test async functions via `asyncio.run()` wrappers or through FastAPI TestClient.

### Material Property Key Convention
The material database (`material_properties.py`) uses lowercase keys (`E_pa`, `density_kg_m3`, `yield_mpa`). Some modules may expect `E_Pa` or `yield_MPa`. Always check and normalize key case when bridging material lookups.

### Frontend-Backend Field Name Alignment
Always verify that Vue template field names match the backend JSON response keys exactly. Use the chain worker result packaging in `fea_process_runner.py` as the source of truth for field names. Check with: `grep "chain_results\[" web/services/fea_process_runner.py`

### Gmsh Thread Safety
Gmsh is not thread-safe. Use `gmsh.initialize(interruptible=False)` and ensure single-threaded access.
**Fix:** FEA process runner uses subprocess isolation (multiprocessing) for all Gmsh operations.

### python vs python3
This system uses `python3`, not `python`. Always use `python3` in commands.

### Context Window Management (for Claude sessions)
Previous sessions crashed with "Prompt is too long" at 80% context usage.
**Strategies:**
1. Delegate independent tasks to subagents (Task tool) — they have their own context
2. Don't read large files (>200 lines) in main context; use targeted reads with offset/limit
3. Don't paste full plan documents; extract relevant task text for subagents
4. Commit frequently to preserve progress across sessions
5. Use `/compact` proactively before 60% context usage

## FEA Implementation Plan
See `docs/plans/2026-02-28-advanced-fea-platform-implementation-plan.md` for the full 27-task plan.

**Status: ALL PHASES COMPLETE**
- Phase 1 (Tasks 1-10): Complete — analysis chain, performance, workbench UI
- Phase 2 (Tasks 12-17): Complete — knurl FEA, STEP export, optimization, knurl workbench
- Phase 3 (Tasks 19-27): Complete — FEniCSx contact, thermal, anvil, contact workbench

## P2 Known Limitations (non-blocking)
- Chain harmonic module re-runs modal eigensolve (doesn't reuse prior modal result)
- No FRF/Von Mises/Goodman chart visualizations in workbench result tabs (metric cards only)
- Docker FEniCSx requires server setup (`docker pull dolfinx/dolfinx:stable`)
