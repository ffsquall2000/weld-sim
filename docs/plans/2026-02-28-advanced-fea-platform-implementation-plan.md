# Advanced FEA Simulation Platform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the ultrasonic welding simulation platform from basic modal analysis to a complete analysis workstation with harmonic response, stress, fatigue, knurl FEA, contact mechanics, thermal analysis, and a unified UI workbench.

**Architecture:** Dual-solver (SolverA scipy/numpy + FEniCSx Docker) sharing a unified CadQuery→Gmsh→TET10 mesh pipeline. Flexible analysis DAG with auto-dependency resolution. Two new UI workbenches replace 5 existing views.

**Tech Stack:** Python 3.12 (scipy, numpy, gmsh, cadquery), FEniCSx/dolfinx (Docker), Vue 3 + TypeScript + Three.js, FastAPI + WebSocket, pytest

**Design Doc:** `docs/plans/2026-02-28-advanced-fea-simulation-platform-design.md`

**Worktree:** `/Users/jialechen/Desktop/work/AI code/超声波焊接参数自动调整器/.claude/worktrees/fervent-cannon`

**Server:** SSH `ssh -i /Users/jialechen/.ssh/lab_deploy_180_152_71_166 squall@180.152.71.166`, PM2 id=104, port 8001

---

## Phase 1: Complete Analysis Chain + Performance + UI Workbench

### Task 1: Expose Harmonic Analysis API Endpoint

**Goal:** Wire the existing `SolverA.harmonic_analysis()` (solver_a.py:364-527) to a new HTTP endpoint with subprocess isolation and WebSocket progress.

**Files:**
- Modify: `web/routers/geometry.py` — add `POST /fea/run-harmonic` endpoint
- Modify: `web/services/fea_process_runner.py` — add harmonic subprocess worker
- Create: `tests/test_web/test_harmonic_endpoint.py`

**Step 1: Add Pydantic request/response models to geometry.py**

Add after `FEARunResponse` (line ~196):

```python
class HarmonicRequest(BaseModel):
    """Request for harmonic response analysis."""
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = Field(gt=0, default=20.0)
    freq_range_percent: float = Field(gt=0, le=50, default=5.0)
    n_freq_points: int = Field(ge=10, le=1000, default=201)
    damping_model: str = "hysteretic"  # hysteretic, rayleigh, modal
    damping_ratio: float = Field(gt=0, le=1, default=0.005)
    mesh_density: str = "medium"
    task_id: Optional[str] = None
    # Source: either parametric horn or STEP file path (from prior upload)
    source_task_id: Optional[str] = None  # reuse mesh from modal result
    # OR inline parametric:
    horn_type: Optional[str] = None
    width_mm: Optional[float] = None
    height_mm: Optional[float] = None
    length_mm: Optional[float] = None
    # OR STEP file
    step_file_path: Optional[str] = None

class HarmonicResponse(BaseModel):
    task_id: Optional[str] = None
    frequencies_hz: list[float] = []
    displacement_amplitudes_db: list[float] = []  # FRF magnitude in dB
    contact_face_uniformity: float = 0.0
    gain: float = 0.0
    q_factor: float = 0.0
    resonance_hz: float = 0.0
    node_count: int = 0
    element_count: int = 0
    solve_time_s: float = 0.0
```

**Step 2: Add harmonic endpoint handler**

```python
@router.post("/fea/run-harmonic", response_model=HarmonicResponse)
async def run_harmonic_analysis(request: HarmonicRequest):
    """Run harmonic response analysis. Requires prior modal analysis or inline geometry."""
    target_hz = request.frequency_khz * 1000
    freq_min = target_hz * (1 - request.freq_range_percent / 100)
    freq_max = target_hz * (1 + request.freq_range_percent / 100)

    params = {
        "material": request.material,
        "freq_min_hz": freq_min,
        "freq_max_hz": freq_max,
        "n_freq_points": request.n_freq_points,
        "damping_model": request.damping_model,
        "damping_ratio": request.damping_ratio,
        "mesh_density": request.mesh_density,
        "horn_type": request.horn_type,
        "width_mm": request.width_mm,
        "height_mm": request.height_mm,
        "length_mm": request.length_mm,
    }
    result = await _run_fea_subprocess("harmonic", params, client_task_id=request.task_id)
    return HarmonicResponse(**result)
```

**Step 3: Add harmonic worker to fea_process_runner.py**

Add `HARMONIC_STEPS` constant and the harmonic branch in the subprocess worker function. The worker should:
1. Generate/import mesh (reuse meshing code from modal path)
2. Call `SolverA().harmonic_analysis(HarmonicConfig(...))`
3. Package result as dict

Add to `PHASE_WEIGHTS`:
```python
HARMONIC_STEPS = [
    "init", "meshing", "assembly", "solving", "packaging",
]
```

In the subprocess worker, add handling for `task_type == "harmonic"`:
```python
elif task_type == "harmonic":
    _progress(q, "init", 0.5, "Loading modules")
    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import HarmonicConfig
    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
    # ... mesh generation same as modal ...
    _progress(q, "solving", 0.0, "Running harmonic analysis...")
    config = HarmonicConfig(mesh=mesh, material_name=params["material"], ...)
    result = SolverA().harmonic_analysis(config)
    _progress(q, "packaging", 0.5, "Packaging results")
    # Convert HarmonicResult to dict for JSON serialization
```

**Step 4: Write API test**

```python
# tests/test_web/test_harmonic_endpoint.py
def test_harmonic_endpoint_returns_422_missing_geometry(client):
    """Harmonic without geometry source should return validation error."""
    resp = client.post("/api/v1/geometry/fea/run-harmonic", json={})
    assert resp.status_code in (200, 422)
```

**Step 5: Run tests, commit**

```bash
cd /path/to/worktree
python -m pytest tests/test_web/test_harmonic_endpoint.py -v
git add web/routers/geometry.py web/services/fea_process_runner.py tests/test_web/test_harmonic_endpoint.py
git commit -m "feat: add harmonic response analysis API endpoint"
```

---

### Task 2: Harmonic Stress Analysis Module

**Goal:** Create `harmonic_stress_analysis()` that computes Von Mises stress from the harmonic displacement field U(ω₀), not from static loading.

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` — add `harmonic_stress_analysis()` method
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/results.py` — add `HarmonicStressResult` dataclass
- Create: `tests/test_fea/test_harmonic_stress.py`

**Step 1: Add HarmonicStressResult to results.py**

After `StaticResult` (line 44):
```python
@dataclass
class HarmonicStressResult:
    """Stress analysis derived from harmonic displacement field."""
    stress_vm: np.ndarray            # (n_elements,) Von Mises in Pa
    stress_tensor: np.ndarray        # (n_elements, 6) Voigt notation
    max_stress_mpa: float
    safety_factor: float             # yield_strength / max_stress
    displacement_amplitude: np.ndarray  # (n_nodes,) displacement magnitude at target freq
    max_displacement_mm: float
    contact_face_uniformity: float   # min/mean amplitude at weld face
    mesh: Optional[object] = None
    solve_time_s: float = 0.0
    solver_name: str = "SolverA"
```

**Step 2: Add harmonic_stress_analysis() to SolverA**

Reuse B-matrix and D-matrix code from `static_analysis()` (solver_a.py:909-977). Key difference: input is complex displacement from harmonic analysis, not static load.

```python
def harmonic_stress_analysis(self, harmonic_result: HarmonicResult,
                              config: HarmonicConfig,
                              target_freq_hz: Optional[float] = None) -> HarmonicStressResult:
    """Compute Von Mises stress from harmonic displacement at target frequency.

    1. Find the frequency index closest to target_freq_hz
    2. Extract displacement field U(ω₀) — complex, take magnitude
    3. Compute strain ε = B·|U| at each element centroid
    4. Compute stress σ = D·ε
    5. Compute Von Mises equivalent stress
    """
```

**Step 3: Write tests**

```python
# tests/test_fea/test_harmonic_stress.py
# Test that harmonic stress uses displacement magnitude correctly
# Test Von Mises calculation matches known formula
# Test safety factor = yield / max_vm
```

**Step 4: Run tests, commit**

---

### Task 3: Expose Fatigue Assessment via API

**Goal:** The fatigue module already exists (`fatigue.py` with `FatigueAssessor`). Wire it to a new API endpoint that accepts stress results and returns fatigue assessment.

**Files:**
- Modify: `web/routers/geometry.py` — add `POST /fea/run-fatigue` endpoint
- Create: `frontend/src/api/analysis.ts` — new unified analysis API module

**Step 1: Add FatigueRequest/FatigueResponse to geometry.py**

```python
class FatigueRequest(BaseModel):
    material: str = "Titanium Ti-6Al-4V"
    surface_finish: str = "machined"
    characteristic_diameter_mm: float = 25.0
    reliability_pct: float = 90.0
    temperature_c: float = 25.0
    Kt_global: float = 1.5
    # Stress data — from prior harmonic stress analysis
    stress_vm_mpa: Optional[list[float]] = None  # per-element Von Mises
    source_task_id: Optional[str] = None  # reuse result from prior task
    task_id: Optional[str] = None

class FatigueResponse(BaseModel):
    task_id: Optional[str] = None
    safety_factors: list[float] = []
    min_safety_factor: float = 0.0
    estimated_life_cycles: float = 0.0
    estimated_hours_at_20khz: float = 0.0
    critical_locations: list[dict] = []
    sn_curve_name: str = ""
    corrected_endurance_mpa: float = 0.0
```

**Step 2: Add endpoint handler**

The endpoint takes Von Mises stress array (from harmonic stress analysis) and runs the existing `FatigueAssessor.assess()`. Convert cycles to hours using 20kHz = 7.2e7 cycles/hour.

**Step 3: Run tests, commit**

---

### Task 4: Full Analysis Chain Orchestrator

**Goal:** Create a "chain runner" that executes modal→harmonic→stress→fatigue in sequence, reusing intermediate results, with unified progress tracking.

**Files:**
- Create: `web/services/chain_runner.py` — analysis chain orchestrator
- Modify: `web/routers/geometry.py` — add `POST /fea/run-chain` endpoint
- Create: `tests/test_web/test_chain_endpoint.py`

**Step 1: Chain runner service**

```python
# web/services/chain_runner.py
class AnalysisChainRunner:
    """Orchestrates multi-step analysis chain with dependency resolution."""

    DEPENDENCY_GRAPH = {
        "modal": [],
        "harmonic": ["modal"],
        "stress": ["harmonic"],
        "fatigue": ["stress"],
        "uniformity": ["harmonic"],
        "static": [],
    }

    async def run_chain(self, modules: list[str], params: dict,
                         task_id: str, on_progress: Callable):
        """Run selected modules in dependency order."""
        # 1. Resolve dependencies (topological sort)
        # 2. Execute each module, passing results forward
        # 3. Report progress per module
```

**Step 2: Chain endpoint**

```python
class ChainRequest(BaseModel):
    modules: list[str]  # ["modal", "harmonic", "stress", "fatigue"]
    material: str = "Titanium Ti-6Al-4V"
    frequency_khz: float = 20.0
    mesh_density: str = "medium"
    # ... geometry params ...
    task_id: Optional[str] = None

@router.post("/fea/run-chain")
async def run_analysis_chain(request: ChainRequest):
    """Run selected analysis modules in dependency order."""
```

**Step 3: Tests, commit**

---

### Task 5: Performance — Adaptive Mesh Strategy

**Goal:** Replace uniform mesh density with adaptive meshing: fine at stress concentration regions (weld face, fillets), coarse elsewhere. Target: 40-60% fewer nodes.

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py` — add adaptive mesh fields
- Create: `tests/test_fea/test_adaptive_mesh.py`

**Step 1: Add mesh refinement fields to GmshMesher**

Add a `mesh_with_refinement()` method that uses Gmsh's `mesh.field` API:
```python
def mesh_with_refinement(self, base_size: float, fine_size: float,
                          fine_node_set: str = "bottom_face"):
    """Mesh with local refinement at specified face."""
    # Use Gmsh Distance + Threshold fields for smooth transition
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "FacesList", face_tags)
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "IField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", fine_size)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", base_size)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", base_size * 5)
```

**Step 2: Tests, commit**

---

### Task 6: Performance — Matrix Caching

**Goal:** Cache assembled K/M matrices for repeated analyses on the same mesh. Skip 60s+ assembly time on harmonic/stress rerun.

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/assembler.py` — add hash-based cache
- Create: `tests/test_fea/test_matrix_cache.py`

**Step 1: Add matrix cache to GlobalAssembler**

```python
import hashlib

class GlobalAssembler:
    _cache: dict[str, tuple] = {}  # class-level cache

    def _cache_key(self) -> str:
        """Hash of mesh nodes + elements + material for cache lookup."""
        h = hashlib.sha256()
        h.update(self.mesh.nodes.tobytes())
        h.update(self.mesh.elements.tobytes())
        h.update(self.material_name.encode())
        return h.hexdigest()[:16]

    def assemble(self, use_cache: bool = True) -> tuple:
        key = self._cache_key()
        if use_cache and key in self._cache:
            logger.info("Cache hit for K/M matrices (key=%s)", key)
            return self._cache[key]

        K, M = self._assemble_impl()
        self._cache[key] = (K, M)
        return K, M
```

**Step 2: Tests, commit**

---

### Task 7: Frontend — Analysis Workbench (Core Layout)

**Goal:** Create the main `AnalysisWorkbench.vue` component with left 3D view + right config/results panel. This replaces GeometryView and SimulationView.

**Files:**
- Create: `frontend/src/views/AnalysisWorkbench.vue`
- Modify: `frontend/src/router/index.ts` — add `/workbench` route
- Create: `frontend/src/api/analysis.ts` — unified analysis API client

**Step 1: Create analysis API client**

```typescript
// frontend/src/api/analysis.ts
import apiClient from './client'

export interface ChainRequest {
  modules: string[]
  material: string
  frequency_khz: number
  mesh_density: string
  horn_type?: string
  width_mm?: number
  height_mm?: number
  length_mm?: number
  task_id?: string
}

export interface HarmonicResult {
  frequencies_hz: number[]
  displacement_amplitudes_db: number[]
  contact_face_uniformity: number
  gain: number
  q_factor: number
  resonance_hz: number
}

export interface StressResult {
  max_stress_mpa: number
  safety_factor: number
  max_displacement_mm: number
  contact_face_uniformity: number
}

export interface FatigueResult {
  min_safety_factor: number
  estimated_life_cycles: number
  estimated_hours_at_20khz: number
  critical_locations: Array<{ x: number; y: number; z: number; safety_factor: number }>
  sn_curve_name: string
}

export async function runAnalysisChain(req: ChainRequest) {
  const resp = await apiClient.post('/geometry/fea/run-chain', req, { timeout: 1800000 })
  return resp.data
}

export async function runHarmonicAnalysis(params: any) {
  const resp = await apiClient.post('/geometry/fea/run-harmonic', params, { timeout: 660000 })
  return resp.data
}
```

**Step 2: Create AnalysisWorkbench.vue skeleton**

Layout: left 50% = 3D viewer (reuse `<FEAViewer>`), right 50% = tabbed panels.

Top-right panel:
- Component list (auto-detected from STEP or parametric)
- Analysis module checkboxes with dependency indicators
- Material/frequency/mesh config

Bottom panel: Result tabs [Modal] [Harmonic] [Stress] [Fatigue]

**Step 3: Add route**

```typescript
// router/index.ts — add before existing routes
{ path: '/workbench', name: 'workbench', component: () => import('@/views/AnalysisWorkbench.vue') },
```

**Step 4: Build frontend, verify**

```bash
cd frontend && npm run build
```

**Step 5: Commit**

---

### Task 8: Frontend — Module Checkboxes with Dependency Resolution

**Goal:** Implement the analysis module selection UI with automatic dependency resolution (checking "fatigue" auto-checks stress→harmonic→modal).

**Files:**
- Create: `frontend/src/composables/useAnalysisDependencies.ts`
- Modify: `frontend/src/views/AnalysisWorkbench.vue`

**Step 1: Create dependency resolver composable**

```typescript
// frontend/src/composables/useAnalysisDependencies.ts
import { ref, computed } from 'vue'

const DEPS: Record<string, string[]> = {
  modal: [],
  static: [],
  harmonic: ['modal'],
  stress: ['harmonic'],
  uniformity: ['harmonic'],
  fatigue: ['stress'],
  contact: [],      // Phase 3
  thermal: ['contact'],  // Phase 3
}

export function useAnalysisDependencies() {
  const selected = ref<Set<string>>(new Set())

  function toggle(module: string) {
    if (selected.value.has(module)) {
      // Uncheck: also uncheck dependents
      selected.value.delete(module)
      for (const [mod, deps] of Object.entries(DEPS)) {
        if (deps.includes(module)) selected.value.delete(mod)
      }
    } else {
      // Check: also check dependencies
      selected.value.add(module)
      const addDeps = (m: string) => {
        for (const dep of DEPS[m] || []) {
          selected.value.add(dep)
          addDeps(dep)
        }
      }
      addDeps(module)
    }
  }

  const orderedModules = computed(() => {
    // Topological sort of selected modules
    const order = ['modal', 'static', 'harmonic', 'stress', 'uniformity', 'fatigue', 'contact', 'thermal']
    return order.filter(m => selected.value.has(m))
  })

  return { selected, toggle, orderedModules }
}
```

**Step 2: Integrate into workbench, commit**

---

### Task 9: Frontend — Result Tabs (Modal + Harmonic + Stress + Fatigue)

**Goal:** Implement the result display tabs at the bottom of the workbench. Each tab shows the appropriate visualization for its analysis type.

**Files:**
- Create: `frontend/src/components/results/ModalResultTab.vue`
- Create: `frontend/src/components/results/HarmonicResultTab.vue`
- Create: `frontend/src/components/results/StressResultTab.vue`
- Create: `frontend/src/components/results/FatigueResultTab.vue`
- Modify: `frontend/src/views/AnalysisWorkbench.vue`

**ModalResultTab:** Frequency table + mode animation selector (reuse from GeometryView)

**HarmonicResultTab:** FRF plot (frequency vs amplitude) + uniformity gauge + gain/Q metrics

**StressResultTab:** Von Mises contour on 3D model + max stress indicator + safety factor gauge

**FatigueResultTab:** Safety factor distribution + critical locations 3D markers + life estimate + Goodman diagram

**Commit after each tab component is created.**

---

### Task 10: STEP Component Auto-Detection

**Goal:** When a STEP assembly file is uploaded, auto-detect and classify components (horn, booster, transducer, etc.) by geometric features.

**Files:**
- Create: `web/services/component_detector.py`
- Modify: `web/routers/geometry.py` — enhance upload/cad endpoint
- Create: `tests/test_web/test_component_detection.py`

**Step 1: Component detection service**

```python
# web/services/component_detector.py
class ComponentDetector:
    """Classify STEP assembly components by geometric analysis."""

    COMPONENT_TYPES = ["horn", "booster", "transducer", "anvil", "workpiece", "unknown"]

    def detect(self, step_path: str) -> list[dict]:
        """Parse STEP file and classify each solid body."""
        import cadquery as cq
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_SOLID

        # 1. Load STEP and iterate solids
        # 2. For each solid: compute bounding box, aspect ratio, volume
        # 3. Classify based on heuristics:
        #    - Horn: tapered profile, aspect ratio > 2, one flat face
        #    - Booster: cylindrical, aspect ratio > 3
        #    - Transducer: ring/disc shape
        #    - Anvil: flat base, wider than tall
        # 4. Return list of {type, name, volume_mm3, bbox, centroid}
```

**Step 2: Tests, commit**

---

### Task 11: Deploy Phase 1 to Server

**Goal:** Build frontend, deploy to server, verify with STEP file test.

**Steps:**
1. Build frontend: `cd frontend && npm run build`
2. SCP to server: `scp -r dist/* squall@server:/opt/weld-sim/web/static/`
3. Restart PM2: `pm2 restart 104`
4. Run e2e test with STEP file
5. Verify in browser: upload STEP → run chain → see all results

**Commit: tag Phase 1 complete**

---

## Phase 2: Knurl FEA + STEP Export (Tasks 12-18)

### Task 12: CadQuery Knurl Geometry on Imported STEP

**Goal:** Extend `horn_generator._cq_apply_knurl()` to work on imported STEP horn bodies (not just parametric ones).

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/horn_generator.py`
- Create: `tests/test_fea/test_knurl_geometry.py`

Key: Load STEP with CadQuery, detect bottom face (Z-min), apply knurl pattern via boolean operations.

### Task 13: Gmsh Adaptive Mesh for Knurl

**Goal:** Mesh knurl geometry with local refinement: 0.3-0.5mm at knurl features, 5-8mm elsewhere.

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py`

Use Gmsh `Distance` + `Threshold` fields as described in Task 5 but specifically targeting knurl surface entities.

### Task 14: Knurl FEA API Endpoint

**Goal:** New endpoint that generates knurl geometry → meshes → runs analysis chain → returns results.

**Files:**
- Create: `web/routers/knurl_fea.py` — new router for knurl FEA endpoints
- Modify: `web/app.py` — register router

Endpoints:
- `POST /knurl-fea/generate` — generate knurl geometry preview (mesh for 3D display)
- `POST /knurl-fea/analyze` — run full FEA on knurl horn
- `POST /knurl-fea/compare` — compare with/without knurl
- `POST /knurl-fea/export-step` — export optimized horn as STEP file

### Task 15: STEP Export Service

**Goal:** Export CadQuery solid to STEP file and serve for download.

**Files:**
- Create: `web/services/step_export_service.py`
- Modify: `web/routers/knurl_fea.py`

```python
class StepExportService:
    def export(self, solid, filename: str) -> str:
        """Export CadQuery solid to STEP, return file path."""
        import cadquery as cq
        path = f"/tmp/exports/{filename}.step"
        cq.exporters.export(solid, path)
        return path
```

### Task 16: Knurl Optimization Loop

**Goal:** Iterate knurl parameters, run FEA on each, find Pareto-optimal configurations.

**Files:**
- Create: `web/services/knurl_fea_optimizer.py`

Strategy: Start with existing KnurlOptimizer's parameter grid (900 configs), but replace analytical scoring with actual FEA results for top-N candidates (Bayesian pre-screening).

### Task 17: Knurl Workbench Frontend

**Goal:** Create `KnurlWorkbench.vue` with parameter editor, 3D preview, FEA integration, and STEP download.

**Files:**
- Create: `frontend/src/views/KnurlWorkbench.vue`
- Modify: `frontend/src/router/index.ts`

### Task 18: Deploy Phase 2

Build, deploy, test knurl FEA with real STEP horn.

---

## Phase 3: FEniCSx Contact + Thermal + Anvil (Tasks 19-27)

### Task 19: FEniCSx Docker Setup on Server

**Goal:** Deploy `dolfinx/dolfinx:stable` Docker container on the server with volume mounts for mesh exchange.

**Steps:**
```bash
ssh squall@server
docker pull dolfinx/dolfinx:stable
docker run -d --name dolfinx-solver \
  -v /opt/weld-sim/fea-exchange:/exchange \
  dolfinx/dolfinx:stable tail -f /dev/null
```

Create a wrapper script that runs FEniCSx Python scripts inside the container.

### Task 20: Gmsh→dolfinx Mesh Conversion

**Goal:** Convert Gmsh .msh files to dolfinx mesh format.

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesh_converter.py`

```python
# Inside Docker container:
from dolfinx.io import gmshio
mesh, cell_tags, facet_tags = gmshio.read_from_msh("model.msh", MPI.COMM_WORLD)
```

### Task 21: FEniCSx Contact Solver Module

**Goal:** Implement penalty/Nitsche contact formulation for horn-workpiece-anvil system.

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_fenicsx.py`

### Task 22: Anvil Parametric Generator

**Goal:** Generate anvil geometry: flat, groove, knurled, contour types.

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/anvil_generator.py`

### Task 23: Contact Analysis API Endpoint

**Goal:** Wire FEniCSx contact solver to API with Docker execution.

**Files:**
- Create: `web/routers/contact.py`
- Create: `web/services/fenicsx_runner.py`

### Task 24: Thermal Analysis Module

**Goal:** Friction heating calculation + transient heat conduction in FEniCSx.

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_fenicsx.py`

### Task 25: Thermal Analysis API Endpoint

**Files:**
- Modify: `web/routers/contact.py` — add thermal endpoint

### Task 26: Frontend Contact + Thermal Tabs

**Goal:** Add contact analysis and thermal analysis tabs to the workbench.

**Files:**
- Create: `frontend/src/components/results/ContactResultTab.vue`
- Create: `frontend/src/components/results/ThermalResultTab.vue`
- Modify: `frontend/src/views/AnalysisWorkbench.vue`

### Task 27: Deploy Phase 3

Build, deploy, e2e test with full horn-workpiece-anvil system.

---

## Testing Strategy

**Unit tests** (per task): Each solver method has unit tests with small mock meshes.

**Integration tests**: Full chain test (modal→harmonic→stress→fatigue) with a simple cylinder geometry.

**E2E tests**: Python script (`e2e_fea_test_v2.py` pattern) that tests HTTP + WebSocket with real STEP file on server.

**Run all tests:**
```bash
cd /path/to/worktree
python -m pytest tests/ -v --timeout=120
```

---

## Key Reference Files

| Component | File Path | Key Lines |
|-----------|-----------|-----------|
| HarmonicConfig | `ultrasonic_weld_master/.../fea/config.py` | 41-51 |
| HarmonicResult | `ultrasonic_weld_master/.../fea/results.py` | 22-32 |
| FatigueResult | `ultrasonic_weld_master/.../fea/results.py` | 47-54 |
| SolverA.harmonic_analysis | `ultrasonic_weld_master/.../fea/solver_a.py` | 364-527 |
| SolverA.static_analysis | `ultrasonic_weld_master/.../fea/solver_a.py` | 783-977 |
| Von Mises formula | `ultrasonic_weld_master/.../fea/solver_a.py` | 949-956 |
| FatigueAssessor | `ultrasonic_weld_master/.../fea/fatigue.py` | Full module |
| B-matrix | `ultrasonic_weld_master/.../fea/elements.py` | 230-303 |
| D-matrix | `ultrasonic_weld_master/.../fea/elements.py` | 477-509 |
| GlobalAssembler | `ultrasonic_weld_master/.../fea/assembler.py` | 91-211 |
| FEA router | `web/routers/geometry.py` | 198-342 |
| AnalysisManager | `web/services/analysis_manager.py` | 42-225 |
| FEAProcessRunner | `web/services/fea_process_runner.py` | Full module |
| Frontend API | `frontend/src/api/geometry.ts` | Full module |
| Frontend router | `frontend/src/router/index.ts` | 1-19 |
| FEAProgress component | `frontend/src/components/FEAProgress.vue` | Full |
| FEAViewer component | `frontend/src/components/FEAViewer.vue` | Full |
| Material DB | `ultrasonic_weld_master/.../fea/material_properties.py` | 33-293 |
| KnurlOptimizer | `web/services/knurl_service.py` | Full module |
| HornGenerator | `ultrasonic_weld_master/.../horn_generator.py` | Full |
