# Phase 2: Core Simulation Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Phase 1 mock endpoints into a working simulation pipeline — real DB persistence, async solver execution via Celery, WebSocket progress streaming, and functional workflow canvas.

**Architecture:** Service layer pattern between routers and models. Routers call services, services use async SQLAlchemy sessions for DB ops and dispatch Celery tasks for solver runs. WebSocket pushes progress via Redis pub/sub. Frontend workflow canvas wired to backend execution.

**Tech Stack:** FastAPI + SQLAlchemy async + Celery + Redis pub/sub + WebSocket + Vue 3 + vue-flow + VTK.js

---

## Task 1: Alembic Initial Migration

**Files:**
- Create: `backend/alembic/versions/001_initial_schema.py`

**Step 1: Generate auto migration**

```bash
cd backend
DATABASE_URL=sqlite+aiosqlite:///./test.db alembic revision --autogenerate -m "initial schema"
```

Note: For development without PostgreSQL, we'll add SQLite fallback support.

**Step 2: Verify migration file contains all tables**

Expected tables: projects, geometry_versions, simulation_cases, runs, artifacts, metrics, comparisons, comparison_results, optimization_studies, materials

**Step 3: Add SQLite compatibility to config**

In `backend/app/config.py`, add SQLite fallback when PostgreSQL unavailable:
```python
@property
def effective_database_url(self) -> str:
    """Use SQLite fallback if PostgreSQL not available."""
    if self.DEBUG:
        return "sqlite+aiosqlite:///./weldsim.db"
    return self.DATABASE_URL
```

**Step 4: Commit**

```bash
git add backend/alembic/ backend/app/config.py
git commit -m "feat: add initial Alembic migration with SQLite fallback"
```

---

## Task 2: Service Layer — Project Service

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/project_service.py`
- Create: `backend/tests/test_services/__init__.py`
- Create: `backend/tests/test_services/test_project_service.py`
- Modify: `backend/app/routers/projects.py`

**Step 1: Write failing tests for project CRUD**

```python
# backend/tests/test_services/test_project_service.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models.base import Base
from app.services.project_service import ProjectService
from app.schemas.project import ProjectCreate

@pytest.fixture
async def db_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
    await engine.dispose()

@pytest.mark.asyncio
async def test_create_project(db_session):
    svc = ProjectService(db_session)
    proj = await svc.create(ProjectCreate(name="Test", application_type="li_battery_tab"))
    assert proj.name == "Test"
    assert proj.id is not None

@pytest.mark.asyncio
async def test_list_projects(db_session):
    svc = ProjectService(db_session)
    await svc.create(ProjectCreate(name="A", application_type="li_battery_tab"))
    await svc.create(ProjectCreate(name="B", application_type="general_metal"))
    items, total = await svc.list_projects(skip=0, limit=10)
    assert total == 2

@pytest.mark.asyncio
async def test_get_project(db_session):
    svc = ProjectService(db_session)
    proj = await svc.create(ProjectCreate(name="Find Me", application_type="li_battery_tab"))
    found = await svc.get(proj.id)
    assert found.name == "Find Me"

@pytest.mark.asyncio
async def test_delete_project(db_session):
    svc = ProjectService(db_session)
    proj = await svc.create(ProjectCreate(name="Delete Me", application_type="li_battery_tab"))
    await svc.delete(proj.id)
    assert await svc.get(proj.id) is None
```

**Step 2: Run tests to verify they fail**

```bash
cd backend && python -m pytest tests/test_services/test_project_service.py -v
```
Expected: FAIL (ProjectService does not exist)

**Step 3: Implement ProjectService**

```python
# backend/app/services/project_service.py
from uuid import UUID
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate

class ProjectService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: ProjectCreate) -> Project:
        project = Project(
            name=data.name,
            description=data.description,
            application_type=data.application_type,
            settings=data.settings or {},
            tags=data.tags or [],
        )
        self.session.add(project)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def get(self, project_id: UUID) -> Project | None:
        return await self.session.get(Project, project_id)

    async def list_projects(self, skip: int = 0, limit: int = 20,
                            application_type: str | None = None,
                            search: str | None = None) -> tuple[list[Project], int]:
        query = select(Project)
        count_query = select(func.count(Project.id))
        if application_type:
            query = query.where(Project.application_type == application_type)
            count_query = count_query.where(Project.application_type == application_type)
        if search:
            query = query.where(Project.name.ilike(f"%{search}%"))
            count_query = count_query.where(Project.name.ilike(f"%{search}%"))
        total = (await self.session.execute(count_query)).scalar() or 0
        query = query.offset(skip).limit(limit).order_by(Project.updated_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all()), total

    async def update(self, project_id: UUID, data: ProjectUpdate) -> Project | None:
        project = await self.get(project_id)
        if not project:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(project, field, value)
        await self.session.commit()
        await self.session.refresh(project)
        return project

    async def delete(self, project_id: UUID) -> bool:
        project = await self.get(project_id)
        if not project:
            return False
        await self.session.delete(project)
        await self.session.commit()
        return True
```

**Step 4: Run tests to verify they pass**

```bash
cd backend && python -m pytest tests/test_services/test_project_service.py -v
```

**Step 5: Wire router to service (replace mock responses)**

Replace mock logic in `backend/app/routers/projects.py` with service calls using dependency injection:
- Inject `AsyncSession` via `get_db_session` dependency
- Create `ProjectService(session)` in each endpoint
- Return real DB results

**Step 6: Commit**

```bash
git add backend/app/services/ backend/tests/test_services/ backend/app/routers/projects.py
git commit -m "feat: add ProjectService with real DB persistence"
```

---

## Task 3: Service Layer — Geometry Service

**Files:**
- Create: `backend/app/services/geometry_service.py`
- Create: `backend/tests/test_services/test_geometry_service.py`
- Modify: `backend/app/routers/geometries.py`

**Step 1: Write tests for geometry CRUD + generation + meshing**

Tests should cover:
- `create_geometry(project_id, data)` → GeometryVersion record
- `generate_parametric(geometry_id)` → calls horn_generator, saves STEP/STL files, updates file_path
- `generate_mesh(geometry_id, config)` → calls mesh_converter, saves mesh file, updates mesh_file_path
- `get_preview(geometry_id)` → returns mesh vertices/faces for VTK.js
- `upload_geometry(project_id, file)` → saves uploaded file, creates GeometryVersion

**Step 2: Implement GeometryService**

Key implementation:
- Uses `domain/horn_generator.py` for parametric generation
- Uses `solvers/mesh_converter.py` for meshing
- Stores files under `storage/geometries/{geometry_id}/`
- Auto-increments version_number per project

**Step 3: Wire router to service**

Replace all mock responses in `geometries.py` router with real service calls.

**Step 4: Commit**

```bash
git commit -m "feat: add GeometryService with parametric generation and meshing"
```

---

## Task 4: Service Layer — Simulation & Run Services

**Files:**
- Create: `backend/app/services/simulation_service.py`
- Create: `backend/app/services/run_service.py`
- Create: `backend/tests/test_services/test_simulation_service.py`
- Create: `backend/tests/test_services/test_run_service.py`
- Modify: `backend/app/routers/simulations.py`
- Modify: `backend/app/routers/runs.py`

**Step 1: Write tests**

SimulationService tests:
- CRUD for simulation cases
- Validate configuration (check solver supports analysis_type, materials exist)

RunService tests:
- `submit_run(simulation_id, geometry_version_id)` → creates Run with status "queued"
- `get_run(run_id)` → returns Run with metrics and artifacts
- `cancel_run(run_id)` → sets status to "cancelled"
- `save_metrics(run_id, metrics_dict)` → creates Metric records
- `save_artifact(run_id, artifact_type, file_path)` → creates Artifact record

**Step 2: Implement services**

SimulationService:
- CRUD operations
- Validation: check solver registered, analysis type supported, materials valid

RunService:
- Submit creates Run record, dispatches Celery task
- Status tracking with timestamps
- Metric and artifact persistence

**Step 3: Wire routers**

Replace mock responses in `simulations.py` and `runs.py`.

**Step 4: Commit**

```bash
git commit -m "feat: add SimulationService and RunService with DB persistence"
```

---

## Task 5: Celery Solver Task

**Files:**
- Create: `backend/app/tasks/__init__.py`
- Create: `backend/app/tasks/solver_tasks.py`
- Create: `backend/tests/test_tasks/__init__.py`
- Create: `backend/tests/test_tasks/test_solver_tasks.py`

**Step 1: Write tests for solver task**

```python
# Test that run_solver_task:
# 1. Loads Run from DB
# 2. Gets solver from registry
# 3. Calls solver.prepare() then solver.run()
# 4. Saves metrics to DB
# 5. Saves result artifacts (VTU file) to storage
# 6. Updates Run status to "completed" or "failed"
# 7. Publishes progress messages to Redis pub/sub
```

**Step 2: Implement solver task**

```python
# backend/app/tasks/solver_tasks.py
from celery import shared_task
from app.dependencies import get_sync_session, redis_client
from app.solvers.registry import get_solver
from app.models.run import Run
from app.models.metric import Metric
from app.models.artifact import Artifact
import json, time

@shared_task(bind=True, max_retries=0)
def run_solver_task(self, run_id: str):
    """Execute solver for a run, stream progress via Redis pub/sub."""
    session = get_sync_session()
    try:
        run = session.get(Run, run_id)
        run.status = "running"
        run.started_at = datetime.utcnow()
        session.commit()

        # Publish progress
        channel = f"run:{run_id}:progress"

        def progress_callback(percent, phase, message):
            redis_client.publish(channel, json.dumps({
                "type": "progress", "run_id": run_id,
                "percent": percent, "phase": phase, "message": message,
                "elapsed_s": (datetime.utcnow() - run.started_at).total_seconds()
            }))

        # Get solver and run
        sim = run.simulation_case
        solver = get_solver(sim.solver_backend)
        config = build_solver_config(sim, run)
        job = solver.prepare_sync(config)
        result = solver.run_sync(job, progress_callback)

        # Save metrics
        for name, value in result.metrics.items():
            session.add(Metric(run_id=run.id, metric_name=name, value=value, unit=guess_unit(name)))

        # Save artifacts
        if result.field_data:
            artifact_path = save_field_data(run.id, result.field_data)
            session.add(Artifact(run_id=run.id, artifact_type="result_vtu", file_path=artifact_path))

        run.status = "completed"
        run.compute_time_s = result.compute_time_s
        run.completed_at = datetime.utcnow()
        run.solver_log = result.solver_log

        # Publish completion
        redis_client.publish(channel, json.dumps({
            "type": "completed", "run_id": run_id, "status": "completed",
            "compute_time_s": result.compute_time_s,
            "metrics_summary": result.metrics
        }))
    except Exception as e:
        run.status = "failed"
        run.error_message = str(e)
        run.completed_at = datetime.utcnow()
        redis_client.publish(channel, json.dumps({
            "type": "error", "run_id": run_id, "error": str(e)
        }))
    finally:
        session.commit()
        session.close()
```

**Step 3: Add sync session factory to dependencies.py**

For Celery tasks (which run in sync context), add a sync session factory.

**Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_tasks/test_solver_tasks.py -v
```

**Step 5: Commit**

```bash
git commit -m "feat: add Celery solver task with Redis progress streaming"
```

---

## Task 6: WebSocket Progress Streaming

**Files:**
- Modify: `backend/app/routers/ws.py`
- Create: `backend/tests/test_routers/test_ws.py`

**Step 1: Write tests for WebSocket endpoint**

Test that connecting to `/ws/runs/{run_id}` receives progress messages published to Redis.

**Step 2: Implement Redis pub/sub WebSocket bridge**

```python
# backend/app/routers/ws.py
import asyncio, json
from fastapi import WebSocket, WebSocketDisconnect
from app.dependencies import get_redis

@router.websocket("/runs/{run_id}")
async def ws_run_progress(websocket: WebSocket, run_id: str):
    await websocket.accept()
    redis = get_redis()
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"run:{run_id}:progress")
    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "message":
                await websocket.send_text(message["data"])
            # Handle client pings
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(f"run:{run_id}:progress")
```

**Step 3: Similarly implement optimization WebSocket**

**Step 4: Commit**

```bash
git commit -m "feat: wire WebSocket to Redis pub/sub for real-time progress"
```

---

## Task 7: FEniCS Solver Integration

**Files:**
- Create: `backend/app/solvers/fenics_solver.py`
- Create: `backend/tests/test_solvers/__init__.py`
- Create: `backend/tests/test_solvers/test_fenics_solver.py`
- Modify: `backend/app/solvers/registry.py`

**Step 1: Write tests with solver abstraction**

Test that FEniCSSolver:
- Reports supported analyses: thermal_steady, thermal_transient, static_structural, coupled_thermo_structural
- prepare() creates XDMF mesh from config
- run() executes FEniCS analysis (with graceful fallback if dolfinx not installed)
- read_results() returns FieldData

**Step 2: Implement FEniCSSolver**

```python
# backend/app/solvers/fenics_solver.py
class FEniCSSolver(SolverBackend):
    """FEniCS/dolfinx solver for thermal and structural analysis."""

    @property
    def name(self) -> str:
        return "fenics"

    @property
    def supported_analyses(self) -> list[AnalysisType]:
        return [
            AnalysisType.thermal_steady,
            AnalysisType.thermal_transient,
            AnalysisType.static_structural,
            AnalysisType.coupled_thermo_structural,
        ]

    def prepare_sync(self, config: SolverConfig) -> PreparedJob:
        # Convert mesh to XDMF format
        # Set up boundary conditions
        # Create work directory
        ...

    def run_sync(self, job: PreparedJob, progress_callback=None) -> SolverResult:
        try:
            import dolfinx
        except ImportError:
            return self._run_fallback(job, progress_callback)
        # Real FEniCS execution
        ...

    def _run_fallback(self, job, progress_callback):
        """Numpy/scipy fallback when dolfinx is not installed."""
        # Use simplified thermal/structural models
        ...
```

**Step 3: Register in init_solvers()**

Add FEniCSSolver to the registry in `registry.py`:
```python
def init_solvers():
    register_solver(PreviewSolver())
    register_solver(FEniCSSolver())
```

**Step 4: Run tests**

```bash
cd backend && python -m pytest tests/test_solvers/test_fenics_solver.py -v
```

**Step 5: Commit**

```bash
git commit -m "feat: add FEniCS solver with numpy/scipy fallback"
```

---

## Task 8: Mesh Conversion Pipeline Enhancement

**Files:**
- Modify: `backend/app/solvers/mesh_converter.py`
- Create: `backend/tests/test_solvers/test_mesh_converter.py`

**Step 1: Write tests for mesh pipeline**

Test the full pipeline:
- `step_to_mesh(step_path, config)` → creates .msh file
- `mesh_to_fenics(msh_path)` → creates XDMF
- `get_mesh_info(mesh_path)` → returns stats
- Fallback when gmsh not available: generate simple hex mesh from dimensions

**Step 2: Add fallback mesh generation**

When gmsh is not installed, generate a structured mesh from parametric horn dimensions using numpy:
```python
def generate_structured_mesh(params: dict, config: MeshConfig) -> MeshResult:
    """Generate structured hex mesh from horn parameters (no gmsh needed)."""
    # Create box/cylinder mesh based on horn_type
    # Return mesh data compatible with solver input
```

**Step 3: Run tests**

```bash
cd backend && python -m pytest tests/test_solvers/test_mesh_converter.py -v
```

**Step 4: Commit**

```bash
git commit -m "feat: enhance mesh converter with structured mesh fallback"
```

---

## Task 9: Material & Comparison Services

**Files:**
- Create: `backend/app/services/material_service.py`
- Create: `backend/app/services/comparison_service.py`
- Modify: `backend/app/routers/materials.py`
- Modify: `backend/app/routers/comparisons.py`

**Step 1: Implement MaterialService**

- `list_materials()` → built-in + DB materials
- `get_material(id)` → by ID
- `create_material(data)` → persist custom material to DB

**Step 2: Implement ComparisonService**

- `create_comparison(project_id, run_ids, metric_names)` → fetch metrics from runs, compute deltas
- `get_comparison(id)` → return with results
- `refresh_comparison(id)` → recompute from current run metrics

**Step 3: Wire routers**

Replace mock responses in `materials.py` and `comparisons.py`.

**Step 4: Commit**

```bash
git commit -m "feat: add MaterialService and ComparisonService"
```

---

## Task 10: Workflow Execution Service

**Files:**
- Create: `backend/app/services/workflow_service.py`
- Modify: `backend/app/routers/workflows.py`

**Step 1: Implement WorkflowService**

```python
class WorkflowService:
    async def validate(self, definition: WorkflowDefinition) -> ValidationResult:
        """Validate DAG: no cycles, required connections, valid node configs."""
        ...

    async def execute(self, definition: WorkflowDefinition, project_id: UUID) -> str:
        """Execute workflow nodes in topological order."""
        # 1. Validate DAG
        # 2. Topological sort
        # 3. For each node in order:
        #    - geometry → call geometry_service.generate_parametric()
        #    - mesh → call geometry_service.generate_mesh()
        #    - material → validate material assignments
        #    - boundary_condition → store BC config
        #    - solver → call run_service.submit_run()
        #    - post_process → extract metrics from run
        # 4. Return workflow execution ID
        ...
```

**Step 2: Wire router**

Replace mock responses in `workflows.py`.

**Step 3: Commit**

```bash
git commit -m "feat: add WorkflowService with DAG validation and execution"
```

---

## Task 11: Frontend — Wire Stores to Real API

**Files:**
- Modify: `frontend/src/stores/project.ts`
- Modify: `frontend/src/stores/geometry.ts`
- Modify: `frontend/src/stores/simulation.ts`
- Modify: `frontend/src/stores/optimization.ts`
- Create: `frontend/src/api/v2.ts`

**Step 1: Create v2 API client**

```typescript
// frontend/src/api/v2.ts
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v2',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' }
})

export default api
```

**Step 2: Update project store**

Ensure all CRUD calls use `/api/v2/projects` and handle real response shapes.

**Step 3: Update geometry store**

Wire `createGeometry`, `generateGeometry`, `generateMesh` to real API.

**Step 4: Update simulation store**

Wire `createSimulation`, `submitRun`, `fetchRunMetrics` to real API.

**Step 5: Commit**

```bash
git commit -m "feat: wire frontend stores to real v2 API endpoints"
```

---

## Task 12: Frontend — WebSocket Integration

**Files:**
- Create: `frontend/src/composables/useWebSocket.ts`
- Modify: `frontend/src/stores/simulation.ts`
- Modify: `frontend/src/components/panels/SolverConsole.vue`

**Step 1: Implement useWebSocket composable**

```typescript
// frontend/src/composables/useWebSocket.ts
export function useWebSocket(url: string) {
  const ws = ref<WebSocket | null>(null)
  const messages = ref<any[]>([])
  const isConnected = ref(false)

  function connect() {
    ws.value = new WebSocket(url)
    ws.value.onopen = () => { isConnected.value = true }
    ws.value.onmessage = (event) => {
      const data = JSON.parse(event.data)
      messages.value.push(data)
    }
    ws.value.onclose = () => { isConnected.value = false }
  }

  function disconnect() { ws.value?.close() }
  function sendPing() { ws.value?.send(JSON.stringify({ type: 'ping' })) }

  return { connect, disconnect, sendPing, messages, isConnected }
}
```

**Step 2: Integrate WebSocket into simulation store**

When a run is submitted:
- Connect to `ws://host/api/v2/ws/runs/{run_id}`
- Update `runProgress`, `runPhase`, `logs` from messages
- On `completed` message, fetch final metrics

**Step 3: Update SolverConsole to show real-time progress**

Wire the console to display WebSocket messages as they arrive.

**Step 4: Commit**

```bash
git commit -m "feat: add WebSocket integration for real-time solver progress"
```

---

## Task 13: Frontend — Workflow Canvas Execution

**Files:**
- Modify: `frontend/src/stores/workflow.ts`
- Modify: `frontend/src/components/workflow/WorkflowCanvas.vue`
- Modify: `frontend/src/components/workflow/WorkflowToolbar.vue`

**Step 1: Wire workflow execution to backend**

Update `executeWorkflow` action in workflow store:
- POST `/api/v2/workflows/execute` with current nodes/edges
- Poll `/api/v2/workflows/{id}/status` for node status updates
- Update node visual status (running → completed/error)

**Step 2: Wire validation to backend**

Update `validateWorkflow` action:
- POST `/api/v2/workflows/validate` with current definition
- Display errors in toolbar

**Step 3: Commit**

```bash
git commit -m "feat: connect workflow canvas to backend execution"
```

---

## Task 14: Frontend — VTK.js Result Loading

**Files:**
- Modify: `frontend/src/composables/useVtkViewer.ts`
- Modify: `frontend/src/stores/viewer3d.ts`

**Step 1: Add API result loading**

When a run completes, fetch field data from `/api/v2/runs/{id}/results/field/{name}` and render in VTK viewport:
- Load mesh geometry (points, cells)
- Load scalar fields (stress, displacement, temperature)
- Auto-select first available field

**Step 2: Wire viewer store to run completion**

Watch for `activeRun` status changes; when `completed`, auto-load results.

**Step 3: Commit**

```bash
git commit -m "feat: auto-load solver results into VTK viewport"
```

---

## Task 15: i18n Updates

**Files:**
- Modify: `frontend/src/i18n/zh-CN.json`
- Modify: `frontend/src/i18n/en.json`

**Step 1: Add translation keys for new features**

Add keys for:
- WebSocket connection status messages
- Solver task progress phases
- Workflow execution status
- Error messages from services

**Step 2: Commit**

```bash
git commit -m "feat: add i18n keys for Phase 2 features"
```

---

## Task 16: Integration Test

**Files:**
- Create: `backend/tests/test_integration/test_simulation_flow.py`

**Step 1: Write end-to-end test**

Test the full simulation flow:
1. Create project
2. Create geometry (parametric)
3. Generate mesh
4. Create simulation case
5. Submit run
6. Verify run completes with metrics
7. Fetch field results

**Step 2: Run full test suite**

```bash
cd backend && python -m pytest -v
```

**Step 3: Commit**

```bash
git commit -m "test: add end-to-end simulation flow integration test"
```

---

## Task 17: Final Review & Cleanup

**Step 1: Verify all mock responses are replaced**

Grep for "TODO" in routers:
```bash
grep -rn "TODO" backend/app/routers/
```
All should be resolved.

**Step 2: Run full test suite**

```bash
cd backend && python -m pytest -v --tb=short
```

**Step 3: Final commit**

```bash
git commit -m "feat: complete Phase 2 core simulation loop"
```
