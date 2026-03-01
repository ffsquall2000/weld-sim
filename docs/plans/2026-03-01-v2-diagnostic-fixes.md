# V2 Diagnostic Report Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 13 issues (P0-P2) identified in the V2 diagnostic report, restoring full platform usability.

**Architecture:** Fixes are organized by severity. P0 blockers first (GZip, Run execution, FEA geometry), then P1 improvements (preview, thresholds, validation), then P2 polish. Each task is independently deployable.

**Tech Stack:** FastAPI + Starlette middleware, Celery + Redis, SQLAlchemy async, Vue 3 + Vite, Nginx

**Diagnostic Source:** `docs/UltrasonicWeldMaster_V2诊断报告.md`

---

## Investigation Summary (pre-plan findings)

Before writing this plan, the codebase was investigated. Key findings:

| Report Claim | Investigation Result |
|---|---|
| P0 #2: Frontend calls V1 API | **FALSE POSITIVE** - All frontend APIs already use `/api/v2`. Grep found zero v1 references. |
| P0 #1: JS 60s timeout | **CONFIRMED** - No GZipMiddleware in `main.py`; nginx.conf exists but nginx is NOT deployed (bare uvicorn). |
| P0 #3: Run stays queued | **ROOT CAUSE** - Celery task IS dispatched in `run_service.py:35` but no Celery worker process is running. The `except` on line 36-38 silently swallows the error. |
| P0 #4: FEA ignores geometry | **CONFIRMED** - `geometry_analysis.py:150-192` accepts only parametric horn params, no `geometry_id`. |
| P1 #5: Preview 404 | **CONFIRMED** - `geometry_service.py:123` only returns preview for `parametric_params`, not imported STEP files. |
| P1 #6: mesh_file_path null | **PARTIALLY CONFIRMED** - Code at `geometry_service.py:108` sets it, but mesh converter may return None if STEP parsing fails. |
| P1 #7: Pressure threshold | **CONFIRMED** - `quality.py:78-79`: 0.1-0.6 MPa (should be ~3-15 MPa for multi-layer Li-battery tab). |
| P1 #8: Reports for queued runs | **CONFIRMED** - `report_service.py` never checks `run.status` before generating. |

---

## Task 1: Add GZip Middleware (P0 #1 + P2 #12)

**Files:**
- Modify: `backend/app/main.py:86-93`

**Step 1: Add GZipMiddleware after CORS**

In `backend/app/main.py`, add GZipMiddleware right after the CORS middleware block (after line 93):

```python
from starlette.middleware.gzip import GZipMiddleware

# ---- GZip compression ----
app.add_middleware(GZipMiddleware, minimum_size=500)
```

**Step 2: Verify locally**

```bash
cd /opt/weld-sim
source venv/bin/activate
python -c "from starlette.middleware.gzip import GZipMiddleware; print('OK')"
```

Expected: `OK`

**Step 3: Test with curl**

After restarting service:
```bash
curl -s -o /dev/null -w '%{size_download}' -H 'Accept-Encoding: gzip' http://localhost:8001/api/v2/health
```

Expected: Response size significantly smaller than uncompressed.

---

## Task 2: Fix Run Execution - Add Inline Fallback + Celery Worker Service (P0 #3)

**Files:**
- Modify: `backend/app/services/run_service.py:32-38`
- Create: `deploy/weld-sim-worker.service`
- Modify: `deploy/deploy.sh`

**Step 1: Fix the silent exception swallow in run_service.py**

Replace lines 32-38 in `backend/app/services/run_service.py`:

```python
        # Dispatch Celery task
        try:
            from backend.app.tasks.solver_tasks import run_solver_task
            run_solver_task.delay(str(run.id))
        except Exception:
            # If Celery/Redis not available, run inline (dev mode)
            pass
```

With:

```python
        # Dispatch execution
        dispatched = False
        try:
            from backend.app.tasks.solver_tasks import run_solver_task
            run_solver_task.delay(str(run.id))
            dispatched = True
            logger.info("Run %s dispatched to Celery worker", run.id)
        except Exception as exc:
            logger.warning("Celery dispatch failed for run %s: %s", run.id, exc)

        if not dispatched:
            # Fallback: run inline via FastAPI BackgroundTasks
            import asyncio
            from backend.app.tasks.solver_tasks import execute_run_inline

            async def _run_inline(run_id_str: str) -> None:
                try:
                    await execute_run_inline(run_id_str)
                except Exception as e:
                    logger.error("Inline run %s failed: %s", run_id_str, e)

            asyncio.create_task(_run_inline(str(run.id)))
            logger.info("Run %s scheduled for inline execution (no Celery)", run.id)
```

Add at top of file:
```python
import logging
logger = logging.getLogger(__name__)
```

**Step 2: Add `execute_run_inline` to solver_tasks.py**

Add an async wrapper function at the end of `backend/app/tasks/solver_tasks.py` that runs the solver logic using async DB session (instead of Celery's sync context):

```python
async def execute_run_inline(run_id_str: str) -> None:
    """Execute a solver run inline (no Celery) using async DB session."""
    from backend.app.dependencies import async_session_factory
    from backend.app.models.run import Run

    async with async_session_factory() as session:
        run = await session.get(Run, uuid.UUID(run_id_str))
        if not run:
            logger.error("Run %s not found for inline execution", run_id_str)
            return
        run.status = "running"
        await session.commit()

        try:
            # Re-use the existing solver pipeline
            result = _execute_solver(run_id_str)  # calls the sync solver
            run.status = "completed"
            run.computed_time_s = result.get("elapsed_s", 0)
        except Exception as exc:
            run.status = "failed"
            run.error_message = str(exc)[:2000]
            logger.error("Inline run %s failed: %s", run_id_str, exc)

        await session.commit()
```

**Step 3: Create Celery worker systemd service**

Create `deploy/weld-sim-worker.service`:

```ini
[Unit]
Description=UltrasonicWeldMaster Celery Worker
After=network.target redis-server.service

[Service]
Type=simple
User=squall
WorkingDirectory=/opt/weld-sim
ExecStart=/opt/weld-sim/venv/bin/celery -A backend.app.tasks.solver_tasks worker --loglevel=info --concurrency=2
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/weld-sim
Environment=UWM_DATA_DIR=/opt/weld-sim/data

[Install]
WantedBy=multi-user.target
```

**Step 4: Update deploy.sh to install worker service**

Add after the `weld-sim.service` copy line:
```bash
sudo cp deploy/weld-sim-worker.service /etc/systemd/system/
sudo systemctl enable --now weld-sim-worker || true
```

---

## Task 3: FEA Endpoint Accept geometry_id (P0 #4)

**Files:**
- Modify: `backend/app/schemas/geometry_analysis.py:14-41` (FEARunRequest)
- Modify: `backend/app/routers/geometry_analysis.py:150-192`

**Step 1: Add geometry_id to FEARunRequest schema**

In `backend/app/schemas/geometry_analysis.py`, add to FEARunRequest:

```python
class FEARunRequest(BaseModel):
    geometry_id: Optional[str] = Field(
        default=None,
        description="UUID of an uploaded geometry to analyze. If provided, parametric horn params are ignored.",
    )
    horn_type: str = Field(...)  # existing
    # ... rest unchanged
```

Add `from typing import Any, Optional` (already imported).

**Step 2: Update FEA endpoint to load geometry mesh**

In `backend/app/routers/geometry_analysis.py`, modify `run_fea_analysis`:

```python
@router.post("/fea/run", response_model=FEARunResponse)
async def run_fea_analysis(request: FEARunRequest):
    if _fea_service is None:
        raise HTTPException(status_code=503, detail="FEA service not available")

    try:
        if request.geometry_id:
            # Load mesh from uploaded geometry
            from backend.app.dependencies import async_session_factory
            from backend.app.services.geometry_service import GeometryService
            async with async_session_factory() as session:
                svc = GeometryService(session)
                geom = await svc.get(uuid.UUID(request.geometry_id))
                if not geom:
                    raise HTTPException(404, "Geometry not found")
                if geom.mesh_file_path and Path(geom.mesh_file_path).exists():
                    result = _fea_service.run_modal_analysis_from_mesh(
                        mesh_path=geom.mesh_file_path,
                        material=request.material,
                        frequency_khz=request.frequency_khz,
                    )
                else:
                    # Fallback: use dimensions from uploaded geometry
                    dims = geom.metadata_json.get("dimensions", {}) if geom.metadata_json else {}
                    result = _fea_service.run_modal_analysis_gmsh(
                        horn_type=dims.get("horn_type", request.horn_type),
                        diameter_mm=dims.get("width_mm", request.width_mm),
                        length_mm=dims.get("height_mm", request.height_mm),
                        material=request.material,
                        frequency_khz=request.frequency_khz,
                        mesh_density=request.mesh_density,
                    )
        elif request.use_gmsh:
            result = _fea_service.run_modal_analysis_gmsh(...)  # existing code
        else:
            result = _fea_service.run_modal_analysis(...)  # existing code

        return FEARunResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("FEA analysis error: %s", exc)
        raise HTTPException(status_code=500, detail=f"FEA failed: {exc}") from exc
```

Add at top: `import uuid` and `from pathlib import Path`

**Step 3: Add `run_modal_analysis_from_mesh` to FEA service**

In `web/services/fea_service.py`, add a method that accepts a mesh file path instead of generating a mesh from parameters. This method should:
1. Load the .msh file (Gmsh format)
2. Extract nodes and elements
3. Run the same eigenvalue solve
4. Return the same result format

---

## Task 4: Fix Geometry Preview for Imported STEP Files (P1 #5)

**Files:**
- Modify: `backend/app/services/geometry_service.py:117-136`

**Step 1: Update get_preview to support imported geometries**

```python
async def get_preview(self, geometry_id: UUID) -> dict | None:
    geom = await self.get(geometry_id)
    if not geom:
        return None

    # Case 1: Parametric geometry
    if geom.parametric_params:
        from backend.app.domain.horn_generator import HornGenerator, HornParams
        params = HornParams(**geom.parametric_params)
        generator = HornGenerator()
        result = generator.generate(params)
        mesh = result.mesh
        return {
            "vertices": mesh.get("vertices", []),
            "faces": mesh.get("faces", []),
            "scalar_field": None,
            "node_count": len(mesh.get("vertices", [])),
            "element_count": len(mesh.get("faces", [])),
        }

    # Case 2: Imported geometry with mesh
    if geom.mesh_file_path and Path(geom.mesh_file_path).exists():
        from backend.app.solvers.mesh_converter import MeshConverter
        converter = MeshConverter()
        preview = converter.mesh_to_preview(geom.mesh_file_path, max_faces=5000)
        return preview

    # Case 3: Imported STEP file without mesh - generate quick preview
    if geom.file_path and Path(geom.file_path).exists():
        from backend.app.solvers.mesh_converter import MeshConverter, MeshConfig
        converter = MeshConverter()
        preview = converter.step_to_preview(
            geom.file_path,
            config=MeshConfig(element_size=5.0),  # coarse for preview
        )
        return preview

    return None
```

---

## Task 5: Calibrate Quality Assessment Thresholds (P1 #7)

**Files:**
- Modify: `backend/app/routers/quality.py:58-112` (li_battery_tab profile)

**Step 1: Update li_battery_tab thresholds**

Per the diagnostic report's industry reference values for multi-layer Li-battery tab welding:

```python
"li_battery_tab": ApplicationProfile(
    label="Li-Ion Battery Tab Welding",
    criteria=[
        CriterionSpec(
            name="amplitude",
            description="Vibration amplitude",
            param_key="amplitude_um",
            unit="um",
            acceptable_min=15.0,
            acceptable_max=50.0,   # was 40 -> 50
            optimal_min=20.0,
            optimal_max=40.0,      # was 30 -> 40
            weight=1.5,
        ),
        CriterionSpec(
            name="pressure",
            description="Clamping pressure",
            param_key="pressure_mpa",
            unit="MPa",
            acceptable_min=1.0,    # was 0.1 -> 1.0
            acceptable_max=15.0,   # was 0.6 -> 15.0
            optimal_min=3.0,       # was 0.2 -> 3.0
            optimal_max=10.0,      # was 0.4 -> 10.0
            weight=1.0,
        ),
        CriterionSpec(
            name="energy",
            description="Welding energy",
            param_key="energy_j",
            unit="J",
            acceptable_min=50.0,   # was 20 -> 50
            acceptable_max=3000.0, # was 200 -> 3000
            optimal_min=200.0,     # was 50 -> 200
            optimal_max=2000.0,    # was 150 -> 2000
            weight=2.0,
        ),
        CriterionSpec(
            name="time",
            description="Weld duration",
            param_key="time_ms",
            unit="ms",
            acceptable_min=100.0,  # unchanged
            acceptable_max=1500.0, # was 500 -> 1500
            optimal_min=200.0,     # was 150 -> 200
            optimal_max=1000.0,    # was 300 -> 1000
            weight=1.0,
        ),
        # frequency_deviation, temperature_rise, amplitude_uniformity unchanged
    ],
    ...
)
```

Also update the `WeldQualityReport.vue` criteria ranges to match:
- `frontend/src/components/common/WeldQualityReport.vue` lines 294-345

---

## Task 6: Validate Run Status Before Report Generation (P1 #8)

**Files:**
- Modify: `backend/app/services/report_service.py` (generate_report function)

**Step 1: Add status check**

In the `generate_report` function, after fetching each run:

```python
for rid in run_ids:
    run = await _fetch_run_with_relations(session, uuid.UUID(rid))
    if run.status not in ("completed",):
        raise ValueError(
            f"Run {rid} has status '{run.status}'. "
            f"Reports can only be generated for completed runs."
        )
    runs_data.append(_run_to_report_data(run))
```

---

## Task 7: Add application_type Enum (P2 #10)

**Files:**
- Modify: `backend/app/schemas/project.py:12-20`
- Create/Modify: `backend/app/schemas/enums.py` (or inline)

**Step 1: Add Literal type constraint**

```python
from typing import Literal, Optional, List
from pydantic import BaseModel

APPLICATION_TYPES = Literal[
    "li_battery_tab",
    "busbar",
    "collector",
    "general_metal",
    "horn_analysis",
]

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    application_type: APPLICATION_TYPES = "general_metal"
    settings: Optional[dict] = None
    tags: Optional[List[str]] = None
```

**Step 2: Add GET /api/v2/application-types endpoint**

In `backend/app/routers/projects.py`, add:

```python
@router.get("/application-types")
async def list_application_types():
    return [
        {"value": "li_battery_tab", "label": "Li-Ion Battery Tab Welding"},
        {"value": "busbar", "label": "Busbar Welding"},
        {"value": "collector", "label": "Collector Plate Welding"},
        {"value": "general_metal", "label": "General Metal Joining"},
        {"value": "horn_analysis", "label": "Horn Analysis & Design"},
    ]
```

---

## Task 8: Add V1 Compatibility Redirect (P2 #13)

**Files:**
- Modify: `backend/app/main.py`

**Step 1: Add V1 redirect route**

After all v2 router registrations:

```python
from fastapi.responses import RedirectResponse

@app.api_route("/api/v1/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def v1_redirect(path: str, request: Request):
    """Redirect V1 API calls to V2 with 308 Permanent Redirect."""
    query = f"?{request.query_params}" if request.query_params else ""
    return RedirectResponse(
        url=f"/api/v2/{path}{query}",
        status_code=308,
    )

# Also redirect old docs URL
@app.get("/docs")
async def old_docs_redirect():
    return RedirectResponse(url="/api/v2/docs", status_code=301)
```

---

## Task 9: Deploy and Verify All Fixes

**Files:**
- Run: `deploy/deploy.sh`

**Step 1: Build and deploy**

```bash
WELD_SIM_DIR="$(pwd)" bash deploy/deploy.sh
```

**Step 2: Verify each fix**

```bash
# P0 #1: GZip
curl -sI -H 'Accept-Encoding: gzip' http://180.152.71.166:8001/ | grep -i content-encoding

# P0 #3: Run execution (create project -> simulation -> run, check status)
PROJECT=$(curl -s -X POST http://localhost:8001/api/v2/projects -H 'Content-Type: application/json' -d '{"name":"test","application_type":"horn_analysis"}')
echo $PROJECT | python3 -m json.tool

# P1 #7: Quality thresholds
curl -s -X POST http://localhost:8001/api/v2/quality/assess \
  -H 'Content-Type: application/json' \
  -d '{"application_type":"li_battery_tab","parameters":{"amplitude_um":30,"pressure_mpa":5.0,"energy_j":800,"time_ms":500}}' \
  | python3 -m json.tool

# P2 #10: Application types
curl -s http://localhost:8001/api/v2/application-types | python3 -m json.tool

# P2 #13: V1 redirect
curl -sI http://localhost:8001/api/v1/health | head -5
```

---

## Execution Order

| Priority | Task | Est. Time | Dependencies |
|----------|------|-----------|--------------|
| 1 | Task 1: GZip Middleware | 5 min | None |
| 2 | Task 5: Quality Thresholds | 10 min | None |
| 3 | Task 6: Report Validation | 5 min | None |
| 4 | Task 7: Application Type Enum | 10 min | None |
| 5 | Task 8: V1 Redirect | 5 min | None |
| 6 | Task 2: Run Execution | 30 min | None |
| 7 | Task 3: FEA geometry_id | 30 min | None |
| 8 | Task 4: Geometry Preview | 20 min | None |
| 9 | Task 9: Deploy & Verify | 15 min | All above |
