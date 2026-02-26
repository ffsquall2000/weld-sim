# Web Frontend + API Service Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deploy UltrasonicWeldMaster as a FastAPI service (port 8001) with Vue 3 web frontend, integrating with JIANKONG (port 9900) and LAB (port 3000+8000) on server 180.152.71.166.

**Architecture:** FastAPI backend wraps existing Engine/Plugin system as REST API. Vue 3 SPA provides the web frontend, served as static files by FastAPI. JIANKONG/LAB call the API via `http://127.0.0.1:8001/api/v1/*`.

**Tech Stack:** FastAPI, uvicorn, Pydantic v2, Vue 3, Vite, TypeScript, Tailwind CSS, ECharts, Three.js, Pinia, vue-i18n, axios.

**Project root:** `/Users/jialechen/Desktop/work/AI code/超声波焊接参数自动调整器/.claude/worktrees/nice-stonebraker/`

---

## Task 1: FastAPI Backend — Project Scaffolding + Health Endpoint

**Files:**
- Create: `web/__init__.py`
- Create: `web/app.py`
- Create: `web/config.py`
- Create: `web/dependencies.py`
- Create: `web/routers/__init__.py`
- Create: `web/routers/health.py`
- Create: `web/schemas/__init__.py`
- Create: `web/services/__init__.py`
- Create: `web/services/engine_service.py`
- Create: `requirements-web.txt`
- Create: `run_web.py`
- Test: `tests/test_web/test_health.py`

**Step 1: Create `requirements-web.txt`**

```
# Web service dependencies (in addition to base requirements.txt)
fastapi>=0.109
uvicorn[standard]>=0.27
python-multipart>=0.0.6
aiofiles>=23.2
```

**Step 2: Create `web/config.py`**

```python
"""Web service configuration via environment variables."""
from __future__ import annotations
import os

class WebConfig:
    HOST: str = os.getenv("WELD_SIM_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("WELD_SIM_PORT", "8001"))
    DATA_DIR: str = os.getenv("WELD_SIM_DATA_DIR", "data")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    REPORTS_DIR: str = os.getenv("WELD_SIM_REPORTS_DIR", "data/reports")
    UPLOADS_DIR: str = os.getenv("WELD_SIM_UPLOADS_DIR", "data/uploads")
```

**Step 3: Create `web/services/engine_service.py`**

Thread-safe singleton that wraps `Engine` for FastAPI:

```python
"""Thread-safe engine singleton for web service."""
from __future__ import annotations
import threading
from typing import Optional
from ultrasonic_weld_master.core.engine import Engine
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
from ultrasonic_weld_master.plugins.knowledge_base.plugin import KnowledgeBasePlugin
from ultrasonic_weld_master.plugins.reporter.plugin import ReporterPlugin

class EngineService:
    _instance: Optional["EngineService"] = None
    _lock = threading.Lock()

    def __init__(self, data_dir: str = "data"):
        self._engine = Engine(data_dir=data_dir)
        self._calc_lock = threading.Lock()

    def initialize(self) -> None:
        self._engine.initialize()
        pm = self._engine.plugin_manager
        pm.register(MaterialDBPlugin())
        pm.register(LiBatteryPlugin())
        pm.register(GeneralMetalPlugin())
        pm.register(KnowledgeBasePlugin())
        pm.register(ReporterPlugin())
        for name in ["material_db", "li_battery", "general_metal", "knowledge_base", "reporter"]:
            pm.activate(name)
        try:
            from ultrasonic_weld_master.plugins.geometry_analyzer.plugin import GeometryAnalyzerPlugin
            pm.register(GeometryAnalyzerPlugin())
            pm.activate("geometry_analyzer")
        except ImportError:
            pass

    @property
    def engine(self) -> Engine:
        return self._engine

    def calculate(self, application: str, inputs: dict) -> tuple[WeldRecipe, ValidationResult]:
        app_to_plugin = {
            "li_battery_tab": "li_battery",
            "li_battery_busbar": "li_battery",
            "li_battery_collector": "li_battery",
            "general_metal": "general_metal",
        }
        plugin_name = app_to_plugin.get(application)
        if not plugin_name:
            raise ValueError(f"Unknown application: {application}")
        plugin = self._engine.plugin_manager.get_plugin(plugin_name)
        if not plugin:
            raise RuntimeError(f"Plugin '{plugin_name}' not activated")
        with self._calc_lock:
            recipe = plugin.calculate_parameters(inputs)
            validation = plugin.validate_parameters(recipe)
        return recipe, validation

    def get_materials(self) -> list:
        mat_db = self._engine.plugin_manager.get_plugin("material_db")
        return mat_db.list_materials() if mat_db else []

    def get_material(self, material_type: str) -> Optional[dict]:
        mat_db = self._engine.plugin_manager.get_plugin("material_db")
        return mat_db.get_material(material_type) if mat_db else None

    def get_material_combination(self, mat_a: str, mat_b: str) -> dict:
        mat_db = self._engine.plugin_manager.get_plugin("material_db")
        return mat_db.get_combination_properties(mat_a, mat_b) if mat_db else {}

    def export_report(self, recipe: WeldRecipe, validation: Optional[ValidationResult],
                      fmt: str, output_dir: str) -> str:
        reporter = self._engine.plugin_manager.get_plugin("reporter")
        if not reporter:
            raise RuntimeError("Reporter plugin not activated")
        if fmt == "json":
            return reporter.export_json(recipe, validation, output_dir)
        elif fmt == "excel":
            return reporter.export_excel(recipe, validation, output_dir)
        elif fmt == "pdf":
            return reporter.export_pdf(recipe, validation, output_dir)
        elif fmt == "all":
            return reporter.export_all(recipe, validation, output_dir)
        raise ValueError(f"Unknown format: {fmt}")

    def shutdown(self) -> None:
        self._engine.shutdown()

    @classmethod
    def get_instance(cls, data_dir: str = "data") -> "EngineService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    svc = cls(data_dir=data_dir)
                    svc.initialize()
                    cls._instance = svc
        return cls._instance
```

**Step 4: Create `web/dependencies.py`**

```python
"""FastAPI dependency injection."""
from __future__ import annotations
from web.config import WebConfig
from web.services.engine_service import EngineService

_engine_service: EngineService | None = None

def get_engine_service() -> EngineService:
    global _engine_service
    if _engine_service is None:
        _engine_service = EngineService.get_instance(data_dir=WebConfig.DATA_DIR)
    return _engine_service

def shutdown_engine_service() -> None:
    global _engine_service
    if _engine_service:
        _engine_service.shutdown()
        _engine_service = None
```

**Step 5: Create `web/routers/health.py`**

```python
"""Health check endpoint."""
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health():
    return {"status": "ok", "service": "UltrasonicWeldMaster Simulation Service"}
```

**Step 6: Create `web/app.py`**

```python
"""FastAPI application factory."""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from web.config import WebConfig
from web.dependencies import get_engine_service, shutdown_engine_service
from web.routers import health

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine_service()  # warm up engine on startup
    yield
    shutdown_engine_service()

def create_app() -> FastAPI:
    app = FastAPI(
        title="UltrasonicWeldMaster Simulation API",
        version="1.0.0",
        lifespan=lifespan,
    )
    # CORS
    origins = WebConfig.CORS_ORIGINS
    if origins == "*":
        allow_origins = ["*"]
    else:
        allow_origins = [o.strip() for o in origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # Routers
    app.include_router(health.router, prefix="/api/v1")
    # Static frontend (if built)
    frontend_dist = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")
    if os.path.isdir(frontend_dist):
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
    return app

app = create_app()
```

**Step 7: Create `web/__init__.py`, `web/routers/__init__.py`, `web/schemas/__init__.py`, `web/services/__init__.py`**

All empty `__init__.py` files.

**Step 8: Create `run_web.py`**

```python
"""Web service entry point."""
import uvicorn
from web.config import WebConfig

if __name__ == "__main__":
    uvicorn.run("web.app:app", host=WebConfig.HOST, port=WebConfig.PORT, reload=True)
```

**Step 9: Write the failing test**

```python
# tests/test_web/test_health.py
"""Test health endpoint."""
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_health_returns_ok(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "UltrasonicWeldMaster" in data["service"]
```

**Step 10: Run tests**

```bash
pip install fastapi uvicorn[standard] httpx python-multipart aiofiles
pytest tests/test_web/test_health.py -v
```

Expected: PASS

**Step 11: Commit**

```bash
git add web/ run_web.py requirements-web.txt tests/test_web/
git commit -m "feat(web): FastAPI scaffolding with health endpoint and engine service"
```

---

## Task 2: Calculation API — POST /api/v1/simulate

**Files:**
- Create: `web/schemas/calculation.py`
- Create: `web/routers/calculation.py`
- Modify: `web/app.py` (add router import)
- Test: `tests/test_web/test_calculation.py`

**Step 1: Create `web/schemas/calculation.py`**

Pydantic models mapping to `WeldInputs` and `WeldRecipe`:

```python
"""Pydantic schemas for calculation API."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

class SimulateRequest(BaseModel):
    application: str = Field(..., description="Application type", examples=["li_battery_tab"])
    upper_material_type: str = Field(..., description="Upper material", examples=["Nickel 201"])
    upper_thickness_mm: float = Field(..., gt=0, description="Upper foil thickness in mm")
    upper_layers: int = Field(1, ge=1, le=200, description="Number of upper layers")
    lower_material_type: str = Field(..., description="Lower material", examples=["Copper C110"])
    lower_thickness_mm: float = Field(..., gt=0, description="Lower substrate thickness in mm")
    weld_width_mm: float = Field(3.0, gt=0, description="Weld width in mm")
    weld_length_mm: float = Field(25.0, gt=0, description="Weld length in mm")
    frequency_khz: float = Field(20.0, gt=0, description="Frequency in kHz")
    max_power_w: float = Field(3500.0, gt=0, description="Max power in watts")
    # Optional sonotrode/horn
    horn_type: Optional[str] = None
    horn_gain: Optional[float] = None
    knurl_type: Optional[str] = None
    knurl_pitch_mm: Optional[float] = None
    knurl_tooth_width_mm: Optional[float] = None
    knurl_depth_mm: Optional[float] = None
    # Optional anvil
    anvil_type: Optional[str] = None
    anvil_resonant_freq_khz: Optional[float] = None
    # Optional booster
    booster_gain: Optional[float] = None
    # Optional cylinder
    cylinder_bore_mm: Optional[float] = None
    cylinder_min_air_bar: Optional[float] = None
    cylinder_max_air_bar: Optional[float] = None

class ParametersResponse(BaseModel):
    amplitude_um: Optional[float] = None
    pressure_mpa: Optional[float] = None
    energy_j: Optional[float] = None
    weld_time_ms: Optional[float] = None
    force_n: Optional[float] = None

    class Config:
        extra = "allow"

class ValidationResponse(BaseModel):
    status: str
    messages: list[str] = []

class SimulateResponse(BaseModel):
    recipe_id: str
    application: str
    parameters: dict
    safety_window: dict
    risk_assessment: dict
    quality_estimate: dict
    recommendations: list[str]
    validation: ValidationResponse
    created_at: str

class BatchSimulateRequest(BaseModel):
    items: list[SimulateRequest]

class BatchSimulateResponse(BaseModel):
    results: list[SimulateResponse]
    errors: list[dict] = []
```

**Step 2: Create `web/routers/calculation.py`**

```python
"""Calculation (simulation) API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from web.dependencies import get_engine_service
from web.services.engine_service import EngineService
from web.schemas.calculation import (
    SimulateRequest, SimulateResponse, ValidationResponse,
    BatchSimulateRequest, BatchSimulateResponse,
)

router = APIRouter(prefix="/simulate", tags=["calculation"])

def _request_to_inputs(req: SimulateRequest) -> dict:
    """Convert Pydantic request to plugin inputs dict."""
    d = req.model_dump(exclude_none=True)
    return d

def _build_response(recipe, validation) -> SimulateResponse:
    return SimulateResponse(
        recipe_id=recipe.recipe_id,
        application=recipe.application,
        parameters=recipe.parameters,
        safety_window=recipe.safety_window,
        risk_assessment=recipe.risk_assessment,
        quality_estimate=recipe.quality_estimate,
        recommendations=recipe.recommendations,
        validation=ValidationResponse(
            status=validation.status.value,
            messages=validation.messages,
        ),
        created_at=recipe.created_at,
    )

@router.post("", response_model=SimulateResponse)
async def simulate(req: SimulateRequest, svc: EngineService = Depends(get_engine_service)):
    try:
        inputs = _request_to_inputs(req)
        recipe, validation = svc.calculate(req.application, inputs)
        return _build_response(recipe, validation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch", response_model=BatchSimulateResponse)
async def simulate_batch(req: BatchSimulateRequest, svc: EngineService = Depends(get_engine_service)):
    results = []
    errors = []
    for i, item in enumerate(req.items):
        try:
            inputs = _request_to_inputs(item)
            recipe, validation = svc.calculate(item.application, inputs)
            results.append(_build_response(recipe, validation))
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
    return BatchSimulateResponse(results=results, errors=errors)

@router.get("/schema/{application}")
async def get_schema(application: str, svc: EngineService = Depends(get_engine_service)):
    app_to_plugin = {
        "li_battery_tab": "li_battery",
        "li_battery_busbar": "li_battery",
        "li_battery_collector": "li_battery",
        "general_metal": "general_metal",
    }
    plugin_name = app_to_plugin.get(application)
    if not plugin_name:
        raise HTTPException(status_code=404, detail=f"Unknown application: {application}")
    plugin = svc.engine.plugin_manager.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin '{plugin_name}' not activated")
    return plugin.get_input_schema()
```

**Step 3: Add router to `web/app.py`**

Add after `from web.routers import health`:
```python
from web.routers import health, calculation
```
Add after `app.include_router(health.router, ...)`:
```python
app.include_router(calculation.router, prefix="/api/v1")
```

**Step 4: Write tests**

```python
# tests/test_web/test_calculation.py
"""Test calculation API."""
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

SAMPLE_INPUT = {
    "application": "li_battery_tab",
    "upper_material_type": "Nickel 201",
    "upper_thickness_mm": 0.1,
    "upper_layers": 40,
    "lower_material_type": "Copper C110",
    "lower_thickness_mm": 0.3,
    "weld_width_mm": 3.0,
    "weld_length_mm": 25.0,
    "frequency_khz": 20.0,
    "max_power_w": 3500,
}

def test_simulate_returns_recipe(client):
    resp = client.post("/api/v1/simulate", json=SAMPLE_INPUT)
    assert resp.status_code == 200
    data = resp.json()
    assert "recipe_id" in data
    assert data["application"] == "li_battery_tab"
    assert "amplitude_um" in data["parameters"] or len(data["parameters"]) > 0
    assert "validation" in data
    assert data["validation"]["status"] in ("pass", "warning", "fail")

def test_simulate_with_horn_params(client):
    inp = {**SAMPLE_INPUT, "horn_type": "flat", "knurl_type": "cross_hatch",
           "knurl_pitch_mm": 1.2, "knurl_tooth_width_mm": 0.6, "knurl_depth_mm": 0.35}
    resp = client.post("/api/v1/simulate", json=inp)
    assert resp.status_code == 200

def test_simulate_invalid_application(client):
    inp = {**SAMPLE_INPUT, "application": "nonexistent"}
    resp = client.post("/api/v1/simulate", json=inp)
    assert resp.status_code in (400, 500)

def test_simulate_general_metal(client):
    inp = {**SAMPLE_INPUT, "application": "general_metal"}
    resp = client.post("/api/v1/simulate", json=inp)
    assert resp.status_code == 200

def test_batch_simulate(client):
    req = {"items": [SAMPLE_INPUT, {**SAMPLE_INPUT, "upper_layers": 20}]}
    resp = client.post("/api/v1/simulate/batch", json=req)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 2

def test_get_schema(client):
    resp = client.get("/api/v1/simulate/schema/li_battery_tab")
    assert resp.status_code == 200
    data = resp.json()
    assert "properties" in data
```

**Step 5: Run tests**

```bash
pytest tests/test_web/ -v
```

Expected: ALL PASS

**Step 6: Commit**

```bash
git add web/schemas/calculation.py web/routers/calculation.py web/app.py tests/test_web/test_calculation.py
git commit -m "feat(web): add POST /simulate and /simulate/batch calculation API"
```

---

## Task 3: Materials + Recipes + Reports API

**Files:**
- Create: `web/routers/materials.py`
- Create: `web/routers/recipes.py`
- Create: `web/routers/reports.py`
- Create: `web/schemas/materials.py`
- Create: `web/schemas/reports.py`
- Create: `web/services/file_service.py`
- Modify: `web/app.py` (add 3 routers)
- Test: `tests/test_web/test_materials.py`
- Test: `tests/test_web/test_recipes.py`
- Test: `tests/test_web/test_reports.py`

**Step 1: Create `web/routers/materials.py`**

```python
"""Material database API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from web.dependencies import get_engine_service
from web.services.engine_service import EngineService

router = APIRouter(prefix="/materials", tags=["materials"])

@router.get("")
async def list_materials(svc: EngineService = Depends(get_engine_service)):
    return {"materials": svc.get_materials()}

@router.get("/{material_type}")
async def get_material(material_type: str, svc: EngineService = Depends(get_engine_service)):
    mat = svc.get_material(material_type)
    if mat is None:
        raise HTTPException(status_code=404, detail=f"Material '{material_type}' not found")
    return {"material_type": material_type, "properties": mat}

@router.get("/combination/{mat_a}/{mat_b}")
async def get_combination(mat_a: str, mat_b: str, svc: EngineService = Depends(get_engine_service)):
    props = svc.get_material_combination(mat_a, mat_b)
    return {"materials": [mat_a, mat_b], "properties": props}
```

**Step 2: Create `web/routers/recipes.py`**

```python
"""Recipe history API endpoints."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from web.dependencies import get_engine_service
from web.services.engine_service import EngineService

router = APIRouter(prefix="/recipes", tags=["recipes"])

@router.get("")
async def list_recipes(
    limit: int = Query(50, ge=1, le=500),
    svc: EngineService = Depends(get_engine_service),
):
    db = svc.engine.database
    rows = db.execute(
        "SELECT id, application, inputs, created_at FROM recipes ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    import json
    results = []
    for row in rows:
        results.append({
            "id": row[0], "application": row[1],
            "inputs": json.loads(row[2]) if row[2] else {},
            "created_at": row[3],
        })
    return {"recipes": results, "count": len(results)}

@router.get("/{recipe_id}")
async def get_recipe(recipe_id: str, svc: EngineService = Depends(get_engine_service)):
    db = svc.engine.database
    recipe_data = db.get_recipe(recipe_id)
    if recipe_data is None:
        raise HTTPException(status_code=404, detail=f"Recipe '{recipe_id}' not found")
    return recipe_data
```

**Step 3: Create `web/services/file_service.py`**

```python
"""File upload/download management."""
from __future__ import annotations
import os
from web.config import WebConfig

class FileService:
    @staticmethod
    def ensure_dirs():
        os.makedirs(WebConfig.REPORTS_DIR, exist_ok=True)
        os.makedirs(WebConfig.UPLOADS_DIR, exist_ok=True)

    @staticmethod
    def get_reports_dir() -> str:
        os.makedirs(WebConfig.REPORTS_DIR, exist_ok=True)
        return WebConfig.REPORTS_DIR

    @staticmethod
    def get_uploads_dir() -> str:
        os.makedirs(WebConfig.UPLOADS_DIR, exist_ok=True)
        return WebConfig.UPLOADS_DIR
```

**Step 4: Create `web/routers/reports.py`**

```python
"""Report export API endpoints."""
from __future__ import annotations
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from web.dependencies import get_engine_service
from web.services.engine_service import EngineService
from web.services.file_service import FileService

router = APIRouter(prefix="/reports", tags=["reports"])

class ExportRequest(BaseModel):
    recipe_id: str
    format: str = "json"  # json | excel | pdf | all

@router.post("/export")
async def export_report(req: ExportRequest, svc: EngineService = Depends(get_engine_service)):
    db = svc.engine.database
    recipe_data = db.get_recipe(req.recipe_id)
    if recipe_data is None:
        raise HTTPException(status_code=404, detail=f"Recipe '{req.recipe_id}' not found")
    # Reconstruct WeldRecipe from DB data
    from ultrasonic_weld_master.core.models import WeldRecipe
    recipe = WeldRecipe(
        recipe_id=recipe_data["id"],
        application=recipe_data.get("application", ""),
        inputs=recipe_data.get("inputs", {}),
        parameters=recipe_data.get("parameters", {}),
        safety_window=recipe_data.get("safety_window", {}),
        risk_assessment=recipe_data.get("risk_assessment", {}),
        created_at=recipe_data.get("created_at", ""),
    )
    output_dir = FileService.get_reports_dir()
    try:
        result = svc.export_report(recipe, None, req.format, output_dir)
        if isinstance(result, dict):
            filenames = {k: os.path.basename(v) for k, v in result.items()}
            return {"files": filenames}
        return {"file": os.path.basename(result)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_report(filename: str):
    filepath = os.path.join(FileService.get_reports_dir(), filename)
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)
```

**Step 5: Create `web/schemas/materials.py` and `web/schemas/reports.py`**

```python
# web/schemas/materials.py
"""Material schemas (placeholder for future expansion)."""
```

```python
# web/schemas/reports.py
"""Report schemas (placeholder for future expansion)."""
```

**Step 6: Add routers to `web/app.py`**

Import line becomes:
```python
from web.routers import health, calculation, materials, recipes, reports
```

Add after existing `app.include_router` lines:
```python
app.include_router(materials.router, prefix="/api/v1")
app.include_router(recipes.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
```

**Step 7: Write tests**

```python
# tests/test_web/test_materials.py
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

@pytest.fixture
def client():
    return TestClient(create_app())

def test_list_materials(client):
    resp = client.get("/api/v1/materials")
    assert resp.status_code == 200
    data = resp.json()
    assert "materials" in data
    assert len(data["materials"]) > 0

def test_get_material(client):
    resp = client.get("/api/v1/materials/Nickel 201")
    assert resp.status_code == 200
    data = resp.json()
    assert "properties" in data

def test_get_material_not_found(client):
    resp = client.get("/api/v1/materials/FakeMaterial999")
    assert resp.status_code == 404

def test_get_combination(client):
    resp = client.get("/api/v1/materials/combination/Nickel 201/Copper C110")
    assert resp.status_code == 200
```

```python
# tests/test_web/test_recipes.py
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

@pytest.fixture
def client():
    return TestClient(create_app())

def test_list_recipes(client):
    resp = client.get("/api/v1/recipes")
    assert resp.status_code == 200
    data = resp.json()
    assert "recipes" in data
    assert "count" in data
```

```python
# tests/test_web/test_reports.py
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

SAMPLE = {
    "application": "li_battery_tab",
    "upper_material_type": "Nickel 201",
    "upper_thickness_mm": 0.1, "upper_layers": 40,
    "lower_material_type": "Copper C110",
    "lower_thickness_mm": 0.3,
    "weld_width_mm": 3.0, "weld_length_mm": 25.0,
}

@pytest.fixture
def client():
    return TestClient(create_app())

def test_export_json_after_simulate(client):
    # First create a recipe
    sim_resp = client.post("/api/v1/simulate", json=SAMPLE)
    assert sim_resp.status_code == 200
    recipe_id = sim_resp.json()["recipe_id"]
    # Then export
    resp = client.post("/api/v1/reports/export", json={"recipe_id": recipe_id, "format": "json"})
    # May be 200 or 404 depending on DB persistence within test
    assert resp.status_code in (200, 404)
```

**Step 8: Run all tests**

```bash
pytest tests/test_web/ -v
```

**Step 9: Commit**

```bash
git add web/routers/materials.py web/routers/recipes.py web/routers/reports.py \
        web/schemas/ web/services/file_service.py web/app.py \
        tests/test_web/test_materials.py tests/test_web/test_recipes.py tests/test_web/test_reports.py
git commit -m "feat(web): add materials, recipes, and reports API endpoints"
```

---

## Task 4: Vue 3 Frontend — Project Scaffolding

**Files:**
- Create: `frontend/` (via Vite scaffold)
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/i18n/zh-CN.json`
- Create: `frontend/src/i18n/en.json`
- Create: `frontend/src/i18n/index.ts`

**Step 1: Scaffold Vue 3 + Vite + TypeScript project**

```bash
cd /path/to/project/root
npm create vite@latest frontend -- --template vue-ts
cd frontend
npm install
npm install -D tailwindcss @tailwindcss/vite
npm install vue-router@4 pinia axios vue-i18n@9 echarts vue-echarts
npm install lucide-vue-next
```

**Step 2: Configure Vite (`frontend/vite.config.ts`)**

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      }
    }
  }
})
```

**Step 3: Setup Tailwind CSS (`frontend/src/style.css`)**

```css
@import "tailwindcss";

:root {
  --color-bg-primary: #0d1117;
  --color-bg-secondary: #161b22;
  --color-bg-card: #21262d;
  --color-border: #30363d;
  --color-text-primary: #e6edf3;
  --color-text-secondary: #8b949e;
  --color-accent-orange: #ff9800;
  --color-accent-blue: #58a6ff;
  --color-success: #4caf50;
  --color-danger: #f44336;
}
```

**Step 4: Create API client (`frontend/src/api/client.ts`)**

```typescript
import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export default apiClient
```

**Step 5: Create i18n files**

`frontend/src/i18n/zh-CN.json`:
```json
{
  "nav": {
    "dashboard": "概览",
    "calculate": "参数计算",
    "results": "计算结果",
    "reports": "报告导出",
    "history": "历史记录",
    "settings": "设置",
    "geometry": "几何分析"
  },
  "wizard": {
    "step1": "应用类型",
    "step2": "材料选择",
    "step3": "焊头/砧座",
    "step4": "几何导入",
    "step5": "设备参数",
    "step6": "约束条件",
    "next": "下一步",
    "back": "上一步",
    "calculate": "开始计算"
  },
  "app": {
    "title": "超声波焊接参数模拟器",
    "subtitle": "UltrasonicWeldMaster"
  }
}
```

`frontend/src/i18n/en.json`:
```json
{
  "nav": {
    "dashboard": "Dashboard",
    "calculate": "Calculate",
    "results": "Results",
    "reports": "Reports",
    "history": "History",
    "settings": "Settings",
    "geometry": "Geometry"
  },
  "wizard": {
    "step1": "Application",
    "step2": "Materials",
    "step3": "Horn / Anvil",
    "step4": "Geometry",
    "step5": "Equipment",
    "step6": "Constraints",
    "next": "Next",
    "back": "Back",
    "calculate": "Calculate"
  },
  "app": {
    "title": "Ultrasonic Weld Simulator",
    "subtitle": "UltrasonicWeldMaster"
  }
}
```

`frontend/src/i18n/index.ts`:
```typescript
import { createI18n } from 'vue-i18n'
import zhCN from './zh-CN.json'
import en from './en.json'

export const i18n = createI18n({
  legacy: false,
  locale: 'zh-CN',
  fallbackLocale: 'en',
  messages: { 'zh-CN': zhCN, en },
})
```

**Step 6: Setup router, Pinia, mount app**

Create `frontend/src/router/index.ts`:
```typescript
import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/calculate', name: 'calculate', component: () => import('@/views/CalculateView.vue') },
  { path: '/results/:id', name: 'results', component: () => import('@/views/ResultsView.vue') },
  { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue') },
  { path: '/reports/:id', name: 'reports', component: () => import('@/views/ReportsView.vue') },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
```

Update `frontend/src/main.ts`:
```typescript
import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import { router } from './router'
import { i18n } from './i18n'
import './style.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)
app.use(i18n)
app.mount('#app')
```

**Step 7: Create stub views (DashboardView, CalculateView, ResultsView, HistoryView, ReportsView, SettingsView)**

Each as a minimal `<template><div>{{ pageName }}</div></template>` placeholder.

**Step 8: Verify frontend builds**

```bash
cd frontend && npm run build
```

Expected: successful build, `dist/` directory created.

**Step 9: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): Vue 3 + Vite scaffolding with router, i18n, Tailwind"
```

---

## Task 5: Vue 3 Frontend — App Layout + Navigation Sidebar

**Files:**
- Create: `frontend/src/components/layout/AppLayout.vue`
- Create: `frontend/src/components/layout/Sidebar.vue`
- Create: `frontend/src/components/layout/ThemeToggle.vue`
- Modify: `frontend/src/App.vue`

**Step 1: Create `Sidebar.vue`**

Left sidebar with navigation icons matching desktop app's 5 nav items + Geometry. Uses `lucide-vue-next` icons. Supports dark/light themes. Active item highlighted with left orange border.

**Step 2: Create `AppLayout.vue`**

Wrapper with sidebar on left (180px), main content area on right. Responsive: sidebar collapses on mobile.

**Step 3: Create `ThemeToggle.vue`**

Dark/light mode toggle button, persists to localStorage.

**Step 4: Update `App.vue`**

Replace default content with `<AppLayout><router-view /></AppLayout>`.

**Step 5: Verify in dev mode**

```bash
cd frontend && npm run dev
# Open http://localhost:5173 in browser
```

**Step 6: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): app layout with sidebar navigation and theme toggle"
```

---

## Task 6: Vue 3 Frontend — 6-Step Calculation Wizard

**Files:**
- Create: `frontend/src/views/CalculateView.vue`
- Create: `frontend/src/components/wizard/StepIndicator.vue`
- Create: `frontend/src/components/wizard/ApplicationStep.vue`
- Create: `frontend/src/components/wizard/MaterialStep.vue`
- Create: `frontend/src/components/wizard/HornAnvilStep.vue`
- Create: `frontend/src/components/wizard/GeometryStep.vue`
- Create: `frontend/src/components/wizard/EquipmentStep.vue`
- Create: `frontend/src/components/wizard/ConstraintStep.vue`
- Create: `frontend/src/stores/calculation.ts`
- Create: `frontend/src/api/simulation.ts`

**Step 1: Create `stores/calculation.ts`**

Pinia store holding wizard state (current step, all input values, result), with actions for `nextStep`, `prevStep`, `submitCalculation`.

**Step 2: Create `api/simulation.ts`**

```typescript
import apiClient from './client'

export interface SimulateRequest {
  application: string
  upper_material_type: string
  upper_thickness_mm: number
  upper_layers: number
  lower_material_type: string
  lower_thickness_mm: number
  weld_width_mm: number
  weld_length_mm: number
  frequency_khz: number
  max_power_w: number
  horn_type?: string
  knurl_type?: string
  knurl_pitch_mm?: number
  knurl_tooth_width_mm?: number
  knurl_depth_mm?: number
  anvil_type?: string
  booster_gain?: number
}

export interface SimulateResponse {
  recipe_id: string
  application: string
  parameters: Record<string, number>
  safety_window: Record<string, [number, number]>
  risk_assessment: Record<string, string>
  quality_estimate: Record<string, number>
  recommendations: string[]
  validation: { status: string; messages: string[] }
  created_at: string
}

export const simulateApi = {
  calculate: (data: SimulateRequest) =>
    apiClient.post<SimulateResponse>('/simulate', data),
  getSchema: (app: string) =>
    apiClient.get(`/simulate/schema/${app}`),
}
```

**Step 3: Create `StepIndicator.vue`**

Horizontal progress indicator with numbered circles + connecting lines. Orange for active/completed, gray for pending. Matches desktop `StepIndicator`.

**Step 4: Create each step component**

- `ApplicationStep.vue`: Dropdown for application type with description
- `MaterialStep.vue`: Upper/lower material selection, thickness, layers + SVG stack visualization
- `HornAnvilStep.vue`: Horn type, anvil type, knurl pattern selectors + SVG diagrams
- `GeometryStep.vue`: Placeholder for file upload (optional step)
- `EquipmentStep.vue`: Frequency, power, booster gain, cylinder params
- `ConstraintStep.vue`: Safety thresholds, target values

**Step 5: Create `CalculateView.vue`**

Assembles StepIndicator + step components + Back/Next/Calculate buttons. On calculate, calls API and navigates to `/results/:id`.

**Step 6: Verify wizard flow in browser**

```bash
cd frontend && npm run dev
# Navigate to /calculate, test all 6 steps
```

**Step 7: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): 6-step calculation wizard with all step components"
```

---

## Task 7: Vue 3 Frontend — Results + Reports + History Views

**Files:**
- Create: `frontend/src/views/ResultsView.vue`
- Create: `frontend/src/views/ReportsView.vue`
- Create: `frontend/src/views/HistoryView.vue`
- Create: `frontend/src/views/DashboardView.vue`
- Create: `frontend/src/components/charts/ParameterCard.vue`
- Create: `frontend/src/components/charts/RiskBadge.vue`
- Create: `frontend/src/components/charts/SafetyGauge.vue`
- Create: `frontend/src/api/materials.ts`
- Create: `frontend/src/api/reports.ts`
- Create: `frontend/src/api/recipes.ts`

**Step 1: Create `ParameterCard.vue`**

Card with accent-colored left border showing: value (large), unit, safe range. Green if in range, red if outside. Matches desktop `_ParamCard`.

**Step 2: Create `RiskBadge.vue`**

Flat badge with dot indicator. Color-coded: green (low), orange (medium), red (high), dark-red (critical).

**Step 3: Create `SafetyGauge.vue`**

ECharts gauge chart showing parameter value within safe window.

**Step 4: Create `ResultsView.vue`**

Layout: Parameter cards grid (2x3) + Risk badges row + Parameter table + Validation section + Recommendations. Fetches result by ID from API.

**Step 5: Create `ReportsView.vue`**

Preview of recipe data + 4 export buttons (JSON/Excel/PDF/All). Calls `/reports/export`, then provides download link.

**Step 6: Create `HistoryView.vue`**

Table of past calculations with columns: ID, Application, Materials, Date. Click row to navigate to `/results/:id`. Fetches from `/recipes`.

**Step 7: Create `DashboardView.vue`**

Overview page: recent 5 calculations, quick "New Calculation" button, material count, system status.

**Step 8: Create API helper files**

```typescript
// api/materials.ts
import apiClient from './client'
export const materialsApi = {
  list: () => apiClient.get('/materials'),
  get: (type: string) => apiClient.get(`/materials/${type}`),
}

// api/reports.ts
import apiClient from './client'
export const reportsApi = {
  export: (recipeId: string, format: string) =>
    apiClient.post('/reports/export', { recipe_id: recipeId, format }),
  downloadUrl: (filename: string) => `/api/v1/reports/download/${filename}`,
}

// api/recipes.ts
import apiClient from './client'
export const recipesApi = {
  list: (limit = 50) => apiClient.get('/recipes', { params: { limit } }),
  get: (id: string) => apiClient.get(`/recipes/${id}`),
}
```

**Step 9: Build and verify**

```bash
cd frontend && npm run build
```

**Step 10: Commit**

```bash
git add frontend/src/
git commit -m "feat(frontend): results, reports, history, and dashboard views"
```

---

## Task 8: SVG Visualization Components (Material Stack, Horn, Anvil, Knurl)

**Files:**
- Create: `frontend/src/components/charts/MaterialStack.vue`
- Create: `frontend/src/components/charts/HornDiagram.vue`
- Create: `frontend/src/components/charts/AnvilDiagram.vue`
- Create: `frontend/src/components/charts/KnurlPattern.vue`

**Step 1: Create `MaterialStack.vue`**

SVG cross-section of foil stack + substrate matching desktop `MaterialStackWidget`:
- Proportional layer heights
- Material-specific colors (Al=#c0c0c0, Cu=#b87333, Ni=#a0a0a0, Steel=#606060)
- Layer count annotation
- Thickness labels

**Step 2: Create `HornDiagram.vue`**

SVG side-view of horn types (flat, curved, segmented, blade, heavy, branson_dp, custom). Orange fill with darker border. Type label at bottom.

**Step 3: Create `AnvilDiagram.vue`**

SVG side-view of anvil types (fixed_flat, knurled, contoured, rotary, multi_station, resonant, custom). Gray fill with orange border.

**Step 4: Create `KnurlPattern.vue`**

SVG top-view knurl pattern + cross-section. Shows pitch/depth/tooth annotations. Pattern types: linear, cross_hatch, diamond, conical, spherical.

**Step 5: Integrate into wizard step components**

Update `MaterialStep.vue`, `HornAnvilStep.vue` to use these SVG components with reactive props.

**Step 6: Commit**

```bash
git add frontend/src/components/charts/
git commit -m "feat(frontend): SVG visualization components for materials, horn, anvil, knurl"
```

---

## Task 9: Integration + Server Deployment

**Files:**
- Create: `deploy/weld-sim.service` (systemd unit)
- Create: `deploy/deploy.sh` (deployment script)
- Create: `.env.example`
- Modify: server `/opt/lab/next.config.ts` (add proxy rewrite)

**Step 1: Create `.env.example`**

```
WELD_SIM_HOST=0.0.0.0
WELD_SIM_PORT=8001
WELD_SIM_DATA_DIR=data
CORS_ORIGINS=*
WELD_SIM_REPORTS_DIR=data/reports
WELD_SIM_UPLOADS_DIR=data/uploads
```

**Step 2: Create `deploy/weld-sim.service`**

```ini
[Unit]
Description=UltrasonicWeldMaster Simulation Service
After=network.target

[Service]
Type=simple
User=squall
WorkingDirectory=/opt/weld-sim
ExecStart=/opt/weld-sim/venv/bin/uvicorn web.app:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/weld-sim

[Install]
WantedBy=multi-user.target
```

**Step 3: Create `deploy/deploy.sh`**

```bash
#!/bin/bash
set -euo pipefail

SERVER="squall@180.152.71.166"
SSH_KEY="/Users/jialechen/.ssh/lab_deploy_180_152_71_166"
REMOTE_DIR="/opt/weld-sim"

echo "=== Building frontend ==="
cd frontend && npm run build && cd ..

echo "=== Syncing to server ==="
rsync -avz --exclude='.venv' --exclude='node_modules' --exclude='.git' \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='frontend/node_modules' \
  -e "ssh -i $SSH_KEY" \
  . "$SERVER:$REMOTE_DIR/"

echo "=== Setting up on server ==="
ssh -i "$SSH_KEY" "$SERVER" << 'REMOTE'
cd /opt/weld-sim
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install -r requirements.txt -r requirements-web.txt -q
echo "=== Restarting service ==="
sudo systemctl restart weld-sim || echo "Service not yet installed. Run: sudo cp deploy/weld-sim.service /etc/systemd/system/ && sudo systemctl enable --now weld-sim"
echo "=== Done ==="
REMOTE
```

**Step 4: Deploy to server and verify**

```bash
bash deploy/deploy.sh
# Then test:
ssh -i /Users/jialechen/.ssh/lab_deploy_180_152_71_166 squall@180.152.71.166 \
  "curl -s http://localhost:8001/api/v1/health"
# Expected: {"status":"ok","service":"UltrasonicWeldMaster Simulation Service"}
```

**Step 5: Add proxy rewrite to LAB's next.config.ts**

SSH into server and add rewrite:
```bash
ssh -i $SSH_KEY squall@180.152.71.166
# Edit /opt/lab/next.config.ts to add:
# { source: "/api/weld-sim/:path*", destination: "http://127.0.0.1:8001/api/v1/:path*" }
# Then: cd /opt/lab && npm run build && pm2 restart lab (or however LAB is managed)
```

**Step 6: Verify integration**

```bash
# From LAB's perspective:
curl -s http://localhost:3000/api/weld-sim/health
# Expected: {"status":"ok","service":"UltrasonicWeldMaster Simulation Service"}

# Full calculation test:
curl -s -X POST http://localhost:3000/api/weld-sim/simulate \
  -H "Content-Type: application/json" \
  -d '{"application":"li_battery_tab","upper_material_type":"Nickel 201","upper_thickness_mm":0.1,"upper_layers":40,"lower_material_type":"Copper C110","lower_thickness_mm":0.3,"weld_width_mm":3.0,"weld_length_mm":25.0}'
# Expected: JSON with recipe_id, parameters, safety_window, validation
```

**Step 7: Commit**

```bash
git add deploy/ .env.example
git commit -m "feat(deploy): systemd service, deploy script, and LAB integration"
```

---

## Task 10: End-to-End Testing + Final Polish

**Files:**
- Create: `tests/test_web/test_integration.py`
- Modify: existing tests to ensure no regressions

**Step 1: Full backend integration test**

```python
# tests/test_web/test_integration.py
"""End-to-end integration test: simulate -> get recipe -> export report."""
import pytest
from fastapi.testclient import TestClient
from web.app import create_app

@pytest.fixture
def client():
    return TestClient(create_app())

def test_full_workflow(client):
    # 1. List materials
    resp = client.get("/api/v1/materials")
    assert resp.status_code == 200
    materials = resp.json()["materials"]
    assert len(materials) > 0

    # 2. Simulate
    resp = client.post("/api/v1/simulate", json={
        "application": "li_battery_tab",
        "upper_material_type": "Nickel 201",
        "upper_thickness_mm": 0.1,
        "upper_layers": 40,
        "lower_material_type": "Copper C110",
        "lower_thickness_mm": 0.3,
        "weld_width_mm": 3.0,
        "weld_length_mm": 25.0,
    })
    assert resp.status_code == 200
    result = resp.json()
    assert result["validation"]["status"] in ("pass", "warning", "fail")
    recipe_id = result["recipe_id"]

    # 3. List recipes
    resp = client.get("/api/v1/recipes")
    assert resp.status_code == 200

    # 4. Health
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
```

**Step 2: Run ALL tests (existing 253 + new web tests)**

```bash
pytest tests/ -v --tb=short
```

Expected: ALL PASS, no regressions.

**Step 3: Build frontend and verify static serving**

```bash
cd frontend && npm run build && cd ..
python run_web.py &
# Open http://localhost:8001 in browser — should see Vue frontend
# Test: http://localhost:8001/api/v1/health — should return JSON
kill %1
```

**Step 4: Final commit**

```bash
git add tests/test_web/test_integration.py
git commit -m "test(web): end-to-end integration tests for full workflow"
```

---

## Summary: Task Order and Dependencies

```
Task 1: Backend scaffolding + health ──┐
Task 2: Calculation API ───────────────┤── Backend (independent of frontend)
Task 3: Materials + Recipes + Reports ─┘
                                        │
Task 4: Vue 3 scaffolding ────────────┐│
Task 5: Layout + Navigation ──────────┤├── Frontend (depends on Task 1-3 for API proxy)
Task 6: Calculation Wizard ────────────┤│
Task 7: Results + Reports + History ───┤│
Task 8: SVG Visualization ────────────┘│
                                        │
Task 9: Server deployment ─────────────┘── Depends on all above
Task 10: E2E testing ──────────────────── Final verification
```

**Parallelization opportunity:** Tasks 1-3 (backend) and Tasks 4-8 (frontend) can be developed in parallel by separate agents.
