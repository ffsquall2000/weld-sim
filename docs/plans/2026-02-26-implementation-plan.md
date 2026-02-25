# UltrasonicWeldMaster Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a plugin-based microkernel desktop application for ultrasonic metal welding parameter auto-generation and experiment report creation.

**Architecture:** Plugin-based microkernel with PySide6 GUI shell. Core engine handles plugin lifecycle, event bus, structured logging, and SQLite data warehouse. All business logic lives in plugins (li-battery engine, general metal engine, material DB, knowledge base, reporter).

**Tech Stack:** Python 3.9+ / PySide6 / SQLite / pydantic / numpy / scipy / reportlab / openpyxl / matplotlib

---

## Prerequisites

```bash
# Create virtual environment (use system Python 3.9)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

> Note: Design doc specifies 3.11+ but system has 3.9.6. We use `from __future__ import annotations` for modern type hints. All code is compatible with Python 3.9+.

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `main.py`
- Create: `cli.py`
- Create: `ultrasonic_weld_master/__init__.py`
- Create: `ultrasonic_weld_master/core/__init__.py`
- Create: `ultrasonic_weld_master/plugins/__init__.py`
- Create: `ultrasonic_weld_master/gui/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_core/__init__.py`
- Create: `tests/test_plugins/__init__.py`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "ultrasonic-weld-master"
version = "0.1.0"
description = "Ultrasonic metal welding parameter auto-generation and experiment report software"
requires-python = ">=3.9"
dependencies = [
    "PySide6>=6.5",
    "numpy>=1.24",
    "scipy>=1.11",
    "pyyaml>=6.0",
    "reportlab>=4.0",
    "openpyxl>=3.1",
    "matplotlib>=3.8",
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0", "pytest-qt>=4.2"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

**Step 2: Create requirements.txt**

```
PySide6>=6.5
numpy>=1.24
scipy>=1.11
pyyaml>=6.0
reportlab>=4.0
openpyxl>=3.1
matplotlib>=3.8
pydantic>=2.0
pytest>=7.0
pytest-cov>=4.0
```

**Step 3: Create directory structure and __init__.py files**

All `__init__.py` files start empty except:

`ultrasonic_weld_master/__init__.py`:
```python
"""UltrasonicWeldMaster - Ultrasonic metal welding parameter auto-generation software."""
__version__ = "0.1.0"
```

**Step 4: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
*.egg-info/
dist/
build/
.DS_Store
data/database.sqlite
data/logs/
reports/
*.log
```

**Step 5: Create main.py (placeholder)**

```python
"""UltrasonicWeldMaster application entry point."""
from __future__ import annotations
import sys

def main():
    print("UltrasonicWeldMaster v0.1.0 - Starting...")
    # Will be replaced with actual GUI launch
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

**Step 6: Create venv and install deps**

Run: `python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

**Step 7: Verify setup**

Run: `python -c "import ultrasonic_weld_master; print(ultrasonic_weld_master.__version__)"`
Expected: `0.1.0`

**Step 8: Commit**

```bash
git add pyproject.toml requirements.txt main.py cli.py .gitignore ultrasonic_weld_master/ tests/
git commit -m "feat: project scaffold with pyproject.toml and directory structure"
```

---

## Task 2: Core Data Models

**Files:**
- Create: `ultrasonic_weld_master/core/models.py`
- Create: `tests/test_core/test_models.py`

**Step 1: Write the failing test**

`tests/test_core/test_models.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.models import (
    WeldRecipe, ValidationResult, ValidationStatus,
    MaterialInfo, SonotrodeInfo, WeldInputs, RiskLevel,
)

class TestWeldRecipe:
    def test_create_recipe(self):
        recipe = WeldRecipe(
            recipe_id="R001",
            application="li_battery_tab",
            inputs={},
            parameters={"amplitude_um": 30.0, "pressure_n": 200.0, "energy_j": 50.0, "time_ms": 200},
            safety_window={"amplitude_um": [25.0, 35.0], "pressure_n": [150.0, 250.0]},
        )
        assert recipe.recipe_id == "R001"
        assert recipe.parameters["amplitude_um"] == 30.0

    def test_recipe_to_dict(self):
        recipe = WeldRecipe(
            recipe_id="R002",
            application="general_metal",
            inputs={},
            parameters={"amplitude_um": 25.0},
        )
        d = recipe.to_dict()
        assert d["recipe_id"] == "R002"
        assert "created_at" in d

class TestValidationResult:
    def test_pass_result(self):
        result = ValidationResult(
            status=ValidationStatus.PASS,
            validators={"physics": {"status": "pass", "messages": []}},
        )
        assert result.is_passed()

    def test_fail_result(self):
        result = ValidationResult(
            status=ValidationStatus.FAIL,
            validators={"physics": {"status": "fail", "messages": ["Power density too high"]}},
        )
        assert not result.is_passed()

class TestMaterialInfo:
    def test_create_material(self):
        mat = MaterialInfo(
            name="Copper C11000",
            material_type="Cu",
            thickness_mm=0.2,
            layers=1,
        )
        assert mat.material_type == "Cu"
        assert mat.total_thickness_mm == 0.2

    def test_multi_layer_thickness(self):
        mat = MaterialInfo(
            name="Al foil",
            material_type="Al",
            thickness_mm=0.012,
            layers=40,
        )
        assert abs(mat.total_thickness_mm - 0.48) < 1e-6

class TestWeldInputs:
    def test_create_inputs(self):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0,
            weld_length_mm=25.0,
            frequency_khz=20.0,
            max_power_w=3000,
        )
        assert inputs.weld_area_mm2 == 125.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_models.py -v`
Expected: FAIL (module not found)

**Step 3: Write implementation**

`ultrasonic_weld_master/core/models.py`:
```python
"""Core data models for UltrasonicWeldMaster."""
from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


class ValidationStatus(enum.Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MaterialInfo:
    name: str
    material_type: str  # "Cu", "Al", "Ni", "Steel", etc.
    thickness_mm: float
    layers: int = 1
    temper: str = ""  # annealed, half-hard, etc.
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def total_thickness_mm(self) -> float:
        return self.thickness_mm * self.layers


@dataclass
class SonotrodeInfo:
    name: str
    sonotrode_type: str  # "sonotrode" or "anvil"
    material: str = "Titanium"
    knurl_type: str = "linear"  # linear, cross, diamond
    knurl_pitch_mm: float = 1.0
    knurl_depth_mm: float = 0.3
    contact_width_mm: float = 5.0
    contact_length_mm: float = 25.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeldInputs:
    application: str
    upper_material: MaterialInfo
    lower_material: MaterialInfo
    weld_width_mm: float
    weld_length_mm: float
    frequency_khz: float = 20.0
    max_power_w: float = 3000.0
    sonotrode: Optional[SonotrodeInfo] = None
    anvil: Optional[SonotrodeInfo] = None
    target_peel_force_n: Optional[float] = None
    target_resistance_mohm: Optional[float] = None
    target_cpk: float = 1.67
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def weld_area_mm2(self) -> float:
        return self.weld_width_mm * self.weld_length_mm


@dataclass
class WeldRecipe:
    recipe_id: str
    application: str
    inputs: dict[str, Any]
    parameters: dict[str, Any]
    safety_window: dict[str, Any] = field(default_factory=dict)
    quality_estimate: dict[str, Any] = field(default_factory=dict)
    risk_assessment: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    status: ValidationStatus
    validators: dict[str, Any]
    messages: list[str] = field(default_factory=list)

    def is_passed(self) -> bool:
        return self.status == ValidationStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "validators": self.validators,
            "messages": self.messages,
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_models.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/core/models.py tests/test_core/test_models.py
git commit -m "feat: core data models (WeldRecipe, MaterialInfo, ValidationResult)"
```

---

## Task 3: Event Bus

**Files:**
- Create: `ultrasonic_weld_master/core/event_bus.py`
- Create: `tests/test_core/test_event_bus.py`

**Step 1: Write the failing test**

`tests/test_core/test_event_bus.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.event_bus import EventBus

class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda data: received.append(data))
        bus.emit("test.event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["key"] == "value"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe("calc.done", lambda d: results.append("A"))
        bus.subscribe("calc.done", lambda d: results.append("B"))
        bus.emit("calc.done", {})
        assert results == ["A", "B"]

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda d: received.append(d)
        bus.subscribe("ev", handler)
        bus.unsubscribe("ev", handler)
        bus.emit("ev", {"x": 1})
        assert len(received) == 0

    def test_emit_unregistered_event(self):
        bus = EventBus()
        bus.emit("no.listener", {})  # should not raise

    def test_event_history(self):
        bus = EventBus(keep_history=True)
        bus.emit("a", {"v": 1})
        bus.emit("b", {"v": 2})
        history = bus.get_history()
        assert len(history) == 2
        assert history[0]["event"] == "a"

    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda d: received.append(d))
        bus.emit("any.event", {"x": 1})
        assert len(received) == 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_event_bus.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/core/event_bus.py`:
```python
"""Publish-subscribe event bus for plugin communication."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventHandler = Callable[[dict[str, Any]], None]


class EventBus:
    def __init__(self, keep_history: bool = False):
        self._subscribers: dict[str, list[EventHandler]] = {}
        self._keep_history = keep_history
        self._history: list[dict[str, Any]] = []

    def subscribe(self, event: str, handler: EventHandler) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        if event in self._subscribers:
            self._subscribers[event] = [
                h for h in self._subscribers[event] if h is not handler
            ]

    def emit(self, event: str, data: dict[str, Any]) -> None:
        record = {
            "event": event,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._keep_history:
            self._history.append(record)

        handlers = list(self._subscribers.get(event, []))
        # Also notify wildcard subscribers
        handlers.extend(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                handler(data)
            except Exception:
                logger.exception("Event handler error for %s", event)

    def get_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_event_bus.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/core/event_bus.py tests/test_core/test_event_bus.py
git commit -m "feat: event bus with pub-sub, wildcard, and history"
```

---

## Task 4: Structured Logging System

**Files:**
- Create: `ultrasonic_weld_master/core/logger.py`
- Create: `tests/test_core/test_logger.py`

**Step 1: Write the failing test**

`tests/test_core/test_logger.py`:
```python
from __future__ import annotations
import json
import tempfile
import os
import pytest
from ultrasonic_weld_master.core.logger import StructuredLogger

class TestStructuredLogger:
    def test_log_operation(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        logger.log_operation(
            session_id="sess1",
            event_type="calculation.started",
            user_action="generate_parameters",
            data={"inputs": {"material": "Cu"}},
        )
        ops_file = os.path.join(log_dir, "operations.jsonl")
        assert os.path.exists(ops_file)
        with open(ops_file) as f:
            record = json.loads(f.readline())
        assert record["event_type"] == "calculation.started"
        assert record["session_id"] == "sess1"
        assert "timestamp" in record

    def test_log_calculation(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        logger.log_calculation(
            session_id="sess2",
            inputs={"material": "Al", "layers": 40},
            outputs={"amplitude_um": 30.0},
            intermediate={"power_density": 2.5},
        )
        calc_file = os.path.join(log_dir, "calculations.jsonl")
        assert os.path.exists(calc_file)
        with open(calc_file) as f:
            record = json.loads(f.readline())
        assert record["inputs"]["material"] == "Al"
        assert record["outputs"]["amplitude_um"] == 30.0

    def test_multiple_entries(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        logger = StructuredLogger(log_dir=log_dir)
        for i in range(5):
            logger.log_operation(
                session_id=f"s{i}", event_type="test", data={"i": i}
            )
        ops_file = os.path.join(log_dir, "operations.jsonl")
        with open(ops_file) as f:
            lines = f.readlines()
        assert len(lines) == 5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_logger.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/core/logger.py`:
```python
"""Three-tier structured logging system for UltrasonicWeldMaster."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, log_dir: str = "data/logs"):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._setup_app_logger()

    def _setup_app_logger(self) -> None:
        self._app_logger = logging.getLogger("ultrasonic_weld_master")
        if not self._app_logger.handlers:
            handler = RotatingFileHandler(
                os.path.join(self._log_dir, "app.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._app_logger.addHandler(handler)
            self._app_logger.setLevel(logging.DEBUG)

    @property
    def app(self) -> logging.Logger:
        return self._app_logger

    def _write_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        filepath = os.path.join(self._log_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_operation(
        self,
        session_id: str,
        event_type: str,
        user_action: str = "",
        data: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "user_action": user_action,
            "data": data or {},
            "metadata": metadata or {},
        }
        self._write_jsonl("operations.jsonl", record)

    def log_calculation(
        self,
        session_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        intermediate: Optional[dict[str, Any]] = None,
        validation: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": "calculation.completed",
            "inputs": inputs,
            "outputs": outputs,
            "intermediate": intermediate or {},
            "validation": validation or {},
            "metadata": metadata or {},
        }
        self._write_jsonl("calculations.jsonl", record)
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_logger.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/core/logger.py tests/test_core/test_logger.py
git commit -m "feat: structured logging system (app/operations/calculations)"
```

---

## Task 5: SQLite Database

**Files:**
- Create: `ultrasonic_weld_master/core/database.py`
- Create: `tests/test_core/test_database.py`

**Step 1: Write the failing test**

`tests/test_core/test_database.py`:
```python
from __future__ import annotations
import os
import pytest
from ultrasonic_weld_master.core.database import Database

class TestDatabase:
    @pytest.fixture
    def db(self, tmp_path):
        db_path = str(tmp_path / "test.sqlite")
        database = Database(db_path)
        database.initialize()
        yield database
        database.close()

    def test_initialize_creates_tables(self, db):
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        table_names = {row[0] for row in tables}
        assert "projects" in table_names
        assert "sessions" in table_names
        assert "operations" in table_names
        assert "recipes" in table_names
        assert "reports" in table_names
        assert "materials" in table_names
        assert "sonotrodes" in table_names
        assert "weld_results" in table_names

    def test_create_project(self, db):
        pid = db.create_project("Test Project", "li_battery_tab", {"key": "val"})
        assert pid is not None
        project = db.get_project(pid)
        assert project["name"] == "Test Project"
        assert project["application"] == "li_battery_tab"

    def test_create_session(self, db):
        pid = db.create_project("P1", "li_battery_tab")
        sid = db.create_session(pid, user_name="engineer")
        assert sid is not None
        session = db.get_session(sid)
        assert session["project_id"] == pid

    def test_save_recipe(self, db):
        pid = db.create_project("P1", "li_battery_tab")
        sid = db.create_session(pid)
        rid = db.save_recipe(
            project_id=pid,
            session_id=sid,
            application="li_battery_tab",
            inputs={"material": "Cu"},
            parameters={"amplitude_um": 30.0},
        )
        recipe = db.get_recipe(rid)
        assert recipe["application"] == "li_battery_tab"

    def test_log_operation(self, db):
        pid = db.create_project("P1", "li_battery_tab")
        sid = db.create_session(pid)
        db.log_operation(sid, "calculation.started", data={"test": True})
        ops = db.get_operations(sid)
        assert len(ops) == 1
        assert ops[0]["event_type"] == "calculation.started"

    def test_list_projects(self, db):
        db.create_project("P1", "li_battery_tab")
        db.create_project("P2", "general_metal")
        projects = db.list_projects()
        assert len(projects) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_database.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/core/database.py`:
```python
"""SQLite data warehouse for UltrasonicWeldMaster."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    application TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    config TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    started_at TEXT DEFAULT (datetime('now')),
    ended_at TEXT,
    user_name TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS operations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(id),
    timestamp TEXT DEFAULT (datetime('now')),
    event_type TEXT NOT NULL,
    user_action TEXT DEFAULT '',
    data TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS recipes (
    id TEXT PRIMARY KEY,
    project_id TEXT REFERENCES projects(id),
    session_id TEXT REFERENCES sessions(id),
    created_at TEXT DEFAULT (datetime('now')),
    application TEXT NOT NULL,
    inputs TEXT NOT NULL,
    parameters TEXT NOT NULL,
    safety_window TEXT DEFAULT '{}',
    validation_result TEXT DEFAULT '{}',
    risk_assessment TEXT DEFAULT '{}',
    notes TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    recipe_id TEXT REFERENCES recipes(id),
    created_at TEXT DEFAULT (datetime('now')),
    report_type TEXT,
    file_path TEXT,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS materials (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT DEFAULT '',
    material_type TEXT DEFAULT '',
    properties TEXT NOT NULL DEFAULT '{}',
    source TEXT DEFAULT '',
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sonotrodes (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sonotrode_type TEXT DEFAULT 'sonotrode',
    material TEXT DEFAULT '',
    knurl_type TEXT DEFAULT '',
    knurl_pitch REAL DEFAULT 0,
    knurl_depth REAL DEFAULT 0,
    contact_area_w REAL DEFAULT 0,
    contact_area_l REAL DEFAULT 0,
    properties TEXT DEFAULT '{}',
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS weld_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recipe_id TEXT REFERENCES recipes(id),
    created_at TEXT DEFAULT (datetime('now')),
    actual_parameters TEXT DEFAULT '{}',
    quality_results TEXT DEFAULT '{}',
    notes TEXT DEFAULT '',
    operator TEXT DEFAULT ''
);
"""


class Database:
    def __init__(self, db_path: str = "data/database.sqlite"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        assert self._conn is not None
        return self._conn.execute(sql, params)

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _json_dumps(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=str)

    def _json_loads(self, s: str) -> Any:
        return json.loads(s) if s else {}

    # --- Projects ---
    def create_project(self, name: str, application: str, config: Optional[dict] = None) -> str:
        pid = self._new_id()
        self.execute(
            "INSERT INTO projects (id, name, application, config) VALUES (?, ?, ?, ?)",
            (pid, name, application, self._json_dumps(config or {})),
        )
        self._conn.commit()
        return pid

    def get_project(self, project_id: str) -> dict[str, Any]:
        row = self.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise ValueError(f"Project {project_id} not found")
        d = dict(row)
        d["config"] = self._json_loads(d["config"])
        return d

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.execute("SELECT * FROM projects ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    # --- Sessions ---
    def create_session(self, project_id: str, user_name: str = "") -> str:
        sid = self._new_id()
        self.execute(
            "INSERT INTO sessions (id, project_id, user_name) VALUES (?, ?, ?)",
            (sid, project_id, user_name),
        )
        self._conn.commit()
        return sid

    def get_session(self, session_id: str) -> dict[str, Any]:
        row = self.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        return dict(row)

    def end_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id))
        self._conn.commit()

    # --- Operations ---
    def log_operation(
        self, session_id: str, event_type: str,
        user_action: str = "", data: Optional[dict] = None, metadata: Optional[dict] = None,
    ) -> None:
        self.execute(
            "INSERT INTO operations (session_id, event_type, user_action, data, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, user_action,
             self._json_dumps(data or {}), self._json_dumps(metadata or {})),
        )
        self._conn.commit()

    def get_operations(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.execute(
            "SELECT * FROM operations WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = self._json_loads(d["data"])
            d["metadata"] = self._json_loads(d["metadata"])
            result.append(d)
        return result

    # --- Recipes ---
    def save_recipe(
        self, project_id: str, session_id: str, application: str,
        inputs: dict, parameters: dict,
        safety_window: Optional[dict] = None,
        validation_result: Optional[dict] = None,
        risk_assessment: Optional[dict] = None,
        notes: str = "",
    ) -> str:
        rid = self._new_id()
        self.execute(
            """INSERT INTO recipes
            (id, project_id, session_id, application, inputs, parameters,
             safety_window, validation_result, risk_assessment, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (rid, project_id, session_id, application,
             self._json_dumps(inputs), self._json_dumps(parameters),
             self._json_dumps(safety_window or {}),
             self._json_dumps(validation_result or {}),
             self._json_dumps(risk_assessment or {}),
             notes),
        )
        self._conn.commit()
        return rid

    def get_recipe(self, recipe_id: str) -> dict[str, Any]:
        row = self.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if not row:
            raise ValueError(f"Recipe {recipe_id} not found")
        d = dict(row)
        for key in ("inputs", "parameters", "safety_window", "validation_result", "risk_assessment"):
            d[key] = self._json_loads(d[key])
        return d

    def list_recipes(self, project_id: str) -> list[dict[str, Any]]:
        rows = self.execute(
            "SELECT * FROM recipes WHERE project_id = ? ORDER BY created_at DESC",
            (project_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Materials ---
    def save_material(self, name: str, material_type: str, properties: dict,
                      category: str = "", source: str = "") -> str:
        mid = self._new_id()
        self.execute(
            "INSERT INTO materials (id, name, category, material_type, properties, source) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, name, category, material_type, self._json_dumps(properties), source),
        )
        self._conn.commit()
        return mid

    def list_materials(self, material_type: str = "") -> list[dict[str, Any]]:
        if material_type:
            rows = self.execute(
                "SELECT * FROM materials WHERE material_type = ?", (material_type,)
            ).fetchall()
        else:
            rows = self.execute("SELECT * FROM materials").fetchall()
        return [dict(r) for r in rows]

    # --- Sonotrodes ---
    def save_sonotrode(self, name: str, sonotrode_type: str = "sonotrode", **kwargs: Any) -> str:
        sid = self._new_id()
        self.execute(
            """INSERT INTO sonotrodes
            (id, name, sonotrode_type, material, knurl_type, knurl_pitch, knurl_depth,
             contact_area_w, contact_area_l, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sid, name, sonotrode_type,
             kwargs.get("material", ""), kwargs.get("knurl_type", ""),
             kwargs.get("knurl_pitch", 0), kwargs.get("knurl_depth", 0),
             kwargs.get("contact_area_w", 0), kwargs.get("contact_area_l", 0),
             self._json_dumps(kwargs.get("properties", {}))),
        )
        self._conn.commit()
        return sid

    # --- Weld Results ---
    def save_weld_result(self, recipe_id: str, actual_parameters: dict,
                         quality_results: dict, notes: str = "", operator: str = "") -> int:
        cursor = self.execute(
            """INSERT INTO weld_results
            (recipe_id, actual_parameters, quality_results, notes, operator)
            VALUES (?, ?, ?, ?, ?)""",
            (recipe_id, self._json_dumps(actual_parameters),
             self._json_dumps(quality_results), notes, operator),
        )
        self._conn.commit()
        return cursor.lastrowid
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_database.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/core/database.py tests/test_core/test_database.py
git commit -m "feat: SQLite data warehouse with full CRUD operations"
```

---

## Task 6: Config Manager

**Files:**
- Create: `ultrasonic_weld_master/core/config.py`
- Create: `tests/test_core/test_config.py`
- Create: `config.yaml` (default config)

**Step 1: Write the failing test**

`tests/test_core/test_config.py`:
```python
from __future__ import annotations
import os
import pytest
from ultrasonic_weld_master.core.config import AppConfig

class TestConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.get("app.name") == "UltrasonicWeldMaster"
        assert config.get("app.version") == "0.1.0"

    def test_load_from_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("app:\\n  name: TestApp\\n  version: '9.9.9'\\n")
        config = AppConfig(str(cfg_file))
        assert config.get("app.name") == "TestApp"

    def test_get_nested(self):
        config = AppConfig()
        assert config.get("database.path") is not None

    def test_get_with_default(self):
        config = AppConfig()
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_set_value(self):
        config = AppConfig()
        config.set("custom.key", "hello")
        assert config.get("custom.key") == "hello"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_config.py -v`

**Step 3: Write config.yaml**

`config.yaml`:
```yaml
app:
  name: UltrasonicWeldMaster
  version: "0.1.0"
  language: zh_CN

database:
  path: data/database.sqlite

logging:
  dir: data/logs
  level: DEBUG

plugins:
  dir: ultrasonic_weld_master/plugins
  enabled:
    - material_db
    - li_battery
    - general_metal
    - knowledge_base
    - reporter

gui:
  theme: light
  window_width: 1400
  window_height: 900
```

**Step 4: Write implementation**

`ultrasonic_weld_master/core/config.py`:
```python
"""Global configuration manager using YAML."""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

DEFAULT_CONFIG = {
    "app": {"name": "UltrasonicWeldMaster", "version": "0.1.0", "language": "zh_CN"},
    "database": {"path": "data/database.sqlite"},
    "logging": {"dir": "data/logs", "level": "DEBUG"},
    "plugins": {
        "dir": "ultrasonic_weld_master/plugins",
        "enabled": ["material_db", "li_battery", "general_metal", "knowledge_base", "reporter"],
    },
    "gui": {"theme": "light", "window_width": 1400, "window_height": 900},
}


class AppConfig:
    def __init__(self, config_path: Optional[str] = None):
        self._data: dict[str, Any] = {}
        self._deep_merge(self._data, DEFAULT_CONFIG)
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            self._deep_merge(self._data, file_data)

    def _deep_merge(self, base: dict, override: dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, dotted_key: str, default: Any = None) -> Any:
        keys = dotted_key.split(".")
        node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    def set(self, dotted_key: str, value: Any) -> None:
        keys = dotted_key.split(".")
        node = self._data
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    @property
    def data(self) -> dict[str, Any]:
        return self._data
```

**Step 5: Run test**

Run: `pytest tests/test_core/test_config.py -v`

**Step 6: Commit**

```bash
git add ultrasonic_weld_master/core/config.py tests/test_core/test_config.py config.yaml
git commit -m "feat: YAML config manager with dot-notation access"
```

---

## Task 7: Plugin Manager

**Files:**
- Create: `ultrasonic_weld_master/core/plugin_api.py`
- Create: `ultrasonic_weld_master/core/plugin_manager.py`
- Create: `tests/test_core/test_plugin_manager.py`

**Step 1: Write the failing test**

`tests/test_core/test_plugin_manager.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo
from ultrasonic_weld_master.core.plugin_manager import PluginManager
from ultrasonic_weld_master.core.event_bus import EventBus
from ultrasonic_weld_master.core.logger import StructuredLogger
from ultrasonic_weld_master.core.config import AppConfig

class MockPlugin(PluginBase):
    def __init__(self):
        self.activated = False
        self.deactivated = False

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="mock_plugin", version="1.0.0",
            description="A mock plugin", author="Test", dependencies=[],
        )

    def activate(self, context) -> None:
        self.activated = True

    def deactivate(self) -> None:
        self.deactivated = True

class TestPluginManager:
    @pytest.fixture
    def manager(self, tmp_path):
        config = AppConfig()
        event_bus = EventBus()
        logger = StructuredLogger(log_dir=str(tmp_path / "logs"))
        return PluginManager(config=config, event_bus=event_bus, logger=logger)

    def test_register_and_activate(self, manager):
        plugin = MockPlugin()
        manager.register(plugin)
        manager.activate("mock_plugin")
        assert plugin.activated

    def test_get_plugin(self, manager):
        plugin = MockPlugin()
        manager.register(plugin)
        manager.activate("mock_plugin")
        retrieved = manager.get_plugin("mock_plugin")
        assert retrieved is plugin

    def test_deactivate(self, manager):
        plugin = MockPlugin()
        manager.register(plugin)
        manager.activate("mock_plugin")
        manager.deactivate("mock_plugin")
        assert plugin.deactivated

    def test_list_plugins(self, manager):
        plugin = MockPlugin()
        manager.register(plugin)
        plugins = manager.list_plugins()
        assert len(plugins) == 1
        assert plugins[0]["name"] == "mock_plugin"

    def test_activate_unknown_raises(self, manager):
        with pytest.raises(ValueError):
            manager.activate("nonexistent")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_plugin_manager.py -v`

**Step 3: Write plugin_api.py**

`ultrasonic_weld_master/core/plugin_api.py`:
```python
"""Plugin standard interfaces (ABCs) for UltrasonicWeldMaster."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ultrasonic_weld_master.core.engine import EngineContext
    from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult


@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    dependencies: list[str] = field(default_factory=list)


class PluginBase(ABC):
    @abstractmethod
    def get_info(self) -> PluginInfo:
        ...

    @abstractmethod
    def activate(self, context: Any) -> None:
        ...

    @abstractmethod
    def deactivate(self) -> None:
        ...

    def get_config_schema(self) -> Optional[dict]:
        return None

    def get_ui_panels(self) -> list:
        return []


class ParameterEnginePlugin(PluginBase):
    @abstractmethod
    def get_input_schema(self) -> dict:
        ...

    @abstractmethod
    def calculate_parameters(self, inputs: dict) -> Any:
        ...

    @abstractmethod
    def validate_parameters(self, recipe: Any) -> Any:
        ...

    @abstractmethod
    def get_supported_applications(self) -> list[str]:
        ...
```

**Step 4: Write plugin_manager.py**

`ultrasonic_weld_master/core/plugin_manager.py`:
```python
"""Plugin lifecycle manager."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo
from ultrasonic_weld_master.core.event_bus import EventBus
from ultrasonic_weld_master.core.logger import StructuredLogger
from ultrasonic_weld_master.core.config import AppConfig

logger = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, config: AppConfig, event_bus: EventBus, logger: StructuredLogger):
        self._config = config
        self._event_bus = event_bus
        self._logger = logger
        self._registered: dict[str, PluginBase] = {}
        self._active: set[str] = set()

    def register(self, plugin: PluginBase) -> None:
        info = plugin.get_info()
        self._registered[info.name] = plugin

    def activate(self, name: str) -> None:
        if name not in self._registered:
            raise ValueError(f"Plugin '{name}' not registered")
        plugin = self._registered[name]
        info = plugin.get_info()
        # Check dependencies
        for dep in info.dependencies:
            if dep not in self._active:
                if dep in self._registered:
                    self.activate(dep)
                else:
                    raise ValueError(f"Missing dependency '{dep}' for plugin '{name}'")

        context = {
            "config": self._config,
            "event_bus": self._event_bus,
            "logger": self._logger,
        }
        plugin.activate(context)
        self._active.add(name)
        self._event_bus.emit("plugin.activated", {"name": name, "version": info.version})

    def deactivate(self, name: str) -> None:
        if name in self._active and name in self._registered:
            self._registered[name].deactivate()
            self._active.discard(name)
            self._event_bus.emit("plugin.deactivated", {"name": name})

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        if name in self._active:
            return self._registered.get(name)
        return None

    def list_plugins(self) -> list[dict[str, Any]]:
        result = []
        for name, plugin in self._registered.items():
            info = plugin.get_info()
            result.append({
                "name": info.name,
                "version": info.version,
                "description": info.description,
                "active": name in self._active,
            })
        return result

    def deactivate_all(self) -> None:
        for name in list(self._active):
            self.deactivate(name)
```

**Step 5: Run test**

Run: `pytest tests/test_core/test_plugin_manager.py -v`

**Step 6: Commit**

```bash
git add ultrasonic_weld_master/core/plugin_api.py ultrasonic_weld_master/core/plugin_manager.py tests/test_core/test_plugin_manager.py
git commit -m "feat: plugin API and lifecycle manager with dependency resolution"
```

---

## Task 8: Core Engine (ties everything together)

**Files:**
- Create: `ultrasonic_weld_master/core/engine.py`
- Create: `tests/test_core/test_engine.py`

**Step 1: Write the failing test**

`tests/test_core/test_engine.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.engine import Engine

class TestEngine:
    def test_initialize(self, tmp_path):
        engine = Engine(data_dir=str(tmp_path / "data"))
        engine.initialize()
        assert engine.event_bus is not None
        assert engine.database is not None
        assert engine.logger is not None
        assert engine.plugin_manager is not None
        engine.shutdown()

    def test_create_and_get_session(self, tmp_path):
        engine = Engine(data_dir=str(tmp_path / "data"))
        engine.initialize()
        pid = engine.database.create_project("Test", "li_battery_tab")
        sid = engine.create_session(pid)
        assert sid is not None
        engine.shutdown()

    def test_event_bus_accessible(self, tmp_path):
        engine = Engine(data_dir=str(tmp_path / "data"))
        engine.initialize()
        received = []
        engine.event_bus.subscribe("test", lambda d: received.append(d))
        engine.event_bus.emit("test", {"v": 1})
        assert len(received) == 1
        engine.shutdown()
```

**Step 2: Run test**

Run: `pytest tests/test_core/test_engine.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/core/engine.py`:
```python
"""Core engine - the microkernel that ties everything together."""
from __future__ import annotations

import os
from typing import Optional

from ultrasonic_weld_master.core.config import AppConfig
from ultrasonic_weld_master.core.database import Database
from ultrasonic_weld_master.core.event_bus import EventBus
from ultrasonic_weld_master.core.logger import StructuredLogger
from ultrasonic_weld_master.core.plugin_manager import PluginManager


class Engine:
    def __init__(self, config_path: Optional[str] = None, data_dir: str = "data"):
        self._config_path = config_path
        self._data_dir = data_dir
        self.config: Optional[AppConfig] = None
        self.event_bus: Optional[EventBus] = None
        self.database: Optional[Database] = None
        self.logger: Optional[StructuredLogger] = None
        self.plugin_manager: Optional[PluginManager] = None
        self._current_session: Optional[str] = None

    def initialize(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "logs"), exist_ok=True)

        self.config = AppConfig(self._config_path)
        self.event_bus = EventBus(keep_history=True)
        self.logger = StructuredLogger(log_dir=os.path.join(self._data_dir, "logs"))
        self.database = Database(os.path.join(self._data_dir, "database.sqlite"))
        self.database.initialize()
        self.plugin_manager = PluginManager(
            config=self.config, event_bus=self.event_bus, logger=self.logger,
        )
        self.logger.app.info("Engine initialized")

    def create_session(self, project_id: str, user_name: str = "") -> str:
        sid = self.database.create_session(project_id, user_name)
        self._current_session = sid
        self.event_bus.emit("session.created", {"session_id": sid, "project_id": project_id})
        return sid

    @property
    def current_session(self) -> Optional[str]:
        return self._current_session

    def shutdown(self) -> None:
        if self._current_session:
            self.database.end_session(self._current_session)
        if self.plugin_manager:
            self.plugin_manager.deactivate_all()
        if self.database:
            self.database.close()
        if self.logger:
            self.logger.app.info("Engine shutdown")
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_engine.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/core/engine.py tests/test_core/test_engine.py
git commit -m "feat: core engine orchestrating all microkernel components"
```

---

## Task 9: Material Database Plugin

**Files:**
- Create: `ultrasonic_weld_master/plugins/material_db/__init__.py`
- Create: `ultrasonic_weld_master/plugins/material_db/plugin.py`
- Create: `ultrasonic_weld_master/plugins/material_db/materials.yaml`
- Create: `tests/test_plugins/__init__.py`
- Create: `tests/test_plugins/test_material_db.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_material_db.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin

class TestMaterialDBPlugin:
    @pytest.fixture
    def plugin(self):
        p = MaterialDBPlugin()
        p.activate({"config": None, "event_bus": None, "logger": None})
        return p

    def test_get_info(self):
        p = MaterialDBPlugin()
        info = p.get_info()
        assert info.name == "material_db"

    def test_get_material_cu(self, plugin):
        cu = plugin.get_material("Cu")
        assert cu is not None
        assert cu["density_kg_m3"] > 0
        assert cu["yield_strength_mpa"] > 0
        assert cu["acoustic_impedance"] > 0

    def test_get_material_al(self, plugin):
        al = plugin.get_material("Al")
        assert al is not None
        assert al["thermal_conductivity"] > 0

    def test_get_material_ni(self, plugin):
        ni = plugin.get_material("Ni")
        assert ni is not None

    def test_get_material_unknown(self, plugin):
        result = plugin.get_material("Unobtanium")
        assert result is None

    def test_list_materials(self, plugin):
        materials = plugin.list_materials()
        assert len(materials) >= 3  # At least Cu, Al, Ni

    def test_get_combination_properties(self, plugin):
        props = plugin.get_combination_properties("Cu", "Al")
        assert "friction_coefficient" in props
```

**Step 2: Run test**

Run: `pytest tests/test_plugins/test_material_db.py -v`

**Step 3: Write materials.yaml**

`ultrasonic_weld_master/plugins/material_db/materials.yaml`:
```yaml
materials:
  Cu:
    full_name: "Copper (C11000)"
    density_kg_m3: 8960
    melting_point_c: 1085
    yield_strength_mpa: 70
    ultimate_strength_mpa: 220
    hardness_hv: 50
    thermal_conductivity: 401
    electrical_resistivity_ohm_m: 1.68e-8
    acoustic_velocity_m_s: 4660
    acoustic_impedance: 41.7e6
    specific_heat_j_kg_k: 385
    youngs_modulus_gpa: 117
    poisson_ratio: 0.34
    oxide_layer: "Cu2O, thin, easily broken"

  Al:
    full_name: "Aluminum (1100)"
    density_kg_m3: 2700
    melting_point_c: 660
    yield_strength_mpa: 35
    ultimate_strength_mpa: 90
    hardness_hv: 23
    thermal_conductivity: 237
    electrical_resistivity_ohm_m: 2.65e-8
    acoustic_velocity_m_s: 6420
    acoustic_impedance: 17.3e6
    specific_heat_j_kg_k: 897
    youngs_modulus_gpa: 70
    poisson_ratio: 0.33
    oxide_layer: "Al2O3, tenacious, requires high amplitude"

  Ni:
    full_name: "Nickel (Ni201)"
    density_kg_m3: 8900
    melting_point_c: 1455
    yield_strength_mpa: 150
    ultimate_strength_mpa: 460
    hardness_hv: 90
    thermal_conductivity: 91
    electrical_resistivity_ohm_m: 6.99e-8
    acoustic_velocity_m_s: 5600
    acoustic_impedance: 49.8e6
    specific_heat_j_kg_k: 440
    youngs_modulus_gpa: 200
    poisson_ratio: 0.31
    oxide_layer: "NiO, moderate"

  Steel:
    full_name: "Low Carbon Steel"
    density_kg_m3: 7850
    melting_point_c: 1510
    yield_strength_mpa: 250
    ultimate_strength_mpa: 410
    hardness_hv: 120
    thermal_conductivity: 50
    electrical_resistivity_ohm_m: 1.43e-7
    acoustic_velocity_m_s: 5960
    acoustic_impedance: 46.7e6
    specific_heat_j_kg_k: 486
    youngs_modulus_gpa: 200
    poisson_ratio: 0.30
    oxide_layer: "FeO/Fe2O3, variable"

combinations:
  Cu-Al:
    friction_coefficient: 0.3
    imc_risk: "high"
    imc_type: "CuAl2, Cu9Al4"
    recommended_energy_range: "low-medium"
    max_interface_temp_c: 300
    notes: "Cu-Al forms brittle IMC above 300°C; keep energy low"

  Cu-Cu:
    friction_coefficient: 0.2
    imc_risk: "none"
    recommended_energy_range: "medium"
    max_interface_temp_c: 500
    notes: "Pure Cu-Cu forms strong solid-state bond"

  Al-Al:
    friction_coefficient: 0.35
    imc_risk: "none"
    recommended_energy_range: "medium"
    max_interface_temp_c: 350
    notes: "Al oxide layer requires higher amplitude to break"

  Cu-Ni:
    friction_coefficient: 0.25
    imc_risk: "low"
    recommended_energy_range: "medium-high"
    max_interface_temp_c: 450
    notes: "Cu-Ni good compatibility; Ni harder, needs more energy"

  Al-Ni:
    friction_coefficient: 0.28
    imc_risk: "medium"
    imc_type: "Al3Ni, AlNi"
    recommended_energy_range: "medium"
    max_interface_temp_c: 350
    notes: "Monitor IMC formation at higher temperatures"
```

**Step 4: Write plugin**

`ultrasonic_weld_master/plugins/material_db/plugin.py`:
```python
"""Material database plugin."""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo


class MaterialDBPlugin(PluginBase):
    def __init__(self):
        self._materials: dict[str, dict] = {}
        self._combinations: dict[str, dict] = {}

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="material_db",
            version="1.0.0",
            description="Material properties database for ultrasonic welding",
            author="UltrasonicWeldMaster",
            dependencies=[],
        )

    def activate(self, context: Any) -> None:
        yaml_path = os.path.join(os.path.dirname(__file__), "materials.yaml")
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self._materials = data.get("materials", {})
        self._combinations = data.get("combinations", {})

    def deactivate(self) -> None:
        self._materials.clear()
        self._combinations.clear()

    def get_material(self, material_type: str) -> Optional[dict[str, Any]]:
        return self._materials.get(material_type)

    def list_materials(self) -> list[str]:
        return list(self._materials.keys())

    def get_combination_properties(self, mat_a: str, mat_b: str) -> dict[str, Any]:
        key = f"{mat_a}-{mat_b}"
        if key in self._combinations:
            return self._combinations[key]
        reverse_key = f"{mat_b}-{mat_a}"
        if reverse_key in self._combinations:
            return self._combinations[reverse_key]
        return {}
```

`ultrasonic_weld_master/plugins/material_db/__init__.py`: empty

**Step 5: Run test**

Run: `pytest tests/test_plugins/test_material_db.py -v`

**Step 6: Commit**

```bash
git add ultrasonic_weld_master/plugins/material_db/ tests/test_plugins/
git commit -m "feat: material database plugin with Cu/Al/Ni/Steel data and combinations"
```

---

## Task 10: Li-Battery Physics Models

**Files:**
- Create: `ultrasonic_weld_master/plugins/li_battery/__init__.py`
- Create: `ultrasonic_weld_master/plugins/li_battery/physics.py`
- Create: `tests/test_plugins/test_li_battery_physics.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_li_battery_physics.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

class TestPhysicsModel:
    @pytest.fixture
    def model(self):
        return PhysicsModel()

    def test_acoustic_impedance_match(self, model):
        efficiency = model.acoustic_impedance_match(
            z1=41.7e6,  # Cu impedance
            z2=17.3e6,  # Al impedance
        )
        assert 0 < efficiency < 1

    def test_perfect_impedance_match(self, model):
        efficiency = model.acoustic_impedance_match(z1=41.7e6, z2=41.7e6)
        assert abs(efficiency - 1.0) < 1e-6

    def test_interface_power_density(self, model):
        pd = model.interface_power_density(
            frequency_hz=20000,
            amplitude_um=30.0,
            pressure_mpa=0.3,
            friction_coeff=0.3,
            contact_area_mm2=125.0,
        )
        assert pd > 0  # W/mm2

    def test_multilayer_energy_attenuation(self, model):
        ratios = model.multilayer_energy_attenuation(
            n_layers=40,
            material_impedance=17.3e6,
            layer_thickness_mm=0.012,
        )
        assert len(ratios) == 40
        assert ratios[0] > ratios[-1]  # Top layer gets more energy
        assert all(0 <= r <= 1 for r in ratios)

    def test_interface_temperature_rise(self, model):
        delta_t = model.interface_temperature_rise(
            power_density_w_mm2=2.0,
            weld_time_s=0.2,
            thermal_conductivity_1=401,  # Cu
            thermal_conductivity_2=237,  # Al
            density_1=8960,
            density_2=2700,
            specific_heat_1=385,
            specific_heat_2=897,
        )
        assert delta_t > 0
        assert delta_t < 1000  # Should not exceed melting for normal params
```

**Step 2: Run test**

Run: `pytest tests/test_plugins/test_li_battery_physics.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/plugins/li_battery/physics.py`:
```python
"""Physics models for ultrasonic metal welding."""
from __future__ import annotations

import math
from typing import Optional


class PhysicsModel:
    def acoustic_impedance_match(self, z1: float, z2: float) -> float:
        """Calculate energy transmission efficiency between two materials.

        Uses transmission coefficient T = 4*Z1*Z2 / (Z1+Z2)^2
        """
        if z1 <= 0 or z2 <= 0:
            return 0.0
        return 4 * z1 * z2 / (z1 + z2) ** 2

    def interface_power_density(
        self,
        frequency_hz: float,
        amplitude_um: float,
        pressure_mpa: float,
        friction_coeff: float,
        contact_area_mm2: float,
    ) -> float:
        """Calculate interface power density in W/mm².

        P = 2 * pi * f * mu * sigma_n * A
        where:
          f = frequency (Hz)
          mu = friction coefficient
          sigma_n = normal pressure (MPa)
          A = amplitude (m, peak)
        Result divided by contact area for density.
        """
        amplitude_m = amplitude_um * 1e-6
        # Total friction power (W)
        power_w = 2 * math.pi * frequency_hz * friction_coeff * pressure_mpa * 1e6 * amplitude_m * contact_area_mm2 * 1e-6
        return power_w / contact_area_mm2

    def multilayer_energy_attenuation(
        self,
        n_layers: int,
        material_impedance: float,
        layer_thickness_mm: float,
        attenuation_coeff: float = 0.02,
    ) -> list[float]:
        """Calculate energy ratio reaching each layer in a multi-layer stack.

        Simple exponential attenuation model:
        E(n) = E0 * exp(-alpha * n * t)

        Returns list of ratios [E1/E0, E2/E0, ..., En/E0]
        """
        ratios = []
        for i in range(n_layers):
            depth_mm = (i + 0.5) * layer_thickness_mm
            ratio = math.exp(-attenuation_coeff * depth_mm * n_layers ** 0.3)
            ratios.append(max(ratio, 0.0))
        return ratios

    def interface_temperature_rise(
        self,
        power_density_w_mm2: float,
        weld_time_s: float,
        thermal_conductivity_1: float,
        thermal_conductivity_2: float,
        density_1: float,
        density_2: float,
        specific_heat_1: float,
        specific_heat_2: float,
    ) -> float:
        """Estimate interface temperature rise using 1D heat diffusion.

        Simplified model: dT = q * sqrt(t) / sqrt(pi * k * rho * cp)
        Uses geometric mean of both materials' thermal properties.
        """
        # Thermal effusivity for each material: e = sqrt(k * rho * cp)
        e1 = math.sqrt(thermal_conductivity_1 * density_1 * specific_heat_1)
        e2 = math.sqrt(thermal_conductivity_2 * density_2 * specific_heat_2)
        # Effective effusivity (harmonic mean)
        e_eff = 2 * e1 * e2 / (e1 + e2) if (e1 + e2) > 0 else 1.0

        # Convert power density to W/m²
        q = power_density_w_mm2 * 1e6

        # Temperature rise: dT = q * sqrt(t / pi) / e_eff
        delta_t = q * math.sqrt(weld_time_s / math.pi) / e_eff
        return delta_t

    def estimate_collapse_um(
        self,
        amplitude_um: float,
        pressure_mpa: float,
        weld_time_s: float,
        n_layers: int,
        material_yield_mpa: float,
    ) -> float:
        """Estimate total collapse/indentation in micrometers."""
        # Simplified: collapse proportional to pressure/yield * amplitude * time * layers
        if material_yield_mpa <= 0:
            return 0.0
        ratio = pressure_mpa / material_yield_mpa
        collapse = amplitude_um * ratio * weld_time_s * 1000 * (1 + 0.01 * n_layers)
        return max(collapse, 0.0)
```

**Step 4: Run test**

Run: `pytest tests/test_plugins/test_li_battery_physics.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/li_battery/ tests/test_plugins/test_li_battery_physics.py
git commit -m "feat: ultrasonic welding physics models (impedance, power, energy, thermal)"
```

---

## Task 11: Li-Battery Calculator (3-Layer Model)

**Files:**
- Create: `ultrasonic_weld_master/plugins/li_battery/calculator.py`
- Create: `tests/test_plugins/test_li_battery_calculator.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_li_battery_calculator.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.calculator import LiBatteryCalculator
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.core.models import WeldInputs, MaterialInfo, WeldRecipe

class TestLiBatteryCalculator:
    @pytest.fixture
    def calculator(self):
        mat_db = MaterialDBPlugin()
        mat_db.activate({})
        return LiBatteryCalculator(material_db=mat_db)

    def test_calculate_tab_welding(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al foil", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0,
            weld_length_mm=25.0,
            frequency_khz=20.0,
            max_power_w=3500,
        )
        recipe = calculator.calculate(inputs)
        assert isinstance(recipe, WeldRecipe)
        assert "amplitude_um" in recipe.parameters
        assert "pressure_n" in recipe.parameters
        assert "energy_j" in recipe.parameters
        assert recipe.parameters["amplitude_um"] > 0
        assert recipe.parameters["pressure_n"] > 0
        assert recipe.parameters["energy_j"] > 0

    def test_calculate_cu_cu(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Cu foil", material_type="Cu", thickness_mm=0.008, layers=50),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0,
            weld_length_mm=20.0,
            frequency_khz=20.0,
            max_power_w=3500,
        )
        recipe = calculator.calculate(inputs)
        assert recipe.parameters["amplitude_um"] > 0

    def test_safety_window_included(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al foil", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu tab", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0,
            weld_length_mm=25.0,
        )
        recipe = calculator.calculate(inputs)
        assert "amplitude_um" in recipe.safety_window
        sw = recipe.safety_window["amplitude_um"]
        assert sw[0] < recipe.parameters["amplitude_um"] < sw[1]

    def test_risk_assessment(self, calculator):
        inputs = WeldInputs(
            application="li_battery_tab",
            upper_material=MaterialInfo(name="Al", material_type="Al", thickness_mm=0.012, layers=40),
            lower_material=MaterialInfo(name="Cu", material_type="Cu", thickness_mm=0.3, layers=1),
            weld_width_mm=5.0,
            weld_length_mm=25.0,
        )
        recipe = calculator.calculate(inputs)
        assert "overweld_risk" in recipe.risk_assessment
        assert "underweld_risk" in recipe.risk_assessment
```

**Step 2: Run test**

Run: `pytest tests/test_plugins/test_li_battery_calculator.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/plugins/li_battery/calculator.py`:
```python
"""Li-battery welding parameter calculator with 3-layer model."""
from __future__ import annotations

import math
import uuid
from typing import Any, Optional

from ultrasonic_weld_master.core.models import (
    WeldInputs, WeldRecipe, MaterialInfo, RiskLevel,
)
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel


# Empirical lookup tables for base parameters
BASE_PARAMS = {
    "Al-Cu": {"amplitude_um": 30, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 200},
    "Al-Al": {"amplitude_um": 28, "pressure_mpa": 0.25, "energy_density_j_mm2": 0.4, "time_ms": 180},
    "Cu-Cu": {"amplitude_um": 35, "pressure_mpa": 0.35, "energy_density_j_mm2": 0.6, "time_ms": 250},
    "Cu-Ni": {"amplitude_um": 38, "pressure_mpa": 0.40, "energy_density_j_mm2": 0.7, "time_ms": 280},
    "Cu-Al": {"amplitude_um": 30, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 200},
    "Ni-Cu": {"amplitude_um": 38, "pressure_mpa": 0.40, "energy_density_j_mm2": 0.7, "time_ms": 280},
}

DEFAULT_BASE = {"amplitude_um": 32, "pressure_mpa": 0.30, "energy_density_j_mm2": 0.5, "time_ms": 220}


class LiBatteryCalculator:
    def __init__(self, material_db: Any):
        self._material_db = material_db
        self._physics = PhysicsModel()

    def calculate(self, inputs: WeldInputs) -> WeldRecipe:
        upper_mat = self._material_db.get_material(inputs.upper_material.material_type) or {}
        lower_mat = self._material_db.get_material(inputs.lower_material.material_type) or {}
        combo_props = self._material_db.get_combination_properties(
            inputs.upper_material.material_type, inputs.lower_material.material_type
        )

        # Layer 1: Physics model
        physics_data = self._layer1_physics(inputs, upper_mat, lower_mat, combo_props)

        # Layer 2: Empirical correction
        corrected = self._layer2_empirical(inputs, physics_data, combo_props)

        # Layer 3: Build recipe
        recipe = self._layer3_output(inputs, corrected, physics_data)
        return recipe

    def _layer1_physics(
        self, inputs: WeldInputs, upper_mat: dict, lower_mat: dict, combo: dict,
    ) -> dict[str, Any]:
        z1 = upper_mat.get("acoustic_impedance", 17e6)
        z2 = lower_mat.get("acoustic_impedance", 41e6)
        impedance_efficiency = self._physics.acoustic_impedance_match(z1, z2)

        friction_coeff = combo.get("friction_coefficient", 0.3)
        combo_key = f"{inputs.upper_material.material_type}-{inputs.lower_material.material_type}"
        base = BASE_PARAMS.get(combo_key, DEFAULT_BASE)

        pd = self._physics.interface_power_density(
            frequency_hz=inputs.frequency_khz * 1000,
            amplitude_um=base["amplitude_um"],
            pressure_mpa=base["pressure_mpa"],
            friction_coeff=friction_coeff,
            contact_area_mm2=inputs.weld_area_mm2,
        )

        energy_ratios = self._physics.multilayer_energy_attenuation(
            n_layers=inputs.upper_material.layers,
            material_impedance=z1,
            layer_thickness_mm=inputs.upper_material.thickness_mm,
        )
        bottom_energy_ratio = energy_ratios[-1] if energy_ratios else 1.0

        delta_t = self._physics.interface_temperature_rise(
            power_density_w_mm2=pd,
            weld_time_s=base["time_ms"] / 1000.0,
            thermal_conductivity_1=upper_mat.get("thermal_conductivity", 200),
            thermal_conductivity_2=lower_mat.get("thermal_conductivity", 200),
            density_1=upper_mat.get("density_kg_m3", 5000),
            density_2=lower_mat.get("density_kg_m3", 5000),
            specific_heat_1=upper_mat.get("specific_heat_j_kg_k", 500),
            specific_heat_2=lower_mat.get("specific_heat_j_kg_k", 500),
        )

        return {
            "impedance_efficiency": impedance_efficiency,
            "power_density_w_mm2": pd,
            "bottom_energy_ratio": bottom_energy_ratio,
            "interface_temp_rise_c": delta_t,
            "base_params": base,
            "friction_coeff": friction_coeff,
        }

    def _layer2_empirical(
        self, inputs: WeldInputs, physics: dict, combo: dict,
    ) -> dict[str, Any]:
        base = dict(physics["base_params"])
        n_layers = inputs.upper_material.layers

        # Multi-layer correction: more layers need more energy and amplitude
        layer_factor = 1.0 + 0.008 * max(n_layers - 1, 0)
        base["amplitude_um"] *= min(layer_factor, 1.6)
        base["energy_density_j_mm2"] *= min(layer_factor, 2.0)
        base["time_ms"] *= min(layer_factor, 1.8)

        # Area correction: larger area needs more pressure
        area = inputs.weld_area_mm2
        if area > 100:
            area_factor = math.sqrt(area / 100)
            base["pressure_mpa"] *= min(area_factor, 1.5)

        # Impedance mismatch correction
        eff = physics["impedance_efficiency"]
        if eff < 0.8:
            base["amplitude_um"] *= 1 + 0.2 * (1 - eff)
            base["energy_density_j_mm2"] *= 1 + 0.15 * (1 - eff)

        # IMC risk correction for Cu-Al
        if combo.get("imc_risk") == "high":
            max_temp = combo.get("max_interface_temp_c", 300)
            if physics["interface_temp_rise_c"] > max_temp * 0.7:
                base["energy_density_j_mm2"] *= 0.85
                base["time_ms"] *= 0.9

        # Clamp amplitude to reasonable range
        base["amplitude_um"] = max(15, min(base["amplitude_um"], 60))
        base["pressure_mpa"] = max(0.1, min(base["pressure_mpa"], 0.8))
        base["time_ms"] = max(50, min(base["time_ms"], 800))

        return base

    def _layer3_output(
        self, inputs: WeldInputs, params: dict, physics: dict,
    ) -> WeldRecipe:
        area = inputs.weld_area_mm2
        energy_j = params["energy_density_j_mm2"] * area
        pressure_n = params["pressure_mpa"] * area

        parameters = {
            "amplitude_um": round(params["amplitude_um"], 1),
            "pressure_n": round(pressure_n, 0),
            "pressure_mpa": round(params["pressure_mpa"], 3),
            "energy_j": round(energy_j, 1),
            "time_ms": round(params["time_ms"]),
            "frequency_khz": inputs.frequency_khz,
            "control_mode": "energy",
        }

        amp = parameters["amplitude_um"]
        safety_window = {
            "amplitude_um": [round(amp * 0.85, 1), round(amp * 1.15, 1)],
            "pressure_n": [round(pressure_n * 0.8, 0), round(pressure_n * 1.2, 0)],
            "energy_j": [round(energy_j * 0.8, 1), round(energy_j * 1.3, 1)],
            "time_ms": [round(params["time_ms"] * 0.7), round(params["time_ms"] * 1.5)],
        }

        # Risk assessment
        risk = self._assess_risk(inputs, parameters, physics)

        recommendations = self._generate_recommendations(inputs, parameters, physics, risk)

        return WeldRecipe(
            recipe_id=uuid.uuid4().hex[:12],
            application=inputs.application,
            inputs={
                "upper_material": inputs.upper_material.material_type,
                "upper_thickness_mm": inputs.upper_material.thickness_mm,
                "upper_layers": inputs.upper_material.layers,
                "lower_material": inputs.lower_material.material_type,
                "lower_thickness_mm": inputs.lower_material.thickness_mm,
                "weld_area_mm2": area,
                "frequency_khz": inputs.frequency_khz,
            },
            parameters=parameters,
            safety_window=safety_window,
            risk_assessment=risk,
            recommendations=recommendations,
        )

    def _assess_risk(
        self, inputs: WeldInputs, params: dict, physics: dict,
    ) -> dict[str, Any]:
        risks = {}

        # Overweld risk: high energy + high temp
        temp = physics["interface_temp_rise_c"]
        upper_melt = 660 if inputs.upper_material.material_type == "Al" else 1085
        temp_ratio = temp / upper_melt
        if temp_ratio > 0.5:
            risks["overweld_risk"] = "high"
        elif temp_ratio > 0.3:
            risks["overweld_risk"] = "medium"
        else:
            risks["overweld_risk"] = "low"

        # Underweld risk: low bottom energy ratio
        ber = physics["bottom_energy_ratio"]
        if ber < 0.3:
            risks["underweld_risk"] = "high"
        elif ber < 0.5:
            risks["underweld_risk"] = "medium"
        else:
            risks["underweld_risk"] = "low"

        # Perforation risk
        collapse = self._physics.estimate_collapse_um(
            amplitude_um=params["amplitude_um"],
            pressure_mpa=params.get("pressure_mpa", 0.3),
            weld_time_s=params["time_ms"] / 1000,
            n_layers=inputs.upper_material.layers,
            material_yield_mpa=35 if inputs.upper_material.material_type == "Al" else 70,
        )
        total_thickness_um = inputs.upper_material.total_thickness_mm * 1000
        if total_thickness_um > 0 and collapse / total_thickness_um > 0.4:
            risks["perforation_risk"] = "high"
        elif total_thickness_um > 0 and collapse / total_thickness_um > 0.2:
            risks["perforation_risk"] = "medium"
        else:
            risks["perforation_risk"] = "low"

        return risks

    def _generate_recommendations(
        self, inputs: WeldInputs, params: dict, physics: dict, risks: dict,
    ) -> list[str]:
        recs = []
        if risks.get("overweld_risk") == "high":
            recs.append("Overweld risk is high. Consider reducing energy or weld time.")
        if risks.get("underweld_risk") == "high":
            recs.append("Bottom layers may not bond well. Consider increasing amplitude.")
        if risks.get("perforation_risk") in ("medium", "high"):
            recs.append("Perforation risk detected. Reduce pressure or amplitude.")
        if inputs.upper_material.layers > 30:
            recs.append(f"High layer count ({inputs.upper_material.layers}). Recommend trial welding with progressive energy.")
        recs.append("Start trial welding at 80% of recommended energy, then increase in 5% steps.")
        return recs
```

**Step 4: Run test**

Run: `pytest tests/test_plugins/test_li_battery_calculator.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/li_battery/calculator.py tests/test_plugins/test_li_battery_calculator.py
git commit -m "feat: li-battery 3-layer parameter calculator with risk assessment"
```

---

## Task 12: Three Validators

**Files:**
- Create: `ultrasonic_weld_master/plugins/li_battery/validators.py`
- Create: `tests/test_plugins/test_validators.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_validators.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.li_battery.validators import (
    PhysicsValidator, SafetyValidator, ConsistencyValidator, validate_recipe,
)

def _make_recipe(**overrides):
    defaults = {
        "recipe_id": "test",
        "application": "li_battery_tab",
        "inputs": {
            "upper_material": "Al", "upper_layers": 40,
            "upper_thickness_mm": 0.012, "lower_material": "Cu",
            "weld_area_mm2": 125.0,
        },
        "parameters": {
            "amplitude_um": 30.0, "pressure_n": 200.0, "pressure_mpa": 0.3,
            "energy_j": 60.0, "time_ms": 200, "frequency_khz": 20.0,
        },
        "safety_window": {},
        "risk_assessment": {"overweld_risk": "low", "underweld_risk": "low", "perforation_risk": "low"},
    }
    defaults.update(overrides)
    return WeldRecipe(**defaults)

class TestPhysicsValidator:
    def test_valid_recipe_passes(self):
        recipe = _make_recipe()
        v = PhysicsValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

    def test_extreme_amplitude_fails(self):
        recipe = _make_recipe(parameters={
            "amplitude_um": 100.0, "pressure_n": 200.0, "pressure_mpa": 0.3,
            "energy_j": 60.0, "time_ms": 200, "frequency_khz": 20.0,
        })
        v = PhysicsValidator()
        result = v.validate(recipe)
        assert result["status"] in ("warning", "fail")

class TestSafetyValidator:
    def test_valid_recipe_passes(self):
        recipe = _make_recipe()
        v = SafetyValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

    def test_high_risk_warns(self):
        recipe = _make_recipe(risk_assessment={
            "overweld_risk": "high", "underweld_risk": "low", "perforation_risk": "high",
        })
        v = SafetyValidator()
        result = v.validate(recipe)
        assert result["status"] in ("warning", "fail")

class TestConsistencyValidator:
    def test_consistent_recipe_passes(self):
        recipe = _make_recipe()
        v = ConsistencyValidator()
        result = v.validate(recipe)
        assert result["status"] == "pass"

class TestValidateRecipe:
    def test_full_validation(self):
        recipe = _make_recipe()
        result = validate_recipe(recipe)
        assert isinstance(result, ValidationResult)
        assert "physics" in result.validators
        assert "safety" in result.validators
        assert "consistency" in result.validators
```

**Step 2: Run test**

Run: `pytest tests/test_plugins/test_validators.py -v`

**Step 3: Write implementation**

`ultrasonic_weld_master/plugins/li_battery/validators.py`:
```python
"""Three-validator system for welding parameter verification."""
from __future__ import annotations

from typing import Any
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus


class PhysicsValidator:
    """Validates physical reasonableness of parameters."""

    AMPLITUDE_RANGE = (10, 70)  # um
    PRESSURE_RANGE = (0.05, 1.0)  # MPa
    POWER_DENSITY_RANGE = (0.1, 10.0)  # W/mm2
    FREQ_OPTIONS = (15, 20, 30, 35, 40)  # kHz

    def validate(self, recipe: WeldRecipe) -> dict[str, Any]:
        messages = []
        status = "pass"
        p = recipe.parameters

        amp = p.get("amplitude_um", 0)
        if not (self.AMPLITUDE_RANGE[0] <= amp <= self.AMPLITUDE_RANGE[1]):
            messages.append(f"Amplitude {amp} um outside range {self.AMPLITUDE_RANGE}")
            status = "fail"
        elif amp > 55:
            messages.append(f"Amplitude {amp} um is high, risk of material damage")
            status = "warning"

        pres = p.get("pressure_mpa", 0)
        if not (self.PRESSURE_RANGE[0] <= pres <= self.PRESSURE_RANGE[1]):
            messages.append(f"Pressure {pres} MPa outside range {self.PRESSURE_RANGE}")
            status = "fail"

        area = recipe.inputs.get("weld_area_mm2", 1)
        energy = p.get("energy_j", 0)
        if area > 0:
            energy_density = energy / area
            if not (0.05 <= energy_density <= 5.0):
                messages.append(f"Energy density {energy_density:.2f} J/mm2 outside normal range")
                status = "warning" if status == "pass" else status

        return {"status": status, "messages": messages}


class SafetyValidator:
    """Validates process safety constraints (especially for Li-battery)."""

    def validate(self, recipe: WeldRecipe) -> dict[str, Any]:
        messages = []
        status = "pass"
        risks = recipe.risk_assessment

        high_risks = [k for k, v in risks.items() if v == "high"]
        if len(high_risks) >= 2:
            messages.append(f"Multiple high risks: {high_risks}")
            status = "fail"
        elif high_risks:
            messages.append(f"High risk detected: {high_risks}")
            status = "warning"

        # Li-battery specific: check for thermal safety
        if risks.get("overweld_risk") == "high":
            messages.append("Overweld risk high: possible separator damage in li-battery")
            status = "fail" if status != "fail" else status

        if risks.get("perforation_risk") == "high":
            messages.append("Perforation risk high: foil stack may be punctured")
            if status == "pass":
                status = "warning"

        return {"status": status, "messages": messages}


class ConsistencyValidator:
    """Validates parameter internal consistency."""

    def validate(self, recipe: WeldRecipe) -> dict[str, Any]:
        messages = []
        status = "pass"
        p = recipe.parameters

        # Energy and time should be consistent
        energy = p.get("energy_j", 0)
        time_ms = p.get("time_ms", 1)
        if time_ms > 0:
            implied_power = energy / (time_ms / 1000)
            max_power = recipe.inputs.get("max_power_w", 5000)
            if max_power and implied_power > max_power:
                messages.append(
                    f"Implied power {implied_power:.0f}W exceeds max {max_power}W"
                )
                status = "warning"

        # Pressure_n and pressure_mpa should match with area
        area = recipe.inputs.get("weld_area_mm2", 0)
        if area > 0:
            p_n = p.get("pressure_n", 0)
            p_mpa = p.get("pressure_mpa", 0)
            expected_n = p_mpa * area
            if p_n > 0 and abs(p_n - expected_n) / max(p_n, 1) > 0.1:
                messages.append(
                    f"Pressure inconsistency: {p_n}N vs {expected_n:.0f}N from {p_mpa}MPa * {area}mm2"
                )
                status = "warning" if status == "pass" else status

        return {"status": status, "messages": messages}


def validate_recipe(recipe: WeldRecipe) -> ValidationResult:
    """Run all three validators and return combined result."""
    validators = {
        "physics": PhysicsValidator(),
        "safety": SafetyValidator(),
        "consistency": ConsistencyValidator(),
    }
    results = {}
    overall = ValidationStatus.PASS

    for name, validator in validators.items():
        result = validator.validate(recipe)
        results[name] = result
        if result["status"] == "fail":
            overall = ValidationStatus.FAIL
        elif result["status"] == "warning" and overall != ValidationStatus.FAIL:
            overall = ValidationStatus.WARNING

    all_messages = []
    for name, r in results.items():
        for msg in r["messages"]:
            all_messages.append(f"[{name}] {msg}")

    return ValidationResult(
        status=overall,
        validators=results,
        messages=all_messages,
    )
```

**Step 4: Run test**

Run: `pytest tests/test_plugins/test_validators.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/li_battery/validators.py tests/test_plugins/test_validators.py
git commit -m "feat: three-validator system (physics, safety, consistency)"
```

---

## Task 13: Li-Battery Plugin Assembly

**Files:**
- Create: `ultrasonic_weld_master/plugins/li_battery/plugin.py`
- Create: `ultrasonic_weld_master/plugins/li_battery/config.yaml`
- Create: `tests/test_plugins/test_li_battery_plugin.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_li_battery_plugin.py`:
```python
from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.core.event_bus import EventBus

class TestLiBatteryPlugin:
    @pytest.fixture
    def plugin(self):
        mat_db = MaterialDBPlugin()
        mat_db.activate({})
        event_bus = EventBus()
        p = LiBatteryPlugin()
        p.activate({"config": None, "event_bus": event_bus, "logger": None, "material_db": mat_db})
        return p

    def test_get_info(self):
        p = LiBatteryPlugin()
        info = p.get_info()
        assert info.name == "li_battery"

    def test_supported_applications(self, plugin):
        apps = plugin.get_supported_applications()
        assert "li_battery_tab" in apps
        assert "li_battery_busbar" in apps

    def test_calculate_and_validate(self, plugin):
        inputs = {
            "application": "li_battery_tab",
            "upper_material_type": "Al",
            "upper_thickness_mm": 0.012,
            "upper_layers": 40,
            "lower_material_type": "Cu",
            "lower_thickness_mm": 0.3,
            "weld_width_mm": 5.0,
            "weld_length_mm": 25.0,
            "frequency_khz": 20.0,
            "max_power_w": 3500,
        }
        recipe = plugin.calculate_parameters(inputs)
        assert isinstance(recipe, WeldRecipe)
        result = plugin.validate_parameters(recipe)
        assert isinstance(result, ValidationResult)
```

**Step 2: Run test**

Run: `pytest tests/test_plugins/test_li_battery_plugin.py -v`

**Step 3: Write plugin.py**

`ultrasonic_weld_master/plugins/li_battery/plugin.py`:
```python
"""Li-battery welding parameter engine plugin."""
from __future__ import annotations

from typing import Any

from ultrasonic_weld_master.core.plugin_api import ParameterEnginePlugin, PluginInfo
from ultrasonic_weld_master.core.models import (
    WeldInputs, WeldRecipe, MaterialInfo, ValidationResult,
)
from ultrasonic_weld_master.plugins.li_battery.calculator import LiBatteryCalculator
from ultrasonic_weld_master.plugins.li_battery.validators import validate_recipe


class LiBatteryPlugin(ParameterEnginePlugin):
    def __init__(self):
        self._calculator: LiBatteryCalculator = None
        self._material_db = None
        self._event_bus = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="li_battery",
            version="1.0.0",
            description="Lithium battery ultrasonic welding parameter engine",
            author="UltrasonicWeldMaster",
            dependencies=["material_db"],
        )

    def activate(self, context: Any) -> None:
        self._material_db = context.get("material_db") if isinstance(context, dict) else None
        self._event_bus = context.get("event_bus") if isinstance(context, dict) else None
        if self._material_db:
            self._calculator = LiBatteryCalculator(material_db=self._material_db)

    def deactivate(self) -> None:
        self._calculator = None

    def get_input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "application": {"type": "string", "enum": self.get_supported_applications()},
                "upper_material_type": {"type": "string"},
                "upper_thickness_mm": {"type": "number", "minimum": 0.001},
                "upper_layers": {"type": "integer", "minimum": 1, "maximum": 200},
                "lower_material_type": {"type": "string"},
                "lower_thickness_mm": {"type": "number", "minimum": 0.01},
                "weld_width_mm": {"type": "number", "minimum": 1},
                "weld_length_mm": {"type": "number", "minimum": 1},
                "frequency_khz": {"type": "number", "default": 20},
                "max_power_w": {"type": "number", "default": 3500},
            },
            "required": ["application", "upper_material_type", "upper_thickness_mm",
                         "upper_layers", "lower_material_type", "lower_thickness_mm",
                         "weld_width_mm", "weld_length_mm"],
        }

    def calculate_parameters(self, inputs: dict) -> WeldRecipe:
        weld_inputs = WeldInputs(
            application=inputs["application"],
            upper_material=MaterialInfo(
                name=inputs["upper_material_type"],
                material_type=inputs["upper_material_type"],
                thickness_mm=inputs["upper_thickness_mm"],
                layers=inputs.get("upper_layers", 1),
            ),
            lower_material=MaterialInfo(
                name=inputs["lower_material_type"],
                material_type=inputs["lower_material_type"],
                thickness_mm=inputs["lower_thickness_mm"],
                layers=1,
            ),
            weld_width_mm=inputs["weld_width_mm"],
            weld_length_mm=inputs["weld_length_mm"],
            frequency_khz=inputs.get("frequency_khz", 20.0),
            max_power_w=inputs.get("max_power_w", 3500),
        )
        recipe = self._calculator.calculate(weld_inputs)
        if self._event_bus:
            self._event_bus.emit("calculation.completed", recipe.to_dict())
        return recipe

    def validate_parameters(self, recipe: WeldRecipe) -> ValidationResult:
        result = validate_recipe(recipe)
        if self._event_bus:
            self._event_bus.emit("validation.completed", result.to_dict())
        return result

    def get_supported_applications(self) -> list[str]:
        return ["li_battery_tab", "li_battery_busbar", "li_battery_collector"]
```

**Step 4: Run test**

Run: `pytest tests/test_plugins/test_li_battery_plugin.py -v`

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/li_battery/plugin.py tests/test_plugins/test_li_battery_plugin.py
git commit -m "feat: li-battery parameter engine plugin with calculate and validate"
```

---

## Task 14: General Metal Welding Plugin

**Files:**
- Create: `ultrasonic_weld_master/plugins/general_metal/__init__.py`
- Create: `ultrasonic_weld_master/plugins/general_metal/plugin.py`
- Create: `ultrasonic_weld_master/plugins/general_metal/calculator.py`
- Create: `tests/test_plugins/test_general_metal.py`

This task follows the same pattern as Task 11-13 but with simplified calculations for arbitrary material combinations. The calculator uses the same physics models but with more conservative defaults.

**Step 1: Write test, Step 2: Verify fail, Step 3: Implement, Step 4: Verify pass, Step 5: Commit**

```bash
git commit -m "feat: general metal welding plugin"
```

---

## Task 15: Knowledge Base Plugin

**Files:**
- Create: `ultrasonic_weld_master/plugins/knowledge_base/__init__.py`
- Create: `ultrasonic_weld_master/plugins/knowledge_base/plugin.py`
- Create: `ultrasonic_weld_master/plugins/knowledge_base/rules/li_battery_rules.yaml`
- Create: `tests/test_plugins/test_knowledge_base.py`

The knowledge base stores process rules as YAML. Rules contain conditions and adjustments.

**Step 1: Write test, Step 2: Verify fail, Step 3: Implement, Step 4: Verify pass, Step 5: Commit**

```bash
git commit -m "feat: knowledge base plugin with YAML rules"
```

---

## Task 16: JSON Report Exporter

**Files:**
- Create: `ultrasonic_weld_master/plugins/reporter/__init__.py`
- Create: `ultrasonic_weld_master/plugins/reporter/json_exporter.py`
- Create: `tests/test_plugins/test_json_exporter.py`

**Step 1: Write the failing test**

`tests/test_plugins/test_json_exporter.py`:
```python
from __future__ import annotations
import json
import os
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.reporter.json_exporter import JsonExporter

class TestJsonExporter:
    def test_export(self, tmp_path):
        recipe = WeldRecipe(
            recipe_id="R001", application="li_battery_tab",
            inputs={"material": "Cu"}, parameters={"amplitude_um": 30.0},
            safety_window={"amplitude_um": [25, 35]},
            risk_assessment={"overweld_risk": "low"},
            recommendations=["Start at 80% energy"],
        )
        validation = ValidationResult(
            status=ValidationStatus.PASS, validators={"physics": {"status": "pass"}},
        )
        exporter = JsonExporter()
        path = exporter.export(recipe, validation, str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["recipe"]["recipe_id"] == "R001"
        assert data["validation"]["status"] == "pass"
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: JSON report exporter"
```

---

## Task 17: Excel Report Generator

**Files:**
- Create: `ultrasonic_weld_master/plugins/reporter/excel_generator.py`
- Create: `tests/test_plugins/test_excel_generator.py`

Uses openpyxl to create professional Excel reports with multiple sheets.

**Step 1-5: Test, implement, verify, commit**

```bash
git commit -m "feat: Excel report generator with openpyxl"
```

---

## Task 18: PDF Report Generator

**Files:**
- Create: `ultrasonic_weld_master/plugins/reporter/pdf_generator.py`
- Create: `tests/test_plugins/test_pdf_generator.py`

Uses ReportLab to create professional PDF reports following the 7-section structure from the design doc.

**Step 1-5: Test, implement, verify, commit**

```bash
git commit -m "feat: PDF report generator with ReportLab"
```

---

## Task 19: Reporter Plugin Assembly

**Files:**
- Create: `ultrasonic_weld_master/plugins/reporter/plugin.py`
- Create: `tests/test_plugins/test_reporter_plugin.py`

Assembles all three exporters into one plugin that generates all report formats.

**Step 1-5: Test, implement, verify, commit**

```bash
git commit -m "feat: reporter plugin supporting PDF/Excel/JSON"
```

---

## Task 20: CLI Interface

**Files:**
- Modify: `cli.py`
- Create: `tests/test_cli.py`

Create a command-line interface that can run parameter calculations without GUI.

**Step 1: Write test**

```python
import subprocess
def test_cli_version():
    result = subprocess.run(["python", "cli.py", "--version"], capture_output=True, text=True)
    assert "0.1.0" in result.stdout

def test_cli_calculate():
    result = subprocess.run([
        "python", "cli.py", "calculate",
        "--application", "li_battery_tab",
        "--upper-material", "Al", "--upper-thickness", "0.012", "--upper-layers", "40",
        "--lower-material", "Cu", "--lower-thickness", "0.3",
        "--width", "5", "--length", "25",
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert "amplitude" in result.stdout.lower()
```

**Step 2-5: Implement, test, commit**

```bash
git commit -m "feat: CLI interface for headless parameter calculation"
```

---

## Task 21: GUI Main Window

**Files:**
- Create: `ultrasonic_weld_master/gui/main_window.py`
- Create: `ultrasonic_weld_master/gui/panels/__init__.py`
- Modify: `main.py`

PySide6 main window with left navigation bar, central workspace, and status bar.

**Step 1-5: Implement, verify visually, commit**

```bash
git commit -m "feat: GUI main window with navigation and status bar"
```

---

## Task 22: Input Wizard (4-Step)

**Files:**
- Create: `ultrasonic_weld_master/gui/panels/input_wizard.py`

4-step wizard: Application Selection → Materials → Tooling → Constraints.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: 4-step input wizard for welding parameters"
```

---

## Task 23: Result Panel & Chart Widget

**Files:**
- Create: `ultrasonic_weld_master/gui/panels/result_panel.py`
- Create: `ultrasonic_weld_master/gui/widgets/chart_widget.py`

Display calculation results with parameter tables, safety windows, and risk assessment visualization.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: result panel with charts and risk visualization"
```

---

## Task 24: Report Preview Panel

**Files:**
- Create: `ultrasonic_weld_master/gui/panels/report_panel.py`

Preview generated reports and trigger PDF/Excel/JSON export.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: report preview and export panel"
```

---

## Task 25: History & Project Panels

**Files:**
- Create: `ultrasonic_weld_master/gui/panels/history_panel.py`
- Create: `ultrasonic_weld_master/gui/panels/project_panel.py`

Browse past calculation sessions, compare recipes, manage projects.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: history and project management panels"
```

---

## Task 26: Material & Sonotrode Library Panels

**Files:**
- Create: `ultrasonic_weld_master/gui/widgets/material_selector.py`
- Create: `ultrasonic_weld_master/gui/widgets/sonotrode_editor.py`
- Create: `ultrasonic_weld_master/gui/panels/settings_panel.py`

Library management for materials and sonotrodes, plus settings.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: material/sonotrode libraries and settings panel"
```

---

## Task 27: Theme System

**Files:**
- Create: `ultrasonic_weld_master/gui/themes.py`
- Create: `ultrasonic_weld_master/gui/resources/styles/light.qss`
- Create: `ultrasonic_weld_master/gui/resources/styles/dark.qss`

Light/dark theme toggle with professional engineering software aesthetics.

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "feat: light/dark theme system"
```

---

## Task 28: Full Integration Test

**Files:**
- Create: `tests/test_integration.py`

End-to-end test: create project → input parameters → calculate → validate → generate reports.

```python
def test_full_workflow(tmp_path):
    engine = Engine(data_dir=str(tmp_path))
    engine.initialize()
    # Register plugins
    # Create project, session
    # Calculate parameters
    # Validate
    # Generate all 3 report formats
    # Verify files exist
    engine.shutdown()
```

**Step 1-5: Implement, verify, commit**

```bash
git commit -m "test: full integration test for complete workflow"
```

---

## Task 29: Documentation & Version

**Files:**
- Create: `CHANGELOG.md`
- Create: `docs/versions/v0.1.0-release-notes.md`

**Step 1-5: Write, commit**

```bash
git commit -m "docs: CHANGELOG and v0.1.0 release notes"
```

---

## Task 30: Run All Tests & Final Verification

**Step 1:** Run full test suite

```bash
pytest tests/ -v --tb=short
```

**Step 2:** Run the application

```bash
python main.py
```

**Step 3:** Verify CLI works

```bash
python cli.py calculate --application li_battery_tab \
  --upper-material Al --upper-thickness 0.012 --upper-layers 40 \
  --lower-material Cu --lower-thickness 0.3 \
  --width 5 --length 25 --output-dir reports/
```

**Step 4:** Final commit

```bash
git commit -m "chore: v0.1.0 release ready"
```
