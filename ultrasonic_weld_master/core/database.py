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

    def get_project(self, project_id: str) -> dict:
        row = self.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not row:
            raise ValueError(f"Project {project_id} not found")
        d = dict(row)
        d["config"] = self._json_loads(d["config"])
        return d

    def list_projects(self) -> list:
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

    def get_session(self, session_id: str) -> dict:
        row = self.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            raise ValueError(f"Session {session_id} not found")
        return dict(row)

    def end_session(self, session_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (now, session_id))
        self._conn.commit()

    # --- Operations ---
    def log_operation(self, session_id: str, event_type: str,
                      user_action: str = "", data: Optional[dict] = None,
                      metadata: Optional[dict] = None) -> None:
        self.execute(
            "INSERT INTO operations (session_id, event_type, user_action, data, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, event_type, user_action,
             self._json_dumps(data or {}), self._json_dumps(metadata or {})),
        )
        self._conn.commit()

    def get_operations(self, session_id: str) -> list:
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
    def save_recipe(self, project_id: str, session_id: str, application: str,
                    inputs: dict, parameters: dict,
                    safety_window: Optional[dict] = None,
                    validation_result: Optional[dict] = None,
                    risk_assessment: Optional[dict] = None,
                    notes: str = "") -> str:
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

    def get_recipe(self, recipe_id: str) -> dict:
        row = self.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        if not row:
            raise ValueError(f"Recipe {recipe_id} not found")
        d = dict(row)
        for key in ("inputs", "parameters", "safety_window", "validation_result", "risk_assessment"):
            d[key] = self._json_loads(d[key])
        return d

    def list_recipes(self, project_id: str) -> list:
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

    def list_materials(self, material_type: str = "") -> list:
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
