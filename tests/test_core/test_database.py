from __future__ import annotations
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
        rid = db.save_recipe(project_id=pid, session_id=sid, application="li_battery_tab",
                             inputs={"material": "Cu"}, parameters={"amplitude_um": 30.0})
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
