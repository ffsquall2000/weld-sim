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
