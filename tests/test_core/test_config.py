from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.config import AppConfig

class TestConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.get("app.name") == "UltrasonicWeldMaster"
        assert config.get("app.version") == "0.1.0"

    def test_load_from_file(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("app:\n  name: TestApp\n  version: '9.9.9'\n")
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
