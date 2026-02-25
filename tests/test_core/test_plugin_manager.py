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
        return PluginInfo(name="mock_plugin", version="1.0.0",
                          description="A mock plugin", author="Test", dependencies=[])

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
