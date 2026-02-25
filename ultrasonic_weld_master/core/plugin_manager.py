"""Plugin lifecycle manager."""
from __future__ import annotations

import logging
from typing import Any, Optional

from ultrasonic_weld_master.core.plugin_api import PluginBase
from ultrasonic_weld_master.core.event_bus import EventBus
from ultrasonic_weld_master.core.logger import StructuredLogger
from ultrasonic_weld_master.core.config import AppConfig

logger = logging.getLogger(__name__)


class PluginManager:
    def __init__(self, config: AppConfig, event_bus: EventBus, logger: StructuredLogger):
        self._config = config
        self._event_bus = event_bus
        self._logger = logger
        self._registered: dict = {}
        self._active: set = set()

    def register(self, plugin: PluginBase) -> None:
        info = plugin.get_info()
        self._registered[info.name] = plugin

    def activate(self, name: str) -> None:
        if name not in self._registered:
            raise ValueError(f"Plugin '{name}' not registered")
        plugin = self._registered[name]
        info = plugin.get_info()
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
        # Provide activated dependency plugins in context
        for dep in info.dependencies:
            if dep in self._registered:
                context[dep] = self._registered[dep]
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

    def list_plugins(self) -> list:
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
