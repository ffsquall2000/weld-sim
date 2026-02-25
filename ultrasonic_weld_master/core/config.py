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
        self._data: dict = {}
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
    def data(self) -> dict:
        return self._data
