"""Material database plugin."""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo


class MaterialDBPlugin(PluginBase):
    def __init__(self):
        self._materials: dict = {}
        self._combinations: dict = {}

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="material_db", version="1.0.0",
            description="Material properties database for ultrasonic welding",
            author="UltrasonicWeldMaster", dependencies=[],
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

    def get_material(self, material_type: str) -> Optional[dict]:
        return self._materials.get(material_type)

    def list_materials(self) -> list:
        return list(self._materials.keys())

    def get_combination_properties(self, mat_a: str, mat_b: str) -> dict:
        key = f"{mat_a}-{mat_b}"
        if key in self._combinations:
            return self._combinations[key]
        reverse_key = f"{mat_b}-{mat_a}"
        if reverse_key in self._combinations:
            return self._combinations[reverse_key]
        return {}
