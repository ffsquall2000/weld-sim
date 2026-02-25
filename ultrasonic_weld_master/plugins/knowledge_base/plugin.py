"""Knowledge base plugin with YAML process rules."""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo


class KnowledgeBasePlugin(PluginBase):
    def __init__(self):
        self._rules: list = []

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="knowledge_base", version="1.0.0",
            description="Process knowledge base with YAML rules for parameter adjustment",
            author="UltrasonicWeldMaster", dependencies=[],
        )

    def activate(self, context: Any) -> None:
        rules_dir = os.path.join(os.path.dirname(__file__), "rules")
        self._rules = []
        if os.path.isdir(rules_dir):
            for fname in sorted(os.listdir(rules_dir)):
                if fname.endswith(".yaml") or fname.endswith(".yml"):
                    path = os.path.join(rules_dir, fname)
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data and "rules" in data:
                        self._rules.extend(data["rules"])

    def deactivate(self) -> None:
        self._rules.clear()

    def get_rules(self) -> list:
        return list(self._rules)

    def get_rule_by_id(self, rule_id: str) -> Optional[dict]:
        for rule in self._rules:
            if rule.get("id") == rule_id:
                return rule
        return None

    def evaluate_rules(self, context: dict) -> list:
        """Evaluate rules against a context dict and return matching rules with adjustments."""
        matched = []
        for rule in self._rules:
            if self._check_conditions(rule.get("conditions", {}), context):
                matched.append({
                    "rule_id": rule["id"],
                    "description": rule.get("description", ""),
                    "adjustments": rule.get("adjustments", {}),
                    "recommendations": rule.get("recommendations", []),
                })
        return matched

    def _check_conditions(self, conditions: dict, context: dict) -> bool:
        for key, value in conditions.items():
            if key.endswith("_gt"):
                field = key[:-3]
                ctx_val = context.get(field, 0)
                if not (ctx_val > value):
                    return False
            elif key.endswith("_lt"):
                field = key[:-3]
                ctx_val = context.get(field, float("inf"))
                if not (ctx_val < value):
                    return False
            elif key == "material_combo":
                combo = context.get("material_combo", "")
                if combo not in value:
                    return False
            elif key == "application":
                app = context.get("application", "")
                if app not in value:
                    return False
            else:
                if context.get(key) != value:
                    return False
        return True
