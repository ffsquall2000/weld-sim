"""Knowledge-based rule engine for parameter adjustment recommendations.

Standalone version extracted from the knowledge_base plugin.
Loads YAML rules and evaluates them against simulation/optimization context.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# Default rules directory (relative to original project root)
_DEFAULT_RULES_DIR = Path(__file__).resolve().parents[3] / "ultrasonic_weld_master" / "plugins" / "knowledge_base" / "rules"


class KnowledgeRuleEngine:
    """Evaluate domain knowledge rules against simulation context."""

    def __init__(self, rules_dir: str | Path | None = None):
        self._rules: list[dict] = []
        self._rules_dir = Path(rules_dir) if rules_dir else _DEFAULT_RULES_DIR
        self._load_rules()

    def _load_rules(self) -> None:
        """Load all YAML rule files from the rules directory."""
        if not self._rules_dir.is_dir():
            logger.warning("Rules directory not found: %s", self._rules_dir)
            return

        for fname in sorted(os.listdir(self._rules_dir)):
            if fname.endswith((".yaml", ".yml")):
                path = self._rules_dir / fname
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                    if data and "rules" in data:
                        self._rules.extend(data["rules"])
                        logger.info("Loaded %d rules from %s", len(data["rules"]), fname)
                except Exception as e:
                    logger.error("Failed to load rules from %s: %s", fname, e)

    def get_rules(self) -> list[dict]:
        """Return all loaded rules."""
        return list(self._rules)

    def get_rule_by_id(self, rule_id: str) -> Optional[dict]:
        """Look up a rule by its ID."""
        for rule in self._rules:
            if rule.get("id") == rule_id:
                return rule
        return None

    def evaluate(self, context: dict[str, Any]) -> list[dict]:
        """Evaluate rules against a context dict.

        Args:
            context: Dictionary with keys like 'application', 'material_combo',
                     'n_layers', 'pressure_mpa', 'amplitude_um', etc.

        Returns:
            List of matched rules with their adjustments and recommendations.
        """
        matched = []
        for rule in self._rules:
            if self._check_conditions(rule.get("conditions", {}), context):
                matched.append({
                    "rule_id": rule["id"],
                    "description": rule.get("description", ""),
                    "adjustments": rule.get("adjustments", {}),
                    "recommendations": rule.get("recommendations", []),
                    "priority": rule.get("priority", 0),
                })
        # Sort by priority (higher = more important)
        matched.sort(key=lambda r: r.get("priority", 0), reverse=True)
        return matched

    def get_suggestions(self, context: dict[str, Any]) -> list[str]:
        """Get human-readable suggestions for the given context.

        Returns a flat list of recommendation strings.
        """
        matched = self.evaluate(context)
        suggestions = []
        for rule in matched:
            suggestions.extend(rule.get("recommendations", []))
        return suggestions

    @staticmethod
    def _check_conditions(conditions: dict, context: dict) -> bool:
        """Check if all conditions match the context."""
        for key, value in conditions.items():
            if key.endswith("_gt"):
                field_name = key[:-3]
                ctx_val = context.get(field_name, 0)
                if not (ctx_val > value):
                    return False
            elif key.endswith("_lt"):
                field_name = key[:-3]
                ctx_val = context.get(field_name, float("inf"))
                if not (ctx_val < value):
                    return False
            elif key.endswith("_gte"):
                field_name = key[:-4]
                ctx_val = context.get(field_name, 0)
                if not (ctx_val >= value):
                    return False
            elif key.endswith("_lte"):
                field_name = key[:-4]
                ctx_val = context.get(field_name, float("inf"))
                if not (ctx_val <= value):
                    return False
            elif key == "material_combo":
                combo = context.get("material_combo", "")
                if isinstance(value, list):
                    if combo not in value:
                        return False
                elif combo != value:
                    return False
            elif key == "application":
                app = context.get("application", "")
                if isinstance(value, list):
                    if app not in value:
                        return False
                elif app != value:
                    return False
            else:
                if context.get(key) != value:
                    return False
        return True
