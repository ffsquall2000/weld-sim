from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.knowledge_base.plugin import KnowledgeBasePlugin

class TestKnowledgeBasePlugin:
    @pytest.fixture
    def plugin(self):
        p = KnowledgeBasePlugin()
        p.activate({})
        return p

    def test_get_info(self):
        p = KnowledgeBasePlugin()
        info = p.get_info()
        assert info.name == "knowledge_base"

    def test_rules_loaded(self, plugin):
        rules = plugin.get_rules()
        assert len(rules) >= 5

    def test_get_rule_by_id(self, plugin):
        rule = plugin.get_rule_by_id("high_layer_count")
        assert rule is not None
        assert "adjustments" in rule
        assert rule["adjustments"]["amplitude_factor"] > 1.0

    def test_get_unknown_rule(self, plugin):
        rule = plugin.get_rule_by_id("nonexistent")
        assert rule is None

    def test_evaluate_high_layer_count(self, plugin):
        context = {
            "application": "li_battery_tab",
            "upper_layers": 50,
            "upper_thickness_mm": 0.012,
        }
        matched = plugin.evaluate_rules(context)
        rule_ids = [m["rule_id"] for m in matched]
        assert "high_layer_count" in rule_ids

    def test_evaluate_cu_al_combo(self, plugin):
        context = {
            "application": "li_battery_tab",
            "material_combo": "Cu-Al",
            "upper_layers": 10,
        }
        matched = plugin.evaluate_rules(context)
        rule_ids = [m["rule_id"] for m in matched]
        assert "cu_al_imc_prevention" in rule_ids

    def test_evaluate_no_match(self, plugin):
        context = {
            "application": "other",
            "upper_layers": 5,
            "upper_thickness_mm": 0.5,
        }
        matched = plugin.evaluate_rules(context)
        assert len(matched) == 0
