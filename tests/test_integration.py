"""End-to-end integration tests."""
from __future__ import annotations

import json
import os
import pytest

from ultrasonic_weld_master.core.engine import Engine
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
from ultrasonic_weld_master.plugins.knowledge_base.plugin import KnowledgeBasePlugin
from ultrasonic_weld_master.plugins.reporter.plugin import ReporterPlugin


class TestFullWorkflow:
    @pytest.fixture
    def engine(self, tmp_path):
        e = Engine(data_dir=str(tmp_path))
        e.initialize()
        # Register all plugins
        e.plugin_manager.register(MaterialDBPlugin())
        e.plugin_manager.register(LiBatteryPlugin())
        e.plugin_manager.register(GeneralMetalPlugin())
        e.plugin_manager.register(KnowledgeBasePlugin())
        e.plugin_manager.register(ReporterPlugin())
        # Activate in dependency order (plugin_manager auto-injects deps)
        e.plugin_manager.activate("material_db")
        e.plugin_manager.activate("li_battery")
        e.plugin_manager.activate("general_metal")
        e.plugin_manager.activate("knowledge_base")
        e.plugin_manager.activate("reporter")
        yield e
        e.shutdown()

    def test_li_battery_full_workflow(self, engine, tmp_path):
        """Full workflow: calculate -> validate -> generate all 3 report formats."""
        li_plugin = engine.plugin_manager.get_plugin("li_battery")
        reporter = engine.plugin_manager.get_plugin("reporter")

        inputs = {
            "application": "li_battery_tab",
            "upper_material_type": "Al", "upper_thickness_mm": 0.012, "upper_layers": 40,
            "lower_material_type": "Cu", "lower_thickness_mm": 0.3,
            "weld_width_mm": 5.0, "weld_length_mm": 25.0,
            "frequency_khz": 20.0, "max_power_w": 3500,
        }

        recipe = li_plugin.calculate_parameters(inputs)
        assert isinstance(recipe, WeldRecipe)
        assert recipe.parameters["amplitude_um"] > 0

        validation = li_plugin.validate_parameters(recipe)
        assert isinstance(validation, ValidationResult)

        report_dir = str(tmp_path / "reports")
        os.makedirs(report_dir)
        paths = reporter.export_all(recipe, validation, report_dir)

        assert os.path.exists(paths["json"])
        assert os.path.exists(paths["excel"])
        assert os.path.exists(paths["pdf"])

        with open(paths["json"]) as f:
            data = json.load(f)
        assert data["recipe"]["recipe_id"] == recipe.recipe_id

    def test_general_metal_workflow(self, engine, tmp_path):
        gm_plugin = engine.plugin_manager.get_plugin("general_metal")
        reporter = engine.plugin_manager.get_plugin("reporter")

        inputs = {
            "application": "general_metal",
            "upper_material_type": "Ni", "upper_thickness_mm": 0.1, "upper_layers": 1,
            "lower_material_type": "Cu", "lower_thickness_mm": 0.5,
            "weld_width_mm": 4.0, "weld_length_mm": 15.0,
        }

        recipe = gm_plugin.calculate_parameters(inputs)
        validation = gm_plugin.validate_parameters(recipe)

        report_dir = str(tmp_path / "reports2")
        os.makedirs(report_dir)
        json_path = reporter.export_json(recipe, validation, report_dir)
        assert os.path.exists(json_path)

    def test_knowledge_base_integration(self, engine):
        kb = engine.plugin_manager.get_plugin("knowledge_base")
        context = {
            "application": "li_battery_tab",
            "upper_layers": 50,
            "material_combo": "Cu-Al",
            "upper_thickness_mm": 0.008,
        }
        matched = kb.evaluate_rules(context)
        assert len(matched) >= 2  # high_layer_count + cu_al_imc_prevention + thin_foil

    def test_all_plugins_registered(self, engine):
        plugins = engine.plugin_manager.list_plugins()
        names = [p["name"] for p in plugins]
        assert "material_db" in names
        assert "li_battery" in names
        assert "general_metal" in names
        assert "knowledge_base" in names
        assert "reporter" in names
