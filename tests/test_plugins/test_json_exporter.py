from __future__ import annotations
import json
import os
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.reporter.json_exporter import JsonExporter

class TestJsonExporter:
    def test_export(self, tmp_path):
        recipe = WeldRecipe(
            recipe_id="R001", application="li_battery_tab",
            inputs={"material": "Cu"}, parameters={"amplitude_um": 30.0},
            safety_window={"amplitude_um": [25, 35]},
            risk_assessment={"overweld_risk": "low"},
            recommendations=["Start at 80% energy"],
        )
        validation = ValidationResult(
            status=ValidationStatus.PASS, validators={"physics": {"status": "pass"}},
        )
        exporter = JsonExporter()
        path = exporter.export(recipe, validation, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".json")
        with open(path) as f:
            data = json.load(f)
        assert data["recipe"]["recipe_id"] == "R001"
        assert data["validation"]["status"] == "pass"

    def test_export_without_validation(self, tmp_path):
        recipe = WeldRecipe(
            recipe_id="R002", application="general_metal",
            inputs={}, parameters={"amplitude_um": 25.0},
            safety_window={}, risk_assessment={},
        )
        exporter = JsonExporter()
        path = exporter.export(recipe, output_dir=str(tmp_path))
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["validation"] is None
