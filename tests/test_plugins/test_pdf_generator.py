from __future__ import annotations
import os
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.reporter.pdf_generator import PdfGenerator

def _make_recipe():
    return WeldRecipe(
        recipe_id="PDF001", application="li_battery_tab",
        inputs={"upper_material": "Al", "weld_area_mm2": 125.0},
        parameters={"amplitude_um": 30.0, "pressure_n": 37.5, "pressure_mpa": 0.3,
                     "energy_j": 60.0, "time_ms": 200, "frequency_khz": 20.0},
        safety_window={"amplitude_um": [25.5, 34.5], "energy_j": [48.0, 78.0]},
        risk_assessment={"overweld_risk": "low", "underweld_risk": "medium", "perforation_risk": "low"},
        recommendations=["Start at 80% energy", "Increase in 5% steps"],
    )

class TestPdfGenerator:
    def test_export_creates_file(self, tmp_path):
        recipe = _make_recipe()
        gen = PdfGenerator()
        path = gen.export(recipe, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".pdf")
        assert os.path.getsize(path) > 1000

    def test_export_with_validation(self, tmp_path):
        recipe = _make_recipe()
        validation = ValidationResult(
            status=ValidationStatus.WARNING,
            validators={"physics": {"status": "pass", "messages": []},
                        "safety": {"status": "warning", "messages": ["Medium risk detected"]}},
            messages=["[safety] Medium risk detected"],
        )
        gen = PdfGenerator()
        path = gen.export(recipe, validation, str(tmp_path))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000
