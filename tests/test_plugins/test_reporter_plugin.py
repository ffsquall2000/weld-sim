from __future__ import annotations
import os
import pytest
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus
from ultrasonic_weld_master.plugins.reporter.plugin import ReporterPlugin

def _make_recipe():
    return WeldRecipe(
        recipe_id="RPT001", application="li_battery_tab",
        inputs={"upper_material": "Al"},
        parameters={"amplitude_um": 30.0, "energy_j": 60.0, "time_ms": 200},
        safety_window={"amplitude_um": [25, 35]},
        risk_assessment={"overweld_risk": "low"},
        recommendations=["Start at 80% energy"],
    )

class TestReporterPlugin:
    @pytest.fixture
    def plugin(self):
        p = ReporterPlugin()
        p.activate({})
        return p

    def test_get_info(self):
        p = ReporterPlugin()
        info = p.get_info()
        assert info.name == "reporter"

    def test_export_json(self, plugin, tmp_path):
        recipe = _make_recipe()
        path = plugin.export_json(recipe, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_export_excel(self, plugin, tmp_path):
        recipe = _make_recipe()
        path = plugin.export_excel(recipe, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".xlsx")

    def test_export_pdf(self, plugin, tmp_path):
        recipe = _make_recipe()
        path = plugin.export_pdf(recipe, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".pdf")

    def test_export_all(self, plugin, tmp_path):
        recipe = _make_recipe()
        validation = ValidationResult(
            status=ValidationStatus.PASS,
            validators={"physics": {"status": "pass"}},
        )
        paths = plugin.export_all(recipe, validation, str(tmp_path))
        assert "json" in paths and os.path.exists(paths["json"])
        assert "excel" in paths and os.path.exists(paths["excel"])
        assert "pdf" in paths and os.path.exists(paths["pdf"])
