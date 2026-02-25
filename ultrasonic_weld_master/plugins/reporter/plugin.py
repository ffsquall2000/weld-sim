"""Reporter plugin assembling JSON, Excel, and PDF exporters."""
from __future__ import annotations

from typing import Any, Optional

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.plugins.reporter.json_exporter import JsonExporter
from ultrasonic_weld_master.plugins.reporter.excel_generator import ExcelGenerator
from ultrasonic_weld_master.plugins.reporter.pdf_generator import PdfGenerator


class ReporterPlugin(PluginBase):
    def __init__(self):
        self._json_exporter: Optional[JsonExporter] = None
        self._excel_generator: Optional[ExcelGenerator] = None
        self._pdf_generator: Optional[PdfGenerator] = None

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="reporter", version="1.0.0",
            description="Report generator supporting JSON, Excel, and PDF formats",
            author="UltrasonicWeldMaster", dependencies=[],
        )

    def activate(self, context: Any) -> None:
        self._json_exporter = JsonExporter()
        self._excel_generator = ExcelGenerator()
        self._pdf_generator = PdfGenerator()

    def deactivate(self) -> None:
        self._json_exporter = None
        self._excel_generator = None
        self._pdf_generator = None

    def export_json(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None,
                    output_dir: str = ".") -> str:
        return self._json_exporter.export(recipe, validation, output_dir)

    def export_excel(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None,
                     output_dir: str = ".") -> str:
        return self._excel_generator.export(recipe, validation, output_dir)

    def export_pdf(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None,
                   output_dir: str = ".") -> str:
        return self._pdf_generator.export(recipe, validation, output_dir)

    def export_all(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None,
                   output_dir: str = ".") -> dict:
        return {
            "json": self.export_json(recipe, validation, output_dir),
            "excel": self.export_excel(recipe, validation, output_dir),
            "pdf": self.export_pdf(recipe, validation, output_dir),
        }
