"""Report preview and export panel."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFileDialog, QMessageBox, QTextEdit,
)
from PySide6.QtCore import Qt

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.plugins.reporter.json_exporter import JsonExporter
from ultrasonic_weld_master.plugins.reporter.excel_generator import ExcelGenerator
from ultrasonic_weld_master.plugins.reporter.pdf_generator import PdfGenerator


class ReportPanel(QWidget):
    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._recipe: Optional[WeldRecipe] = None
        self._validation: Optional[ValidationResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Report Generation")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._empty_label = QLabel("No recipe loaded. Run a calculation first.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._empty_label)

        self._content = QWidget()
        self._content.setVisible(False)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Info
        info_grp = QGroupBox("Recipe Info")
        info_layout = QVBoxLayout(info_grp)
        self._info_label = QLabel("")
        info_layout.addWidget(self._info_label)
        content_layout.addWidget(info_grp)

        # Preview
        preview_grp = QGroupBox("Report Preview")
        preview_layout = QVBoxLayout(preview_grp)
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        preview_layout.addWidget(self._preview)
        content_layout.addWidget(preview_grp, 1)

        # Export buttons
        btn_grp = QGroupBox("Export")
        btn_layout = QHBoxLayout(btn_grp)
        self._json_btn = QPushButton("Export JSON")
        self._json_btn.clicked.connect(self._export_json)
        self._excel_btn = QPushButton("Export Excel")
        self._excel_btn.clicked.connect(self._export_excel)
        self._pdf_btn = QPushButton("Export PDF")
        self._pdf_btn.clicked.connect(self._export_pdf)
        self._all_btn = QPushButton("Export All")
        self._all_btn.clicked.connect(self._export_all)
        btn_layout.addWidget(self._json_btn)
        btn_layout.addWidget(self._excel_btn)
        btn_layout.addWidget(self._pdf_btn)
        btn_layout.addWidget(self._all_btn)
        content_layout.addWidget(btn_grp)

        layout.addWidget(self._content, 1)

    def set_recipe(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None):
        self._recipe = recipe
        self._validation = validation
        self._empty_label.setVisible(False)
        self._content.setVisible(True)
        self._info_label.setText("Recipe: %s | Application: %s" % (recipe.recipe_id, recipe.application))

        # Build text preview
        lines = ["=== Welding Parameter Report ===", ""]
        lines.append("Recipe ID: %s" % recipe.recipe_id)
        lines.append("Application: %s" % recipe.application)
        lines.append("")
        lines.append("--- Parameters ---")
        for k, v in recipe.parameters.items():
            sw = recipe.safety_window.get(k)
            if sw and len(sw) >= 2:
                lines.append("  %s: %s  [%s - %s]" % (k, v, sw[0], sw[1]))
            else:
                lines.append("  %s: %s" % (k, v))
        lines.append("")
        lines.append("--- Risk Assessment ---")
        for k, v in recipe.risk_assessment.items():
            lines.append("  %s: %s" % (k, v.upper()))
        lines.append("")
        lines.append("--- Recommendations ---")
        for r in recipe.recommendations:
            lines.append("  - %s" % r)
        if validation:
            lines.append("")
            lines.append("--- Validation: %s ---" % validation.status.value.upper())
            for msg in validation.messages:
                lines.append("  %s" % msg)

        self._preview.setPlainText("\n".join(lines))

    def _get_output_dir(self) -> Optional[str]:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        return path if path else None

    def _export_json(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = JsonExporter().export(self._recipe, self._validation, out)
            QMessageBox.information(self, "Export", "JSON saved:\n%s" % path)

    def _export_excel(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = ExcelGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, "Export", "Excel saved:\n%s" % path)

    def _export_pdf(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = PdfGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, "Export", "PDF saved:\n%s" % path)

    def _export_all(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            paths = {}
            paths["json"] = JsonExporter().export(self._recipe, self._validation, out)
            paths["excel"] = ExcelGenerator().export(self._recipe, self._validation, out)
            paths["pdf"] = PdfGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, "Export All",
                                   "All reports saved:\n%s\n%s\n%s" % (paths["json"], paths["excel"], paths["pdf"]))
