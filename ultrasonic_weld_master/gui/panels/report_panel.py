"""Report preview and export panel."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView,
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
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        self._label_title = QLabel("")
        self._label_title.setObjectName("sectionTitle")
        layout.addWidget(self._label_title)

        self._empty_label = QLabel("")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #484f58; font-size: 14px; padding: 60px;")
        layout.addWidget(self._empty_label)

        self._content = QWidget()
        self._content.setVisible(False)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        # Info header
        self._grp_info = QGroupBox("")
        info_layout = QVBoxLayout(self._grp_info)
        self._info_label = QLabel("")
        self._info_label.setStyleSheet("font-family: 'Menlo', 'SF Mono', monospace; font-size: 12px;")
        info_layout.addWidget(self._info_label)
        content_layout.addWidget(self._grp_info)

        # Structured parameter preview table
        self._grp_preview = QGroupBox("")
        preview_layout = QVBoxLayout(self._grp_preview)
        self._preview_table = QTableWidget()
        self._preview_table.setColumnCount(3)
        self._preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self._preview_table)
        content_layout.addWidget(self._grp_preview, 1)

        # Risk + Validation summary
        summary_layout = QHBoxLayout()
        self._grp_risk = QGroupBox("")
        risk_layout = QVBoxLayout(self._grp_risk)
        self._risk_label = QLabel("")
        self._risk_label.setWordWrap(True)
        self._risk_label.setStyleSheet("font-size: 12px;")
        risk_layout.addWidget(self._risk_label)
        summary_layout.addWidget(self._grp_risk)

        self._grp_validation = QGroupBox("")
        val_layout = QVBoxLayout(self._grp_validation)
        self._val_label = QLabel("")
        self._val_label.setWordWrap(True)
        self._val_label.setStyleSheet("font-size: 12px;")
        val_layout.addWidget(self._val_label)
        summary_layout.addWidget(self._grp_validation)
        content_layout.addLayout(summary_layout)

        # Export buttons
        self._grp_export = QGroupBox("")
        btn_layout = QHBoxLayout(self._grp_export)
        btn_layout.setSpacing(12)
        self._json_btn = QPushButton("")
        self._json_btn.clicked.connect(self._export_json)
        self._excel_btn = QPushButton("")
        self._excel_btn.clicked.connect(self._export_excel)
        self._pdf_btn = QPushButton("")
        self._pdf_btn.clicked.connect(self._export_pdf)
        self._all_btn = QPushButton("")
        self._all_btn.clicked.connect(self._export_all)
        btn_layout.addWidget(self._json_btn)
        btn_layout.addWidget(self._excel_btn)
        btn_layout.addWidget(self._pdf_btn)
        btn_layout.addWidget(self._all_btn)
        content_layout.addWidget(self._grp_export)

        layout.addWidget(self._content, 1)

        self.retranslateUi()

    def retranslateUi(self):
        self._label_title.setText(self.tr("REPORT GENERATION"))
        self._empty_label.setText(self.tr("No recipe loaded. Run a calculation first."))
        self._grp_info.setTitle(self.tr("RECIPE INFO"))
        self._grp_preview.setTitle(self.tr("PARAMETER PREVIEW"))
        self._preview_table.setHorizontalHeaderLabels([
            self.tr("PARAMETER"), self.tr("VALUE"), self.tr("SAFE RANGE"),
        ])
        self._grp_risk.setTitle(self.tr("RISK ASSESSMENT"))
        self._grp_validation.setTitle(self.tr("VALIDATION"))
        self._grp_export.setTitle(self.tr("EXPORT"))
        self._json_btn.setText(self.tr("Export JSON"))
        self._excel_btn.setText(self.tr("Export Excel"))
        self._pdf_btn.setText(self.tr("Export PDF"))
        self._all_btn.setText(self.tr("Export All"))

    def set_recipe(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None):
        self._recipe = recipe
        self._validation = validation
        self._empty_label.setVisible(False)
        self._content.setVisible(True)
        self._info_label.setText(
            "Recipe: %s  |  Application: %s  |  Created: %s" % (
                recipe.recipe_id, recipe.application, recipe.created_at[:19]))

        # Fill parameter preview table
        self._preview_table.setRowCount(0)
        for key, val in recipe.parameters.items():
            row = self._preview_table.rowCount()
            self._preview_table.insertRow(row)
            self._preview_table.setItem(row, 0, QTableWidgetItem(key))
            self._preview_table.setItem(row, 1, QTableWidgetItem(
                "%.4g" % val if isinstance(val, float) else str(val)))
            sw = recipe.safety_window.get(key, [])
            if len(sw) >= 2:
                self._preview_table.setItem(row, 2, QTableWidgetItem(
                    "[%.4g — %.4g]" % (sw[0], sw[1])))
            else:
                self._preview_table.setItem(row, 2, QTableWidgetItem("—"))

        # Risk assessment
        risk_lines = []
        _risk_colors = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336", "critical": "#d32f2f"}
        for k, v in recipe.risk_assessment.items():
            color = _risk_colors.get(v.lower(), "#8b949e")
            label = k.replace("_", " ").title()
            risk_lines.append('<span style="color:%s;">&#9679;</span> %s: <b>%s</b>' % (color, label, v.upper()))
        self._risk_label.setText("<br>".join(risk_lines) if risk_lines else self.tr("No risk data"))

        # Validation
        if validation:
            status_color = "#4caf50" if validation.is_passed() else "#f44336"
            val_lines = ['<span style="color:%s; font-weight:bold;">%s</span>' % (
                status_color, validation.status.value.upper())]
            for msg in validation.messages:
                val_lines.append(msg)
            self._val_label.setText("<br>".join(val_lines))
        else:
            self._val_label.setText('<span style="color:#8b949e;">%s</span>' % self.tr("No validation data"))

    def _get_output_dir(self) -> Optional[str]:
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Output Directory"))
        return path if path else None

    def _export_json(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = JsonExporter().export(self._recipe, self._validation, out)
            QMessageBox.information(self, self.tr("Export"), "JSON saved:\n%s" % path)

    def _export_excel(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = ExcelGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, self.tr("Export"), "Excel saved:\n%s" % path)

    def _export_pdf(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            path = PdfGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, self.tr("Export"), "PDF saved:\n%s" % path)

    def _export_all(self):
        if not self._recipe:
            return
        out = self._get_output_dir()
        if out:
            paths = {}
            paths["json"] = JsonExporter().export(self._recipe, self._validation, out)
            paths["excel"] = ExcelGenerator().export(self._recipe, self._validation, out)
            paths["pdf"] = PdfGenerator().export(self._recipe, self._validation, out)
            QMessageBox.information(self, self.tr("Export All"),
                                   "All reports saved:\n%s\n%s\n%s" % (paths["json"], paths["excel"], paths["pdf"]))
