"""Result display panel with gauge widgets, risk indicators, and parameter table."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QGroupBox, QHeaderView, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.gui.widgets.gauge_widget import GaugeWidget
from ultrasonic_weld_master.gui.widgets.risk_indicator import RiskIndicator


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipe: Optional[WeldRecipe] = None
        self._validation: Optional[ValidationResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        self._title_label = QLabel("")
        self._title_label.setObjectName("sectionTitle")
        layout.addWidget(self._title_label)

        self._empty_label = QLabel("")
        self._empty_label.setAlignment(Qt.AlignCenter)
        self._empty_label.setStyleSheet("color: #484f58; font-size: 14px; padding: 60px;")
        layout.addWidget(self._empty_label)

        # Scrollable content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._content = QWidget()
        self._content.setVisible(False)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(16)

        # ── Row 1: Gauge Widgets ──
        self._gauge_grp = QGroupBox("")
        gauge_layout = QHBoxLayout(self._gauge_grp)
        gauge_layout.setSpacing(8)

        self._gauge_amplitude = GaugeWidget(label="Amplitude", unit="um", min_val=10, max_val=65)
        self._gauge_pressure = GaugeWidget(label="Pressure", unit="MPa", min_val=0.05, max_val=0.80)
        self._gauge_energy = GaugeWidget(label="Energy", unit="J", min_val=0, max_val=200)
        gauge_layout.addWidget(self._gauge_amplitude)
        gauge_layout.addWidget(self._gauge_pressure)
        gauge_layout.addWidget(self._gauge_energy)
        content_layout.addWidget(self._gauge_grp)

        # ── Row 2: Risk Indicators ──
        self._risk_grp = QGroupBox("")
        risk_layout = QHBoxLayout(self._risk_grp)
        risk_layout.setSpacing(16)

        self._risk_overweld = RiskIndicator(label="overweld_risk")
        self._risk_underweld = RiskIndicator(label="underweld_risk")
        self._risk_perforation = RiskIndicator(label="perforation_risk")
        risk_layout.addStretch()
        risk_layout.addWidget(self._risk_overweld)
        risk_layout.addWidget(self._risk_underweld)
        risk_layout.addWidget(self._risk_perforation)
        risk_layout.addStretch()
        content_layout.addWidget(self._risk_grp)

        # ── Row 3: Full Parameter Table ──
        self._table_grp = QGroupBox("")
        table_layout = QVBoxLayout(self._table_grp)
        self._param_table = QTableWidget()
        self._param_table.setColumnCount(4)
        self._param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._param_table.setAlternatingRowColors(True)
        self._param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._param_table.verticalHeader().setVisible(False)
        self._param_table.setMaximumHeight(220)
        table_layout.addWidget(self._param_table)
        content_layout.addWidget(self._table_grp)

        # ── Row 4: Validation + Recommendations ──
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(12)

        self._val_grp = QGroupBox("")
        val_layout = QVBoxLayout(self._val_grp)
        self._val_label = QLabel("")
        self._val_label.setWordWrap(True)
        self._val_label.setStyleSheet("font-family: 'Menlo', 'SF Mono', monospace; font-size: 12px;")
        val_layout.addWidget(self._val_label)
        bottom_layout.addWidget(self._val_grp)

        self._rec_grp = QGroupBox("")
        rec_layout = QVBoxLayout(self._rec_grp)
        self._rec_label = QLabel("")
        self._rec_label.setWordWrap(True)
        self._rec_label.setStyleSheet("font-size: 12px; line-height: 1.4;")
        rec_layout.addWidget(self._rec_label)
        bottom_layout.addWidget(self._rec_grp)

        content_layout.addLayout(bottom_layout)
        content_layout.addStretch()

        scroll.setWidget(self._content)
        layout.addWidget(scroll, 1)

        self.retranslateUi()

    def retranslateUi(self):
        """Re-set all translatable text for i18n support."""
        self._title_label.setText(self.tr("CALCULATION RESULTS"))
        self._empty_label.setText(self.tr("No results yet. Run a calculation from the wizard."))
        self._gauge_grp.setTitle(self.tr("PARAMETERS"))
        self._risk_grp.setTitle(self.tr("RISK ASSESSMENT"))
        self._table_grp.setTitle(self.tr("ALL PARAMETERS"))
        self._param_table.setHorizontalHeaderLabels([
            self.tr("PARAMETER"),
            self.tr("VALUE"),
            self.tr("SAFE MIN"),
            self.tr("SAFE MAX"),
        ])
        self._val_grp.setTitle(self.tr("VALIDATION"))
        self._rec_grp.setTitle(self.tr("RECOMMENDATIONS"))

    def show_results(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None):
        self._recipe = recipe
        self._validation = validation
        self._empty_label.setVisible(False)
        self._content.setVisible(True)

        p = recipe.parameters
        sw = recipe.safety_window

        # Gauge: Amplitude
        amp = p.get("amplitude_um", 0)
        amp_sw = sw.get("amplitude_um", [10, 65])
        self._gauge_amplitude.set_range(10, 65)
        self._gauge_amplitude.set_safe_window(amp_sw[0] if len(amp_sw) >= 2 else 10,
                                               amp_sw[1] if len(amp_sw) >= 2 else 65)
        self._gauge_amplitude.set_value(amp)

        # Gauge: Pressure
        pres = p.get("pressure_mpa", 0)
        pres_sw = sw.get("pressure_n", [])  # show pressure in MPa gauge
        self._gauge_pressure.set_range(0.05, 0.80)
        self._gauge_pressure.set_safe_window(0.1, 0.7)
        self._gauge_pressure.set_value(pres)

        # Gauge: Energy
        energy = p.get("energy_j", 0)
        energy_sw = sw.get("energy_j", [0, 200])
        max_e = max(energy * 1.8, 100)
        self._gauge_energy.set_range(0, max_e)
        self._gauge_energy.set_safe_window(
            energy_sw[0] if len(energy_sw) >= 2 else 0,
            energy_sw[1] if len(energy_sw) >= 2 else max_e,
        )
        self._gauge_energy.set_value(energy)

        # Risk indicators
        risk = recipe.risk_assessment
        self._risk_overweld.set_level(risk.get("overweld_risk", "low"))
        self._risk_underweld.set_level(risk.get("underweld_risk", "low"))
        self._risk_perforation.set_level(risk.get("perforation_risk", "low"))

        # Parameter table
        self._param_table.setRowCount(0)
        for key, val in p.items():
            row = self._param_table.rowCount()
            self._param_table.insertRow(row)
            self._param_table.setItem(row, 0, QTableWidgetItem(key))
            self._param_table.setItem(row, 1, QTableWidgetItem("%.4g" % val if isinstance(val, float) else str(val)))
            sw_range = sw.get(key, [])
            self._param_table.setItem(row, 2, QTableWidgetItem(
                "%.4g" % sw_range[0] if len(sw_range) >= 2 else "—"))
            self._param_table.setItem(row, 3, QTableWidgetItem(
                "%.4g" % sw_range[1] if len(sw_range) >= 2 else "—"))
            # Color rows outside safe window
            if isinstance(val, (int, float)) and len(sw_range) >= 2:
                if val < sw_range[0] or val > sw_range[1]:
                    for col in range(4):
                        item = self._param_table.item(row, col)
                        if item:
                            item.setForeground(QColor("#f44336"))

        # Validation summary
        if validation:
            status_color = "#4caf50" if validation.is_passed() else "#f44336"
            val_lines = ['<span style="color:%s; font-weight:bold;">%s</span>' % (
                status_color, validation.status.value.upper())]
            for msg in validation.messages:
                val_lines.append(msg)
            self._val_label.setText("<br>".join(val_lines))
        else:
            self._val_label.setText('<span style="color:#8b949e;">%s</span>' % self.tr("No validation data"))

        # Recommendations
        if recipe.recommendations:
            rec_lines = ['<span style="color:#ff9800;">&#9679;</span> ' + r for r in recipe.recommendations]
            self._rec_label.setText("<br>".join(rec_lines))
        else:
            self._rec_label.setText('<span style="color:#8b949e;">%s</span>' % self.tr("No recommendations"))
