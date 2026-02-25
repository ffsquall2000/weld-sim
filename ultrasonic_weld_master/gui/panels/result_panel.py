"""Result display panel with parameter table, safety windows, and risk assessment."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QGroupBox, QHeaderView, QSplitter,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult


_RISK_COLORS = {"low": QColor("#27ae60"), "medium": QColor("#f39c12"), "high": QColor("#e74c3c"), "critical": QColor("#8b0000")}


class ResultPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._recipe: Optional[WeldRecipe] = None
        self._validation: Optional[ValidationResult] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Calculation Results")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._empty_label = QLabel("No results yet. Run a calculation from the wizard.")
        self._empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._empty_label)

        self._content = QSplitter(Qt.Horizontal)
        self._content.setVisible(False)
        layout.addWidget(self._content, 1)

        # Left: Parameters table
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        param_grp = QGroupBox("Welding Parameters")
        param_layout = QVBoxLayout(param_grp)
        self._param_table = QTableWidget()
        self._param_table.setColumnCount(4)
        self._param_table.setHorizontalHeaderLabels(["Parameter", "Value", "Safe Min", "Safe Max"])
        self._param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._param_table.setAlternatingRowColors(True)
        self._param_table.setEditTriggers(QTableWidget.NoEditTriggers)
        param_layout.addWidget(self._param_table)
        left_layout.addWidget(param_grp)
        self._content.addWidget(left)

        # Right: Risk + Validation + Recommendations
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        risk_grp = QGroupBox("Risk Assessment")
        risk_layout = QVBoxLayout(risk_grp)
        self._risk_table = QTableWidget()
        self._risk_table.setColumnCount(2)
        self._risk_table.setHorizontalHeaderLabels(["Risk Type", "Level"])
        self._risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._risk_table.setEditTriggers(QTableWidget.NoEditTriggers)
        risk_layout.addWidget(self._risk_table)
        right_layout.addWidget(risk_grp)

        val_grp = QGroupBox("Validation")
        val_layout = QVBoxLayout(val_grp)
        self._val_label = QLabel("")
        self._val_label.setWordWrap(True)
        val_layout.addWidget(self._val_label)
        right_layout.addWidget(val_grp)

        rec_grp = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_grp)
        self._rec_label = QLabel("")
        self._rec_label.setWordWrap(True)
        rec_layout.addWidget(self._rec_label)
        right_layout.addWidget(rec_grp)

        self._content.addWidget(right)
        self._content.setSizes([500, 400])

    def show_results(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None):
        self._recipe = recipe
        self._validation = validation
        self._empty_label.setVisible(False)
        self._content.setVisible(True)

        # Parameters table
        params = recipe.parameters
        self._param_table.setRowCount(len(params))
        for row, (key, value) in enumerate(params.items()):
            self._param_table.setItem(row, 0, QTableWidgetItem(key))
            self._param_table.setItem(row, 1, QTableWidgetItem(str(value)))
            sw = recipe.safety_window.get(key)
            if sw and isinstance(sw, (list, tuple)) and len(sw) >= 2:
                self._param_table.setItem(row, 2, QTableWidgetItem(str(sw[0])))
                self._param_table.setItem(row, 3, QTableWidgetItem(str(sw[1])))
            else:
                self._param_table.setItem(row, 2, QTableWidgetItem("-"))
                self._param_table.setItem(row, 3, QTableWidgetItem("-"))

        # Risk table
        risks = recipe.risk_assessment
        self._risk_table.setRowCount(len(risks))
        for row, (key, level) in enumerate(risks.items()):
            name_item = QTableWidgetItem(key.replace("_", " ").title())
            level_item = QTableWidgetItem(level.upper())
            color = _RISK_COLORS.get(level, QColor("#888"))
            level_item.setForeground(color)
            self._risk_table.setItem(row, 0, name_item)
            self._risk_table.setItem(row, 1, level_item)

        # Validation
        if validation:
            status = validation.status.value.upper()
            msgs = "\n".join(validation.messages) if validation.messages else "All checks passed."
            self._val_label.setText("Status: %s\n\n%s" % (status, msgs))
        else:
            self._val_label.setText("No validation data.")

        # Recommendations
        recs = "\n".join("- " + r for r in recipe.recommendations) if recipe.recommendations else "None."
        self._rec_label.setText(recs)
