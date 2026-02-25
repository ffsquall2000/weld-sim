"""History panel for browsing past calculation sessions."""
from __future__ import annotations

import json
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QGroupBox, QHeaderView, QPushButton,
)
from PySide6.QtCore import Qt


class HistoryPanel(QWidget):
    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        self._label_title = QLabel("")
        self._label_title.setObjectName("sectionTitle")
        layout.addWidget(self._label_title)

        # Toolbar
        toolbar = QHBoxLayout()
        self._refresh_btn = QPushButton("")
        self._refresh_btn.setProperty("secondary", True)
        self._refresh_btn.clicked.connect(self._load_history)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addStretch()
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #8b949e; font-size: 12px;")
        toolbar.addWidget(self._count_label)
        layout.addLayout(toolbar)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, 1)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #484f58; font-size: 12px;")
        layout.addWidget(self._status_label)

        self.retranslateUi()

    def retranslateUi(self):
        self._label_title.setText(self.tr("CALCULATION HISTORY"))
        self._refresh_btn.setText(self.tr("Refresh"))
        self._table.setHorizontalHeaderLabels([
            self.tr("ID"), self.tr("APPLICATION"), self.tr("MATERIALS"),
            self.tr("DATE"), self.tr("STATUS"),
        ])
        self._status_label.setText(
            self.tr("No history loaded. Click Refresh or run a calculation."))

    def refresh(self):
        """Public method to trigger a history reload."""
        self._load_history()

    def _load_history(self):
        if not self._engine or not hasattr(self._engine, "database"):
            self._status_label.setText(self.tr("Engine not connected."))
            return

        try:
            db = self._engine.database
            rows = db.execute(
                "SELECT id, application, inputs, created_at FROM recipes ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
            self._table.setRowCount(0)
            if not rows:
                self._status_label.setText(self.tr("No calculations found."))
                self._count_label.setText(self.tr("%d records") % 0)
                return

            for row_data in rows:
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(str(row_data[0])[:12]))
                self._table.setItem(row, 1, QTableWidgetItem(str(row_data[1])))
                # Parse inputs JSON to show material summary
                try:
                    inputs = json.loads(row_data[2]) if isinstance(row_data[2], str) else row_data[2]
                    mat_text = "%s → %s" % (
                        inputs.get("upper_material_type", "?"),
                        inputs.get("lower_material_type", "?"),
                    )
                except (json.JSONDecodeError, TypeError):
                    mat_text = str(row_data[2])[:30]
                self._table.setItem(row, 2, QTableWidgetItem(mat_text))
                self._table.setItem(row, 3, QTableWidgetItem(str(row_data[3])))
                self._table.setItem(row, 4, QTableWidgetItem("OK"))

            count = self._table.rowCount()
            self._status_label.setText(self.tr("Loaded %d records.") % count)
            self._count_label.setText(self.tr("%d records") % count)
        except Exception as e:
            self._status_label.setText(self.tr("Error: %s") % str(e))
