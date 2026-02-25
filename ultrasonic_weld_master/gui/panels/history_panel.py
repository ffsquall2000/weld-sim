"""History panel for browsing past calculation sessions."""
from __future__ import annotations

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
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Calculation History")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Toolbar
        toolbar = QHBoxLayout()
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setProperty("secondary", True)
        self._refresh_btn.clicked.connect(self._load_history)
        toolbar.addWidget(self._refresh_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Table
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["Recipe ID", "Application", "Materials", "Date", "Status"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self._table, 1)

        self._status_label = QLabel("No history loaded. Click Refresh or connect an engine.")
        layout.addWidget(self._status_label)

    def _load_history(self):
        if not self._engine or not hasattr(self._engine, "database"):
            self._status_label.setText("Engine not connected.")
            return

        try:
            db = self._engine.database
            # Query recent recipes from database
            rows = db.execute("SELECT recipe_id, application, inputs_json, created_at FROM recipes ORDER BY created_at DESC LIMIT 50")
            self._table.setRowCount(0)
            if not rows:
                self._status_label.setText("No calculations found.")
                return

            for row_data in rows:
                row = self._table.rowCount()
                self._table.insertRow(row)
                self._table.setItem(row, 0, QTableWidgetItem(str(row_data[0])))
                self._table.setItem(row, 1, QTableWidgetItem(str(row_data[1])))
                self._table.setItem(row, 2, QTableWidgetItem(str(row_data[2])[:40]))
                self._table.setItem(row, 3, QTableWidgetItem(str(row_data[3])))
                self._table.setItem(row, 4, QTableWidgetItem("OK"))

            self._status_label.setText("Loaded %d records." % self._table.rowCount())
        except Exception as e:
            self._status_label.setText("Error: %s" % str(e))
