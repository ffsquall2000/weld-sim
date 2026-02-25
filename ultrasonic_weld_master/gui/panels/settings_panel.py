"""Settings panel with theme selection and configuration options."""
from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QComboBox, QSpinBox, QLineEdit, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt


class SettingsPanel(QWidget):
    def __init__(self, on_theme_change: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._on_theme_change = on_theme_change
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Settings")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Appearance
        appearance_grp = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_grp)
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Light", "Dark"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        appearance_form.addRow("Theme:", self._theme_combo)
        layout.addWidget(appearance_grp)

        # Defaults
        defaults_grp = QGroupBox("Default Values")
        defaults_form = QFormLayout(defaults_grp)
        self._default_freq = QComboBox()
        self._default_freq.addItems(["20.0 kHz", "30.0 kHz", "35.0 kHz", "40.0 kHz"])
        defaults_form.addRow("Default Frequency:", self._default_freq)
        self._default_power = QSpinBox()
        self._default_power.setRange(500, 10000)
        self._default_power.setValue(3500)
        self._default_power.setSuffix(" W")
        defaults_form.addRow("Default Max Power:", self._default_power)
        layout.addWidget(defaults_grp)

        # Paths
        paths_grp = QGroupBox("File Paths")
        paths_form = QFormLayout(paths_grp)
        self._report_dir = QLineEdit()
        self._report_dir.setPlaceholderText("Default: ./reports")
        browse_btn = QPushButton("Browse...")
        browse_btn.setProperty("secondary", True)
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_report_dir)
        paths_form.addRow("Report Output:", self._report_dir)
        paths_form.addRow("", browse_btn)
        layout.addWidget(paths_grp)

        layout.addStretch()

    def _on_theme_selected(self, text: str):
        theme = text.lower()
        if self._on_theme_change:
            self._on_theme_change(theme)

    def _browse_report_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Report Directory")
        if path:
            self._report_dir.setText(path)
