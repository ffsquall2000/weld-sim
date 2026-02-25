"""Settings panel with theme selection and configuration options."""
from __future__ import annotations

from typing import Optional, Callable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout,
    QComboBox, QSpinBox, QLineEdit, QPushButton, QFileDialog,
)
from PySide6.QtCore import Signal, Qt


class SettingsPanel(QWidget):
    defaults_changed = Signal(float, float)  # (frequency_khz, max_power_w)

    def __init__(self, on_theme_change: Optional[Callable] = None,
                 on_language_change: Optional[Callable] = None, parent=None):
        super().__init__(parent)
        self._on_theme_change = on_theme_change
        self._on_language_change = on_language_change
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        title = QLabel("SETTINGS")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Appearance
        appearance_grp = QGroupBox("APPEARANCE")
        appearance_form = QFormLayout(appearance_grp)
        appearance_form.setLabelAlignment(Qt.AlignRight)
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        appearance_form.addRow("Theme:", self._theme_combo)
        layout.addWidget(appearance_grp)

        # Defaults
        defaults_grp = QGroupBox("DEFAULT VALUES")
        defaults_form = QFormLayout(defaults_grp)
        defaults_form.setLabelAlignment(Qt.AlignRight)
        self._default_freq = QComboBox()
        self._default_freq.addItems(["20.0 kHz", "30.0 kHz", "35.0 kHz", "40.0 kHz"])
        self._default_freq.currentTextChanged.connect(self._emit_defaults)
        defaults_form.addRow("Default Frequency:", self._default_freq)
        self._default_power = QSpinBox()
        self._default_power.setRange(500, 10000)
        self._default_power.setValue(3500)
        self._default_power.setSuffix(" W")
        self._default_power.valueChanged.connect(self._emit_defaults)
        defaults_form.addRow("Default Max Power:", self._default_power)
        layout.addWidget(defaults_grp)

        # Paths
        paths_grp = QGroupBox("FILE PATHS")
        paths_form = QFormLayout(paths_grp)
        paths_form.setLabelAlignment(Qt.AlignRight)
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

    def _emit_defaults(self):
        freq_text = self._default_freq.currentText()
        freq = float(freq_text.replace(" kHz", ""))
        power = float(self._default_power.value())
        self.defaults_changed.emit(freq, power)

    def _browse_report_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Report Directory")
        if path:
            self._report_dir.setText(path)
