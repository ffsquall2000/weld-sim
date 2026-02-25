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

        self._label_title = QLabel("")
        self._label_title.setObjectName("sectionTitle")
        layout.addWidget(self._label_title)

        # Appearance
        self._grp_appearance = QGroupBox("")
        appearance_form = QFormLayout(self._grp_appearance)
        appearance_form.setLabelAlignment(Qt.AlignRight)
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["Dark", "Light"])
        self._theme_combo.currentTextChanged.connect(self._on_theme_selected)
        self._label_theme = QLabel("")
        appearance_form.addRow(self._label_theme, self._theme_combo)
        layout.addWidget(self._grp_appearance)

        # Language
        self._grp_language = QGroupBox("")
        language_form = QFormLayout(self._grp_language)
        language_form.setLabelAlignment(Qt.AlignRight)
        self._language_combo = QComboBox()
        self._language_combo.addItems(["\u4e2d\u6587", "English"])
        self._language_combo.currentIndexChanged.connect(self._on_language_selected)
        self._label_language = QLabel("")
        language_form.addRow(self._label_language, self._language_combo)
        layout.addWidget(self._grp_language)

        # Defaults
        self._grp_defaults = QGroupBox("")
        defaults_form = QFormLayout(self._grp_defaults)
        defaults_form.setLabelAlignment(Qt.AlignRight)
        self._default_freq = QComboBox()
        self._default_freq.addItems(["20.0 kHz", "30.0 kHz", "35.0 kHz", "40.0 kHz"])
        self._default_freq.currentTextChanged.connect(self._emit_defaults)
        self._label_default_freq = QLabel("")
        defaults_form.addRow(self._label_default_freq, self._default_freq)
        self._default_power = QSpinBox()
        self._default_power.setRange(500, 10000)
        self._default_power.setValue(3500)
        self._default_power.setSuffix(" W")
        self._default_power.valueChanged.connect(self._emit_defaults)
        self._label_default_power = QLabel("")
        defaults_form.addRow(self._label_default_power, self._default_power)
        layout.addWidget(self._grp_defaults)

        # Paths
        self._grp_paths = QGroupBox("")
        paths_form = QFormLayout(self._grp_paths)
        paths_form.setLabelAlignment(Qt.AlignRight)
        self._report_dir = QLineEdit()
        self._report_dir.setPlaceholderText("Default: ./reports")
        self._browse_btn = QPushButton("")
        self._browse_btn.setProperty("secondary", True)
        self._browse_btn.setFixedWidth(80)
        self._browse_btn.clicked.connect(self._browse_report_dir)
        self._label_report_output = QLabel("")
        paths_form.addRow(self._label_report_output, self._report_dir)
        paths_form.addRow("", self._browse_btn)
        layout.addWidget(self._grp_paths)

        layout.addStretch()

        self.retranslateUi()

    def retranslateUi(self):
        self._label_title.setText(self.tr("SETTINGS"))
        self._grp_appearance.setTitle(self.tr("APPEARANCE"))
        self._label_theme.setText(self.tr("Theme:"))
        self._grp_language.setTitle(self.tr("LANGUAGE"))
        self._label_language.setText(self.tr("Language:"))
        self._grp_defaults.setTitle(self.tr("DEFAULT VALUES"))
        self._label_default_freq.setText(self.tr("Default Frequency:"))
        self._label_default_power.setText(self.tr("Default Max Power:"))
        self._grp_paths.setTitle(self.tr("FILE PATHS"))
        self._label_report_output.setText(self.tr("Report Output:"))
        self._browse_btn.setText(self.tr("Browse..."))

    def _on_theme_selected(self, text: str):
        theme = text.lower()
        if self._on_theme_change:
            self._on_theme_change(theme)

    def _on_language_selected(self, index: int):
        if self._on_language_change:
            if index == 0:
                self._on_language_change("zh_CN")
            else:
                self._on_language_change("en")

    def _emit_defaults(self):
        freq_text = self._default_freq.currentText()
        freq = float(freq_text.replace(" kHz", ""))
        power = float(self._default_power.value())
        self.defaults_changed.emit(freq, power)

    def _browse_report_dir(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("Select Report Directory"))
        if path:
            self._report_dir.setText(path)
