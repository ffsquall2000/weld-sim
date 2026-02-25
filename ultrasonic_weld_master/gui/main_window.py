"""Main application window with navigation and central workspace."""
from __future__ import annotations

import os
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QStackedWidget, QStatusBar, QLabel, QListWidgetItem, QPushButton,
    QMessageBox, QApplication,
)
from PySide6.QtCore import Qt, QSize, QTranslator, QLocale, QLibraryInfo
from PySide6.QtGui import QAction

from ultrasonic_weld_master.gui.themes import get_theme
from ultrasonic_weld_master.gui.panels.input_wizard import InputWizardPanel
from ultrasonic_weld_master.gui.panels.result_panel import ResultPanel
from ultrasonic_weld_master.gui.panels.report_panel import ReportPanel
from ultrasonic_weld_master.gui.panels.history_panel import HistoryPanel
from ultrasonic_weld_master.gui.panels.settings_panel import SettingsPanel
from ultrasonic_weld_master.gui.widgets.status_led import StatusLED


def _translations_path() -> str:
    """Return path to translations directory, works both in dev and PyInstaller."""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'translations')
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'translations')


class MainWindow(QMainWindow):
    NAV_KEYS = ["input_wizard", "results", "reports", "history", "settings"]

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._current_theme = "dark"
        self._translator = QTranslator()
        self._qt_translator = QTranslator()
        self._setup_ui()
        self._apply_theme("dark")

    def _setup_ui(self):
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Menu bar
        menubar = self.menuBar()
        self._file_menu = menubar.addMenu("")
        self._new_action = QAction("", self)
        self._new_action.triggered.connect(lambda: self._navigate_to(0))
        self._file_menu.addAction(self._new_action)
        self._file_menu.addSeparator()
        self._quit_action = QAction("", self)
        self._quit_action.triggered.connect(self.close)
        self._file_menu.addAction(self._quit_action)

        self._view_menu = menubar.addMenu("")
        self._toggle_theme_action = QAction("", self)
        self._toggle_theme_action.triggered.connect(self._toggle_theme)
        self._view_menu.addAction(self._toggle_theme_action)

        self._help_menu = menubar.addMenu("")
        self._about_action = QAction("", self)
        self._about_action.triggered.connect(self._show_about)
        self._help_menu.addAction(self._about_action)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left navigation
        nav_container = QWidget()
        nav_container.setFixedWidth(180)
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(0)

        self._nav_list = QListWidget()
        self._nav_list.setIconSize(QSize(24, 24))
        for _ in self.NAV_KEYS:
            item = QListWidgetItem("")
            item.setSizeHint(QSize(170, 44))
            self._nav_list.addItem(item)
        self._nav_list.currentRowChanged.connect(self._navigate_to)
        nav_layout.addWidget(self._nav_list)
        main_layout.addWidget(nav_container)

        # Stacked workspace
        self._stack = QStackedWidget()
        self._input_wizard = InputWizardPanel(engine=self._engine)
        self._result_panel = ResultPanel()
        self._report_panel = ReportPanel(engine=self._engine)
        self._history_panel = HistoryPanel(engine=self._engine)
        self._settings_panel = SettingsPanel(
            on_theme_change=self._apply_theme,
            on_language_change=self._on_language_changed,
        )

        self._panels = [
            self._input_wizard, self._result_panel, self._report_panel,
            self._history_panel, self._settings_panel,
        ]
        for p in self._panels:
            self._stack.addWidget(p)
        main_layout.addWidget(self._stack, 1)

        # Connections
        self._input_wizard.calculation_done.connect(self._on_calculation_done)
        self._settings_panel.defaults_changed.connect(self._on_defaults_changed)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("")
        self._status_bar.addWidget(self._status_label)
        self._status_bar.addPermanentWidget(self._make_status_separator())
        self._status_mode = QLabel("—")
        self._status_bar.addPermanentWidget(self._status_mode)
        self._status_bar.addPermanentWidget(self._make_status_separator())
        self._status_material = QLabel("—")
        self._status_bar.addPermanentWidget(self._status_material)
        self._status_bar.addPermanentWidget(self._make_status_separator())
        self._status_freq = QLabel("20 kHz")
        self._status_bar.addPermanentWidget(self._status_freq)
        self._status_bar.addPermanentWidget(self._make_status_separator())
        self._status_power = QLabel("3500 W")
        self._status_bar.addPermanentWidget(self._status_power)

        # Set all translatable text
        self.retranslateUi()

        # Default selection
        self._nav_list.setCurrentRow(0)

    def retranslateUi(self):
        self.setWindowTitle(self.tr("UltrasonicWeldMaster v0.1.0"))
        self._file_menu.setTitle(self.tr("File"))
        self._new_action.setText(self.tr("New Calculation"))
        self._quit_action.setText(self.tr("Quit"))
        self._view_menu.setTitle(self.tr("View"))
        self._toggle_theme_action.setText(self.tr("Toggle Dark/Light Theme"))
        self._help_menu.setTitle(self.tr("Help"))
        self._about_action.setText(self.tr("About"))

        nav_labels = [
            self.tr("New Calculation"),
            self.tr("Results"),
            self.tr("Reports"),
            self.tr("History"),
            self.tr("Settings"),
        ]
        for i, label in enumerate(nav_labels):
            self._nav_list.item(i).setText(label)

        self._status_label.setText(self.tr("Ready"))

    def _make_status_separator(self) -> QLabel:
        sep = QLabel("|")
        sep.setStyleSheet("color: #30363d; padding: 0 6px;")
        return sep

    def _navigate_to(self, index: int):
        self._stack.setCurrentIndex(index)
        self._nav_list.setCurrentRow(index)

    def _on_calculation_done(self, recipe, validation):
        self._result_panel.show_results(recipe, validation)
        self._report_panel.set_recipe(recipe, validation)
        self._navigate_to(1)

        inputs = self._input_wizard.get_inputs()
        self._status_label.setText(self.tr("Calculation complete: %s") % recipe.recipe_id[:12])
        self._status_mode.setText(recipe.application.replace("_", " ").title())
        self._status_material.setText(
            "%s → %s" % (inputs.get("upper_material_type", "?"), inputs.get("lower_material_type", "?")))
        self._status_freq.setText("%.0f kHz" % inputs.get("frequency_khz", 20))
        self._status_power.setText("%.0f W" % inputs.get("max_power_w", 3500))

        self._history_panel.refresh()

    def _on_defaults_changed(self, frequency: float, max_power: float):
        self._input_wizard.set_defaults(frequency=frequency, max_power=max_power)
        self._status_freq.setText("%.0f kHz" % frequency)
        self._status_power.setText("%.0f W" % max_power)

    def _on_language_changed(self, lang_code: str):
        """Switch application language. lang_code is 'zh_CN' or 'en'."""
        app = QApplication.instance()
        app.removeTranslator(self._translator)
        app.removeTranslator(self._qt_translator)

        tr_path = _translations_path()
        self._translator.load("app_%s" % lang_code, tr_path)
        app.installTranslator(self._translator)

        # Also load Qt's own translations (button labels etc)
        qt_tr_path = QLibraryInfo.path(QLibraryInfo.TranslationsPath)
        self._qt_translator.load("qt_%s" % lang_code, qt_tr_path)
        app.installTranslator(self._qt_translator)

        self.retranslateUi()
        for panel in self._panels:
            if hasattr(panel, 'retranslateUi'):
                panel.retranslateUi()

    def _toggle_theme(self):
        new_theme = "dark" if self._current_theme == "light" else "light"
        self._apply_theme(new_theme)

    def _apply_theme(self, theme_name: str):
        self._current_theme = theme_name
        self.setStyleSheet(get_theme(theme_name))

    def _show_about(self):
        QMessageBox.about(self, self.tr("About UltrasonicWeldMaster"),
                          self.tr("UltrasonicWeldMaster v0.1.0\n\n"
                                  "Ultrasonic metal welding parameter\n"
                                  "auto-generation and experiment report tool.\n\n"
                                  "Plugin-based microkernel architecture."))
