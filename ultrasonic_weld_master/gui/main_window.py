"""Main application window with navigation and central workspace."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QListWidget,
    QStackedWidget, QStatusBar, QLabel, QListWidgetItem, QPushButton,
    QMessageBox,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction

from ultrasonic_weld_master.gui.themes import get_theme
from ultrasonic_weld_master.gui.panels.input_wizard import InputWizardPanel
from ultrasonic_weld_master.gui.panels.result_panel import ResultPanel
from ultrasonic_weld_master.gui.panels.report_panel import ReportPanel
from ultrasonic_weld_master.gui.panels.history_panel import HistoryPanel
from ultrasonic_weld_master.gui.panels.settings_panel import SettingsPanel


class MainWindow(QMainWindow):
    NAV_ITEMS = [
        ("New Calculation", "input_wizard"),
        ("Results", "results"),
        ("Reports", "reports"),
        ("History", "history"),
        ("Settings", "settings"),
    ]

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._current_theme = "light"
        self._setup_ui()
        self._apply_theme("light")

    def _setup_ui(self):
        self.setWindowTitle("UltrasonicWeldMaster v0.1.0")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 800)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        self._new_action = QAction("New Calculation", self)
        self._new_action.triggered.connect(lambda: self._navigate_to(0))
        file_menu.addAction(self._new_action)
        file_menu.addSeparator()
        self._quit_action = QAction("Quit", self)
        self._quit_action.triggered.connect(self.close)
        file_menu.addAction(self._quit_action)

        view_menu = menubar.addMenu("View")
        self._toggle_theme_action = QAction("Toggle Dark/Light Theme", self)
        self._toggle_theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self._toggle_theme_action)

        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left navigation
        self._nav_list = QListWidget()
        self._nav_list.setFixedWidth(180)
        self._nav_list.setIconSize(QSize(24, 24))
        for label, _ in self.NAV_ITEMS:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(170, 44))
            self._nav_list.addItem(item)
        self._nav_list.currentRowChanged.connect(self._navigate_to)
        main_layout.addWidget(self._nav_list)

        # Stacked workspace
        self._stack = QStackedWidget()
        self._input_wizard = InputWizardPanel(engine=self._engine)
        self._result_panel = ResultPanel()
        self._report_panel = ReportPanel(engine=self._engine)
        self._history_panel = HistoryPanel(engine=self._engine)
        self._settings_panel = SettingsPanel(on_theme_change=self._apply_theme)

        self._stack.addWidget(self._input_wizard)
        self._stack.addWidget(self._result_panel)
        self._stack.addWidget(self._report_panel)
        self._stack.addWidget(self._history_panel)
        self._stack.addWidget(self._settings_panel)
        main_layout.addWidget(self._stack, 1)

        # Connect wizard to result panel
        self._input_wizard.calculation_done.connect(self._on_calculation_done)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_label = QLabel("Ready")
        self._status_bar.addWidget(self._status_label)

        # Default selection
        self._nav_list.setCurrentRow(0)

    def _navigate_to(self, index: int):
        self._stack.setCurrentIndex(index)
        self._nav_list.setCurrentRow(index)

    def _on_calculation_done(self, recipe, validation):
        self._result_panel.show_results(recipe, validation)
        self._report_panel.set_recipe(recipe, validation)
        self._navigate_to(1)
        self._status_label.setText("Calculation complete: %s" % recipe.recipe_id)

    def _toggle_theme(self):
        new_theme = "dark" if self._current_theme == "light" else "light"
        self._apply_theme(new_theme)

    def _apply_theme(self, theme_name: str):
        self._current_theme = theme_name
        self.setStyleSheet(get_theme(theme_name))

    def _show_about(self):
        QMessageBox.about(self, "About UltrasonicWeldMaster",
                          "UltrasonicWeldMaster v0.1.0\n\n"
                          "Ultrasonic metal welding parameter\n"
                          "auto-generation and experiment report tool.\n\n"
                          "Plugin-based microkernel architecture.")
