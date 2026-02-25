"""Light and dark theme system for the GUI."""
from __future__ import annotations

import os

_STYLES_DIR = os.path.join(os.path.dirname(__file__), "resources", "styles")

LIGHT_THEME = """
QMainWindow { background-color: #f5f5f5; }
QWidget { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; }
QMenuBar { background-color: #ffffff; border-bottom: 1px solid #ddd; }
QMenuBar::item:selected { background-color: #e0e0e0; }
QToolBar { background-color: #fafafa; border-bottom: 1px solid #ddd; spacing: 4px; }
QStatusBar { background-color: #f0f0f0; border-top: 1px solid #ddd; color: #555; }
QListWidget { background-color: #ffffff; border: 1px solid #ddd; border-radius: 4px; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background-color: #4472C4; color: white; }
QStackedWidget { background-color: #ffffff; }
QGroupBox { border: 1px solid #ccc; border-radius: 6px; margin-top: 12px; padding-top: 16px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #ccc; border-radius: 4px; padding: 6px 8px;
    background-color: #ffffff; min-height: 24px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #4472C4;
}
QPushButton {
    background-color: #4472C4; color: white; border: none; border-radius: 4px;
    padding: 8px 20px; font-weight: bold; min-height: 28px;
}
QPushButton:hover { background-color: #3a62a8; }
QPushButton:pressed { background-color: #305090; }
QPushButton:disabled { background-color: #ccc; color: #888; }
QPushButton[secondary="true"] {
    background-color: #e0e0e0; color: #333; border: 1px solid #bbb;
}
QPushButton[secondary="true"]:hover { background-color: #d0d0d0; }
QTableWidget {
    gridline-color: #ddd; background-color: #fff; border: 1px solid #ddd;
    alternate-background-color: #f9f9f9;
}
QHeaderView::section {
    background-color: #4472C4; color: white; padding: 6px;
    border: none; font-weight: bold;
}
QTabWidget::pane { border: 1px solid #ddd; background-color: #fff; }
QTabBar::tab {
    padding: 8px 16px; background-color: #e8e8e8; border: 1px solid #ddd;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #fff; border-bottom-color: #fff; }
QProgressBar { border: 1px solid #ccc; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background-color: #4472C4; border-radius: 3px; }
QLabel#sectionTitle { font-size: 16px; font-weight: bold; color: #333; }
"""

DARK_THEME = """
QMainWindow { background-color: #1e1e1e; }
QWidget { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; font-size: 13px; color: #ddd; }
QMenuBar { background-color: #2d2d2d; border-bottom: 1px solid #444; color: #ddd; }
QMenuBar::item:selected { background-color: #3e3e3e; }
QToolBar { background-color: #252525; border-bottom: 1px solid #444; spacing: 4px; }
QStatusBar { background-color: #252525; border-top: 1px solid #444; color: #aaa; }
QListWidget { background-color: #2d2d2d; border: 1px solid #444; border-radius: 4px; color: #ddd; }
QListWidget::item { padding: 8px; }
QListWidget::item:selected { background-color: #4472C4; color: white; }
QStackedWidget { background-color: #2d2d2d; }
QGroupBox { border: 1px solid #555; border-radius: 6px; margin-top: 12px; padding-top: 16px; font-weight: bold; color: #ddd; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #555; border-radius: 4px; padding: 6px 8px;
    background-color: #3a3a3a; color: #ddd; min-height: 24px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #5b9bd5;
}
QPushButton {
    background-color: #4472C4; color: white; border: none; border-radius: 4px;
    padding: 8px 20px; font-weight: bold; min-height: 28px;
}
QPushButton:hover { background-color: #5b9bd5; }
QPushButton:pressed { background-color: #3a62a8; }
QPushButton:disabled { background-color: #555; color: #888; }
QPushButton[secondary="true"] {
    background-color: #3a3a3a; color: #ddd; border: 1px solid #555;
}
QPushButton[secondary="true"]:hover { background-color: #454545; }
QTableWidget {
    gridline-color: #444; background-color: #2d2d2d; border: 1px solid #444;
    alternate-background-color: #333; color: #ddd;
}
QHeaderView::section {
    background-color: #4472C4; color: white; padding: 6px;
    border: none; font-weight: bold;
}
QTabWidget::pane { border: 1px solid #444; background-color: #2d2d2d; }
QTabBar::tab {
    padding: 8px 16px; background-color: #333; border: 1px solid #444;
    border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; color: #ddd;
}
QTabBar::tab:selected { background-color: #2d2d2d; border-bottom-color: #2d2d2d; }
QProgressBar { border: 1px solid #555; border-radius: 4px; text-align: center; color: #ddd; }
QProgressBar::chunk { background-color: #5b9bd5; border-radius: 3px; }
QLabel#sectionTitle { font-size: 16px; font-weight: bold; color: #eee; }
"""


def get_theme(name: str = "light") -> str:
    if name == "dark":
        return DARK_THEME
    return LIGHT_THEME
