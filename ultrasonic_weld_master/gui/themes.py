"""Industrial dark theme system — oscilloscope orange on near-black."""
from __future__ import annotations

DARK_THEME = """
/* ── Global ── */
QMainWindow { background-color: #0d1117; }
QWidget {
    font-family: "SF Pro", "Helvetica Neue", "Segoe UI", system-ui, sans-serif;
    font-size: 13px; color: #e6edf3;
}

/* ── Menu Bar ── */
QMenuBar { background-color: #0d1117; border-bottom: 1px solid #30363d; color: #e6edf3; }
QMenuBar::item { padding: 6px 12px; }
QMenuBar::item:selected { background-color: #21262d; color: #ff9800; }
QMenu { background-color: #161b22; border: 1px solid #30363d; color: #e6edf3; padding: 4px 0; }
QMenu::item { padding: 6px 24px; }
QMenu::item:selected { background-color: #ff9800; color: #0d1117; }

/* ── Status Bar ── */
QStatusBar {
    background-color: #0d1117; border-top: 1px solid #30363d;
    color: #ff9800; font-family: "Menlo", "SF Mono", "Courier New", monospace; font-size: 12px;
}
QStatusBar QLabel { color: #ff9800; }

/* ── Navigation List ── */
QListWidget {
    background-color: #0d1117; border: none; border-right: 1px solid #30363d;
    outline: none; color: #8b949e;
}
QListWidget::item {
    padding: 12px 16px; border-left: 3px solid transparent;
    font-weight: bold; font-size: 13px;
}
QListWidget::item:hover {
    background-color: #161b22; color: #e6edf3; border-left: 3px solid #30363d;
}
QListWidget::item:selected {
    background-color: #161b22; color: #ff9800; border-left: 3px solid #ff9800;
}

/* ── Stacked Widget / Panels ── */
QStackedWidget { background-color: #0d1117; }

/* ── Group Box (Cards) ── */
QGroupBox {
    background-color: #161b22; border: 1px solid #30363d; border-radius: 8px;
    margin-top: 16px; padding: 20px 16px 12px 16px;
    font-weight: bold; color: #e6edf3;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 16px; padding: 0 8px;
    color: #ff9800; font-size: 12px; text-transform: uppercase;
}

/* ── Inputs ── */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px;
    padding: 8px 10px; color: #e6edf3; min-height: 26px;
    font-family: "Menlo", "SF Mono", monospace;
    selection-background-color: #ff9800; selection-color: #0d1117;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #ff9800;
}
QComboBox::drop-down {
    border: none; width: 24px;
}
QComboBox QAbstractItemView {
    background-color: #161b22; border: 1px solid #30363d; color: #e6edf3;
    selection-background-color: #ff9800; selection-color: #0d1117;
}

/* ── Buttons ── */
QPushButton {
    background-color: #ff9800; color: #0d1117; border: none; border-radius: 6px;
    padding: 10px 24px; font-weight: bold; min-height: 30px; font-size: 13px;
}
QPushButton:hover { background-color: #ffb74d; }
QPushButton:pressed { background-color: #e68900; }
QPushButton:disabled { background-color: #21262d; color: #484f58; }
QPushButton[secondary="true"] {
    background-color: #21262d; color: #8b949e; border: 1px solid #30363d;
}
QPushButton[secondary="true"]:hover { background-color: #30363d; color: #e6edf3; }

/* ── Tables ── */
QTableWidget {
    background-color: #0d1117; gridline-color: #21262d; border: 1px solid #30363d;
    alternate-background-color: #161b22; color: #e6edf3; border-radius: 4px;
    font-family: "Menlo", "SF Mono", monospace; font-size: 12px;
}
QTableWidget::item { padding: 4px 8px; }
QTableWidget::item:selected { background-color: #ff9800; color: #0d1117; }
QHeaderView::section {
    background-color: #21262d; color: #ff9800; padding: 8px 6px;
    border: none; border-bottom: 2px solid #ff9800; font-weight: bold;
    font-size: 11px; text-transform: uppercase;
}

/* ── Progress Bar ── */
QProgressBar {
    background-color: #21262d; border: none; border-radius: 4px;
    text-align: center; color: #0d1117; font-weight: bold; min-height: 8px; max-height: 8px;
}
QProgressBar::chunk { background-color: #ff9800; border-radius: 4px; }

/* ── Text Edit / Browser ── */
QTextEdit, QTextBrowser {
    background-color: #0d1117; border: 1px solid #30363d; border-radius: 4px;
    color: #e6edf3; font-family: "Menlo", "SF Mono", monospace; font-size: 12px;
    padding: 8px;
}

/* ── Splitter ── */
QSplitter::handle { background-color: #30363d; width: 1px; }
QSplitter::handle:hover { background-color: #ff9800; }

/* ── Scroll Bars ── */
QScrollBar:vertical {
    background-color: #0d1117; width: 8px; border: none;
}
QScrollBar::handle:vertical {
    background-color: #30363d; border-radius: 4px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background-color: #484f58; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background-color: #0d1117; height: 8px; border: none;
}
QScrollBar::handle:horizontal {
    background-color: #30363d; border-radius: 4px; min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background-color: #484f58; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Section Title ── */
QLabel#sectionTitle {
    font-size: 18px; font-weight: bold; color: #ff9800;
    font-family: "SF Pro", "Helvetica Neue", system-ui;
    padding-bottom: 4px;
}

/* ── Form Labels ── */
QLabel { color: #8b949e; }
"""

LIGHT_THEME = """
QMainWindow { background-color: #f8f9fa; }
QWidget { font-family: "SF Pro", "Helvetica Neue", "Segoe UI", system-ui, sans-serif; font-size: 13px; color: #1a1a2e; }
QMenuBar { background-color: #fff; border-bottom: 1px solid #e0e0e0; }
QMenuBar::item:selected { background-color: #f0f0f0; }
QStatusBar { background-color: #f0f0f0; border-top: 1px solid #e0e0e0; color: #555; }
QListWidget { background-color: #fff; border: none; border-right: 1px solid #e0e0e0; }
QListWidget::item { padding: 12px 16px; border-left: 3px solid transparent; font-weight: bold; }
QListWidget::item:selected { background-color: #fff3e0; color: #e65100; border-left: 3px solid #ff9800; }
QStackedWidget { background-color: #f8f9fa; }
QGroupBox { background-color: #fff; border: 1px solid #e0e0e0; border-radius: 8px; margin-top: 16px; padding: 20px 16px 12px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; color: #e65100; font-size: 12px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 4px; padding: 8px; min-height: 26px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #ff9800; }
QPushButton { background-color: #ff9800; color: white; border: none; border-radius: 6px; padding: 10px 24px; font-weight: bold; min-height: 30px; }
QPushButton:hover { background-color: #ffb74d; }
QPushButton:pressed { background-color: #e68900; }
QPushButton:disabled { background-color: #e0e0e0; color: #999; }
QPushButton[secondary="true"] { background-color: #f0f0f0; color: #555; border: 1px solid #ddd; }
QTableWidget { background-color: #fff; gridline-color: #e8e8e8; border: 1px solid #e0e0e0; alternate-background-color: #fafafa; }
QHeaderView::section { background-color: #ff9800; color: white; padding: 8px 6px; border: none; font-weight: bold; }
QProgressBar { background-color: #e0e0e0; border: none; border-radius: 4px; min-height: 8px; max-height: 8px; }
QProgressBar::chunk { background-color: #ff9800; border-radius: 4px; }
QLabel#sectionTitle { font-size: 18px; font-weight: bold; color: #e65100; }
"""


def get_theme(name: str = "dark") -> str:
    if name == "light":
        return LIGHT_THEME
    return DARK_THEME
