"""Circular risk indicator with glow effect — industrial style."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QFont, QRadialGradient


_RISK_COLORS = {
    "low": QColor("#4caf50"),
    "medium": QColor("#ff9800"),
    "high": QColor("#f44336"),
    "critical": QColor("#d32f2f"),
}


class RiskIndicator(QWidget):
    """Circular risk level indicator with glow effect."""

    def __init__(self, label: str = "", parent=None):
        super().__init__(parent)
        self._label = label
        self._level = "low"
        self.setMinimumSize(90, 100)
        self.setMaximumSize(140, 130)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def set_level(self, level: str):
        self._level = level.lower()
        self.update()

    def set_label(self, label: str):
        self._label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        color = _RISK_COLORS.get(self._level, QColor("#8b949e"))

        cx = w / 2
        cy = h / 2 - 8
        radius = min(w, h - 30) * 0.28

        # Outer glow
        glow = QRadialGradient(QPointF(cx, cy), radius * 2.2)
        glow.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 60))
        glow.setColorAt(0.5, QColor(color.red(), color.green(), color.blue(), 20))
        glow.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setBrush(glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius * 2.2, radius * 2.2)

        # Dark ring
        painter.setBrush(QColor("#21262d"))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius * 1.2, radius * 1.2)

        # Inner colored circle
        inner_glow = QRadialGradient(QPointF(cx - radius * 0.2, cy - radius * 0.2), radius * 1.0)
        bright = QColor(color.red(), color.green(), color.blue(), 255)
        inner_glow.setColorAt(0, bright.lighter(140))
        inner_glow.setColorAt(0.7, bright)
        inner_glow.setColorAt(1.0, bright.darker(130))
        painter.setBrush(inner_glow)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Level text inside circle
        level_font = QFont("Menlo, SF Mono, Courier", max(int(radius * 0.55), 7))
        level_font.setBold(True)
        painter.setFont(level_font)
        painter.setPen(QColor("#0d1117"))
        painter.drawText(QRectF(cx - radius, cy - radius, radius * 2, radius * 2),
                        Qt.AlignCenter, self._level.upper()[:3])

        # Label below
        label_font = QFont("system-ui, Helvetica", max(int(radius * 0.45), 8))
        painter.setFont(label_font)
        painter.setPen(QColor("#8b949e"))
        text = self._label.replace("_", " ").title()
        painter.drawText(QRectF(0, cy + radius * 1.5, w, 20),
                        Qt.AlignHCenter | Qt.AlignTop, text)

        painter.end()
