"""Small LED indicator dot for navigation items."""
from __future__ import annotations

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QRadialGradient


_LED_COLORS = {
    "off": QColor("#30363d"),
    "active": QColor("#ff9800"),
    "success": QColor("#4caf50"),
    "error": QColor("#f44336"),
}


class StatusLED(QWidget):
    """Tiny LED dot indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._state = "off"
        self.setFixedSize(12, 12)

    def set_state(self, state: str):
        self._state = state
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        color = _LED_COLORS.get(self._state, _LED_COLORS["off"])
        cx = self.width() / 2
        cy = self.height() / 2
        radius = 4

        if self._state != "off":
            # Glow
            glow = QRadialGradient(QPointF(cx, cy), radius * 2)
            glow.setColorAt(0, QColor(color.red(), color.green(), color.blue(), 80))
            glow.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            painter.setBrush(glow)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy), radius * 2, radius * 2)

        # Dot
        dot_grad = QRadialGradient(QPointF(cx - 1, cy - 1), radius)
        dot_grad.setColorAt(0, color.lighter(150))
        dot_grad.setColorAt(1, color)
        painter.setBrush(dot_grad)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        painter.end()
