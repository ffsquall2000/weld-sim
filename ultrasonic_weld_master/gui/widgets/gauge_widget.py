"""Half-arc gauge widget with QPainter — industrial oscilloscope style."""
from __future__ import annotations

import math

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, QRectF, QPointF, Property
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QRadialGradient, QConicalGradient


class GaugeWidget(QWidget):
    """Half-arc gauge displaying a value with safe window range."""

    def __init__(self, label: str = "", unit: str = "", min_val: float = 0,
                 max_val: float = 100, parent=None):
        super().__init__(parent)
        self._label = label
        self._unit = unit
        self._min_val = min_val
        self._max_val = max_val
        self._value = 0.0
        self._safe_min = min_val
        self._safe_max = max_val
        self.setMinimumSize(180, 160)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_value(self, value: float):
        self._value = max(self._min_val, min(value, self._max_val))
        self.update()

    def set_range(self, min_val: float, max_val: float):
        self._min_val = min_val
        self._max_val = max_val
        self.update()

    def set_safe_window(self, safe_min: float, safe_max: float):
        self._safe_min = safe_min
        self._safe_max = safe_max
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        side = min(w, h - 30)
        cx = w / 2
        cy = h / 2 + 5

        # Arc geometry
        arc_rect = QRectF(cx - side * 0.42, cy - side * 0.42, side * 0.84, side * 0.84)
        start_angle = 180  # degrees (left)
        span_angle = 180   # half circle

        # Background arc (dark track)
        pen = QPen(QColor("#30363d"), max(side * 0.06, 4))
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(arc_rect, start_angle * 16, span_angle * 16)

        # Safe window arc (green segment)
        if self._max_val > self._min_val:
            safe_start_frac = (self._safe_min - self._min_val) / (self._max_val - self._min_val)
            safe_end_frac = (self._safe_max - self._min_val) / (self._max_val - self._min_val)
            safe_start_frac = max(0, min(1, safe_start_frac))
            safe_end_frac = max(0, min(1, safe_end_frac))

            safe_start_deg = start_angle + (1 - safe_end_frac) * span_angle
            safe_span_deg = (safe_end_frac - safe_start_frac) * span_angle

            pen_safe = QPen(QColor("#4caf50"), max(side * 0.06, 4))
            pen_safe.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_safe)
            painter.drawArc(arc_rect, int(safe_start_deg * 16), int(safe_span_deg * 16))

        # Value arc (orange, from left to value)
        if self._max_val > self._min_val:
            val_frac = (self._value - self._min_val) / (self._max_val - self._min_val)
            val_frac = max(0, min(1, val_frac))

            # Color: green if in safe, orange if outside, red if way outside
            if self._safe_min <= self._value <= self._safe_max:
                arc_color = QColor("#ff9800")
            else:
                arc_color = QColor("#f44336")

            val_span_deg = val_frac * span_angle
            pen_val = QPen(arc_color, max(side * 0.035, 3))
            pen_val.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_val)
            # Draw from left (180°) to value position
            painter.drawArc(arc_rect, int((start_angle + span_angle - val_span_deg) * 16),
                          int(val_span_deg * 16))

        # Pointer needle
        if self._max_val > self._min_val:
            val_frac = (self._value - self._min_val) / (self._max_val - self._min_val)
            angle_rad = math.pi * (1 - val_frac)  # 180° to 0°
            needle_len = side * 0.32
            nx = cx + needle_len * math.cos(angle_rad)
            ny = cy - needle_len * math.sin(angle_rad)

            pen_needle = QPen(QColor("#ff9800"), max(side * 0.02, 2))
            pen_needle.setCapStyle(Qt.RoundCap)
            painter.setPen(pen_needle)
            painter.drawLine(QPointF(cx, cy), QPointF(nx, ny))

            # Center dot
            painter.setBrush(QColor("#ff9800"))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(cx, cy), side * 0.035, side * 0.035)

        # Digital value readout
        font_size = max(int(side * 0.14), 14)
        font = QFont("Menlo, SF Mono, Courier", font_size)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#ff9800"))

        val_text = "%.1f" % self._value
        painter.drawText(QRectF(0, cy + side * 0.08, w, font_size + 8),
                        Qt.AlignHCenter | Qt.AlignTop, val_text)

        # Unit
        unit_font = QFont("Menlo, SF Mono, Courier", max(int(side * 0.08), 9))
        painter.setFont(unit_font)
        painter.setPen(QColor("#8b949e"))
        painter.drawText(QRectF(0, cy + side * 0.08 + font_size + 4, w, 20),
                        Qt.AlignHCenter | Qt.AlignTop, self._unit)

        # Label at top
        label_font = QFont("system-ui, Helvetica", max(int(side * 0.08), 9))
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.setPen(QColor("#8b949e"))
        painter.drawText(QRectF(0, cy - side * 0.50, w, 20),
                        Qt.AlignHCenter | Qt.AlignBottom, self._label.upper())

        # Min/Max labels
        tiny_font = QFont("Menlo", max(int(side * 0.06), 7))
        painter.setFont(tiny_font)
        painter.setPen(QColor("#484f58"))
        painter.drawText(QRectF(cx - side * 0.48, cy + 2, 50, 14),
                        Qt.AlignLeft | Qt.AlignTop, str(round(self._min_val, 1)))
        painter.drawText(QRectF(cx + side * 0.28, cy + 2, 50, 14),
                        Qt.AlignRight | Qt.AlignTop, str(round(self._max_val, 1)))

        painter.end()
