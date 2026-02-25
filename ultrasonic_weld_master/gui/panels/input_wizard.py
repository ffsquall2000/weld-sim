"""4-step input wizard: Application -> Materials -> Tooling -> Constraints."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout,
    QPushButton, QStackedWidget, QMessageBox, QFrame,
)
from PySide6.QtCore import Signal, Qt, QCoreApplication
from PySide6.QtGui import QPainter, QColor, QFont, QPen

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult


class StepIndicator(QWidget):
    """Numbered step circles connected by lines."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current = 0
        self._labels = ["Application", "Materials", "Tooling", "Constraints"]
        self.setFixedHeight(60)

    def set_step(self, step: int):
        self._current = step
        self.update()

    def set_labels(self, labels: list):
        self._labels = labels
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        n = len(self._labels)
        spacing = (w - 80) / max(n - 1, 1)
        start_x = 40
        cy = 22
        radius = 14

        for i in range(n - 1):
            x1 = start_x + i * spacing + radius
            x2 = start_x + (i + 1) * spacing - radius
            color = QColor("#ff9800") if i < self._current else QColor("#30363d")
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(int(x1), cy, int(x2), cy)

        for i in range(n):
            cx = start_x + i * spacing
            if i <= self._current:
                painter.setBrush(QColor("#ff9800"))
                painter.setPen(Qt.NoPen)
            else:
                painter.setBrush(QColor("#21262d"))
                painter.setPen(QPen(QColor("#30363d"), 2))

            painter.drawEllipse(int(cx - radius), int(cy - radius), radius * 2, radius * 2)

            font = QFont("Menlo", 10)
            font.setBold(True)
            painter.setFont(font)
            text_color = QColor("#0d1117") if i <= self._current else QColor("#8b949e")
            painter.setPen(text_color)
            painter.drawText(int(cx - radius), int(cy - radius), radius * 2, radius * 2,
                           Qt.AlignCenter, str(i + 1))

            label_font = QFont("system-ui", 9)
            painter.setFont(label_font)
            painter.setPen(QColor("#8b949e") if i != self._current else QColor("#ff9800"))
            painter.drawText(int(cx - 45), cy + radius + 4, 90, 18,
                           Qt.AlignHCenter | Qt.AlignTop, self._labels[i])

        painter.end()


class InputWizardPanel(QWidget):
    calculation_done = Signal(object, object)  # (WeldRecipe, ValidationResult)

    APP_KEYS = ["li_battery_tab", "li_battery_busbar", "li_battery_collector", "general_metal"]
    MATERIALS = ["Al", "Cu", "Ni", "Steel"]

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._current_step = 0
        self._setup_ui()

    def set_defaults(self, frequency: float = None, max_power: float = None):
        if frequency is not None:
            self._frequency.setValue(frequency)
        if max_power is not None:
            self._max_power.setValue(max_power)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        self._title = QLabel("")
        self._title.setObjectName("sectionTitle")
        layout.addWidget(self._title)

        self._step_indicator = StepIndicator()
        layout.addWidget(self._step_indicator)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #30363d;")
        layout.addWidget(sep)

        # Build step widgets (store group/label refs for retranslation)
        self._steps = QStackedWidget()
        self._steps.addWidget(self._build_step1_application())
        self._steps.addWidget(self._build_step2_materials())
        self._steps.addWidget(self._build_step3_tooling())
        self._steps.addWidget(self._build_step4_constraints())
        layout.addWidget(self._steps, 1)

        btn_layout = QHBoxLayout()
        self._back_btn = QPushButton("")
        self._back_btn.setProperty("secondary", True)
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton("")
        self._next_btn.clicked.connect(self._go_next)
        self._calc_btn = QPushButton("")
        self._calc_btn.clicked.connect(self._do_calculate)
        self._calc_btn.setVisible(False)

        btn_layout.addStretch()
        btn_layout.addWidget(self._back_btn)
        btn_layout.addWidget(self._next_btn)
        btn_layout.addWidget(self._calc_btn)
        layout.addLayout(btn_layout)

        self.retranslateUi()
        self._update_buttons()

    def _build_step1_application(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(16)

        self._grp_app = QGroupBox("")
        form = QFormLayout(self._grp_app)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(12)
        form.setContentsMargins(16, 20, 16, 16)

        self._app_combo = QComboBox()
        # Items set in retranslateUi
        for _ in self.APP_KEYS:
            self._app_combo.addItem("")
        self._label_app = QLabel("")
        form.addRow(self._label_app, self._app_combo)

        self._desc_label = QLabel("")
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #8b949e; font-size: 12px; padding: 8px 0;")
        form.addRow(self._desc_label)

        layout.addWidget(self._grp_app)
        layout.addStretch()
        return w

    def _build_step2_materials(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(12)

        # Upper material
        self._grp_upper = QGroupBox("")
        form1 = QFormLayout(self._grp_upper)
        form1.setLabelAlignment(Qt.AlignRight)
        form1.setSpacing(12)
        form1.setContentsMargins(16, 20, 16, 16)
        self._upper_mat = QComboBox()
        self._upper_mat.addItems(self.MATERIALS)
        self._label_upper_mat = QLabel("")
        form1.addRow(self._label_upper_mat, self._upper_mat)
        self._upper_thick = QDoubleSpinBox()
        self._upper_thick.setRange(0.001, 10.0)
        self._upper_thick.setDecimals(3)
        self._upper_thick.setValue(0.012)
        self._upper_thick.setSuffix(" mm")
        self._label_upper_thick = QLabel("")
        form1.addRow(self._label_upper_thick, self._upper_thick)
        self._upper_layers = QSpinBox()
        self._upper_layers.setRange(1, 200)
        self._upper_layers.setValue(40)
        self._label_upper_layers = QLabel("")
        form1.addRow(self._label_upper_layers, self._upper_layers)
        layout.addWidget(self._grp_upper)

        # Lower material
        self._grp_lower = QGroupBox("")
        form2 = QFormLayout(self._grp_lower)
        form2.setLabelAlignment(Qt.AlignRight)
        form2.setSpacing(12)
        form2.setContentsMargins(16, 20, 16, 16)
        self._lower_mat = QComboBox()
        self._lower_mat.addItems(self.MATERIALS)
        self._lower_mat.setCurrentIndex(1)  # Cu
        self._label_lower_mat = QLabel("")
        form2.addRow(self._label_lower_mat, self._lower_mat)
        self._lower_thick = QDoubleSpinBox()
        self._lower_thick.setRange(0.01, 20.0)
        self._lower_thick.setDecimals(2)
        self._lower_thick.setValue(0.30)
        self._lower_thick.setSuffix(" mm")
        self._label_lower_thick = QLabel("")
        form2.addRow(self._label_lower_thick, self._lower_thick)
        layout.addWidget(self._grp_lower)

        layout.addStretch()
        return w

    def _build_step3_tooling(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(16)

        self._grp_geometry = QGroupBox("")
        form = QFormLayout(self._grp_geometry)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(12)
        form.setContentsMargins(16, 20, 16, 16)
        self._weld_width = QDoubleSpinBox()
        self._weld_width.setRange(1.0, 50.0)
        self._weld_width.setValue(5.0)
        self._weld_width.setSuffix(" mm")
        self._label_width = QLabel("")
        form.addRow(self._label_width, self._weld_width)
        self._weld_length = QDoubleSpinBox()
        self._weld_length.setRange(1.0, 100.0)
        self._weld_length.setValue(25.0)
        self._weld_length.setSuffix(" mm")
        self._label_length = QLabel("")
        form.addRow(self._label_length, self._weld_length)

        self._area_label = QLabel("")
        self._area_label.setStyleSheet("color: #ff9800; font-family: 'Menlo', monospace; font-size: 12px;")
        self._label_area = QLabel("")
        form.addRow(self._label_area, self._area_label)
        self._update_area_display()
        self._weld_width.valueChanged.connect(self._update_area_display)
        self._weld_length.valueChanged.connect(self._update_area_display)

        layout.addWidget(self._grp_geometry)
        layout.addStretch()
        return w

    def _update_area_display(self):
        area = self._weld_width.value() * self._weld_length.value()
        self._area_label.setText("%.1f mm²" % area)

    def _build_step4_constraints(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(16)

        self._grp_constraints = QGroupBox("")
        form = QFormLayout(self._grp_constraints)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(12)
        form.setContentsMargins(16, 20, 16, 16)
        self._frequency = QDoubleSpinBox()
        self._frequency.setRange(15.0, 40.0)
        self._frequency.setValue(20.0)
        self._frequency.setSuffix(" kHz")
        self._label_freq = QLabel("")
        form.addRow(self._label_freq, self._frequency)
        self._max_power = QDoubleSpinBox()
        self._max_power.setRange(500, 10000)
        self._max_power.setValue(3500)
        self._max_power.setSuffix(" W")
        self._label_power = QLabel("")
        form.addRow(self._label_power, self._max_power)
        layout.addWidget(self._grp_constraints)

        self._grp_summary = QGroupBox("")
        summary_form = QFormLayout(self._grp_summary)
        summary_form.setContentsMargins(16, 20, 16, 16)
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setStyleSheet("font-family: 'Menlo', 'SF Mono', monospace; font-size: 12px; color: #e6edf3;")
        summary_form.addRow(self._summary_label)
        layout.addWidget(self._grp_summary)

        layout.addStretch()
        return w

    def retranslateUi(self):
        self._title.setText(self.tr("NEW CALCULATION"))

        self._step_indicator.set_labels([
            self.tr("Application"), self.tr("Materials"),
            self.tr("Tooling"), self.tr("Constraints"),
        ])

        self._back_btn.setText(self.tr("Back"))
        self._next_btn.setText(self.tr("Next Step"))
        self._calc_btn.setText(self.tr("Calculate Parameters"))

        # Step 1
        self._grp_app.setTitle(self.tr("APPLICATION TYPE"))
        self._label_app.setText(self.tr("Application:"))
        app_names = [
            self.tr("Li-Battery Tab Welding"),
            self.tr("Li-Battery Busbar Welding"),
            self.tr("Li-Battery Collector"),
            self.tr("General Metal Welding"),
        ]
        for i, name in enumerate(app_names):
            self._app_combo.setItemText(i, name)
        self._desc_label.setText(self.tr(
            "Select the welding application type. This determines the "
            "physics model and parameter ranges used for calculation."))

        # Step 2
        self._grp_upper.setTitle(self.tr("UPPER MATERIAL (FOIL STACK)"))
        self._label_upper_mat.setText(self.tr("Material:"))
        self._label_upper_thick.setText(self.tr("Foil Thickness:"))
        self._label_upper_layers.setText(self.tr("Number of Layers:"))
        self._grp_lower.setTitle(self.tr("LOWER MATERIAL (TAB / BUSBAR)"))
        self._label_lower_mat.setText(self.tr("Material:"))
        self._label_lower_thick.setText(self.tr("Thickness:"))

        # Step 3
        self._grp_geometry.setTitle(self.tr("WELD GEOMETRY"))
        self._label_width.setText(self.tr("Weld Width:"))
        self._label_length.setText(self.tr("Weld Length:"))
        self._label_area.setText(self.tr("Contact Area:"))

        # Step 4
        self._grp_constraints.setTitle(self.tr("EQUIPMENT CONSTRAINTS"))
        self._label_freq.setText(self.tr("Frequency:"))
        self._label_power.setText(self.tr("Max Power:"))
        self._grp_summary.setTitle(self.tr("INPUT SUMMARY"))

    def _update_summary(self):
        inputs = self.get_inputs()
        lines = [
            "App: %s" % inputs["application"],
            "Upper: %s x%d @ %.3f mm" % (inputs["upper_material_type"], inputs["upper_layers"], inputs["upper_thickness_mm"]),
            "Lower: %s @ %.2f mm" % (inputs["lower_material_type"], inputs["lower_thickness_mm"]),
            "Weld: %.1f x %.1f mm" % (inputs["weld_width_mm"], inputs["weld_length_mm"]),
            "Freq: %.1f kHz | Power: %.0f W" % (inputs["frequency_khz"], inputs["max_power_w"]),
        ]
        self._summary_label.setText("\n".join(lines))

    def _go_back(self):
        if self._current_step > 0:
            self._current_step -= 1
            self._steps.setCurrentIndex(self._current_step)
            self._update_buttons()

    def _go_next(self):
        if self._current_step < 3:
            self._current_step += 1
            self._steps.setCurrentIndex(self._current_step)
            self._update_buttons()
            if self._current_step == 3:
                self._update_summary()

    def _update_buttons(self):
        self._back_btn.setEnabled(self._current_step > 0)
        self._next_btn.setVisible(self._current_step < 3)
        self._calc_btn.setVisible(self._current_step == 3)
        self._step_indicator.set_step(self._current_step)

    def _do_calculate(self):
        app_index = self._app_combo.currentIndex()
        app_key = self.APP_KEYS[app_index]

        inputs = {
            "application": app_key,
            "upper_material_type": self._upper_mat.currentText(),
            "upper_thickness_mm": self._upper_thick.value(),
            "upper_layers": self._upper_layers.value(),
            "lower_material_type": self._lower_mat.currentText(),
            "lower_thickness_mm": self._lower_thick.value(),
            "weld_width_mm": self._weld_width.value(),
            "weld_length_mm": self._weld_length.value(),
            "frequency_khz": self._frequency.value(),
            "max_power_w": self._max_power.value(),
        }

        try:
            if self._engine:
                plugin_name = "li_battery" if app_key.startswith("li_battery") else "general_metal"
                plugin = self._engine.plugin_manager.get_plugin(plugin_name)
                recipe = plugin.calculate_parameters(inputs)
                validation = plugin.validate_parameters(recipe)
                self.calculation_done.emit(recipe, validation)
            else:
                QMessageBox.warning(self, self.tr("No Engine"),
                                    self.tr("Engine not connected. Cannot calculate."))
        except Exception as e:
            QMessageBox.critical(self, self.tr("Calculation Error"), str(e))

    def get_inputs(self) -> dict:
        app_index = self._app_combo.currentIndex()
        app_key = self.APP_KEYS[app_index]
        return {
            "application": app_key,
            "upper_material_type": self._upper_mat.currentText(),
            "upper_thickness_mm": self._upper_thick.value(),
            "upper_layers": self._upper_layers.value(),
            "lower_material_type": self._lower_mat.currentText(),
            "lower_thickness_mm": self._lower_thick.value(),
            "weld_width_mm": self._weld_width.value(),
            "weld_length_mm": self._weld_length.value(),
            "frequency_khz": self._frequency.value(),
            "max_power_w": self._max_power.value(),
        }
