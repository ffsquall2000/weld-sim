"""4-step input wizard: Application -> Materials -> Tooling -> Constraints."""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QSpinBox, QGroupBox, QFormLayout,
    QPushButton, QStackedWidget, QProgressBar, QMessageBox,
)
from PySide6.QtCore import Signal

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult


class InputWizardPanel(QWidget):
    calculation_done = Signal(object, object)  # (WeldRecipe, ValidationResult)

    APPLICATIONS = [
        ("Li-Battery Tab Welding", "li_battery_tab"),
        ("Li-Battery Busbar Welding", "li_battery_busbar"),
        ("Li-Battery Collector", "li_battery_collector"),
        ("General Metal Welding", "general_metal"),
    ]

    MATERIALS = ["Al", "Cu", "Ni", "Steel"]

    def __init__(self, engine=None, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._current_step = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        # Title
        title = QLabel("New Welding Parameter Calculation")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 3)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Step %v of 4")
        layout.addWidget(self._progress)

        # Steps stack
        self._steps = QStackedWidget()
        self._steps.addWidget(self._build_step1_application())
        self._steps.addWidget(self._build_step2_materials())
        self._steps.addWidget(self._build_step3_tooling())
        self._steps.addWidget(self._build_step4_constraints())
        layout.addWidget(self._steps, 1)

        # Navigation buttons
        btn_layout = QHBoxLayout()
        self._back_btn = QPushButton("Back")
        self._back_btn.setProperty("secondary", True)
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._go_next)
        self._calc_btn = QPushButton("Calculate")
        self._calc_btn.clicked.connect(self._do_calculate)
        self._calc_btn.setVisible(False)

        btn_layout.addStretch()
        btn_layout.addWidget(self._back_btn)
        btn_layout.addWidget(self._next_btn)
        btn_layout.addWidget(self._calc_btn)
        layout.addLayout(btn_layout)

        self._update_buttons()

    def _build_step1_application(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        grp = QGroupBox("Step 1: Select Application")
        form = QFormLayout(grp)
        self._app_combo = QComboBox()
        for label, _ in self.APPLICATIONS:
            self._app_combo.addItem(label)
        form.addRow("Application Type:", self._app_combo)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_step2_materials(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        grp_upper = QGroupBox("Step 2a: Upper Material (Foil Stack)")
        form1 = QFormLayout(grp_upper)
        self._upper_mat = QComboBox()
        self._upper_mat.addItems(self.MATERIALS)
        form1.addRow("Material:", self._upper_mat)
        self._upper_thick = QDoubleSpinBox()
        self._upper_thick.setRange(0.001, 10.0)
        self._upper_thick.setDecimals(3)
        self._upper_thick.setValue(0.012)
        self._upper_thick.setSuffix(" mm")
        form1.addRow("Foil Thickness:", self._upper_thick)
        self._upper_layers = QSpinBox()
        self._upper_layers.setRange(1, 200)
        self._upper_layers.setValue(40)
        form1.addRow("Number of Layers:", self._upper_layers)
        layout.addWidget(grp_upper)

        grp_lower = QGroupBox("Step 2b: Lower Material (Tab/Busbar)")
        form2 = QFormLayout(grp_lower)
        self._lower_mat = QComboBox()
        self._lower_mat.addItems(self.MATERIALS)
        self._lower_mat.setCurrentIndex(1)  # Cu
        form2.addRow("Material:", self._lower_mat)
        self._lower_thick = QDoubleSpinBox()
        self._lower_thick.setRange(0.01, 20.0)
        self._lower_thick.setDecimals(2)
        self._lower_thick.setValue(0.30)
        self._lower_thick.setSuffix(" mm")
        form2.addRow("Thickness:", self._lower_thick)
        layout.addWidget(grp_lower)

        layout.addStretch()
        return w

    def _build_step3_tooling(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        grp = QGroupBox("Step 3: Weld Geometry")
        form = QFormLayout(grp)
        self._weld_width = QDoubleSpinBox()
        self._weld_width.setRange(1.0, 50.0)
        self._weld_width.setValue(5.0)
        self._weld_width.setSuffix(" mm")
        form.addRow("Weld Width:", self._weld_width)
        self._weld_length = QDoubleSpinBox()
        self._weld_length.setRange(1.0, 100.0)
        self._weld_length.setValue(25.0)
        self._weld_length.setSuffix(" mm")
        form.addRow("Weld Length:", self._weld_length)
        layout.addWidget(grp)
        layout.addStretch()
        return w

    def _build_step4_constraints(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        grp = QGroupBox("Step 4: Equipment Constraints")
        form = QFormLayout(grp)
        self._frequency = QDoubleSpinBox()
        self._frequency.setRange(15.0, 40.0)
        self._frequency.setValue(20.0)
        self._frequency.setSuffix(" kHz")
        form.addRow("Frequency:", self._frequency)
        self._max_power = QDoubleSpinBox()
        self._max_power.setRange(500, 10000)
        self._max_power.setValue(3500)
        self._max_power.setSuffix(" W")
        form.addRow("Max Power:", self._max_power)
        layout.addWidget(grp)
        layout.addStretch()
        return w

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

    def _update_buttons(self):
        self._back_btn.setEnabled(self._current_step > 0)
        self._next_btn.setVisible(self._current_step < 3)
        self._calc_btn.setVisible(self._current_step == 3)
        self._progress.setValue(self._current_step)
        self._progress.setFormat("Step %d of 4" % (self._current_step + 1))

    def _do_calculate(self):
        app_index = self._app_combo.currentIndex()
        _, app_key = self.APPLICATIONS[app_index]

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
                QMessageBox.warning(self, "No Engine", "Engine not connected. Cannot calculate.")
        except Exception as e:
            QMessageBox.critical(self, "Calculation Error", str(e))

    def get_inputs(self) -> dict:
        app_index = self._app_combo.currentIndex()
        _, app_key = self.APPLICATIONS[app_index]
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
