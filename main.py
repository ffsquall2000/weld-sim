"""UltrasonicWeldMaster GUI application entry point."""
from __future__ import annotations

import sys
import os


def main():
    from PySide6.QtWidgets import QApplication

    from ultrasonic_weld_master.core.engine import Engine
    from ultrasonic_weld_master.gui.main_window import MainWindow
    from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
    from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
    from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
    from ultrasonic_weld_master.plugins.knowledge_base.plugin import KnowledgeBasePlugin
    from ultrasonic_weld_master.plugins.reporter.plugin import ReporterPlugin

    app = QApplication(sys.argv)
    app.setApplicationName("UltrasonicWeldMaster")
    app.setApplicationVersion("0.1.0")

    # Initialize engine
    data_dir = os.path.join(os.path.expanduser("~"), ".ultrasonic_weld_master")
    engine = Engine(data_dir=data_dir)
    engine.initialize()

    # Register and activate plugins
    engine.plugin_manager.register(MaterialDBPlugin())
    engine.plugin_manager.register(LiBatteryPlugin())
    engine.plugin_manager.register(GeneralMetalPlugin())
    engine.plugin_manager.register(KnowledgeBasePlugin())
    engine.plugin_manager.register(ReporterPlugin())
    engine.plugin_manager.activate("material_db")
    engine.plugin_manager.activate("li_battery")
    engine.plugin_manager.activate("general_metal")
    engine.plugin_manager.activate("knowledge_base")
    engine.plugin_manager.activate("reporter")

    # Launch GUI
    window = MainWindow(engine=engine)
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
