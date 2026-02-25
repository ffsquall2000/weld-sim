"""UltrasonicWeldMaster GUI application entry point."""
from __future__ import annotations

import sys
import os


def _translations_path() -> str:
    """Return path to translations directory, works in dev and PyInstaller."""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'translations')
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'translations')


def _detect_language() -> str:
    """Detect language from config or system locale. Returns 'zh_CN' or 'en'."""
    # 1. Check config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    if getattr(sys, 'frozen', False):
        config_path = os.path.join(sys._MEIPASS, 'config.yaml')
    try:
        import yaml
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        lang = cfg.get('language', '')
        if lang in ('zh_CN', 'en'):
            return lang
    except Exception:
        pass

    # 2. Fall back to system locale
    from PySide6.QtCore import QLocale
    locale_name = QLocale.system().name()  # e.g. "zh_CN", "en_US"
    if locale_name.startswith('zh'):
        return 'zh_CN'
    return 'en'


def main():
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTranslator, QLibraryInfo

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

    # Load translations based on detected language
    lang = _detect_language()
    tr_path = _translations_path()

    app_translator = QTranslator()
    if app_translator.load("app_%s" % lang, tr_path):
        app.installTranslator(app_translator)

    qt_translator = QTranslator()
    qt_tr_path = QLibraryInfo.path(QLibraryInfo.TranslationsPath)
    if qt_translator.load("qt_%s" % lang, qt_tr_path):
        app.installTranslator(qt_translator)

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
