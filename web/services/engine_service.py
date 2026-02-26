"""Thread-safe singleton wrapping the core Engine for web use."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from ultrasonic_weld_master.core.engine import Engine
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin
from ultrasonic_weld_master.plugins.li_battery.plugin import LiBatteryPlugin
from ultrasonic_weld_master.plugins.general_metal.plugin import GeneralMetalPlugin
from ultrasonic_weld_master.plugins.knowledge_base.plugin import KnowledgeBasePlugin
from ultrasonic_weld_master.plugins.reporter.plugin import ReporterPlugin

logger = logging.getLogger(__name__)

# Application name -> plugin name mapping
_APP_PLUGIN_MAP: dict[str, str] = {
    "li_battery_tab": "li_battery",
    "li_battery_busbar": "li_battery",
    "li_battery_collector": "li_battery",
    "general_metal": "general_metal",
}


class EngineService:
    """Thread-safe singleton that manages the core Engine lifecycle."""

    _instance: Optional[EngineService] = None
    _lock = threading.Lock()

    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._engine: Optional[Engine] = None

    # ------------------------------------------------------------------
    # Singleton access
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, data_dir: str = "data") -> EngineService:
        """Return (or create) the singleton EngineService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(data_dir=data_dir)
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Create the Engine, register and activate all plugins."""
        with self._lock:
            if self._engine is not None:
                return  # already initialised

            engine = Engine(data_dir=self._data_dir)
            engine.initialize()

            # Core plugins
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

            # Optional geometry analyser (heavy deps may not be installed)
            try:
                from ultrasonic_weld_master.plugins.geometry_analyzer.plugin import (
                    GeometryAnalyzerPlugin,
                )
                engine.plugin_manager.register(GeometryAnalyzerPlugin())
                engine.plugin_manager.activate("geometry_analyzer")
            except ImportError:
                logger.info("geometry_analyzer plugin not available, skipping")

            self._engine = engine
            logger.info("EngineService initialised (data_dir=%s)", self._data_dir)

    def shutdown(self) -> None:
        """Shutdown the engine and release the singleton."""
        with self._lock:
            if self._engine is not None:
                self._engine.shutdown()
                self._engine = None
            EngineService._instance = None
            logger.info("EngineService shut down")

    # ------------------------------------------------------------------
    # Calculation
    # ------------------------------------------------------------------
    def calculate(
        self, application: str, inputs: dict
    ) -> tuple[WeldRecipe, ValidationResult]:
        """Run calculation and validation for the given application.

        *application* is mapped to a plugin name via ``_APP_PLUGIN_MAP``.
        Returns ``(recipe, validation_result)``.
        """
        plugin_name = _APP_PLUGIN_MAP.get(application)
        if plugin_name is None:
            raise ValueError(
                f"Unknown application '{application}'. "
                f"Supported: {list(_APP_PLUGIN_MAP.keys())}"
            )

        plugin = self._engine.plugin_manager.get_plugin(plugin_name)
        recipe = plugin.calculate_parameters(inputs)
        validation = plugin.validate_parameters(recipe)
        return recipe, validation

    # ------------------------------------------------------------------
    # Material queries
    # ------------------------------------------------------------------
    def get_materials(self) -> list[str]:
        """Return list of material type names."""
        plugin = self._engine.plugin_manager.get_plugin("material_db")
        return plugin.list_materials()

    def get_material(self, material_type: str) -> Optional[dict]:
        """Return properties for a single material type."""
        plugin = self._engine.plugin_manager.get_plugin("material_db")
        return plugin.get_material(material_type)

    def get_material_combination(self, mat_a: str, mat_b: str) -> dict:
        """Return combination properties for two material types."""
        plugin = self._engine.plugin_manager.get_plugin("material_db")
        return plugin.get_combination_properties(mat_a, mat_b)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def export_report(
        self,
        recipe: WeldRecipe,
        validation: Optional[ValidationResult] = None,
        fmt: str = "json",
        output_dir: str = ".",
    ) -> str:
        """Export a report in the requested format. Returns the output path."""
        plugin = self._engine.plugin_manager.get_plugin("reporter")
        exporters = {
            "json": plugin.export_json,
            "excel": plugin.export_excel,
            "pdf": plugin.export_pdf,
        }
        exporter = exporters.get(fmt)
        if exporter is None:
            raise ValueError(
                f"Unknown report format '{fmt}'. Supported: {list(exporters.keys())}"
            )
        return exporter(recipe, validation, output_dir)
