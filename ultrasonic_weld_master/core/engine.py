"""Core engine - the microkernel that ties everything together."""
from __future__ import annotations

import os
from typing import Optional

from ultrasonic_weld_master.core.config import AppConfig
from ultrasonic_weld_master.core.database import Database
from ultrasonic_weld_master.core.event_bus import EventBus
from ultrasonic_weld_master.core.logger import StructuredLogger
from ultrasonic_weld_master.core.plugin_manager import PluginManager


class Engine:
    def __init__(self, config_path: Optional[str] = None, data_dir: str = "data",
                 db_check_same_thread: bool = True):
        self._config_path = config_path
        self._data_dir = data_dir
        self._db_check_same_thread = db_check_same_thread
        self.config: Optional[AppConfig] = None
        self.event_bus: Optional[EventBus] = None
        self.database: Optional[Database] = None
        self.logger: Optional[StructuredLogger] = None
        self.plugin_manager: Optional[PluginManager] = None
        self._current_session: Optional[str] = None

    def initialize(self) -> None:
        os.makedirs(self._data_dir, exist_ok=True)
        os.makedirs(os.path.join(self._data_dir, "logs"), exist_ok=True)

        self.config = AppConfig(self._config_path)
        self.event_bus = EventBus(keep_history=True)
        self.logger = StructuredLogger(log_dir=os.path.join(self._data_dir, "logs"))
        self.database = Database(
            os.path.join(self._data_dir, "database.sqlite"),
            check_same_thread=self._db_check_same_thread,
        )
        self.database.initialize()
        self.plugin_manager = PluginManager(
            config=self.config, event_bus=self.event_bus, logger=self.logger,
        )
        self.logger.app.info("Engine initialized")

    def create_session(self, project_id: str, user_name: str = "") -> str:
        sid = self.database.create_session(project_id, user_name)
        self._current_session = sid
        self.event_bus.emit("session.created", {"session_id": sid, "project_id": project_id})
        return sid

    @property
    def current_session(self) -> Optional[str]:
        return self._current_session

    def shutdown(self) -> None:
        if self._current_session and self.database:
            self.database.end_session(self._current_session)
        if self.plugin_manager:
            self.plugin_manager.deactivate_all()
        if self.database:
            self.database.close()
        if self.logger:
            self.logger.app.info("Engine shutdown")
