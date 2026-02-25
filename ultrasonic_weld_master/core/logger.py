"""Three-tier structured logging system for UltrasonicWeldMaster."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Optional


class StructuredLogger:
    def __init__(self, log_dir: str = "data/logs"):
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._setup_app_logger()

    def _setup_app_logger(self) -> None:
        self._app_logger = logging.getLogger("uwm." + str(id(self)))
        if not self._app_logger.handlers:
            handler = RotatingFileHandler(
                os.path.join(self._log_dir, "app.log"),
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            self._app_logger.addHandler(handler)
            self._app_logger.setLevel(logging.DEBUG)

    @property
    def app(self) -> logging.Logger:
        return self._app_logger

    def _write_jsonl(self, filename: str, record: dict) -> None:
        filepath = os.path.join(self._log_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def log_operation(
        self,
        session_id: str,
        event_type: str,
        user_action: str = "",
        data: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": event_type,
            "user_action": user_action,
            "data": data or {},
            "metadata": metadata or {},
        }
        self._write_jsonl("operations.jsonl", record)

    def log_calculation(
        self,
        session_id: str,
        inputs: dict,
        outputs: dict,
        intermediate: Optional[dict] = None,
        validation: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "event_type": "calculation.completed",
            "inputs": inputs,
            "outputs": outputs,
            "intermediate": intermediate or {},
            "validation": validation or {},
            "metadata": metadata or {},
        }
        self._write_jsonl("calculations.jsonl", record)
