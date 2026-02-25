"""Publish-subscribe event bus for plugin communication."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)

EventHandler = Callable


class EventBus:
    def __init__(self, keep_history: bool = False):
        self._subscribers: dict = {}
        self._keep_history = keep_history
        self._history: list = []

    def subscribe(self, event: str, handler: EventHandler) -> None:
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: EventHandler) -> None:
        if event in self._subscribers:
            self._subscribers[event] = [
                h for h in self._subscribers[event] if h is not handler
            ]

    def emit(self, event: str, data: dict) -> None:
        record = {
            "event": event,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self._keep_history:
            self._history.append(record)

        handlers = list(self._subscribers.get(event, []))
        handlers.extend(self._subscribers.get("*", []))

        for handler in handlers:
            try:
                handler(data)
            except Exception:
                logger.exception("Event handler error for %s", event)

    def get_history(self) -> list:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()
