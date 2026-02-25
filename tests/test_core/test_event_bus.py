from __future__ import annotations
import pytest
from ultrasonic_weld_master.core.event_bus import EventBus

class TestEventBus:
    def test_subscribe_and_emit(self):
        bus = EventBus()
        received = []
        bus.subscribe("test.event", lambda data: received.append(data))
        bus.emit("test.event", {"key": "value"})
        assert len(received) == 1
        assert received[0]["key"] == "value"

    def test_multiple_subscribers(self):
        bus = EventBus()
        results = []
        bus.subscribe("calc.done", lambda d: results.append("A"))
        bus.subscribe("calc.done", lambda d: results.append("B"))
        bus.emit("calc.done", {})
        assert results == ["A", "B"]

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        handler = lambda d: received.append(d)
        bus.subscribe("ev", handler)
        bus.unsubscribe("ev", handler)
        bus.emit("ev", {"x": 1})
        assert len(received) == 0

    def test_emit_unregistered_event(self):
        bus = EventBus()
        bus.emit("no.listener", {})

    def test_event_history(self):
        bus = EventBus(keep_history=True)
        bus.emit("a", {"v": 1})
        bus.emit("b", {"v": 2})
        history = bus.get_history()
        assert len(history) == 2
        assert history[0]["event"] == "a"

    def test_wildcard_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda d: received.append(d))
        bus.emit("any.event", {"x": 1})
        assert len(received) == 1
