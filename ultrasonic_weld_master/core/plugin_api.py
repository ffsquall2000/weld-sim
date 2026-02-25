"""Plugin standard interfaces (ABCs) for UltrasonicWeldMaster."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    dependencies: list = field(default_factory=list)


class PluginBase(ABC):
    @abstractmethod
    def get_info(self) -> PluginInfo:
        ...

    @abstractmethod
    def activate(self, context: Any) -> None:
        ...

    @abstractmethod
    def deactivate(self) -> None:
        ...

    def get_config_schema(self) -> Optional[dict]:
        return None

    def get_ui_panels(self) -> list:
        return []


class ParameterEnginePlugin(PluginBase):
    @abstractmethod
    def get_input_schema(self) -> dict:
        ...

    @abstractmethod
    def calculate_parameters(self, inputs: dict) -> Any:
        ...

    @abstractmethod
    def validate_parameters(self, recipe: Any) -> Any:
        ...

    @abstractmethod
    def get_supported_applications(self) -> list:
        ...
