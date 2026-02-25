"""Core data models for UltrasonicWeldMaster."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional


class ValidationStatus(enum.Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class RiskLevel(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MaterialInfo:
    name: str
    material_type: str
    thickness_mm: float
    layers: int = 1
    temper: str = ""
    properties: dict = field(default_factory=dict)

    @property
    def total_thickness_mm(self) -> float:
        return self.thickness_mm * self.layers


@dataclass
class SonotrodeInfo:
    name: str
    sonotrode_type: str
    material: str = "Titanium"
    knurl_type: str = "linear"
    knurl_pitch_mm: float = 1.0
    knurl_depth_mm: float = 0.3
    contact_width_mm: float = 5.0
    contact_length_mm: float = 25.0
    properties: dict = field(default_factory=dict)


@dataclass
class WeldInputs:
    application: str
    upper_material: MaterialInfo
    lower_material: MaterialInfo
    weld_width_mm: float
    weld_length_mm: float
    frequency_khz: float = 20.0
    max_power_w: float = 3000.0
    sonotrode: Optional[SonotrodeInfo] = None
    anvil: Optional[SonotrodeInfo] = None
    target_peel_force_n: Optional[float] = None
    target_resistance_mohm: Optional[float] = None
    target_cpk: float = 1.67
    extra: dict = field(default_factory=dict)

    @property
    def weld_area_mm2(self) -> float:
        return self.weld_width_mm * self.weld_length_mm


@dataclass
class WeldRecipe:
    recipe_id: str
    application: str
    inputs: dict
    parameters: dict
    safety_window: dict = field(default_factory=dict)
    quality_estimate: dict = field(default_factory=dict)
    risk_assessment: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationResult:
    status: ValidationStatus
    validators: dict
    messages: list = field(default_factory=list)

    def is_passed(self) -> bool:
        return self.status == ValidationStatus.PASS

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "validators": self.validators,
            "messages": self.messages,
        }
