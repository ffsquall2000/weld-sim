"""Domain logic modules migrated from the original ultrasonic_weld_master package."""

from backend.app.domain.physics import PhysicsModel
from backend.app.domain.horn_generator import HornGenerator, HornParams, HornGenerationResult
from backend.app.domain.knurl_optimizer import KnurlOptimizer, KnurlConfig, KnurlScore
from backend.app.domain.material_properties import FEA_MATERIALS, get_material, list_materials
from backend.app.domain.knowledge_rules import KnowledgeRuleEngine

__all__ = [
    "PhysicsModel",
    "HornGenerator",
    "HornParams",
    "HornGenerationResult",
    "KnurlOptimizer",
    "KnurlConfig",
    "KnurlScore",
    "FEA_MATERIALS",
    "get_material",
    "list_materials",
    "KnowledgeRuleEngine",
]
