"""JSON report exporter."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult


class JsonExporter:
    def export(self, recipe: WeldRecipe, validation: Optional[ValidationResult] = None,
               output_dir: str = ".") -> str:
        report = {
            "report_format": "json",
            "generated_at": datetime.now().isoformat(),
            "recipe": recipe.to_dict(),
            "validation": validation.to_dict() if validation else None,
        }

        filename = "report_%s_%s.json" % (recipe.recipe_id, datetime.now().strftime("%Y%m%d_%H%M%S"))
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return path
