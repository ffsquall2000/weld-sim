"""Horn generation service with temporary file management."""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Optional

from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
    HornGenerator,
    HornParams,
    HornGenerationResult,
)
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

logger = logging.getLogger(__name__)


class HornService:
    """Service for horn generation and chamfer analysis."""

    def __init__(self):
        self._generator = HornGenerator()
        self._physics = PhysicsModel()
        self._file_cache: dict[str, dict] = {}  # file_id -> {path, format, created}
        self._cache_dir = tempfile.mkdtemp(prefix="horn_export_")

    def generate_horn(
        self, params: HornParams
    ) -> tuple[HornGenerationResult, Optional[str]]:
        """Generate a horn and cache export files.

        Returns:
            (result, download_id) where download_id is None if no CAD
            export is available.
        """
        result = self._generator.generate(params)
        download_id = None

        if result.has_cad_export and result.step_data:
            download_id = uuid.uuid4().hex[:12]
            # Save STEP file
            step_path = os.path.join(self._cache_dir, f"{download_id}.step")
            with open(step_path, "wb") as f:
                f.write(result.step_data)
            self._file_cache[f"{download_id}_step"] = {
                "path": step_path,
                "format": "step",
            }
            # Save STL file
            if result.stl_data:
                stl_path = os.path.join(
                    self._cache_dir, f"{download_id}.stl"
                )
                with open(stl_path, "wb") as f:
                    f.write(result.stl_data)
                self._file_cache[f"{download_id}_stl"] = {
                    "path": stl_path,
                    "format": "stl",
                }

        return result, download_id

    def get_download_path(
        self, file_id: str, fmt: str = "step"
    ) -> Optional[str]:
        """Get cached file path for download."""
        key = f"{file_id}_{fmt}"
        entry = self._file_cache.get(key)
        if entry and os.path.exists(entry["path"]):
            return entry["path"]
        return None

    def analyze_chamfer(
        self,
        contact_width_mm: float,
        contact_length_mm: float,
        chamfer_radius_mm: float,
        chamfer_angle_deg: float,
        edge_treatment: str,
        pressure_mpa: float,
        amplitude_um: float,
        material_yield_mpa: float,
        weld_time_s: float,
    ) -> dict:
        """Analyze chamfer impact on welding.

        Returns dict with stress_concentration_factor, risk assessment,
        area corrections, and recommendations.
        """
        nominal_area = contact_width_mm * contact_length_mm

        kt = self._physics.chamfer_stress_concentration_factor(
            chamfer_radius_mm=chamfer_radius_mm,
            contact_width_mm=contact_width_mm,
            edge_treatment=edge_treatment,
        )

        damage = self._physics.chamfer_material_damage_risk(
            kt=kt,
            pressure_mpa=pressure_mpa,
            amplitude_um=amplitude_um,
            material_yield_mpa=material_yield_mpa,
            weld_time_s=weld_time_s,
        )

        corrected_area = self._physics.chamfer_contact_area_correction(
            nominal_area_mm2=nominal_area,
            chamfer_radius_mm=chamfer_radius_mm,
            contact_width_mm=contact_width_mm,
            contact_length_mm=contact_length_mm,
            edge_treatment=edge_treatment,
            chamfer_angle_deg=chamfer_angle_deg,
        )

        area_reduction = (
            (1 - corrected_area / nominal_area) * 100
            if nominal_area > 0
            else 0
        )

        # Generate recommendations
        recommendations = []
        if damage["risk_level"] in ("high", "critical"):
            recommendations.append(
                f"Edge damage risk is {damage['risk_level']}. "
                f"Consider increasing chamfer radius to >=0.5 mm "
                f"or using fillet treatment."
            )
        if edge_treatment == "none":
            recommendations.append(
                "No edge treatment specified. Adding a fillet (r=0.3-0.5 mm) "
                "is recommended to reduce material damage."
            )
        if area_reduction > 15:
            recommendations.append(
                f"Chamfer reduces contact area by {area_reduction:.1f}%. "
                "Consider reducing chamfer radius to maintain weld quality."
            )
        if kt > 2.0 and edge_treatment == "chamfer":
            recommendations.append(
                "Pure chamfer still has significant stress concentration. "
                "Consider switching to fillet or compound treatment."
            )
        if not recommendations:
            recommendations.append(
                "Edge treatment is appropriate for current parameters."
            )

        return {
            "stress_concentration_factor": round(kt, 3),
            "risk_level": damage["risk_level"],
            "peak_stress_mpa": damage["peak_stress_mpa"],
            "damage_ratio": damage["damage_ratio"],
            "contact_area_nominal_mm2": round(nominal_area, 2),
            "contact_area_corrected_mm2": round(corrected_area, 2),
            "area_reduction_percent": round(area_reduction, 2),
            "energy_redistribution_factor": damage[
                "energy_redistribution_factor"
            ],
            "recommendations": recommendations,
        }
