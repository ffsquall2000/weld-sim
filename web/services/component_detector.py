"""STEP assembly component auto-detection service.

Classifies solid bodies in a STEP assembly file by geometric analysis,
identifying horns, boosters, transducers, anvils, and workpieces based
on bounding-box dimensions, aspect ratios, and volume heuristics.

Uses cadquery/OCC when available; falls back to a STEP-text parser for
basic classification when those libraries are not installed.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Component type definitions
# ---------------------------------------------------------------------------

COMPONENT_TYPES = ["horn", "booster", "transducer", "anvil", "workpiece", "unknown"]


class ComponentDetector:
    """Classify STEP assembly components by geometric analysis.

    For each solid body found in a STEP file the detector computes:
      - bounding box, aspect ratio, volume
      - a heuristic classification based on shape characteristics

    Classification heuristics:
      - Horn: tapered profile, aspect ratio > 2, one flat face at tip
      - Booster: cylindrical, aspect ratio > 3, uniform cross section
      - Transducer: ring/disc shape, shorter
      - Anvil: flat base, wider than tall
      - Workpiece: small thin shape
    """

    COMPONENT_TYPES = COMPONENT_TYPES

    def detect(self, step_path: str) -> list[dict[str, Any]]:
        """Parse STEP file and classify each solid body.

        Returns a list of dicts, each containing::

            {
                "type": str,          # one of COMPONENT_TYPES
                "name": str,          # human-readable label
                "volume_mm3": float,
                "bbox": list[float],  # [xmin, ymin, zmin, xmax, ymax, zmax]
                "centroid": list[float],  # [cx, cy, cz]
                "dimensions": dict,   # {width_mm, height_mm, length_mm}
            }
        """
        # 1. Try cadquery / OCC (full geometry)
        try:
            return self._detect_with_cadquery(step_path)
        except Exception as exc:
            logger.info("cadquery detection unavailable: %s", exc)

        # 2. Fallback: text-based STEP parser
        try:
            return self._detect_from_step_text(step_path)
        except Exception as exc:
            logger.warning("STEP text parser failed: %s", exc)

        return [{
            "type": "unknown",
            "name": "parse_error",
            "volume_mm3": 0.0,
            "bbox": [0, 0, 0, 0, 0, 0],
            "centroid": [0, 0, 0],
            "dimensions": {"width_mm": 0, "height_mm": 0, "length_mm": 0},
        }]

    # ------------------------------------------------------------------
    # cadquery / OCC path
    # ------------------------------------------------------------------

    def _detect_with_cadquery(self, step_path: str) -> list[dict[str, Any]]:
        """Detect components using cadquery and OCC."""
        import cadquery as cq  # type: ignore[import-untyped]
        from OCP.BRepGProp import BRepGProp  # type: ignore[import-untyped]
        from OCP.GProp import GProp_GProps  # type: ignore[import-untyped]
        from OCP.TopAbs import TopAbs_SOLID  # type: ignore[import-untyped]
        from OCP.TopExp import TopExp_Explorer  # type: ignore[import-untyped]

        assembly = cq.importers.importStep(step_path)
        compound = assembly.val().wrapped

        # Iterate over solids in the compound
        solids: list[Any] = []
        explorer = TopExp_Explorer(compound, TopAbs_SOLID)
        while explorer.More():
            solids.append(explorer.Current())
            explorer.Next()

        # If no sub-solids found, treat the whole shape as a single solid
        if not solids:
            solids = [compound]

        results: list[dict[str, Any]] = []
        for idx, solid in enumerate(solids):
            # Volume
            vol_props = GProp_GProps()
            BRepGProp.VolumeProperties_s(solid, vol_props)
            volume = float(vol_props.Mass())
            centroid_pnt = vol_props.CentreOfMass()
            centroid = [
                round(centroid_pnt.X(), 3),
                round(centroid_pnt.Y(), 3),
                round(centroid_pnt.Z(), 3),
            ]

            # Bounding box
            from OCP.Bnd import Bnd_Box  # type: ignore[import-untyped]
            from OCP.BRepBndLib import BRepBndLib  # type: ignore[import-untyped]

            bnd = Bnd_Box()
            BRepBndLib.Add_s(solid, bnd)
            xmin, ymin, zmin, xmax, ymax, zmax = bnd.Get()
            bbox = [
                round(float(xmin), 3), round(float(ymin), 3), round(float(zmin), 3),
                round(float(xmax), 3), round(float(ymax), 3), round(float(zmax), 3),
            ]

            width = float(xmax - xmin)
            height = float(ymax - ymin)
            length = float(zmax - zmin)
            dims = {
                "width_mm": round(width, 3),
                "height_mm": round(height, 3),
                "length_mm": round(length, 3),
            }

            comp_type = self._classify_component(width, height, length, volume)
            name = f"{comp_type}_{idx + 1}"

            results.append({
                "type": comp_type,
                "name": name,
                "volume_mm3": round(volume, 2),
                "bbox": bbox,
                "centroid": centroid,
                "dimensions": dims,
            })

        # Refine names by stacking order (top-down along the longest axis)
        self._refine_names(results)
        return results

    # ------------------------------------------------------------------
    # STEP text fallback
    # ------------------------------------------------------------------

    def _detect_from_step_text(self, step_path: str) -> list[dict[str, Any]]:
        """Parse STEP text to extract approximate geometry for a single body."""
        import numpy as np

        with open(step_path, "r", errors="ignore") as f:
            content = f.read()

        # Extract CARTESIAN_POINT coordinates
        point_pattern = re.compile(
            r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
            r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)\s*\)"
        )
        points: list[tuple[float, float, float]] = []
        for match in point_pattern.finditer(content):
            try:
                x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                points.append((x, y, z))
            except ValueError:
                continue

        if not points:
            raise RuntimeError("No geometry data found in STEP file")

        pts = np.array(points)
        xmin, ymin, zmin = pts.min(axis=0).tolist()
        xmax, ymax, zmax = pts.max(axis=0).tolist()

        width = xmax - xmin
        height = ymax - ymin
        length = zmax - zmin

        # Rough volume estimate (bounding box * fill factor)
        volume = width * height * length * 0.7

        centroid = [
            round((xmin + xmax) / 2, 3),
            round((ymin + ymax) / 2, 3),
            round((zmin + zmax) / 2, 3),
        ]
        bbox = [
            round(xmin, 3), round(ymin, 3), round(zmin, 3),
            round(xmax, 3), round(ymax, 3), round(zmax, 3),
        ]
        dims = {
            "width_mm": round(width, 3),
            "height_mm": round(height, 3),
            "length_mm": round(length, 3),
        }

        comp_type = self._classify_component(width, height, length, volume)

        return [{
            "type": comp_type,
            "name": f"{comp_type}_1",
            "volume_mm3": round(volume, 2),
            "bbox": bbox,
            "centroid": centroid,
            "dimensions": dims,
        }]

    # ------------------------------------------------------------------
    # Classification heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_component(
        width: float, height: float, length: float, volume: float
    ) -> str:
        """Classify a single solid body based on dimensional heuristics.

        The classification uses aspect ratios and volume-to-bounding-box
        fill ratios to distinguish between ultrasonic welding stack
        components.
        """
        if width <= 0 or height <= 0 or length <= 0:
            return "unknown"

        # Sort dimensions: smallest, mid, largest
        dims_sorted = sorted([width, height, length])
        smallest, mid, largest = dims_sorted

        # Aspect ratio: longest / shortest
        aspect_ratio = largest / smallest if smallest > 0 else 1.0
        # Fill ratio: actual volume / bounding box volume
        bb_vol = width * height * length
        fill_ratio = volume / bb_vol if bb_vol > 0 else 0.0

        # Cross-section aspect: mid / smallest
        cross_aspect = mid / smallest if smallest > 0 else 1.0

        # Workpiece: very small volume and thin
        if volume < 5000 and smallest < 10:
            return "workpiece"

        # Workpiece: thin plate-like shape (high aspect, very thin)
        if aspect_ratio >= 2.0 and smallest < 5:
            return "workpiece"

        # Anvil: flat base shape -- smallest dimension is much less than
        # both other dimensions AND cross-section is wide (not cylindrical)
        if smallest < mid * 0.3 and cross_aspect > 2.0:
            return "anvil"

        # Booster: long cylinder, aspect > 3, nearly round cross-section
        if aspect_ratio > 3.0 and cross_aspect < 1.5:
            return "booster"

        # Horn: moderately elongated, tapered (lower fill ratio)
        if aspect_ratio >= 2.0 and fill_ratio < 0.65:
            return "horn"

        # Horn: elongated with moderate fill and round-ish cross-section
        if aspect_ratio >= 2.0 and fill_ratio < 0.85 and cross_aspect < 1.5:
            return "horn"

        # Transducer: disc/ring shape, short and wide
        if aspect_ratio < 1.8 and cross_aspect < 1.5:
            return "transducer"

        # Default for unrecognized shapes
        return "unknown"

    @staticmethod
    def _refine_names(components: list[dict[str, Any]]) -> None:
        """Refine component names based on type counts.

        If there is only one component of a given type, drop the numeric
        suffix for cleaner display.
        """
        from collections import Counter
        type_counts = Counter(c["type"] for c in components)
        type_seen: dict[str, int] = {}

        for comp in components:
            ctype = comp["type"]
            type_seen[ctype] = type_seen.get(ctype, 0) + 1
            if type_counts[ctype] == 1:
                comp["name"] = ctype
            else:
                comp["name"] = f"{ctype}_{type_seen[ctype]}"
