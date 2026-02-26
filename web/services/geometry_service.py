"""Geometry analysis service -- STEP file parsing and PDF drawing analysis.

Uses a simplified STEP parser (text-based) when cadquery/OCP is not available.
Falls back to PyMuPDF for PDF analysis.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GeometryService:
    """Service for analyzing CAD files and PDF drawings."""

    def analyze_step_file(self, file_path: str) -> dict[str, Any]:
        """Analyze a STEP file and classify horn geometry.

        First tries cadquery (full OCC), then falls back to text-based STEP parsing.
        """
        # Try full cadquery import first
        try:
            return self._analyze_with_cadquery(file_path)
        except (ImportError, RuntimeError):
            pass

        # Fallback: parse STEP text for geometry parameters
        return self._analyze_step_text(file_path)

    def _analyze_with_cadquery(self, file_path: str) -> dict[str, Any]:
        """Full analysis using cadquery/OCP (if available)."""
        import cadquery as cq  # type: ignore[import-untyped]
        from OCP.BRepGProp import BRepGProp  # type: ignore[import-untyped]
        from OCP.GProp import GProp_GProps  # type: ignore[import-untyped]

        shape = cq.importers.importStep(file_path)
        bb = shape.val().BoundingBox()

        vol_props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape.val().wrapped, vol_props)
        volume = vol_props.Mass()

        surf_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape.val().wrapped, surf_props)
        surface_area = surf_props.Mass()

        bbox = [bb.xmin, bb.ymin, bb.zmin, bb.xmax, bb.ymax, bb.zmax]
        dims = {
            "width_mm": bb.xmax - bb.xmin,
            "height_mm": bb.ymax - bb.ymin,
            "length_mm": bb.zmax - bb.zmin,
        }

        horn_type, gain, confidence = self._classify_from_dimensions(dims, volume)
        mesh_data = self._generate_visualization_mesh(dims, horn_type)

        return {
            "horn_type": horn_type,
            "dimensions": dims,
            "gain_estimate": gain,
            "confidence": confidence,
            "knurl": None,
            "bounding_box": bbox,
            "volume_mm3": round(volume, 2),
            "surface_area_mm2": round(surface_area, 2),
            "mesh": mesh_data,
        }

    def _analyze_step_text(self, file_path: str) -> dict[str, Any]:
        """Parse STEP file text to extract basic geometry info."""
        with open(file_path, "r", errors="ignore") as f:
            content = f.read()

        # Extract CARTESIAN_POINT coordinates from STEP
        points: list[tuple[float, float, float]] = []
        point_pattern = re.compile(
            r"CARTESIAN_POINT\s*\(\s*'[^']*'\s*,\s*\(\s*"
            r"([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*,\s*([-\d.eE+]+)\s*\)\s*\)"
        )
        for match in point_pattern.finditer(content):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                points.append((x, y, z))
            except ValueError:
                continue

        if not points:
            raise RuntimeError(
                "No geometry data found in STEP file. "
                "The file may be corrupt or empty."
            )

        pts = np.array(points)
        xmin, ymin, zmin = pts.min(axis=0)
        xmax, ymax, zmax = pts.max(axis=0)

        bbox = [
            float(xmin), float(ymin), float(zmin),
            float(xmax), float(ymax), float(zmax),
        ]
        dims = {
            "width_mm": float(xmax - xmin),
            "height_mm": float(ymax - ymin),
            "length_mm": float(zmax - zmin),
        }

        # Estimate volume from bounding box with fill factor
        bb_volume = dims["width_mm"] * dims["height_mm"] * dims["length_mm"]

        # Estimate actual volume by analyzing point distribution
        # Split into height slices and compute cross-section area variance
        n_slices = 20
        heights = np.linspace(ymin, ymax, n_slices + 1)
        areas: list[float] = []
        for i in range(n_slices):
            mask = (pts[:, 1] >= heights[i]) & (pts[:, 1] < heights[i + 1])
            slice_pts = pts[mask]
            if len(slice_pts) >= 3:
                x_range = float(slice_pts[:, 0].max() - slice_pts[:, 0].min())
                z_range = float(slice_pts[:, 2].max() - slice_pts[:, 2].min())
                areas.append(x_range * z_range)
            else:
                areas.append(0.0)

        areas_arr = np.array(areas)
        if areas_arr.max() > 0:
            fill_factor = float(
                np.mean(areas_arr[areas_arr > 0])
                / (dims["width_mm"] * dims["length_mm"])
            )
            fill_factor = max(0.3, min(fill_factor, 1.0))
        else:
            fill_factor = 0.7  # typical horn fill factor

        # 0.85 for typical cylindrical deviation
        volume = bb_volume * fill_factor * 0.85
        surface_area = (
            2
            * (
                dims["width_mm"] * dims["height_mm"]
                + dims["height_mm"] * dims["length_mm"]
                + dims["width_mm"] * dims["length_mm"]
            )
            * fill_factor
        )

        horn_type, gain, confidence = self._classify_from_dimensions(dims, volume)
        mesh_data = self._generate_visualization_mesh(dims, horn_type)

        return {
            "horn_type": horn_type,
            "dimensions": dims,
            "gain_estimate": round(gain, 3),
            "confidence": round(confidence * 0.7, 2),  # lower confidence for text-based
            "knurl": None,
            "bounding_box": bbox,
            "volume_mm3": round(volume, 2),
            "surface_area_mm2": round(surface_area, 2),
            "mesh": mesh_data,
        }

    def _classify_from_dimensions(
        self, dims: dict[str, float], volume: float
    ) -> tuple[str, float, float]:
        """Classify horn type from dimensions and volume."""
        w = dims["width_mm"]
        h = dims["height_mm"]
        l_ = dims["length_mm"]

        if w <= 0 or h <= 0 or l_ <= 0:
            return "unknown", 1.0, 0.1

        aspect_wl = w / l_ if l_ > 0 else 1.0
        aspect_wh = w / h if h > 0 else 1.0
        bb_vol = w * h * l_
        fill_ratio = volume / bb_vol if bb_vol > 0 else 1.0

        # Classification heuristics
        if 0.8 <= aspect_wl <= 1.2 and aspect_wh < 0.5:
            horn_type = "cylindrical"
            confidence = 0.75
        elif min(w, l_) > 0 and max(w, l_) / min(w, l_) > 3.0:
            horn_type = "blade"
            confidence = 0.7
        elif fill_ratio < 0.5:
            horn_type = "exponential"
            confidence = 0.6
        elif aspect_wl > 2.0:
            horn_type = "block"
            confidence = 0.65
        else:
            horn_type = "flat"
            confidence = 0.6

        # Gain estimate from fill ratio (tapered horns have lower fill)
        gain = max(1.0, 1.0 / max(fill_ratio, 0.15))
        gain = min(gain, 5.0)

        return horn_type, gain, confidence

    def _generate_visualization_mesh(
        self, dims: dict[str, float], horn_type: str
    ) -> dict[str, list]:
        """Generate a simplified 3D mesh for Three.js visualization."""
        w = dims["width_mm"]
        h = dims["height_mm"]
        l_ = dims["length_mm"]

        if horn_type == "cylindrical":
            return self._gen_cylinder_mesh(w / 2, h, 32)
        elif horn_type == "blade":
            return self._gen_box_mesh(w, h, l_)
        elif horn_type == "exponential":
            return self._gen_tapered_mesh(w, h, l_, taper=0.5)
        else:
            return self._gen_box_mesh(w, h, l_)

    def _gen_box_mesh(self, w: float, h: float, l_: float) -> dict[str, list]:
        """Generate box (rectangular horn) mesh."""
        hw, hh, hl = w / 2, h / 2, l_ / 2
        vertices = [
            [-hw, -hh, -hl],
            [hw, -hh, -hl],
            [hw, hh, -hl],
            [-hw, hh, -hl],
            [-hw, -hh, hl],
            [hw, -hh, hl],
            [hw, hh, hl],
            [-hw, hh, hl],
        ]
        faces = [
            [0, 1, 2],
            [0, 2, 3],  # front
            [4, 6, 5],
            [4, 7, 6],  # back
            [0, 4, 5],
            [0, 5, 1],  # bottom
            [2, 6, 7],
            [2, 7, 3],  # top
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ]
        return {"vertices": vertices, "faces": faces}

    def _gen_cylinder_mesh(
        self, radius: float, height: float, segments: int = 32
    ) -> dict[str, list]:
        """Generate cylinder mesh."""
        vertices: list[list[float]] = []
        faces: list[list[int]] = []

        # Generate rings
        for j in range(2):  # bottom and top
            y = -height / 2 + j * height
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                vertices.append([round(x, 3), round(y, 3), round(z, 3)])

        # Center points for caps
        vertices.append([0, -height / 2, 0])  # bottom center
        vertices.append([0, height / 2, 0])  # top center

        bc = 2 * segments  # bottom center index
        tc = bc + 1  # top center index

        # Side faces
        for i in range(segments):
            i_next = (i + 1) % segments
            b1, b2 = i, i_next
            t1, t2 = i + segments, i_next + segments
            faces.append([b1, b2, t2])
            faces.append([b1, t2, t1])

        # Bottom cap
        for i in range(segments):
            i_next = (i + 1) % segments
            faces.append([bc, i_next, i])

        # Top cap
        for i in range(segments):
            i_next = (i + 1) % segments
            faces.append([tc, i + segments, i_next + segments])

        return {"vertices": vertices, "faces": faces}

    def _gen_tapered_mesh(
        self, w: float, h: float, l_: float, taper: float = 0.5
    ) -> dict[str, list]:
        """Generate tapered (exponential) horn mesh."""
        n_sections = 10
        vertices: list[list[float]] = []

        for j in range(n_sections + 1):
            t = j / n_sections
            y = -h / 2 + t * h
            # Exponential taper
            exp_denom = 1.0 - math.exp(-3)
            scale = 1.0 - (1.0 - taper) * (1.0 - math.exp(-3 * t)) / exp_denom
            hw = w / 2 * scale
            hl = l_ / 2 * scale
            # 4 corners per section
            vertices.extend(
                [
                    [round(-hw, 3), round(y, 3), round(-hl, 3)],
                    [round(hw, 3), round(y, 3), round(-hl, 3)],
                    [round(hw, 3), round(y, 3), round(hl, 3)],
                    [round(-hw, 3), round(y, 3), round(hl, 3)],
                ]
            )

        faces: list[list[int]] = []
        for j in range(n_sections):
            base = j * 4
            top = (j + 1) * 4
            for i in range(4):
                i_next = (i + 1) % 4
                b1, b2 = base + i, base + i_next
                t1, t2 = top + i, top + i_next
                faces.append([b1, b2, t2])
                faces.append([b1, t2, t1])

        # Bottom cap
        faces.append([0, 1, 2])
        faces.append([0, 2, 3])
        # Top cap
        last = n_sections * 4
        faces.append([last, last + 2, last + 1])
        faces.append([last, last + 3, last + 2])

        return {"vertices": vertices, "faces": faces}

    def analyze_pdf(self, file_path: str) -> dict[str, Any]:
        """Analyze a PDF engineering drawing."""
        try:
            import fitz  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required for PDF analysis. "
                "Install with: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(file_path)
        page_count = len(doc)

        all_dimensions: list[dict[str, Any]] = []
        all_notes: list[str] = []

        dim_pattern = re.compile(
            r"(\d+\.?\d*)\s*(?:mm|MM|cm|CM)"
            r"(?:\s*[+-]\s*(\d+\.?\d*))?"
        )
        diameter_pattern = re.compile(r"[\u00d8\u03a6\u03c6]\s*(\d+\.?\d*)")
        radius_pattern = re.compile(r"[Rr]\s*(\d+\.?\d*)")

        for page_num in range(page_count):
            page = doc[page_num]
            text = page.get_text()

            # Dimensions
            for match in dim_pattern.finditer(text):
                value = float(match.group(1))
                tol = float(match.group(2)) if match.group(2) else 0.0
                all_dimensions.append(
                    {
                        "label": match.group(0).strip(),
                        "value_mm": value,
                        "tolerance_mm": tol,
                        "type": "linear",
                        "confidence": 0.7,
                        "page": page_num + 1,
                    }
                )

            # Diameters
            for match in diameter_pattern.finditer(text):
                value = float(match.group(1))
                all_dimensions.append(
                    {
                        "label": f"\u00d8{value}",
                        "value_mm": value,
                        "tolerance_mm": 0.0,
                        "type": "diameter",
                        "confidence": 0.8,
                        "page": page_num + 1,
                    }
                )

            # Radii
            for match in radius_pattern.finditer(text):
                value = float(match.group(1))
                all_dimensions.append(
                    {
                        "label": f"R{value}",
                        "value_mm": value,
                        "tolerance_mm": 0.0,
                        "type": "radius",
                        "confidence": 0.75,
                        "page": page_num + 1,
                    }
                )

            # Notes
            for line in text.splitlines():
                stripped = line.strip()
                if stripped and len(stripped) > 8:
                    lower = stripped.lower()
                    if any(
                        kw in lower
                        for kw in (
                            "note",
                            "material",
                            "finish",
                            "tolerance",
                            "heat treat",
                            "hardness",
                            "surface",
                            "roughness",
                            "\u6ce8",       # Chinese: note
                            "\u6750\u6599", # Chinese: material
                            "\u516c\u5dee", # Chinese: tolerance
                            "\u786c\u5ea6", # Chinese: hardness
                            "\u7c97\u7cd9\u5ea6", # Chinese: roughness
                            "\u8868\u9762\u5904\u7406", # Chinese: surface treatment
                        )
                    ):
                        all_notes.append(stripped)

        tolerances = [
            {"label": d["label"], "tolerance_mm": d["tolerance_mm"]}
            for d in all_dimensions
            if d["tolerance_mm"] > 0
        ]

        doc.close()

        return {
            "detected_dimensions": all_dimensions,
            "tolerances": tolerances,
            "notes": all_notes,
            "confidence": 0.6 if all_dimensions else 0.0,
            "page_count": page_count,
        }
