"""Parametric anvil geometry generator for ultrasonic welding simulation.

Generates anvil geometry with four profile types:
  - **flat** -- plain rectangular block (default).
  - **groove** -- rectangular block with parallel grooves cut into the
    top face for energy-director focusing.
  - **knurled** -- cross-hatch knurl pattern on the top face for grip.
  - **contour** -- top face with a cylindrical concave contour for
    curved workpieces.

Uses CadQuery when available for proper CAD output (STEP/STL), falling
back to a numpy-based mesh generation for Three.js preview.

Usage
-----
::

    from ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator import (
        AnvilGenerator,
        AnvilParams,
    )

    gen = AnvilGenerator()
    result = gen.generate(AnvilParams(anvil_type="groove", groove_count=5))
    # result["mesh_preview"]  -> {vertices, faces} for Three.js
    # result["params"]        -> parameter dict
    # result["contact_face"]  -> {center, width, length}
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded CadQuery import
# ---------------------------------------------------------------------------

HAS_CADQUERY = False
try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    cq = None  # type: ignore[assignment]
    logger.info("CadQuery not available; using numpy mesh fallback for anvil generation")


# ---------------------------------------------------------------------------
# Dataclass: AnvilParams
# ---------------------------------------------------------------------------


@dataclass
class AnvilParams:
    """Parameters for anvil geometry generation.

    All dimensions are in millimetres.
    """

    anvil_type: str = "flat"  # flat | groove | knurled | contour

    # Overall dimensions
    width_mm: float = 50.0   # X extent
    depth_mm: float = 30.0   # Y extent
    height_mm: float = 20.0  # Z extent

    # Groove parameters (for anvil_type="groove")
    groove_width_mm: float = 5.0
    groove_depth_mm: float = 3.0
    groove_count: int = 3

    # Knurl parameters (for anvil_type="knurled")
    knurl_pitch_mm: float = 1.0
    knurl_depth_mm: float = 0.3

    # Contour parameters (for anvil_type="contour")
    contour_radius_mm: float = 25.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise parameters to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# AnvilGenerator
# ---------------------------------------------------------------------------


class AnvilGenerator:
    """Generate parametric anvil geometry for welding simulation."""

    # Valid anvil types
    VALID_TYPES = ("flat", "groove", "knurled", "contour")

    def generate(self, params: AnvilParams) -> dict[str, Any]:
        """Generate anvil geometry.

        Parameters
        ----------
        params:
            Anvil parameters.

        Returns
        -------
        dict
            Dictionary with keys:
            - ``solid`` -- CadQuery solid (if CadQuery available, else ``None``).
            - ``mesh_preview`` -- ``{vertices, faces}`` for Three.js rendering.
            - ``step_path`` -- path to exported STEP file (if CadQuery
              available, else ``None``).
            - ``params`` -- parameter dict.
            - ``contact_face`` -- ``{center, width, length}`` describing
              the top face.
            - ``volume_mm3`` -- approximate volume.
            - ``surface_area_mm2`` -- approximate surface area.

        Raises
        ------
        ValueError
            If ``params.anvil_type`` is not one of the supported types.
        """
        if params.anvil_type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid anvil_type: {params.anvil_type!r}. "
                f"Must be one of {self.VALID_TYPES}."
            )

        # Try CadQuery first
        if HAS_CADQUERY:
            try:
                return self._generate_cadquery(params)
            except Exception as exc:
                logger.warning(
                    "CadQuery anvil generation failed, using numpy fallback: %s",
                    exc,
                )

        return self._generate_numpy(params)

    # ------------------------------------------------------------------
    # CadQuery path
    # ------------------------------------------------------------------

    def _generate_cadquery(self, params: AnvilParams) -> dict[str, Any]:
        """Generate anvil geometry using CadQuery."""
        dispatch = {
            "flat": self._generate_flat,
            "groove": self._generate_groove,
            "knurled": self._generate_knurled,
            "contour": self._generate_contour,
        }
        body = dispatch[params.anvil_type](params)

        # Tessellate for preview
        mesh_preview = self._cq_tessellate(body)

        # Export STEP to a temp file
        step_path = None
        try:
            fd, step_path = tempfile.mkstemp(
                suffix=".step", prefix=f"anvil-{params.anvil_type}-"
            )
            os.close(fd)
            cq.exporters.export(body, step_path, exportType="STEP")
        except Exception as exc:
            logger.warning("STEP export failed: %s", exc)
            step_path = None

        # Volume and surface area
        vol = body.val().Volume()
        sa = body.val().Area()

        # Contact face info (top face)
        contact_face = {
            "center": [0.0, 0.0, params.height_mm],
            "width": params.width_mm,
            "length": params.depth_mm,
        }

        return {
            "solid": body,
            "mesh_preview": mesh_preview,
            "step_path": step_path,
            "params": params.to_dict(),
            "contact_face": contact_face,
            "volume_mm3": float(vol),
            "surface_area_mm2": float(sa),
        }

    def _generate_flat(self, params: AnvilParams):
        """Generate a flat rectangular anvil block."""
        return (
            cq.Workplane("XY")
            .box(params.width_mm, params.depth_mm, params.height_mm)
        )

    def _generate_groove(self, params: AnvilParams):
        """Generate an anvil with parallel grooves on the top face."""
        # Start with base block
        body = (
            cq.Workplane("XY")
            .box(params.width_mm, params.depth_mm, params.height_mm)
        )

        # Cut grooves along X-axis (parallel to depth/Y)
        if params.groove_count <= 0:
            return body

        # Distribute grooves evenly across the width
        total_groove_span = params.groove_count * params.groove_width_mm
        if total_groove_span > params.width_mm:
            logger.warning(
                "Groove span (%.1f mm) exceeds anvil width (%.1f mm); "
                "reducing groove count",
                total_groove_span, params.width_mm,
            )
            params.groove_count = int(params.width_mm / params.groove_width_mm)

        spacing = params.width_mm / (params.groove_count + 1)
        z_top = params.height_mm / 2.0  # CQ box is centered

        for i in range(params.groove_count):
            x_pos = -params.width_mm / 2.0 + spacing * (i + 1)
            groove = (
                cq.Workplane("XY")
                .center(x_pos, 0)
                .rect(params.groove_width_mm, params.depth_mm + 2)
                .extrude(params.groove_depth_mm)
                .translate((0, 0, z_top - params.groove_depth_mm))
            )
            try:
                body = body.cut(groove)
            except Exception as exc:
                logger.warning("Groove cut %d failed: %s", i, exc)

        return body

    def _generate_knurled(self, params: AnvilParams):
        """Generate an anvil with cross-hatch knurl on top face."""
        body = (
            cq.Workplane("XY")
            .box(params.width_mm, params.depth_mm, params.height_mm)
        )

        z_top = params.height_mm / 2.0
        pitch = params.knurl_pitch_mm
        depth = params.knurl_depth_mm
        groove_w = pitch * 0.4  # groove width = 40% of pitch

        hw = params.width_mm / 2.0
        hd = params.depth_mm / 2.0
        margin = pitch

        grooves = None

        # Grooves along Y-axis (cutting across X)
        x = -hw - margin
        while x <= hw + margin:
            groove = (
                cq.Workplane("XY")
                .center(x, 0)
                .rect(groove_w, params.depth_mm + 2 * margin)
                .extrude(depth)
                .translate((0, 0, z_top - depth))
            )
            grooves = groove if grooves is None else grooves.union(groove)
            x += pitch

        # Grooves along X-axis (cutting across Y)
        y = -hd - margin
        while y <= hd + margin:
            groove = (
                cq.Workplane("XY")
                .center(0, y)
                .rect(params.width_mm + 2 * margin, groove_w)
                .extrude(depth)
                .translate((0, 0, z_top - depth))
            )
            grooves = groove if grooves is None else grooves.union(groove)
            y += pitch

        if grooves is not None:
            try:
                body = body.cut(grooves)
            except Exception as exc:
                logger.warning("Knurl cut failed: %s", exc)

        return body

    def _generate_contour(self, params: AnvilParams):
        """Generate an anvil with a concave cylindrical contour on top."""
        body = (
            cq.Workplane("XY")
            .box(params.width_mm, params.depth_mm, params.height_mm)
        )

        # Cut a cylindrical shape from the top to create a concave surface
        # The cylinder axis runs along the Y direction.
        r = params.contour_radius_mm
        z_top = params.height_mm / 2.0

        # Depth of the concave cut (sagitta)
        half_w = params.width_mm / 2.0
        if r <= half_w:
            # Radius too small for the width; use a semicircle
            sagitta = r
        else:
            sagitta = r - math.sqrt(r**2 - half_w**2)

        # Create a cylinder positioned so that its bottom tangent
        # touches z_top (or slightly below for the contour cut)
        cutter = (
            cq.Workplane("XZ")
            .center(0, z_top + r - sagitta)
            .circle(r)
            .extrude(params.depth_mm + 2, both=True)
        )

        try:
            body = body.cut(cutter)
        except Exception as exc:
            logger.warning("Contour cut failed: %s", exc)

        return body

    @staticmethod
    def _cq_tessellate(body) -> dict[str, list]:
        """Tessellate a CadQuery body into vertices and faces lists."""
        vertices = []
        faces = []
        tess = body.val().tessellate(0.1)
        for v in tess[0]:
            vertices.append([v.x, v.y, v.z])
        for f in tess[1]:
            faces.append(list(f))
        return {"vertices": vertices, "faces": faces}

    # ------------------------------------------------------------------
    # Numpy fallback path
    # ------------------------------------------------------------------

    def _generate_numpy(self, params: AnvilParams) -> dict[str, Any]:
        """Generate anvil geometry using numpy (no CAD export)."""
        dispatch = {
            "flat": self._np_flat,
            "groove": self._np_groove,
            "knurled": self._np_knurled,
            "contour": self._np_contour,
        }
        mesh = dispatch[params.anvil_type](params)

        vertices = np.array(mesh["vertices"])
        faces_arr = np.array(mesh["faces"])

        vol = self._np_volume(vertices, faces_arr)
        sa = self._np_surface_area(vertices, faces_arr)

        contact_face = {
            "center": [0.0, 0.0, params.height_mm],
            "width": params.width_mm,
            "length": params.depth_mm,
        }

        return {
            "solid": None,
            "mesh_preview": {
                "vertices": [v.tolist() for v in vertices],
                "faces": [f.tolist() for f in faces_arr],
            },
            "step_path": None,
            "params": params.to_dict(),
            "contact_face": contact_face,
            "volume_mm3": float(vol),
            "surface_area_mm2": float(sa),
        }

    # ------ Numpy shape generators ------

    @staticmethod
    def _np_flat(params: AnvilParams) -> dict:
        """Generate a flat rectangular box mesh."""
        w, d, h = params.width_mm / 2, params.depth_mm / 2, params.height_mm
        vertices = [
            [-w, -d, 0], [w, -d, 0], [w, d, 0], [-w, d, 0],  # bottom
            [-w, -d, h], [w, -d, h], [w, d, h], [-w, d, h],    # top
        ]
        faces = [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [1, 5, 6], [1, 6, 2],  # right
            [0, 3, 7], [0, 7, 4],  # left
        ]
        return {"vertices": vertices, "faces": faces}

    @staticmethod
    def _np_groove(params: AnvilParams) -> dict:
        """Generate a grooved anvil mesh.

        Models the top face with alternating raised and lowered strips
        to represent the groove pattern.
        """
        w, d, h = params.width_mm / 2, params.depth_mm / 2, params.height_mm
        n_grooves = max(params.groove_count, 0)
        g_depth = params.groove_depth_mm
        g_width = params.groove_width_mm

        if n_grooves == 0:
            return AnvilGenerator._np_flat(params)

        vertices = []
        faces = []

        # Bottom face (4 vertices)
        vertices.extend([
            [-w, -d, 0], [w, -d, 0], [w, d, 0], [-w, d, 0],
        ])

        # Create top face with groove indentations
        # Divide width into strips: land | groove | land | groove | ... | land
        spacing = params.width_mm / (n_grooves + 1)
        half_gw = g_width / 2

        # Build top vertices as a series of x-positions with z-heights
        x_positions = []
        z_heights = []

        # Start from left edge
        x = -w
        x_positions.append(x)
        z_heights.append(h)

        for i in range(n_grooves):
            center_x = -w + spacing * (i + 1)
            # Left edge of groove
            x_left = center_x - half_gw
            x_right = center_x + half_gw

            if x_left > x:
                x_positions.append(x_left)
                z_heights.append(h)

            # Groove bottom
            x_positions.append(x_left)
            z_heights.append(h - g_depth)
            x_positions.append(x_right)
            z_heights.append(h - g_depth)

            # Right edge of groove
            x_positions.append(x_right)
            z_heights.append(h)
            x = x_right

        if x < w:
            x_positions.append(w)
            z_heights.append(h)

        # Create top face vertices (front and back rows)
        top_start = len(vertices)
        for x_pos, z_h in zip(x_positions, z_heights):
            vertices.append([x_pos, -d, z_h])  # front row
        front_end = len(vertices)
        for x_pos, z_h in zip(x_positions, z_heights):
            vertices.append([x_pos, d, z_h])   # back row

        n_top = len(x_positions)

        # Top face triangles
        for i in range(n_top - 1):
            fi = top_start + i
            bi = top_start + n_top + i
            # Two triangles per quad
            faces.append([fi, fi + 1, bi])
            faces.append([fi + 1, bi + 1, bi])

        # Bottom face
        faces.extend([
            [0, 1, 2], [0, 2, 3],
        ])

        # Side faces (front)
        faces.append([0, top_start, top_start + n_top - 1])
        faces.append([0, top_start + n_top - 1, 1])

        # Side faces (back)
        back_start = top_start + n_top
        faces.append([3, 2, back_start + n_top - 1])
        faces.append([3, back_start + n_top - 1, back_start])

        # Left side
        faces.append([0, 3, back_start])
        faces.append([0, back_start, top_start])

        # Right side
        faces.append([1, top_start + n_top - 1, back_start + n_top - 1])
        faces.append([1, back_start + n_top - 1, 2])

        return {"vertices": vertices, "faces": faces}

    @staticmethod
    def _np_knurled(params: AnvilParams) -> dict:
        """Generate a knurled anvil mesh.

        Modulates top face vertex heights in a cross-hatch pattern.
        """
        w, d, h = params.width_mm / 2, params.depth_mm / 2, params.height_mm
        pitch = params.knurl_pitch_mm
        depth = params.knurl_depth_mm

        # Create a grid of vertices on the top face
        n_x = max(int(params.width_mm / pitch) + 1, 4)
        n_y = max(int(params.depth_mm / pitch) + 1, 4)

        vertices = []
        faces = []

        # Bottom face (4 corners)
        vertices.extend([
            [-w, -d, 0], [w, -d, 0], [w, d, 0], [-w, d, 0],
        ])

        # Top face grid
        top_start = len(vertices)
        for iy in range(n_y):
            for ix in range(n_x):
                x = -w + ix * params.width_mm / (n_x - 1)
                y = -d + iy * params.depth_mm / (n_y - 1)
                # Cross-hatch height modulation
                z_mod = (
                    math.sin(2 * math.pi * x / pitch) * depth * 0.5
                    + math.sin(2 * math.pi * y / pitch) * depth * 0.5
                )
                vertices.append([x, y, h + z_mod])

        # Top face triangles
        for iy in range(n_y - 1):
            for ix in range(n_x - 1):
                v00 = top_start + iy * n_x + ix
                v10 = v00 + 1
                v01 = v00 + n_x
                v11 = v01 + 1
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        # Bottom face
        faces.extend([
            [0, 1, 2], [0, 2, 3],
        ])

        # Side faces (simplified: connect bottom corners to nearest top edges)
        # Front side (y = -d)
        front_left = top_start
        front_right = top_start + n_x - 1
        faces.append([0, front_left, front_right])
        faces.append([0, front_right, 1])

        # Back side (y = d)
        back_left = top_start + (n_y - 1) * n_x
        back_right = back_left + n_x - 1
        faces.append([3, 2, back_right])
        faces.append([3, back_right, back_left])

        # Left side (x = -w)
        faces.append([0, 3, back_left])
        faces.append([0, back_left, front_left])

        # Right side (x = w)
        faces.append([1, front_right, back_right])
        faces.append([1, back_right, 2])

        return {"vertices": vertices, "faces": faces}

    @staticmethod
    def _np_contour(params: AnvilParams) -> dict:
        """Generate a contour anvil mesh.

        Top face follows a concave cylindrical curve along the X axis.
        """
        w, d, h = params.width_mm / 2, params.depth_mm / 2, params.height_mm
        r = params.contour_radius_mm

        n_x = 20  # resolution along width
        n_y = 4   # resolution along depth

        vertices = []
        faces = []

        # Bottom face
        vertices.extend([
            [-w, -d, 0], [w, -d, 0], [w, d, 0], [-w, d, 0],
        ])

        # Top face with concave contour
        top_start = len(vertices)
        for iy in range(n_y):
            y = -d + iy * params.depth_mm / (n_y - 1)
            for ix in range(n_x):
                x = -w + ix * params.width_mm / (n_x - 1)
                # Concave surface: z = h - sagitta(x)
                if abs(x) < r:
                    z = h - (r - math.sqrt(max(r**2 - x**2, 0)))
                else:
                    z = h - r  # clamp for large x
                vertices.append([x, y, z])

        # Top face triangles
        for iy in range(n_y - 1):
            for ix in range(n_x - 1):
                v00 = top_start + iy * n_x + ix
                v10 = v00 + 1
                v01 = v00 + n_x
                v11 = v01 + 1
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        # Bottom face
        faces.extend([
            [0, 1, 2], [0, 2, 3],
        ])

        # Side faces
        front_left = top_start
        front_right = top_start + n_x - 1
        back_left = top_start + (n_y - 1) * n_x
        back_right = back_left + n_x - 1

        faces.append([0, front_left, front_right])
        faces.append([0, front_right, 1])
        faces.append([3, 2, back_right])
        faces.append([3, back_right, back_left])
        faces.append([0, 3, back_left])
        faces.append([0, back_left, front_left])
        faces.append([1, front_right, back_right])
        faces.append([1, back_right, 2])

        return {"vertices": vertices, "faces": faces}

    # ------ Numpy mesh utilities ------

    @staticmethod
    def _np_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute mesh volume via signed tetrahedron method."""
        if len(faces) == 0:
            return 0.0
        total = 0.0
        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            total += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(total)

    @staticmethod
    def _np_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute mesh surface area."""
        if len(faces) == 0:
            return 0.0
        total = 0.0
        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            total += np.linalg.norm(np.cross(edge1, edge2)) / 2.0
        return total
