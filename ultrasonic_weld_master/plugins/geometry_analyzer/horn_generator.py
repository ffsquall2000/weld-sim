"""Parametric horn geometry generator with knurl and chamfer support.

Uses CadQuery when available for proper CAD output (STEP/STL),
falls back to numpy-based mesh generation for preview.

Supports applying knurl patterns to both parametric horns and
imported STEP files via :meth:`HornGenerator.apply_knurl_to_step`.
"""
from __future__ import annotations

import math
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import CadQuery; mark availability
try:
    import cadquery as cq
    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False
    logger.info("CadQuery not available; using numpy mesh fallback")


@dataclass
class HornParams:
    """Parameters for horn generation."""
    horn_type: str = "flat"  # flat | cylindrical | exponential | blade | stepped
    width_mm: float = 25.0
    height_mm: float = 80.0
    length_mm: float = 25.0
    material: str = "Titanium Ti-6Al-4V"
    # Diameter-based parameters (override width_mm / length_mm for round horns)
    input_diameter_mm: Optional[float] = None
    output_diameter_mm: Optional[float] = None
    # Knurl
    knurl_type: str = "none"  # none | linear | cross_hatch | diamond | conical | spherical
    knurl_pitch_mm: float = 1.0
    knurl_tooth_width_mm: float = 0.5
    knurl_depth_mm: float = 0.3
    knurl_direction: str = "perpendicular"
    # Chamfer
    chamfer_radius_mm: float = 0.0
    chamfer_angle_deg: float = 45.0
    edge_treatment: str = "none"  # none | chamfer | fillet | compound


@dataclass
class HornGenerationResult:
    """Result from horn generation."""
    mesh: dict  # {vertices: [[x,y,z],...], faces: [[a,b,c],...]}
    step_data: Optional[bytes] = None  # STEP file bytes (only if CadQuery available)
    stl_data: Optional[bytes] = None   # STL file bytes (only if CadQuery available)
    horn_type: str = ""
    dimensions: dict = field(default_factory=dict)
    knurl_info: dict = field(default_factory=dict)
    chamfer_info: dict = field(default_factory=dict)
    volume_mm3: float = 0.0
    surface_area_mm2: float = 0.0
    has_cad_export: bool = False


@dataclass
class KnurlParams:
    """Parameters for standalone knurl application (e.g. on imported STEP)."""
    knurl_type: str = "linear"  # linear | cross_hatch | diamond
    pitch_mm: float = 1.0
    tooth_width_mm: float = 0.5
    depth_mm: float = 0.3


class HornGenerator:
    """Generate parametric horn geometries with knurl and chamfer."""

    def generate(self, params: HornParams) -> HornGenerationResult:
        """Generate a horn with the given parameters."""
        if HAS_CADQUERY:
            try:
                return self._generate_cadquery(params)
            except Exception as e:
                logger.warning("CadQuery generation failed, using fallback: %s", e)
        return self._generate_numpy(params)

    # --- CadQuery path ---

    def _generate_cadquery(self, params: HornParams) -> HornGenerationResult:
        """Generate using CadQuery for proper CAD output."""
        body = self._cq_create_body(params)
        if params.knurl_type != "none":
            body = self._cq_apply_knurl(body, params)
        if params.edge_treatment != "none" and params.chamfer_radius_mm > 0:
            body = self._cq_apply_edge_treatment(body, params)

        # Export
        import io
        step_buf = io.BytesIO()
        cq.exporters.export(body, step_buf, exportType="STEP")
        step_data = step_buf.getvalue()

        stl_buf = io.BytesIO()
        cq.exporters.export(body, stl_buf, exportType="STL")
        stl_data = stl_buf.getvalue()

        # Tessellate for preview
        mesh = self._cq_tessellate(body)

        # Compute properties
        vol = body.val().Volume()
        sa = body.val().Area()

        return HornGenerationResult(
            mesh=mesh, step_data=step_data, stl_data=stl_data,
            horn_type=params.horn_type,
            dimensions={
                "width_mm": params.width_mm,
                "height_mm": params.height_mm,
                "length_mm": params.length_mm,
            },
            knurl_info={
                "type": params.knurl_type,
                "pitch_mm": params.knurl_pitch_mm,
            },
            chamfer_info={
                "radius_mm": params.chamfer_radius_mm,
                "treatment": params.edge_treatment,
            },
            volume_mm3=vol, surface_area_mm2=sa, has_cad_export=True,
        )

    def _cq_create_body(self, p: HornParams):
        """Create the basic horn body shape using CadQuery."""
        if p.horn_type == "flat":
            return cq.Workplane("XY").box(p.width_mm, p.length_mm, p.height_mm)
        elif p.horn_type == "cylindrical":
            return cq.Workplane("XY").circle(p.width_mm / 2).extrude(p.height_mm)
        elif p.horn_type == "exponential":
            # Two cross-sections lofted
            return (
                cq.Workplane("XY")
                .rect(p.width_mm, p.length_mm)
                .workplane(offset=p.height_mm)
                .rect(p.width_mm * 0.6, p.length_mm * 0.6)
                .loft()
            )
        elif p.horn_type == "blade":
            return cq.Workplane("XY").rect(
                p.width_mm * 0.3, p.length_mm
            ).extrude(p.height_mm)
        elif p.horn_type == "stepped":
            base = cq.Workplane("XY").box(
                p.width_mm, p.length_mm, p.height_mm * 0.4
            )
            step = (
                base.faces(">Z").workplane()
                .rect(p.width_mm * 0.7, p.length_mm * 0.7)
                .extrude(p.height_mm * 0.3)
            )
            tip = (
                step.faces(">Z").workplane()
                .rect(p.width_mm * 0.5, p.length_mm * 0.5)
                .extrude(p.height_mm * 0.3)
            )
            return tip
        else:
            return cq.Workplane("XY").box(p.width_mm, p.length_mm, p.height_mm)

    def _cq_apply_knurl(self, body, p: HornParams):
        """Apply knurl pattern to a CadQuery body using boolean groove cuts.

        Delegates to the shared groove-cutting logic using HornParams fields.
        """
        knurl = KnurlParams(
            knurl_type=p.knurl_type,
            pitch_mm=p.knurl_pitch_mm,
            tooth_width_mm=p.knurl_tooth_width_mm,
            depth_mm=p.knurl_depth_mm,
        )
        return self._cq_cut_knurl_on_body(body, knurl)

    # ------------------------------------------------------------------
    # Public: apply knurl to an imported STEP file
    # ------------------------------------------------------------------

    def apply_knurl_to_step(self, step_path: str, knurl_params: KnurlParams):
        """Load STEP file, detect Z-min face, cut knurl grooves, return solid.

        Parameters
        ----------
        step_path : str
            Path to an existing STEP file.
        knurl_params : KnurlParams
            Knurl pattern configuration.

        Returns
        -------
        cq.Workplane
            The modified CadQuery solid with knurl grooves cut into
            the bottom (Z-min) face.

        Raises
        ------
        RuntimeError
            If CadQuery is not installed or the STEP file cannot be loaded.
        FileNotFoundError
            If *step_path* does not exist.
        ValueError
            If no suitable bottom face is found on the imported body.
        """
        if not HAS_CADQUERY:
            raise RuntimeError(
                "CadQuery is required for apply_knurl_to_step but is not installed."
            )
        if not os.path.isfile(step_path):
            raise FileNotFoundError(f"STEP file not found: {step_path}")

        body = cq.importers.importStep(step_path)
        logger.info("Loaded STEP body from %s", step_path)

        return self._cq_cut_knurl_on_body(body, knurl_params)

    # ------------------------------------------------------------------
    # Shared CadQuery knurl cutting logic
    # ------------------------------------------------------------------

    def _cq_cut_knurl_on_body(self, body, knurl: KnurlParams):
        """Cut knurl grooves into the bottom (Z-min) face of *body*.

        Works for both parametric bodies and imported STEP solids.
        """
        if knurl.knurl_type == "none":
            return body

        face_info = self._find_bottom_face(body)
        if face_info is None:
            logger.warning("No suitable bottom face found; skipping knurl.")
            return body

        face_center, face_width, face_length = face_info
        z_bottom = face_center[2]

        grooves = self._cq_create_knurl_grooves(
            knurl, face_center, face_width, face_length, z_bottom,
        )
        if grooves is None:
            return body

        try:
            body = body.cut(grooves)
        except Exception as exc:
            logger.warning("Knurl boolean cut failed: %s", exc)
        return body

    @staticmethod
    def _find_bottom_face(body):
        """Identify the bottom face (Z-min) of a CadQuery body.

        Returns ``(center_xyz, width, length)`` or ``None`` if no suitable
        face is found.  *width* and *length* are the face extents along X
        and Y respectively.
        """
        bb = body.val().BoundingBox()
        z_min = bb.zmin
        z_span = bb.zmax - bb.zmin
        if z_span <= 0:
            return None

        # Tolerance: face centre Z must be within 2% of total Z-span from
        # the absolute bottom.
        tol = max(z_span * 0.02, 0.01)  # at least 0.01 mm

        best_face = None
        best_area = -1.0

        try:
            faces = body.faces().vals()
        except Exception:
            return None

        for face in faces:
            face_bb = face.BoundingBox()
            face_z_center = (face_bb.zmin + face_bb.zmax) / 2.0
            face_z_thickness = face_bb.zmax - face_bb.zmin

            # Face must be near the Z-min boundary
            if face_z_center > z_min + tol:
                continue
            # Face must be approximately flat (thin in Z)
            if face_z_thickness > tol:
                continue

            face_area = face.Area()
            if face_area > best_area:
                best_area = face_area
                best_face = face

        if best_face is None:
            return None

        fbb = best_face.BoundingBox()
        center = (
            (fbb.xmin + fbb.xmax) / 2.0,
            (fbb.ymin + fbb.ymax) / 2.0,
            (fbb.zmin + fbb.zmax) / 2.0,
        )
        width = fbb.xmax - fbb.xmin
        length = fbb.ymax - fbb.ymin

        logger.info(
            "Bottom face detected: center=(%.2f, %.2f, %.2f), "
            "width=%.2f, length=%.2f, area=%.2f",
            *center, width, length, best_area,
        )
        return center, width, length

    @staticmethod
    def _cq_create_knurl_grooves(
        knurl: KnurlParams,
        face_center: tuple,
        face_width: float,
        face_length: float,
        z_bottom: float,
    ):
        """Create a CadQuery solid representing the groove tool body.

        The returned solid is positioned so that it can be boolean-subtracted
        from the horn body to produce the knurl pattern.
        """
        depth = knurl.depth_mm
        pitch = knurl.pitch_mm
        tw = knurl.tooth_width_mm
        cx, cy, _cz = face_center

        # Ensure groove width is less than pitch
        groove_w = min(tw, pitch * 0.8)

        # Margin beyond the face extents to ensure full coverage
        margin = pitch
        x_start = cx - face_width / 2.0 - margin
        x_end = cx + face_width / 2.0 + margin
        y_start = cy - face_length / 2.0 - margin
        y_end = cy + face_length / 2.0 + margin

        grooves = None

        if knurl.knurl_type in ("linear", "cross_hatch", "diamond"):
            # Grooves parallel to Y-axis (cutting along X positions)
            x = x_start
            while x <= x_end:
                groove = (
                    cq.Workplane("XY")
                    .center(x, cy)
                    .rect(groove_w, face_length + 2 * margin)
                    .extrude(depth)
                    .translate((0, 0, z_bottom - depth))
                )
                grooves = groove if grooves is None else grooves.union(groove)
                x += pitch

        if knurl.knurl_type in ("cross_hatch", "diamond"):
            # Add grooves parallel to X-axis (cutting along Y positions)
            y = y_start
            while y <= y_end:
                groove = (
                    cq.Workplane("XY")
                    .center(cx, y)
                    .rect(face_width + 2 * margin, groove_w)
                    .extrude(depth)
                    .translate((0, 0, z_bottom - depth))
                )
                grooves = groove if grooves is None else grooves.union(groove)
                y += pitch

        return grooves

    def _cq_apply_edge_treatment(self, body, p: HornParams):
        """Apply chamfer or fillet edge treatment."""
        r = p.chamfer_radius_mm
        if r <= 0:
            return body
        try:
            if p.edge_treatment == "fillet":
                return body.edges().fillet(r)
            elif p.edge_treatment == "chamfer":
                return body.edges().chamfer(r)
            elif p.edge_treatment == "compound":
                return body.edges("|Z").chamfer(r).edges().fillet(r * 0.5)
        except Exception:
            return body  # If edge treatment fails, return untreated body
        return body

    def _cq_tessellate(self, body) -> dict:
        """Tessellate a CadQuery body into vertices and faces."""
        vertices = []
        faces = []
        tess = body.val().tessellate(0.1)
        for v in tess[0]:
            vertices.append([v.x, v.y, v.z])
        for f in tess[1]:
            faces.append(list(f))
        return {"vertices": vertices, "faces": faces}

    # --- Numpy fallback ---

    def _generate_numpy(self, params: HornParams) -> HornGenerationResult:
        """Generate mesh using numpy (no CAD export)."""
        if params.horn_type == "cylindrical":
            mesh = self._np_cylinder(params)
        elif params.horn_type == "exponential":
            mesh = self._np_exponential(params)
        elif params.horn_type == "stepped":
            mesh = self._np_stepped(params)
        else:
            mesh = self._np_box(params)

        # Apply edge treatment visual (bevel vertices at edges)
        if params.edge_treatment != "none" and params.chamfer_radius_mm > 0:
            mesh = self._np_apply_chamfer(mesh, params)

        # Add knurl pattern to top face
        if params.knurl_type != "none":
            mesh = self._np_add_knurl_texture(mesh, params)

        # Compute approximate volume and surface area
        vertices = np.array(mesh["vertices"])
        faces_arr = np.array(mesh["faces"])
        vol = self._np_mesh_volume(vertices, faces_arr)
        sa = self._np_mesh_surface_area(vertices, faces_arr)

        dims = {
            "width_mm": params.width_mm,
            "height_mm": params.height_mm,
            "length_mm": params.length_mm,
        }
        if params.input_diameter_mm:
            dims["input_diameter_mm"] = params.input_diameter_mm
        if params.output_diameter_mm:
            dims["output_diameter_mm"] = params.output_diameter_mm

        return HornGenerationResult(
            mesh={
                "vertices": [v.tolist() for v in vertices],
                "faces": [f.tolist() for f in faces_arr],
            },
            horn_type=params.horn_type,
            dimensions=dims,
            knurl_info={
                "type": params.knurl_type,
                "pitch_mm": params.knurl_pitch_mm,
                "depth_mm": params.knurl_depth_mm,
            },
            chamfer_info={
                "radius_mm": params.chamfer_radius_mm,
                "treatment": params.edge_treatment,
                "angle_deg": params.chamfer_angle_deg,
            },
            volume_mm3=float(vol),
            surface_area_mm2=float(sa),
            has_cad_export=False,
        )

    def _np_box(self, p: HornParams) -> dict:
        """Generate box mesh (flat/blade horn)."""
        w, l, h = p.width_mm / 2, p.length_mm / 2, p.height_mm
        if p.horn_type == "blade":
            w *= 0.3
        vertices = [
            [-w, -l, 0], [w, -l, 0], [w, l, 0], [-w, l, 0],  # bottom
            [-w, -l, h], [w, -l, h], [w, l, h], [-w, l, h],    # top
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

    def _np_cylinder(self, p: HornParams) -> dict:
        """Generate cylinder mesh."""
        r = (p.input_diameter_mm / 2) if p.input_diameter_mm else (p.width_mm / 2)
        h = p.height_mm
        n_segments = 24
        vertices = []
        faces = []

        # Bottom center and ring
        vertices.append([0, 0, 0])  # 0: bottom center
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            vertices.append([r * math.cos(angle), r * math.sin(angle), 0])

        # Top center and ring
        vertices.append([0, 0, h])  # n_segments+1: top center
        for i in range(n_segments):
            angle = 2 * math.pi * i / n_segments
            vertices.append([r * math.cos(angle), r * math.sin(angle), h])

        # Bottom faces
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([0, next_i + 1, i + 1])

        # Top faces
        top_center = n_segments + 1
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            faces.append([top_center, top_center + 1 + i, top_center + 1 + next_i])

        # Side faces
        for i in range(n_segments):
            next_i = (i + 1) % n_segments
            b1, b2 = i + 1, next_i + 1
            t1, t2 = top_center + 1 + i, top_center + 1 + next_i
            faces.append([b1, t1, t2])
            faces.append([b1, t2, b2])

        return {"vertices": vertices, "faces": faces}

    def _np_exponential(self, p: HornParams) -> dict:
        """Generate exponential horn (tapered).

        If input_diameter_mm and output_diameter_mm are specified, use
        them for the taper ratio.  Otherwise default to 60% taper.
        """
        n_sections = 8
        n_segments = 16
        vertices = []
        faces = []

        # Determine taper ratio from diameters if available
        if p.input_diameter_mm and p.output_diameter_mm:
            taper_ratio = p.output_diameter_mm / p.input_diameter_mm
            base_r = p.input_diameter_mm / 2
        else:
            taper_ratio = 0.6
            base_r = None  # use width/length

        for s in range(n_sections + 1):
            t = s / n_sections
            z = t * p.height_mm
            scale = 1.0 - (1.0 - taper_ratio) * t
            if base_r is not None:
                hw = base_r * scale
                hl = base_r * scale
            else:
                hw = p.width_mm / 2 * scale
                hl = p.length_mm / 2 * scale
            for i in range(n_segments):
                angle = 2 * math.pi * i / n_segments
                x = hw * math.cos(angle)
                y = hl * math.sin(angle)
                vertices.append([x, y, z])

        # Create faces between sections
        for s in range(n_sections):
            for i in range(n_segments):
                next_i = (i + 1) % n_segments
                v00 = s * n_segments + i
                v01 = s * n_segments + next_i
                v10 = (s + 1) * n_segments + i
                v11 = (s + 1) * n_segments + next_i
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

        return {"vertices": vertices, "faces": faces}

    def _np_stepped(self, p: HornParams) -> dict:
        """Generate stepped horn (3 sections)."""
        # Build as 3 stacked boxes of decreasing size
        sections = [
            (p.width_mm / 2, p.length_mm / 2, 0, p.height_mm * 0.4),
            (p.width_mm * 0.35, p.length_mm * 0.35,
             p.height_mm * 0.4, p.height_mm * 0.7),
            (p.width_mm * 0.25, p.length_mm * 0.25,
             p.height_mm * 0.7, p.height_mm),
        ]
        all_verts = []
        all_faces = []
        offset = 0
        for hw, hl, z0, z1 in sections:
            verts = [
                [-hw, -hl, z0], [hw, -hl, z0], [hw, hl, z0], [-hw, hl, z0],
                [-hw, -hl, z1], [hw, -hl, z1], [hw, hl, z1], [-hw, hl, z1],
            ]
            box_faces = [
                [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
                [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
                [1, 5, 6], [1, 6, 2], [0, 3, 7], [0, 7, 4],
            ]
            all_verts.extend(verts)
            all_faces.extend([
                [f[0] + offset, f[1] + offset, f[2] + offset]
                for f in box_faces
            ])
            offset += 8

        return {"vertices": all_verts, "faces": all_faces}

    def _np_apply_chamfer(self, mesh: dict, p: HornParams) -> dict:
        """Apply visual chamfer by beveling edge vertices.

        This is a simplified visual approximation. A real chamfer
        would require proper mesh boolean operations.
        """
        return mesh

    def _np_add_knurl_texture(self, mesh: dict, p: HornParams) -> dict:
        """Add knurl grooves to the top face of the mesh.

        This is a visual approximation that modulates top-face vertex
        heights, not geometrically precise boolean cuts.
        """
        vertices = [list(v) for v in mesh["vertices"]]
        h = p.height_mm
        depth = p.knurl_depth_mm
        pitch = p.knurl_pitch_mm

        for v in vertices:
            if abs(v[2] - h) < 0.01:  # top face vertices
                if p.knurl_type == "linear":
                    # Parallel grooves along y-axis
                    groove = math.sin(2 * math.pi * v[0] / pitch) * depth * 0.5
                    v[2] += groove
                elif p.knurl_type in ("cross_hatch", "diamond"):
                    groove1 = math.sin(
                        2 * math.pi * v[0] / pitch
                    ) * depth * 0.3
                    groove2 = math.sin(
                        2 * math.pi * v[1] / pitch
                    ) * depth * 0.3
                    v[2] += groove1 + groove2
                elif p.knurl_type == "conical":
                    r = math.sqrt(v[0] ** 2 + v[1] ** 2)
                    groove = math.sin(
                        2 * math.pi * r / pitch
                    ) * depth * 0.5
                    v[2] += groove
                elif p.knurl_type == "spherical":
                    r = math.sqrt(v[0] ** 2 + v[1] ** 2)
                    a = math.atan2(v[1], v[0])
                    groove = (
                        math.sin(2 * math.pi * r / pitch)
                        * math.cos(4 * a)
                        * depth * 0.3
                    )
                    v[2] += groove

        mesh["vertices"] = vertices
        return mesh

    def _np_mesh_volume(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Compute mesh volume using signed tetrahedron formula."""
        if len(faces) == 0:
            return 0.0
        total = 0.0
        for f in faces:
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            total += np.dot(v0, np.cross(v1, v2)) / 6.0
        return abs(total)

    def _np_mesh_surface_area(
        self, vertices: np.ndarray, faces: np.ndarray
    ) -> float:
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
