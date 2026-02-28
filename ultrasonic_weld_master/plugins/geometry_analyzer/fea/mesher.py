"""Gmsh-based mesh generator for ultrasonic horn geometries.

Generates TET4/TET10 tetrahedral meshes from parametric horn descriptions
using the Gmsh Python API with the OpenCASCADE (OCC) geometry kernel.

All geometry is built in meters (input dimensions in mm are converted).
The Y-axis is the longitudinal/axial direction.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import gmsh
import numpy as np

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh

logger = logging.getLogger(__name__)

# Gmsh element type codes
_GMSH_TET4 = 4
_GMSH_TET10 = 11
_GMSH_TRI3 = 2
_GMSH_TRI6 = 9

# Gmsh TET10 node ordering (parametric u,v,w):
#   0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(0,0,1)
#   4:(½,0,0)  5:(½,½,0)  6:(0,½,0)  7:(0,0,½)
#   8:(0,½,½)  9:(½,0,½)
#
# Bathe/standard TET10 node ordering (barycentric L1=xi, L2=eta, L3=zeta):
#   0:(1,0,0)  1:(0,1,0)  2:(0,0,1)  3:(0,0,0)
#   4:(½,½,0)  5:(0,½,½)  6:(½,0,½)  7:(½,0,0)
#   8:(0,½,0)  9:(0,0,½)
#
# Mapping: Bathe node i is at Gmsh node j:
#   Bathe 0 -> Gmsh 1, Bathe 1 -> Gmsh 2, Bathe 2 -> Gmsh 3, Bathe 3 -> Gmsh 0
#   Bathe 4 -> Gmsh 5, Bathe 5 -> Gmsh 8, Bathe 6 -> Gmsh 9, Bathe 7 -> Gmsh 4
#   Bathe 8 -> Gmsh 6, Bathe 9 -> Gmsh 7
_GMSH_TO_BATHE_TET10 = [1, 2, 3, 0, 5, 8, 9, 4, 6, 7]


class GmshMesher:
    """Generate finite element meshes for ultrasonic horn geometries.

    Uses the Gmsh Python API with the OpenCASCADE kernel to create
    parametric 3D geometry and produce tetrahedral meshes (TET4 or TET10).

    The generated meshes include:
    - Volume elements (TET4 or TET10)
    - Surface triangulation for visualization
    - Auto-detected node sets for top_face and bottom_face

    All coordinates in the output mesh are in meters.
    """

    # Supported horn types
    HORN_TYPES = ("cylindrical", "flat")

    # Mesh density name -> element size in mm
    MESH_DENSITY_MAP: dict[str, float] = {
        "coarse": 8.0,
        "medium": 5.0,
        "fine": 3.0,
    }

    def mesh_parametric_horn(
        self,
        horn_type: str,
        dimensions: dict[str, float],
        mesh_size: float = 2.0,
        order: int = 2,
        mesh_density: str | None = None,
        knurl_info: dict | None = None,
    ) -> FEAMesh:
        """Generate a mesh for a parametric horn geometry.

        Parameters
        ----------
        horn_type : str
            Type of horn geometry: ``"cylindrical"`` or ``"flat"``.
        dimensions : dict
            Dimensions in **millimeters**.

            For ``"cylindrical"``:
                ``{"diameter_mm": float, "length_mm": float}``

            For ``"flat"`` (rectangular block):
                ``{"width_mm": float, "depth_mm": float, "length_mm": float}``
        mesh_size : float
            Characteristic element size in mm (default 2.0).
            Ignored when *mesh_density* is set to a named preset.
        order : int
            Element order: 1 for TET4, 2 for TET10 (default 2).
        mesh_density : str or None
            Named density preset: ``"coarse"`` (8 mm), ``"medium"`` (5 mm),
            ``"fine"`` (3 mm), or ``"adaptive"``.  When ``"adaptive"`` is
            chosen the mesher applies field-based refinement with fine
            elements at the weld face (Y-min) and coarse elements
            elsewhere.  If *None*, *mesh_size* is used directly.
        knurl_info : dict or None
            Optional knurl description for knurl-aware mesh refinement.
            When provided, the mesher applies local refinement near the
            knurl region (bottom/weld face) with 0.3-0.5 mm element size
            at knurl features and 5-8 mm elsewhere.  Expected keys:

            - ``"type"`` : str -- knurl type (e.g. ``"linear"``,
              ``"cross_hatch"``).  If ``"none"`` or absent, no knurl
              refinement is applied.
            - ``"pitch_mm"`` : float -- knurl pitch in mm (used to set
              the fine element size).
            - ``"depth_mm"`` : float -- knurl depth in mm (used to set
              the refinement transition distance).

            This parameter takes priority over *mesh_density*; when both
            *knurl_info* and ``mesh_density="adaptive"`` are set, the
            knurl refinement fields are used.

        Returns
        -------
        FEAMesh
            Mesh dataclass with nodes in meters, element connectivity,
            node sets, surface triangulation, and statistics.

        Raises
        ------
        ValueError
            If *horn_type* is unsupported or required dimension keys are
            missing.
        RuntimeError
            If mesh generation fails inside Gmsh.
        """
        self._validate_inputs(horn_type, dimensions, order)

        is_knurl = self._has_knurl(knurl_info)
        is_adaptive = mesh_density == "adaptive"

        # Resolve effective mesh size
        if mesh_density is not None and mesh_density != "adaptive":
            mesh_size = self.MESH_DENSITY_MAP.get(mesh_density, mesh_size)

        # Convert mm -> m
        mesh_size_m = mesh_size / 1000.0

        gmsh.initialize(interruptible=False)
        try:
            # Run headless -- no terminal output, no GUI
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("horn")

            # Build geometry (in meters)
            if horn_type == "cylindrical":
                self._build_cylinder(dimensions)
            elif horn_type == "flat":
                self._build_box(dimensions)

            gmsh.model.occ.synchronize()

            if is_knurl:
                # Knurl-aware refinement: very fine at knurl features,
                # standard size elsewhere.
                bottom_faces = self._find_bottom_face_tags()
                fine_m, coarse_m = self._knurl_mesh_sizes(knurl_info)
                self._apply_knurl_refinement_fields(
                    bottom_faces,
                    knurl_info=knurl_info,
                    fine_size=fine_m,
                    coarse_size=coarse_m,
                )
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, coarse_m)
                mesh_size_m = coarse_m
            elif is_adaptive:
                # Adaptive: field-based size control
                fine_size_m = 3.0 / 1000.0   # 3 mm in meters
                coarse_size_m = 8.0 / 1000.0  # 8 mm in meters
                bottom_faces = self._find_bottom_face_tags()
                self._apply_adaptive_fields(
                    bottom_faces,
                    fine_size=fine_size_m,
                    coarse_size=coarse_size_m,
                )
                # Still need a fallback max size on geometry points
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, coarse_size_m)
                # Use coarse size for node-set detection tolerance
                mesh_size_m = coarse_size_m
            else:
                # Uniform: set mesh size on all points
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, mesh_size_m)

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Set element order (1 = linear, 2 = quadratic)
            if order == 2:
                gmsh.model.mesh.setOrder(2)

            # Extract mesh data
            nodes, coords = self._extract_nodes()
            vol_elements, vol_etype = self._extract_volume_elements(order)
            surface_tris = self._extract_surface_tris()

            # Remap Gmsh node tags to 0-based indices
            nodes_coords, elements, surface_tris_remapped = self._remap_indices(
                nodes, coords, vol_elements, surface_tris
            )

            # Reorder TET10 element nodes from Gmsh convention to Bathe
            # convention so that the connectivity is compatible with the
            # TET10Element formulation (shape functions, B-matrix, etc.).
            if order == 2:
                elements = elements[:, _GMSH_TO_BATHE_TET10]

            # Determine element type string
            element_type = "TET4" if order == 1 else "TET10"

            # Detect top and bottom face node sets
            node_sets = self._detect_face_node_sets(
                nodes_coords, mesh_size_m
            )

            mesh_stats: dict[str, Any] = {
                "num_nodes": nodes_coords.shape[0],
                "num_elements": elements.shape[0],
                "num_surface_tris": surface_tris_remapped.shape[0],
                "element_type": element_type,
                "order": order,
                "mesh_size_mm": mesh_size,
            }
            if mesh_density is not None:
                mesh_stats["mesh_density"] = mesh_density
            if is_knurl:
                mesh_stats["knurl_refinement"] = True
                mesh_stats["knurl_info"] = knurl_info

            logger.info(
                "Generated %s mesh: %d nodes, %d elements",
                element_type,
                mesh_stats["num_nodes"],
                mesh_stats["num_elements"],
            )

            return FEAMesh(
                nodes=nodes_coords,
                elements=elements,
                element_type=element_type,
                node_sets=node_sets,
                element_sets={},
                surface_tris=surface_tris_remapped,
                mesh_stats=mesh_stats,
            )
        except Exception:
            logger.exception("Mesh generation failed")
            raise
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------
    # Geometry builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cylinder(dimensions: dict[str, float]) -> None:
        """Create a cylinder aligned along the Y-axis.

        The cylinder is centred on the X-Z plane with:
        - base at y = 0
        - top  at y = length

        Parameters
        ----------
        dimensions : dict
            Must contain ``diameter_mm`` and ``length_mm``.
        """
        radius_m = (dimensions["diameter_mm"] / 2.0) / 1000.0
        length_m = dimensions["length_mm"] / 1000.0

        # addCylinder(x, y, z, dx, dy, dz, r)
        # Origin at (0, 0, 0), extending along +Y
        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, length_m, 0.0, radius_m)

    @staticmethod
    def _build_box(dimensions: dict[str, float]) -> None:
        """Create a rectangular box (flat horn) aligned along the Y-axis.

        The box is centred on the X-Z plane with:
        - base at y = 0
        - top  at y = length

        Parameters
        ----------
        dimensions : dict
            Must contain ``width_mm``, ``depth_mm``, and ``length_mm``.
        """
        width_m = dimensions["width_mm"] / 1000.0
        depth_m = dimensions["depth_mm"] / 1000.0
        length_m = dimensions["length_mm"] / 1000.0

        # Centre the box in X and Z, with y from 0 to length_m
        x0 = -width_m / 2.0
        z0 = -depth_m / 2.0

        gmsh.model.occ.addBox(x0, 0.0, z0, width_m, length_m, depth_m)

    # ------------------------------------------------------------------
    # Mesh data extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_nodes() -> tuple[np.ndarray, np.ndarray]:
        """Return (node_tags, coordinates) from the Gmsh model.

        Returns
        -------
        node_tags : np.ndarray, shape (N,)
            Gmsh 1-based node tags.
        coords : np.ndarray, shape (N, 3)
            Node coordinates in meters.
        """
        node_tags, coord_flat, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coords = np.asarray(coord_flat, dtype=np.float64).reshape(-1, 3)
        return node_tags, coords

    @staticmethod
    def _extract_volume_elements(order: int) -> tuple[np.ndarray, int]:
        """Extract 3D tetrahedral elements.

        Parameters
        ----------
        order : int
            1 for TET4, 2 for TET10.

        Returns
        -------
        connectivity : np.ndarray, shape (E, nodes_per_elem)
            Element connectivity using Gmsh node tags.
        etype : int
            Gmsh element type code.
        """
        target_etype = _GMSH_TET4 if order == 1 else _GMSH_TET10
        nodes_per_elem = 4 if order == 1 else 10

        elem_types, elem_tags, elem_nodes = gmsh.model.mesh.getElements(dim=3)

        for i, etype in enumerate(elem_types):
            if etype == target_etype:
                conn = np.asarray(elem_nodes[i], dtype=np.int64).reshape(
                    -1, nodes_per_elem
                )
                return conn, etype

        raise RuntimeError(
            f"No {('TET4' if order == 1 else 'TET10')} elements found in mesh"
        )

    @staticmethod
    def _extract_surface_tris() -> np.ndarray:
        """Extract surface triangulation (TRI3) for visualization.

        Even when the volume mesh is TET10, we extract the linear
        triangles (TRI3) for lightweight visualisation. If only TRI6
        elements are present (quadratic surface mesh), we extract the
        corner nodes of the TRI6 elements.

        Returns
        -------
        tris : np.ndarray, shape (F, 3)
            Surface triangle connectivity using Gmsh node tags.
        """
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=2)

        for i, etype in enumerate(elem_types):
            if etype == _GMSH_TRI3:
                return np.asarray(elem_nodes[i], dtype=np.int64).reshape(-1, 3)

        # Fallback: extract corner nodes from TRI6 (quadratic triangles)
        for i, etype in enumerate(elem_types):
            if etype == _GMSH_TRI6:
                tri6 = np.asarray(elem_nodes[i], dtype=np.int64).reshape(-1, 6)
                # TRI6 node ordering: first 3 are corner nodes
                return tri6[:, :3]

        raise RuntimeError("No surface triangles found in mesh")

    @staticmethod
    def _remap_indices(
        node_tags: np.ndarray,
        coords: np.ndarray,
        vol_conn: np.ndarray,
        surf_tris: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remap Gmsh 1-based node tags to 0-based contiguous indices.

        Parameters
        ----------
        node_tags : np.ndarray
            Gmsh node tags (1-based, possibly non-contiguous).
        coords : np.ndarray, shape (N, 3)
            Coordinates matching *node_tags*.
        vol_conn : np.ndarray, shape (E, nodes_per_elem)
            Volume element connectivity in Gmsh tags.
        surf_tris : np.ndarray, shape (F, 3)
            Surface triangle connectivity in Gmsh tags.

        Returns
        -------
        new_coords : np.ndarray, shape (N, 3)
        new_vol_conn : np.ndarray, shape (E, nodes_per_elem)
        new_surf_tris : np.ndarray, shape (F, 3)
        """
        # Build a mapping: gmsh_tag -> 0-based index
        max_tag = int(node_tags.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        for new_idx, tag in enumerate(node_tags):
            tag_to_idx[tag] = new_idx

        new_coords = coords.copy()
        new_vol_conn = tag_to_idx[vol_conn]
        new_surf_tris = tag_to_idx[surf_tris]

        # Sanity check -- all indices should be >= 0
        if np.any(new_vol_conn < 0):
            raise RuntimeError("Volume element references unmapped node tag")
        if np.any(new_surf_tris < 0):
            raise RuntimeError("Surface triangle references unmapped node tag")

        return new_coords, new_vol_conn, new_surf_tris

    @staticmethod
    def _detect_face_node_sets(
        coords: np.ndarray,
        mesh_size_m: float,
    ) -> dict[str, np.ndarray]:
        """Identify top_face and bottom_face node sets.

        Nodes are classified by their Y-coordinate:
        - ``bottom_face``: y == y_min within tolerance
        - ``top_face``: y == y_max within tolerance

        Parameters
        ----------
        coords : np.ndarray, shape (N, 3)
            Node coordinates in meters.
        mesh_size_m : float
            Characteristic mesh size in meters, used to set tolerance.

        Returns
        -------
        dict mapping set name to 0-based node index arrays.
        """
        y_coords = coords[:, 1]
        y_min = y_coords.min()
        y_max = y_coords.max()
        tol = mesh_size_m * 0.1

        bottom_mask = np.abs(y_coords - y_min) < tol
        top_mask = np.abs(y_coords - y_max) < tol

        return {
            "bottom_face": np.where(bottom_mask)[0],
            "top_face": np.where(top_mask)[0],
        }

    # ------------------------------------------------------------------
    # Adaptive mesh refinement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_bottom_face_tags() -> list[int]:
        """Find Gmsh face tags at the bottom (Y-min) of the model.

        After geometry is built and synchronised, this method inspects all
        surface entities and returns those whose bounding-box centroid
        Y-coordinate is within tolerance of the global Y-min.  For the
        parametric horns the Y-axis is the longitudinal direction, so
        Y-min corresponds to the weld face (bottom face).

        Returns
        -------
        list[int]
            Gmsh surface entity tags at the bottom of the model.
        """
        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
            return []

        # Compute Y-centroid for every surface
        face_y: list[tuple[int, float]] = []
        for _dim, tag in surfaces:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                    2, tag
                )
                y_mid = (ymin + ymax) / 2.0
                face_y.append((tag, y_mid))
            except Exception:
                continue

        if not face_y:
            return []

        y_min_global = min(y for _, y in face_y)

        # Global bounding box for tolerance reference
        bbox = gmsh.model.getBoundingBox(-1, -1)
        height = abs(bbox[4] - bbox[1])  # ymax - ymin
        tol = height * 0.01 if height > 0 else 1e-6

        bottom_tags = [
            tag for tag, y in face_y if abs(y - y_min_global) < tol
        ]
        return bottom_tags

    @staticmethod
    def _apply_adaptive_fields(
        fine_face_tags: list[int],
        fine_size: float = 3.0,
        coarse_size: float = 8.0,
    ) -> None:
        """Apply Gmsh mesh size fields for adaptive refinement.

        Uses a Distance field from the specified faces combined with a
        Threshold field to produce a smooth transition from *fine_size*
        at the target faces to *coarse_size* far away from them.

        Parameters
        ----------
        fine_face_tags : list[int]
            Gmsh surface entity tags where fine mesh is desired.
        fine_size : float
            Element size at the target faces (mm converted to meters
            by the caller, or in native model units).
        coarse_size : float
            Element size far from the target faces.
        """
        if not fine_face_tags:
            logger.warning(
                "No face tags provided for adaptive refinement; "
                "falling back to uniform coarse mesh."
            )
            return

        # Distance field from target faces
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "SurfacesList", fine_face_tags)

        # Threshold field for smooth transition
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", fine_size)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", coarse_size)
        gmsh.model.mesh.field.setNumber(2, "DistMin", fine_size * 2)
        gmsh.model.mesh.field.setNumber(2, "DistMax", coarse_size * 5)

        # Set as background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # Disable default mesh size from geometry so the field controls
        # element sizes exclusively
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        logger.info(
            "Adaptive mesh fields applied: fine=%.2f at %d face(s), coarse=%.2f",
            fine_size,
            len(fine_face_tags),
            coarse_size,
        )

    # ------------------------------------------------------------------
    # Knurl-aware mesh refinement
    # ------------------------------------------------------------------

    @staticmethod
    def _has_knurl(knurl_info: dict | None) -> bool:
        """Return ``True`` if *knurl_info* describes an active knurl pattern.

        A knurl is considered active when:
        - *knurl_info* is not ``None``
        - ``knurl_info["type"]`` is present and is not ``"none"``
        """
        if knurl_info is None:
            return False
        knurl_type = knurl_info.get("type", "none")
        return knurl_type not in ("none", "", None)

    @staticmethod
    def _knurl_mesh_sizes(
        knurl_info: dict | None,
    ) -> tuple[float, float]:
        """Compute fine and coarse element sizes for knurl refinement.

        The fine size is derived from the knurl pitch so that at least
        3--4 elements span each groove (clamped to 0.3--0.5 mm).
        The coarse size for bulk material is 5--8 mm.

        Parameters
        ----------
        knurl_info : dict or None
            Knurl description with optional ``"pitch_mm"`` key.

        Returns
        -------
        (fine_size_m, coarse_size_m)
            Element sizes in **meters**.
        """
        # Defaults: 0.4 mm fine, 6 mm coarse
        fine_mm = 0.4
        coarse_mm = 6.0

        if knurl_info is not None:
            pitch = knurl_info.get("pitch_mm", 1.0)
            # Aim for ~3 elements per pitch, clamp to 0.3-0.5 mm
            fine_mm = max(0.3, min(0.5, pitch / 3.0))
            # Coarse: 5-8 mm depending on pitch
            coarse_mm = max(5.0, min(8.0, pitch * 6.0))

        return fine_mm / 1000.0, coarse_mm / 1000.0

    @staticmethod
    def _apply_knurl_refinement_fields(
        knurl_face_tags: list[int],
        knurl_info: dict | None = None,
        fine_size: float = 0.0004,
        coarse_size: float = 0.006,
    ) -> None:
        """Apply Gmsh mesh size fields for knurl-aware refinement.

        Uses a ``Distance`` field from the knurl faces combined with a
        ``Threshold`` field to produce a smooth transition from
        *fine_size* at the knurl surfaces to *coarse_size* far from them.

        The transition distances are tuned for typical knurl geometries:
        the fine zone extends slightly beyond the knurl depth, and the
        transition to coarse occurs gradually to avoid abrupt element
        size jumps.

        Parameters
        ----------
        knurl_face_tags : list[int]
            Gmsh surface entity tags identifying the knurl region
            (typically the bottom/weld face of the horn).
        knurl_info : dict or None
            Knurl description; ``"depth_mm"`` is used to set the
            transition distance.  Falls back to sensible defaults.
        fine_size : float
            Element size at the knurl surface (in model native units,
            typically meters).
        coarse_size : float
            Element size far from the knurl region.
        """
        if not knurl_face_tags:
            logger.warning(
                "No knurl face tags provided for knurl refinement; "
                "falling back to uniform mesh."
            )
            return

        # Determine transition distances from knurl depth
        depth_mm = 0.3
        if knurl_info is not None:
            depth_mm = knurl_info.get("depth_mm", 0.3)

        # DistMin: fine region extends 2x the knurl depth from the surface
        # DistMax: transition completes at ~10x the knurl depth
        # Both in the same unit system as fine_size/coarse_size
        # (caller is responsible for unit consistency)
        dist_min = fine_size * 3.0   # ~3 fine elements worth of fine zone
        dist_max = coarse_size * 3.0  # smooth transition to coarse

        # Distance field from knurl faces
        f_dist = gmsh.model.mesh.field.add("Distance", 100)
        gmsh.model.mesh.field.setNumbers(
            f_dist, "SurfacesList", knurl_face_tags
        )

        # Threshold field for smooth size transition
        f_thresh = gmsh.model.mesh.field.add("Threshold", 101)
        gmsh.model.mesh.field.setNumber(f_thresh, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMin", fine_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "SizeMax", coarse_size)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMin", dist_min)
        gmsh.model.mesh.field.setNumber(f_thresh, "DistMax", dist_max)

        # Set as background mesh
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh)

        # Disable default mesh size sources so the field controls sizes
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        logger.info(
            "Knurl mesh refinement applied: fine=%.4f at %d face(s), "
            "coarse=%.4f, depth_mm=%.2f",
            fine_size,
            len(knurl_face_tags),
            coarse_size,
            depth_mm,
        )

    # ------------------------------------------------------------------
    # STEP import
    # ------------------------------------------------------------------

    def mesh_from_step(
        self,
        step_path: str,
        mesh_size: float = 3.0,
        order: int = 2,
        mesh_density: str | None = None,
        knurl_info: dict | None = None,
    ) -> FEAMesh:
        """Import a STEP file and generate a tetrahedral mesh.

        Parameters
        ----------
        step_path : str
            Path to the STEP file (``*.step`` or ``*.stp``).
        mesh_size : float
            Characteristic element size in mm (default 3.0).
            Ignored when *mesh_density* is set to a named preset.
        order : int
            Element order: 1 for TET4, 2 for TET10 (default 2).
        mesh_density : str or None
            Named density preset: ``"coarse"``, ``"medium"``,
            ``"fine"``, or ``"adaptive"``.  See
            :meth:`mesh_parametric_horn` for details.
        knurl_info : dict or None
            Optional knurl description for knurl-aware mesh refinement.
            See :meth:`mesh_parametric_horn` for details.

        Returns
        -------
        FEAMesh
            Mesh dataclass with nodes in meters, element connectivity,
            auto-detected node sets, surface triangulation, and statistics.

        Raises
        ------
        FileNotFoundError
            If *step_path* does not exist.
        ValueError
            If *step_path* is not a valid STEP file or *order* is invalid.
        RuntimeError
            If STEP import or mesh generation fails.
        """
        self._validate_step_inputs(step_path, order)

        is_knurl = self._has_knurl(knurl_info)
        is_adaptive = mesh_density == "adaptive"

        # Resolve effective mesh size from density name
        if mesh_density is not None and mesh_density != "adaptive":
            mesh_size = self.MESH_DENSITY_MAP.get(mesh_density, mesh_size)

        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("step_import")

            # Import STEP using OCC kernel
            try:
                shapes = gmsh.model.occ.importShapes(step_path)
            except Exception as exc:
                raise ValueError(
                    f"Failed to import STEP file {step_path!r}: {exc}"
                ) from exc

            if not shapes:
                raise ValueError(
                    f"No shapes found in STEP file {step_path!r}"
                )

            gmsh.model.occ.synchronize()

            # Auto-detect volumes
            volumes = gmsh.model.getEntities(dim=3)
            logger.info(
                "Imported STEP with %d volume(s) from %s",
                len(volumes),
                step_path,
            )

            # Detect coordinate scale: STEP files from CAD software are
            # typically in millimeters but some may use meters.  We check
            # the bounding box and if the max dimension > 0.5 we assume mm.
            bbox = gmsh.model.getBoundingBox(-1, -1)  # (xmin,ymin,zmin,xmax,ymax,zmax)
            max_dim = max(
                abs(bbox[3] - bbox[0]),
                abs(bbox[4] - bbox[1]),
                abs(bbox[5] - bbox[2]),
            )
            step_in_mm = max_dim > 0.5  # If max dim > 0.5 then coords are in mm
            logger.info(
                "STEP bounding box max dim=%.4f, detected unit=%s",
                max_dim,
                "mm" if step_in_mm else "m",
            )

            # Mesh size must match the STEP coordinate system.
            # mesh_size is specified in mm by the caller.
            if step_in_mm:
                mesh_size_native = mesh_size  # Use mm directly
            else:
                mesh_size_native = mesh_size / 1000.0  # Convert to m

            # Auto-identify faces
            face_sets = self._auto_identify_faces(volumes)

            if is_knurl:
                # Knurl-aware refinement: very fine at knurl features
                bottom_faces = self._find_bottom_face_tags()
                fine_m, coarse_m = self._knurl_mesh_sizes(knurl_info)
                # Convert to native STEP units if needed
                if step_in_mm:
                    fine_native = fine_m * 1000.0
                    coarse_native = coarse_m * 1000.0
                else:
                    fine_native = fine_m
                    coarse_native = coarse_m
                self._apply_knurl_refinement_fields(
                    bottom_faces,
                    knurl_info=knurl_info,
                    fine_size=fine_native,
                    coarse_size=coarse_native,
                )
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, coarse_native)
                mesh_size_native = coarse_native
            elif is_adaptive:
                # Adaptive: field-based size control
                fine_native = 3.0 if step_in_mm else 3.0 / 1000.0
                coarse_native = 8.0 if step_in_mm else 8.0 / 1000.0
                bottom_faces = self._find_bottom_face_tags()
                self._apply_adaptive_fields(
                    bottom_faces,
                    fine_size=fine_native,
                    coarse_size=coarse_native,
                )
                # Fallback max size on geometry points
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, coarse_native)
                mesh_size_native = coarse_native
            else:
                # Uniform: set mesh size on all points
                entities = gmsh.model.getEntities(0)
                gmsh.model.mesh.setSize(entities, mesh_size_native)

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)
            if order == 2:
                gmsh.model.mesh.setOrder(2)

            # Extract mesh data
            nodes, coords = self._extract_nodes()
            vol_elements, _ = self._extract_volume_elements(order)
            surface_tris = self._extract_surface_tris()

            # Remap to 0-based
            nodes_coords, elements, surface_tris_remapped = self._remap_indices(
                nodes, coords, vol_elements, surface_tris
            )

            # Convert node coordinates to meters (SI) if STEP was in mm
            if step_in_mm:
                nodes_coords = nodes_coords / 1000.0

            if order == 2:
                elements = elements[:, _GMSH_TO_BATHE_TET10]

            element_type = "TET4" if order == 1 else "TET10"

            # Build node sets: Y-based top/bottom detection (coords now in meters)
            mesh_size_m = mesh_size / 1000.0
            node_sets = self._detect_face_node_sets(nodes_coords, mesh_size_m)

            # Add face-tag-based node sets from auto identification
            tag_to_idx = self._build_tag_map(nodes)
            for set_name, face_tags in face_sets.items():
                face_node_indices = self._get_face_node_indices(
                    face_tags, tag_to_idx
                )
                if len(face_node_indices) > 0:
                    node_sets[set_name] = face_node_indices

            mesh_stats = {
                "num_nodes": nodes_coords.shape[0],
                "num_elements": elements.shape[0],
                "num_surface_tris": surface_tris_remapped.shape[0],
                "element_type": element_type,
                "order": order,
                "mesh_size_mm": mesh_size,
                "num_volumes": len(volumes),
                "source": step_path,
            }
            if is_knurl:
                mesh_stats["knurl_refinement"] = True
                mesh_stats["knurl_info"] = knurl_info

            logger.info(
                "Generated %s mesh from STEP: %d nodes, %d elements",
                element_type,
                mesh_stats["num_nodes"],
                mesh_stats["num_elements"],
            )

            return FEAMesh(
                nodes=nodes_coords,
                elements=elements,
                element_type=element_type,
                node_sets=node_sets,
                element_sets={},
                surface_tris=surface_tris_remapped,
                mesh_stats=mesh_stats,
            )
        except Exception:
            logger.exception("STEP mesh generation failed")
            raise
        finally:
            gmsh.finalize()

    def mesh_multi_body_step(
        self,
        step_path: str,
        mesh_size: float = 3.0,
        order: int = 2,
    ) -> list[FEAMesh]:
        """Import a multi-body STEP file and mesh each body separately.

        Parameters
        ----------
        step_path : str
            Path to the STEP file containing one or more bodies.
        mesh_size : float
            Characteristic element size in mm (default 3.0).
        order : int
            Element order: 1 for TET4, 2 for TET10 (default 2).

        Returns
        -------
        list[FEAMesh]
            One FEAMesh per body. Each mesh includes an
            ``"interface_faces"`` node set for faces shared between bodies.

        Raises
        ------
        FileNotFoundError
            If *step_path* does not exist.
        ValueError
            If *step_path* is not valid or *order* is invalid.
        RuntimeError
            If mesh generation fails.
        """
        self._validate_step_inputs(step_path, order)

        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("multi_body_step")

            try:
                shapes = gmsh.model.occ.importShapes(step_path)
            except Exception as exc:
                raise ValueError(
                    f"Failed to import STEP file {step_path!r}: {exc}"
                ) from exc

            if not shapes:
                raise ValueError(
                    f"No shapes found in STEP file {step_path!r}"
                )

            gmsh.model.occ.synchronize()

            volumes = gmsh.model.getEntities(dim=3)
            n_bodies = len(volumes)
            logger.info(
                "Multi-body STEP import: %d volume(s) from %s",
                n_bodies,
                step_path,
            )

            if n_bodies == 0:
                raise RuntimeError(
                    f"No 3D volumes found in STEP file {step_path!r}"
                )

            # Detect coordinate scale: STEP files typically use mm
            bbox = gmsh.model.getBoundingBox(-1, -1)
            max_dim = max(
                abs(bbox[3] - bbox[0]),
                abs(bbox[4] - bbox[1]),
                abs(bbox[5] - bbox[2]),
            )
            step_in_mm = max_dim > 0.5

            if step_in_mm:
                mesh_size_native = mesh_size  # mm
            else:
                mesh_size_native = mesh_size / 1000.0  # m

            # Detect interface faces between adjacent bodies
            interface_faces = self._detect_interface_faces(volumes)

            # Set mesh size and generate mesh for the entire model
            entities = gmsh.model.getEntities(0)
            gmsh.model.mesh.setSize(entities, mesh_size_native)
            gmsh.model.mesh.generate(3)
            if order == 2:
                gmsh.model.mesh.setOrder(2)

            element_type = "TET4" if order == 1 else "TET10"
            nodes_per_elem = 4 if order == 1 else 10
            target_etype = _GMSH_TET4 if order == 1 else _GMSH_TET10

            # Get all nodes globally for remap
            all_node_tags, all_coord_flat, _ = gmsh.model.mesh.getNodes()
            all_node_tags = np.asarray(all_node_tags, dtype=np.int64)
            all_coords = np.asarray(all_coord_flat, dtype=np.float64).reshape(-1, 3)
            # Convert to meters if STEP was in mm
            if step_in_mm:
                all_coords = all_coords / 1000.0
            mesh_size_m = mesh_size / 1000.0
            max_tag = int(all_node_tags.max())
            global_tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
            for idx, tag in enumerate(all_node_tags):
                global_tag_to_idx[tag] = idx

            meshes: list[FEAMesh] = []

            for vol_dim, vol_tag in volumes:
                # Get elements for this volume
                elem_types, _, elem_nodes = gmsh.model.mesh.getElements(
                    dim=3, tag=vol_tag
                )

                vol_conn = None
                for i, etype in enumerate(elem_types):
                    if etype == target_etype:
                        vol_conn = np.asarray(
                            elem_nodes[i], dtype=np.int64
                        ).reshape(-1, nodes_per_elem)
                        break

                if vol_conn is None:
                    logger.warning(
                        "Volume %d has no %s elements, skipping",
                        vol_tag,
                        element_type,
                    )
                    continue

                # Collect unique node tags for this body
                unique_tags = np.unique(vol_conn)

                # Build local node set
                local_tag_to_idx = {}
                local_coords = np.zeros((len(unique_tags), 3), dtype=np.float64)
                for local_idx, tag in enumerate(unique_tags):
                    local_tag_to_idx[int(tag)] = local_idx
                    global_idx = global_tag_to_idx[tag]
                    local_coords[local_idx] = all_coords[global_idx]

                # Remap element connectivity to local indices
                local_conn = np.zeros_like(vol_conn)
                for i in range(vol_conn.shape[0]):
                    for j in range(vol_conn.shape[1]):
                        local_conn[i, j] = local_tag_to_idx[int(vol_conn[i, j])]

                if order == 2:
                    local_conn = local_conn[:, _GMSH_TO_BATHE_TET10]

                # Extract surface tris for this volume's boundary faces
                faces = gmsh.model.getBoundary(
                    [(vol_dim, vol_tag)], oriented=False, recursive=False
                )
                face_tags = [abs(f[1]) for f in faces]

                tri_list = []
                for ftag in face_tags:
                    try:
                        et, _, en = gmsh.model.mesh.getElements(dim=2, tag=ftag)
                    except Exception:
                        continue
                    for i, etype in enumerate(et):
                        if etype == _GMSH_TRI3:
                            raw = np.asarray(en[i], dtype=np.int64).reshape(-1, 3)
                            tri_list.append(raw)
                        elif etype == _GMSH_TRI6:
                            raw = np.asarray(en[i], dtype=np.int64).reshape(-1, 6)
                            tri_list.append(raw[:, :3])

                if tri_list:
                    surf_tris = np.vstack(tri_list)
                    # Remap surface tris to local indices
                    local_tris = np.zeros_like(surf_tris)
                    valid = True
                    for i in range(surf_tris.shape[0]):
                        for j in range(surf_tris.shape[1]):
                            t = int(surf_tris[i, j])
                            if t in local_tag_to_idx:
                                local_tris[i, j] = local_tag_to_idx[t]
                            else:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        local_tris = np.zeros((0, 3), dtype=np.int64)
                else:
                    local_tris = np.zeros((0, 3), dtype=np.int64)

                # Detect node sets
                node_sets = self._detect_face_node_sets(
                    local_coords, mesh_size_m
                )

                # Add interface face nodes if applicable
                vol_interface_tags = interface_faces.get(vol_tag, [])
                if vol_interface_tags:
                    iface_nodes = set()
                    for iftag in vol_interface_tags:
                        try:
                            ntags, _, _ = gmsh.model.mesh.getNodes(
                                dim=2, tag=iftag, includeBoundary=True
                            )
                        except Exception:
                            continue
                        for nt in ntags:
                            nt_int = int(nt)
                            if nt_int in local_tag_to_idx:
                                iface_nodes.add(local_tag_to_idx[nt_int])
                    if iface_nodes:
                        node_sets["interface_faces"] = np.array(
                            sorted(iface_nodes), dtype=np.int64
                        )

                mesh_stats = {
                    "num_nodes": local_coords.shape[0],
                    "num_elements": local_conn.shape[0],
                    "num_surface_tris": local_tris.shape[0],
                    "element_type": element_type,
                    "order": order,
                    "mesh_size_mm": mesh_size,
                    "body_tag": vol_tag,
                    "source": step_path,
                }

                meshes.append(
                    FEAMesh(
                        nodes=local_coords,
                        elements=local_conn,
                        element_type=element_type,
                        node_sets=node_sets,
                        element_sets={},
                        surface_tris=local_tris,
                        mesh_stats=mesh_stats,
                    )
                )

            logger.info(
                "Multi-body meshing complete: %d meshes from %d volumes",
                len(meshes),
                n_bodies,
            )
            return meshes
        except Exception:
            logger.exception("Multi-body STEP mesh generation failed")
            raise
        finally:
            gmsh.finalize()

    @staticmethod
    def _auto_identify_faces(
        volumes: list[tuple[int, int]],
    ) -> dict[str, list[int]]:
        """Identify named face groups by geometric analysis.

        For each volume, classifies boundary faces by Y-coordinate
        extrema and surface type (flat vs cylindrical).

        Parameters
        ----------
        volumes : list of (dim, tag) tuples
            Volume entities from Gmsh model.

        Returns
        -------
        dict mapping face set names to lists of Gmsh face tags.
        """
        face_sets: dict[str, list[int]] = {
            "top_faces": [],
            "bottom_faces": [],
            "cylindrical_faces": [],
            "flat_faces": [],
        }

        for vol_dim, vol_tag in volumes:
            # Get boundary faces for this volume
            boundary = gmsh.model.getBoundary(
                [(vol_dim, vol_tag)], oriented=False, recursive=False
            )
            face_tags = [abs(f[1]) for f in boundary if f[0] == 2]

            if not face_tags:
                continue

            # Get bounding box of the volume to identify extrema
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(
                vol_dim, vol_tag
            )
            vol_height = ymax - ymin
            tol = vol_height * 0.01 if vol_height > 0 else 1e-6

            for ftag in face_tags:
                try:
                    fxmin, fymin, fzmin, fxmax, fymax, fzmax = (
                        gmsh.model.getBoundingBox(2, ftag)
                    )
                except Exception:
                    continue

                face_y_span = fymax - fymin
                face_is_flat_y = face_y_span < tol

                if face_is_flat_y:
                    # This face lies in a plane of constant Y
                    face_y_mid = (fymin + fymax) / 2.0
                    if abs(face_y_mid - ymax) < tol:
                        face_sets["top_faces"].append(ftag)
                        face_sets["flat_faces"].append(ftag)
                    elif abs(face_y_mid - ymin) < tol:
                        face_sets["bottom_faces"].append(ftag)
                        face_sets["flat_faces"].append(ftag)
                    else:
                        face_sets["flat_faces"].append(ftag)
                else:
                    # Check if face spans full height -> likely cylindrical
                    if (
                        abs(fymin - ymin) < tol
                        and abs(fymax - ymax) < tol
                    ):
                        face_sets["cylindrical_faces"].append(ftag)
                    else:
                        # Partial height face -- classify by type
                        # Use parametric type to detect cylindrical surfaces
                        try:
                            surf_type = gmsh.model.getType(2, ftag)
                            if surf_type in ("Cylinder", "Cone", "Torus"):
                                face_sets["cylindrical_faces"].append(ftag)
                            else:
                                face_sets["flat_faces"].append(ftag)
                        except Exception:
                            face_sets["flat_faces"].append(ftag)

        return face_sets

    @staticmethod
    def _defeature(tolerance_mm: float = 0.5) -> None:
        """Remove small features below the given tolerance.

        Applies Gmsh OCC heal/defeaturing to simplify the geometry by
        removing small edges and faces (fillets, chamfers, rounds).

        Must be called after ``importShapes`` and before ``synchronize``.

        Parameters
        ----------
        tolerance_mm : float
            Features smaller than this size in mm will be removed.
            Default 0.5 mm.
        """
        tolerance_m = tolerance_mm / 1000.0

        # Heal shapes -- fixes topology and removes degeneracies
        gmsh.model.occ.healShapes(
            dimTags=[],
            tolerance=tolerance_m,
            fixDegenerated=True,
            fixSmallEdges=True,
            fixSmallFaces=True,
            sewFaces=True,
        )

        gmsh.model.occ.synchronize()

        # Remove small edges below tolerance
        edges = gmsh.model.getEntities(dim=1)
        removed_edges = 0
        for dim, tag in edges:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    gmsh.model.getBoundingBox(dim, tag)
                )
                edge_len = (
                    (xmax - xmin) ** 2
                    + (ymax - ymin) ** 2
                    + (zmax - zmin) ** 2
                ) ** 0.5
                if edge_len < tolerance_m:
                    try:
                        gmsh.model.occ.remove([(dim, tag)], recursive=False)
                        removed_edges += 1
                    except Exception:
                        pass
            except Exception:
                pass

        if removed_edges > 0:
            gmsh.model.occ.synchronize()

        # Remove small faces below tolerance
        faces = gmsh.model.getEntities(dim=2)
        removed_faces = 0
        for dim, tag in faces:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    gmsh.model.getBoundingBox(dim, tag)
                )
                diag = (
                    (xmax - xmin) ** 2
                    + (ymax - ymin) ** 2
                    + (zmax - zmin) ** 2
                ) ** 0.5
                if diag < tolerance_m:
                    try:
                        gmsh.model.occ.remove([(dim, tag)], recursive=False)
                        removed_faces += 1
                    except Exception:
                        pass
            except Exception:
                pass

        if removed_faces > 0:
            gmsh.model.occ.synchronize()

        logger.info(
            "Defeaturing (tol=%.2fmm): removed %d small edges, %d small faces",
            tolerance_mm,
            removed_edges,
            removed_faces,
        )

    @staticmethod
    def _detect_interface_faces(
        volumes: list[tuple[int, int]],
    ) -> dict[int, list[int]]:
        """Detect faces shared between adjacent volumes.

        After STEP export/reimport, coincident faces between volumes may
        no longer share the same tag. This method uses geometric proximity
        (bounding box overlap) to find face pairs belonging to different
        volumes that are coincident (i.e., interface faces).

        Parameters
        ----------
        volumes : list of (dim, tag) tuples
            Volume entities from the Gmsh model.

        Returns
        -------
        dict mapping volume tag to list of its interface face tags.
        """
        interface_map: dict[int, list[int]] = {}

        if len(volumes) < 2:
            return interface_map

        # First check for directly shared face tags
        face_to_volumes: dict[int, list[int]] = {}
        vol_faces: dict[int, list[int]] = {}

        for vol_dim, vol_tag in volumes:
            boundary = gmsh.model.getBoundary(
                [(vol_dim, vol_tag)], oriented=False, recursive=False
            )
            ftags = []
            for face_dim, face_tag in boundary:
                face_tag_abs = abs(face_tag)
                if face_dim == 2:
                    face_to_volumes.setdefault(face_tag_abs, []).append(vol_tag)
                    ftags.append(face_tag_abs)
            vol_faces[vol_tag] = ftags

        # Directly shared faces
        for face_tag, vol_tags in face_to_volumes.items():
            if len(vol_tags) >= 2:
                for vt in vol_tags:
                    interface_map.setdefault(vt, []).append(face_tag)

        if interface_map:
            return interface_map

        # Geometric proximity approach: find coincident faces between
        # different volumes by comparing bounding boxes
        face_bboxes: dict[int, tuple[int, tuple[float, ...]]] = {}
        for vol_tag, ftags in vol_faces.items():
            for ftag in ftags:
                try:
                    bbox = gmsh.model.getBoundingBox(2, ftag)
                    face_bboxes[ftag] = (vol_tag, bbox)
                except Exception:
                    pass

        ftag_list = list(face_bboxes.keys())
        for i in range(len(ftag_list)):
            for j in range(i + 1, len(ftag_list)):
                ft_i = ftag_list[i]
                ft_j = ftag_list[j]
                vol_i = face_bboxes[ft_i][0]
                vol_j = face_bboxes[ft_j][0]
                if vol_i == vol_j:
                    continue
                bb_i = face_bboxes[ft_i][1]
                bb_j = face_bboxes[ft_j][1]
                # Check if bounding boxes are coincident (within tolerance)
                tol = 1e-8
                if all(abs(a - b) < tol for a, b in zip(bb_i, bb_j)):
                    interface_map.setdefault(vol_i, []).append(ft_i)
                    interface_map.setdefault(vol_j, []).append(ft_j)

        return interface_map

    @staticmethod
    def _build_tag_map(node_tags: np.ndarray) -> np.ndarray:
        """Build a Gmsh tag -> 0-based index mapping array.

        Parameters
        ----------
        node_tags : np.ndarray
            Gmsh 1-based node tags.

        Returns
        -------
        np.ndarray
            Mapping array where ``result[gmsh_tag] = 0-based index``.
        """
        max_tag = int(node_tags.max())
        tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
        for new_idx, tag in enumerate(node_tags):
            tag_to_idx[tag] = new_idx
        return tag_to_idx

    @staticmethod
    def _get_face_node_indices(
        face_tags: list[int],
        tag_to_idx: np.ndarray,
    ) -> np.ndarray:
        """Get 0-based node indices for a set of Gmsh face tags.

        Parameters
        ----------
        face_tags : list[int]
            Gmsh surface entity tags.
        tag_to_idx : np.ndarray
            Mapping from Gmsh node tag to 0-based index.

        Returns
        -------
        np.ndarray
            Sorted unique 0-based node indices.
        """
        node_indices = set()
        for ftag in face_tags:
            try:
                ntags, _, _ = gmsh.model.mesh.getNodes(
                    dim=2, tag=ftag, includeBoundary=True
                )
                for nt in ntags:
                    idx = tag_to_idx[int(nt)]
                    if idx >= 0:
                        node_indices.add(int(idx))
            except Exception:
                pass
        return np.array(sorted(node_indices), dtype=np.int64)

    @staticmethod
    def _validate_step_inputs(step_path: str, order: int) -> None:
        """Validate inputs for STEP import methods.

        Raises
        ------
        FileNotFoundError
            If the STEP file does not exist.
        ValueError
            If order is invalid or file extension is wrong.
        """
        if not os.path.isfile(step_path):
            raise FileNotFoundError(
                f"STEP file not found: {step_path!r}"
            )

        ext = os.path.splitext(step_path)[1].lower()
        if ext not in (".step", ".stp"):
            raise ValueError(
                f"Expected a STEP file (.step or .stp), got {ext!r}"
            )

        if order not in (1, 2):
            raise ValueError(
                f"Element order must be 1 (TET4) or 2 (TET10), got {order}."
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        horn_type: str,
        dimensions: dict[str, float],
        order: int,
    ) -> None:
        """Validate input parameters before meshing."""
        if horn_type not in self.HORN_TYPES:
            raise ValueError(
                f"Unsupported horn_type {horn_type!r}. "
                f"Must be one of {self.HORN_TYPES}."
            )

        if order not in (1, 2):
            raise ValueError(
                f"Element order must be 1 (TET4) or 2 (TET10), got {order}."
            )

        if horn_type == "cylindrical":
            required = {"diameter_mm", "length_mm"}
        else:
            required = {"width_mm", "depth_mm", "length_mm"}

        missing = required - set(dimensions.keys())
        if missing:
            raise ValueError(
                f"Missing dimension keys for {horn_type!r} horn: {missing}"
            )

        for key in required:
            val = dimensions[key]
            if not isinstance(val, (int, float)) or val <= 0:
                raise ValueError(
                    f"Dimension {key!r} must be a positive number, got {val!r}."
                )
