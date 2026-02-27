"""Gmsh-based mesh generator for ultrasonic horn geometries.

Generates TET4/TET10 tetrahedral meshes from parametric horn descriptions
using the Gmsh Python API with the OpenCASCADE (OCC) geometry kernel.

All geometry is built in meters (input dimensions in mm are converted).
The Y-axis is the longitudinal/axial direction.
"""
from __future__ import annotations

import logging
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

    def mesh_parametric_horn(
        self,
        horn_type: str,
        dimensions: dict[str, float],
        mesh_size: float = 2.0,
        order: int = 2,
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
        order : int
            Element order: 1 for TET4, 2 for TET10 (default 2).

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

        # Convert mm -> m
        mesh_size_m = mesh_size / 1000.0

        gmsh.initialize()
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

            # Set mesh size on all points
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

            mesh_stats = {
                "num_nodes": nodes_coords.shape[0],
                "num_elements": elements.shape[0],
                "num_surface_tris": surface_tris_remapped.shape[0],
                "element_type": element_type,
                "order": order,
                "mesh_size_mm": mesh_size,
            }

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
