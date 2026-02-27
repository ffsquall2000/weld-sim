"""Parametric booster geometry generator for ultrasonic welding stacks.

A booster (velocity/amplitude transformer) sits between the transducer and
horn in the ultrasonic stack.  It amplifies or attenuates the displacement
amplitude delivered by the transducer.

Four profile types are supported:
- **uniform** (cylindrical): constant diameter, gain = 1.0
- **stepped**: two cylinders joined at a step, gain = (D_in/D_out)^2
- **exponential**: smooth exponential taper, gain = D_in/D_out
- **catenoidal**: smooth catenoidal taper (optimal for minimum stress),
  gain = D_in/D_out

All geometry is built in meters (inputs in mm are converted).
The Y-axis is the longitudinal direction, matching the existing mesher
convention.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

import gmsh
import numpy as np

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import (
    GmshMesher,
    _GMSH_TO_BATHE_TET10,
)

logger = logging.getLogger(__name__)

# Number of points used to discretise smooth profile curves
_N_PROFILE_POINTS = 50


class BoosterGenerator:
    """Parametric booster geometry generator for ultrasonic welding stacks.

    Generates 3D tetrahedral meshes (TET4/TET10) of axisymmetric booster
    geometries using the Gmsh OCC kernel.  The resulting :class:`FEAMesh`
    is directly compatible with the existing FEA pipeline.
    """

    PROFILES = ("uniform", "stepped", "exponential", "catenoidal")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def half_wavelength_length(
        self,
        material_name: str,
        frequency_hz: float = 20000.0,
    ) -> float:
        """Compute half-wavelength resonant length in **mm**.

        Parameters
        ----------
        material_name : str
            Material name recognised by :func:`get_material`.
        frequency_hz : float
            Operating frequency in Hz (default 20 000).

        Returns
        -------
        float
            Half-wavelength length in mm.

        Raises
        ------
        ValueError
            If the material is unknown.
        """
        mat = get_material(material_name)
        if mat is None:
            raise ValueError(
                f"Unknown material {material_name!r}. "
                "Use list_materials() to see available materials."
            )
        E = mat["E_pa"]
        rho = mat["rho_kg_m3"]
        c = math.sqrt(E / rho)  # longitudinal bar wave speed [m/s]
        length_m = c / (2.0 * frequency_hz)
        return length_m * 1000.0  # convert to mm

    def theoretical_gain(
        self,
        profile: str,
        d_input_mm: float,
        d_output_mm: float,
    ) -> float:
        """Compute theoretical amplitude gain for the given profile.

        Parameters
        ----------
        profile : str
            One of ``"uniform"``, ``"stepped"``, ``"exponential"``,
            ``"catenoidal"``.
        d_input_mm : float
            Input (transducer-side) diameter in mm.
        d_output_mm : float
            Output (horn-side) diameter in mm.

        Returns
        -------
        float
            Theoretical gain ratio (>1 means amplification).

        Raises
        ------
        ValueError
            If profile is unsupported or diameters are invalid.
        """
        if profile not in self.PROFILES:
            raise ValueError(
                f"Unsupported profile {profile!r}. "
                f"Must be one of {self.PROFILES}."
            )
        if d_input_mm <= 0:
            raise ValueError(
                f"d_input_mm must be positive, got {d_input_mm}."
            )
        if d_output_mm <= 0:
            raise ValueError(
                f"d_output_mm must be positive, got {d_output_mm}."
            )

        if profile == "uniform":
            return 1.0
        elif profile == "stepped":
            return (d_input_mm / d_output_mm) ** 2
        else:
            # exponential and catenoidal
            return d_input_mm / d_output_mm

    def generate_mesh(
        self,
        profile: str,
        d_input_mm: float,
        d_output_mm: float,
        length_mm: Optional[float] = None,
        material_name: str = "Titanium Ti-6Al-4V",
        frequency_hz: float = 20000.0,
        fillet_radius_mm: float = 2.0,
        mesh_size: float = 2.0,
        order: int = 2,
    ) -> FEAMesh:
        """Generate a meshed booster with the specified profile.

        Parameters
        ----------
        profile : str
            Profile type: ``"uniform"``, ``"stepped"``, ``"exponential"``,
            or ``"catenoidal"``.
        d_input_mm : float
            Input diameter in mm (transducer side, at y=0).
        d_output_mm : float
            Output diameter in mm (horn side, at y=L).
        length_mm : float or None
            Total booster length in mm.  ``None`` auto-computes the
            half-wavelength length for the given material and frequency.
        material_name : str
            Material for half-wavelength computation (default Ti-6Al-4V).
        frequency_hz : float
            Operating frequency in Hz (default 20 000).
        fillet_radius_mm : float
            Fillet radius at the step transition for stepped profiles
            (default 2.0 mm).
        mesh_size : float
            Characteristic element size in mm (default 2.0).
        order : int
            Element order: 1 for TET4, 2 for TET10 (default 2).

        Returns
        -------
        FEAMesh
            Mesh with nodes in meters, element connectivity, node sets
            (``top_face``, ``bottom_face``), surface triangulation, and
            mesh statistics.

        Raises
        ------
        ValueError
            If inputs are invalid.
        RuntimeError
            If mesh generation fails.
        """
        # --- Validation ---
        self._validate_inputs(profile, d_input_mm, d_output_mm, length_mm, order)

        # --- Auto half-wavelength length ---
        if length_mm is None:
            length_mm = self.half_wavelength_length(material_name, frequency_hz)
            logger.info(
                "Auto half-wavelength length: %.1f mm (%s @ %.0f Hz)",
                length_mm,
                material_name,
                frequency_hz,
            )

        # Convert mm -> m
        mesh_size_m = mesh_size / 1000.0

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("booster")

            # Build OCC geometry (all dimensions in meters internally)
            if profile == "uniform":
                self._build_uniform(d_input_mm, length_mm)
            elif profile == "stepped":
                self._build_stepped(
                    d_input_mm, d_output_mm, length_mm, fillet_radius_mm
                )
            elif profile == "exponential":
                self._build_exponential(d_input_mm, d_output_mm, length_mm)
            elif profile == "catenoidal":
                self._build_catenoidal(d_input_mm, d_output_mm, length_mm)

            gmsh.model.occ.synchronize()

            # Set mesh size on all points
            entities = gmsh.model.getEntities(0)
            gmsh.model.mesh.setSize(entities, mesh_size_m)

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Set element order
            if order == 2:
                gmsh.model.mesh.setOrder(2)

            # Extract mesh data (reuse GmshMesher static helpers)
            nodes, coords = GmshMesher._extract_nodes()
            vol_elements, vol_etype = GmshMesher._extract_volume_elements(order)
            surface_tris = GmshMesher._extract_surface_tris()

            # Remap to 0-based indices
            nodes_coords, elements, surface_tris_remapped = (
                GmshMesher._remap_indices(nodes, coords, vol_elements, surface_tris)
            )

            # Reorder TET10 nodes from Gmsh to Bathe convention
            if order == 2:
                elements = elements[:, _GMSH_TO_BATHE_TET10]

            element_type = "TET4" if order == 1 else "TET10"

            # Detect top/bottom face node sets
            node_sets = GmshMesher._detect_face_node_sets(
                nodes_coords, mesh_size_m
            )

            mesh_stats = {
                "num_nodes": nodes_coords.shape[0],
                "num_elements": elements.shape[0],
                "num_surface_tris": surface_tris_remapped.shape[0],
                "element_type": element_type,
                "order": order,
                "mesh_size_mm": mesh_size,
                "profile": profile,
                "d_input_mm": d_input_mm,
                "d_output_mm": d_output_mm,
                "length_mm": length_mm,
            }

            logger.info(
                "Generated %s booster mesh (%s profile): %d nodes, %d elements",
                element_type,
                profile,
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
            logger.exception("Booster mesh generation failed")
            raise
        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------
    # Geometry builders (all dimensions converted from mm to meters)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_uniform(d_input_mm: float, length_mm: float) -> None:
        """Build a uniform (cylindrical) booster.

        Parameters
        ----------
        d_input_mm : float
            Diameter in mm (constant along length).
        length_mm : float
            Total length in mm.
        """
        radius_m = (d_input_mm / 2.0) / 1000.0
        length_m = length_mm / 1000.0
        # Cylinder along +Y axis
        gmsh.model.occ.addCylinder(
            0.0, 0.0, 0.0, 0.0, length_m, 0.0, radius_m
        )

    @staticmethod
    def _build_stepped(
        d_input_mm: float,
        d_output_mm: float,
        length_mm: float,
        fillet_radius_mm: float,
    ) -> None:
        """Build a stepped booster from two cylinders joined at mid-length.

        The input (larger) section is at the bottom (y=0 to y=L/2) and
        the output (smaller) section is at the top (y=L/2 to y=L).

        Parameters
        ----------
        d_input_mm : float
            Input diameter in mm.
        d_output_mm : float
            Output diameter in mm.
        length_mm : float
            Total length in mm.
        fillet_radius_mm : float
            Fillet radius at the step transition.
        """
        r_in_m = (d_input_mm / 2.0) / 1000.0
        r_out_m = (d_output_mm / 2.0) / 1000.0
        half_len_m = (length_mm / 2.0) / 1000.0
        fillet_m = fillet_radius_mm / 1000.0

        # Bottom cylinder (input side)
        cyl1 = gmsh.model.occ.addCylinder(
            0.0, 0.0, 0.0, 0.0, half_len_m, 0.0, r_in_m
        )
        # Top cylinder (output side)
        cyl2 = gmsh.model.occ.addCylinder(
            0.0, half_len_m, 0.0, 0.0, half_len_m, 0.0, r_out_m
        )

        # Fuse both cylinders into a single volume
        result, result_map = gmsh.model.occ.fuse(
            [(3, cyl1)], [(3, cyl2)]
        )

        # Apply fillet at the step transition if requested and the step
        # radius difference can accommodate it.
        if fillet_m > 0 and abs(r_in_m - r_out_m) > fillet_m:
            gmsh.model.occ.synchronize()
            # Find edges near the step transition plane (y ~ half_len_m)
            # by querying all edges and checking bounding boxes.
            edges = gmsh.model.getEntities(1)
            fillet_edges = []
            for dim, tag in edges:
                bbox = gmsh.model.getBoundingBox(dim, tag)
                y_min_edge, y_max_edge = bbox[1], bbox[4]
                # Edge spans the step transition plane
                if (
                    abs(y_min_edge - half_len_m) < 1e-6
                    and abs(y_max_edge - half_len_m) < 1e-6
                ):
                    fillet_edges.append(tag)

            if fillet_edges:
                try:
                    volumes = gmsh.model.getEntities(3)
                    if volumes:
                        gmsh.model.occ.fillet(
                            [volumes[0][1]],
                            fillet_edges,
                            [fillet_m],
                        )
                except Exception:
                    # If filleting fails (e.g., radius too large), proceed
                    # without fillet. The mesh is still valid.
                    logger.warning(
                        "Fillet at step transition failed; proceeding "
                        "without fillet."
                    )

    @staticmethod
    def _build_exponential(
        d_input_mm: float,
        d_output_mm: float,
        length_mm: float,
    ) -> None:
        """Build an exponential-taper booster via revolution.

        Profile:  R(y) = (D_input/2) * exp(-alpha * y)
        where alpha = ln(D_input / D_output) / L

        Parameters
        ----------
        d_input_mm : float
            Input diameter in mm.
        d_output_mm : float
            Output diameter in mm.
        length_mm : float
            Total length in mm.
        """
        r_in_m = (d_input_mm / 2.0) / 1000.0
        r_out_m = (d_output_mm / 2.0) / 1000.0
        length_m = length_mm / 1000.0

        if abs(d_input_mm - d_output_mm) < 1e-9:
            # Degenerate to uniform
            gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, 0.0, length_m, 0.0, r_in_m
            )
            return

        alpha = math.log(d_input_mm / d_output_mm) / length_m

        BoosterGenerator._build_revolved_profile(
            length_m,
            lambda y: r_in_m * math.exp(-alpha * y),
        )

    @staticmethod
    def _build_catenoidal(
        d_input_mm: float,
        d_output_mm: float,
        length_mm: float,
    ) -> None:
        """Build a catenoidal-taper booster via revolution.

        Profile:  R(y) = (D_output/2) * cosh(beta * (1 - y/L))
        where cosh(beta) = D_input / D_output

        Parameters
        ----------
        d_input_mm : float
            Input diameter in mm.
        d_output_mm : float
            Output diameter in mm.
        length_mm : float
            Total length in mm.
        """
        r_in_m = (d_input_mm / 2.0) / 1000.0
        r_out_m = (d_output_mm / 2.0) / 1000.0
        length_m = length_mm / 1000.0

        if abs(d_input_mm - d_output_mm) < 1e-9:
            gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, 0.0, length_m, 0.0, r_in_m
            )
            return

        ratio = d_input_mm / d_output_mm
        beta = math.acosh(ratio)

        BoosterGenerator._build_revolved_profile(
            length_m,
            lambda y: r_out_m * math.cosh(beta * (1.0 - y / length_m)),
        )

    @staticmethod
    def _build_revolved_profile(
        length_m: float,
        radius_func,
    ) -> None:
        """Build a body of revolution from a profile radius function.

        Creates a 2D cross-section in the X-Y plane (X = radius, Y = axial
        position), then revolves it 360 degrees around the Y-axis.

        Parameters
        ----------
        length_m : float
            Axial length in meters.
        radius_func : callable
            Function mapping axial position y (meters) to radius (meters).
        """
        n = _N_PROFILE_POINTS

        # Generate profile points (x=radius, y=axial, z=0)
        ys = [i * length_m / (n - 1) for i in range(n)]
        rs = [radius_func(y) for y in ys]

        # Create Gmsh points for the outer profile curve
        profile_pts = []
        for y, r in zip(ys, rs):
            pt = gmsh.model.occ.addPoint(r, y, 0.0)
            profile_pts.append(pt)

        # Spline through the profile points
        profile_spline = gmsh.model.occ.addSpline(profile_pts)

        # Bottom axis point (0, 0, 0) and top axis point (0, L, 0)
        pt_axis_bottom = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
        pt_axis_top = gmsh.model.occ.addPoint(0.0, length_m, 0.0)

        # Lines to close the profile:
        # top edge: from last profile point to axis top
        line_top = gmsh.model.occ.addLine(profile_pts[-1], pt_axis_top)
        # axis line: from axis top to axis bottom
        line_axis = gmsh.model.occ.addLine(pt_axis_top, pt_axis_bottom)
        # bottom edge: from axis bottom to first profile point
        line_bottom = gmsh.model.occ.addLine(pt_axis_bottom, profile_pts[0])

        # Create wire loop and surface
        wire = gmsh.model.occ.addCurveLoop(
            [profile_spline, line_top, line_axis, line_bottom]
        )
        surface = gmsh.model.occ.addPlaneSurface([wire])

        # Revolve surface 360 degrees around Y-axis
        # revolve(dimTags, x, y, z, ax, ay, az, angle)
        # Axis of revolution: Y-axis through origin
        gmsh.model.occ.revolve(
            [(2, surface)],
            0.0, 0.0, 0.0,  # point on axis
            0.0, 1.0, 0.0,  # axis direction
            2.0 * math.pi,  # full revolution
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        profile: str,
        d_input_mm: float,
        d_output_mm: float,
        length_mm: Optional[float],
        order: int,
    ) -> None:
        """Validate generate_mesh inputs before proceeding."""
        if profile not in self.PROFILES:
            raise ValueError(
                f"Unsupported profile {profile!r}. "
                f"Must be one of {self.PROFILES}."
            )
        if d_input_mm <= 0:
            raise ValueError(
                f"d_input_mm must be positive, got {d_input_mm}."
            )
        if d_output_mm <= 0:
            raise ValueError(
                f"d_output_mm must be positive, got {d_output_mm}."
            )
        if length_mm is not None and length_mm <= 0:
            raise ValueError(
                f"length_mm must be positive (or None for auto), "
                f"got {length_mm}."
            )
        if order not in (1, 2):
            raise ValueError(
                f"Element order must be 1 (TET4) or 2 (TET10), got {order}."
            )
