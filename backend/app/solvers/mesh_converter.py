"""Mesh conversion utilities using gmsh and meshio.

Provides a pipeline for converting CAD geometry (STEP files) into FEA-ready
meshes in multiple formats:

    STEP  -->  gmsh (.msh)  -->  meshio  -->  XDMF / Elmer / Abaqus .inp

Both ``gmsh`` and ``meshio`` are optional dependencies.  When they are
unavailable the converter raises an :class:`ImportError` with an
installation hint.  Basic mesh-info queries (element counts, bounding box)
work without either dependency if a ``.msh`` file is already present.
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MeshConfig:
    """Parameters that control mesh generation."""

    element_size: float = 2.0       # target element size [mm]
    min_element_size: float = 0.5   # minimum element size [mm]
    max_element_size: float = 5.0   # maximum element size [mm]
    element_order: int = 1          # 1 = linear, 2 = quadratic
    mesh_algorithm: int = 6         # gmsh algorithm (6 = Frontal-Delaunay)
    refinement_regions: list[dict] = field(default_factory=list)
    # Each entry: {"center": [x,y,z], "radius": float, "size": float}


# ---------------------------------------------------------------------------
# Lazy dependency helpers
# ---------------------------------------------------------------------------

def _require_gmsh():
    """Import and return the ``gmsh`` module, or raise with install hint."""
    try:
        import gmsh  # type: ignore[import-untyped]
        return gmsh
    except ImportError as exc:
        raise ImportError(
            "gmsh is required for STEP-to-mesh conversion.  "
            "Install it with:  pip install gmsh"
        ) from exc


def _require_meshio():
    """Import and return the ``meshio`` module, or raise with install hint."""
    try:
        import meshio  # type: ignore[import-untyped]
        return meshio
    except ImportError as exc:
        raise ImportError(
            "meshio is required for mesh format conversion.  "
            "Install it with:  pip install meshio"
        ) from exc


# ---------------------------------------------------------------------------
# MeshConverter
# ---------------------------------------------------------------------------

class MeshConverter:
    """Convert STEP / STL geometry to FEA mesh formats.

    All public methods accept plain strings (or :class:`Path` objects) so
    that the converter can be used from both sync and async contexts.
    """

    # ------------------------------------------------------------------
    # STEP --> .msh  (gmsh)
    # ------------------------------------------------------------------

    def step_to_mesh(
        self,
        step_path: str,
        output_path: str,
        config: Optional[MeshConfig] = None,
    ) -> dict[str, Any]:
        """Convert a STEP file to a gmsh ``.msh`` mesh.

        Parameters
        ----------
        step_path:
            Path to the input STEP file.
        output_path:
            Desired path for the output ``.msh`` file.
        config:
            Optional :class:`MeshConfig`; defaults are used if *None*.

        Returns
        -------
        dict
            Mesh statistics: ``n_nodes``, ``n_elements``, ``element_types``,
            ``bounding_box``, ``file_size_bytes``.
        """
        gmsh = _require_gmsh()
        cfg = config or MeshConfig()

        step_file = Path(step_path)
        out_file = Path(output_path)

        if not step_file.exists():
            raise FileNotFoundError(f"STEP file not found: {step_file}")

        out_file.parent.mkdir(parents=True, exist_ok=True)

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("horn_mesh")

            # Import STEP geometry
            gmsh.model.occ.importShapes(str(step_file))
            gmsh.model.occ.synchronize()

            # Global mesh size
            gmsh.option.setNumber("Mesh.CharacteristicLengthMin", cfg.min_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", cfg.max_element_size)
            gmsh.option.setNumber("Mesh.CharacteristicLengthFactor", 1.0)
            gmsh.option.setNumber("Mesh.Algorithm3D", cfg.mesh_algorithm)
            gmsh.option.setNumber("Mesh.ElementOrder", cfg.element_order)

            # Apply refinement regions (sphere-based size fields)
            if cfg.refinement_regions:
                field_ids: list[int] = []
                for i, region in enumerate(cfg.refinement_regions):
                    center = region.get("center", [0.0, 0.0, 0.0])
                    radius = region.get("radius", 5.0)
                    size = region.get("size", cfg.min_element_size)

                    # Ball field
                    fid = gmsh.model.mesh.field.add("Ball")
                    gmsh.model.mesh.field.setNumber(fid, "VIn", size)
                    gmsh.model.mesh.field.setNumber(fid, "VOut", cfg.element_size)
                    gmsh.model.mesh.field.setNumber(fid, "XCenter", center[0])
                    gmsh.model.mesh.field.setNumber(fid, "YCenter", center[1])
                    gmsh.model.mesh.field.setNumber(fid, "ZCenter", center[2])
                    gmsh.model.mesh.field.setNumber(fid, "Radius", radius)
                    field_ids.append(fid)

                if field_ids:
                    # Combine with Min field
                    min_fid = gmsh.model.mesh.field.add("Min")
                    gmsh.model.mesh.field.setNumbers(min_fid, "FieldsList", field_ids)
                    gmsh.model.mesh.field.setAsBackgroundMesh(min_fid)

            # Generate 3D mesh
            gmsh.model.mesh.generate(3)

            # Optimize
            gmsh.model.mesh.optimize("Netgen")

            # Write output
            gmsh.write(str(out_file))

            # Gather statistics
            node_tags, _, _ = gmsh.model.mesh.getNodes()
            elem_types, _, _ = gmsh.model.mesh.getElements()

            # Bounding box
            bb = gmsh.model.getBoundingBox(-1, -1)  # xmin, ymin, zmin, xmax, ymax, zmax

            stats = {
                "n_nodes": len(node_tags),
                "n_elements": sum(
                    len(gmsh.model.mesh.getElementsByType(et)[0])
                    for et in elem_types
                ),
                "element_types": [
                    gmsh.model.mesh.getElementProperties(et)[0]
                    for et in elem_types
                ],
                "bounding_box": {
                    "min": [bb[0], bb[1], bb[2]],
                    "max": [bb[3], bb[4], bb[5]],
                },
                "file_size_bytes": out_file.stat().st_size,
                "element_order": cfg.element_order,
                "mesh_algorithm": cfg.mesh_algorithm,
            }

            logger.info(
                "Mesh generated: %d nodes, %d elements -> %s",
                stats["n_nodes"],
                stats["n_elements"],
                out_file,
            )
            return stats

        finally:
            gmsh.finalize()

    # ------------------------------------------------------------------
    # .msh --> XDMF  (FEniCS / dolfinx)
    # ------------------------------------------------------------------

    def mesh_to_fenics(self, mesh_path: str, output_path: str) -> str:
        """Convert a gmsh ``.msh`` mesh to XDMF format for FEniCS.

        Parameters
        ----------
        mesh_path:
            Path to the input ``.msh`` file.
        output_path:
            Desired path for the output ``.xdmf`` file.

        Returns
        -------
        str
            Absolute path of the written XDMF file.
        """
        meshio = _require_meshio()
        msh_file = Path(mesh_path)
        out_file = Path(output_path)

        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_file}")

        out_file.parent.mkdir(parents=True, exist_ok=True)

        mesh = meshio.read(str(msh_file))

        # Extract volume cells (tetra or hexahedra) for a 3D domain
        volume_types = {"tetra", "tetra10", "hexahedron", "hexahedron27"}
        volume_cells = []
        for cell_block in mesh.cells:
            if cell_block.type in volume_types:
                volume_cells.append(cell_block)

        if not volume_cells:
            # Fall back to all cells
            logger.warning(
                "No volume cells found in %s; writing all cells to XDMF.",
                msh_file,
            )
            volume_cells = list(mesh.cells)

        # Build a new mesh with only volume cells
        out_mesh = meshio.Mesh(
            points=mesh.points,
            cells=volume_cells,
        )

        meshio.xdmf.write(str(out_file), out_mesh)
        logger.info("XDMF written: %s", out_file)
        return str(out_file.resolve())

    # ------------------------------------------------------------------
    # .msh --> Elmer  (ElmerGrid)
    # ------------------------------------------------------------------

    def mesh_to_elmer(self, mesh_path: str, output_dir: str) -> str:
        """Convert a gmsh ``.msh`` mesh to Elmer format via ElmerGrid.

        Parameters
        ----------
        mesh_path:
            Path to the input ``.msh`` file (gmsh format 4).
        output_dir:
            Directory where Elmer mesh files will be written.

        Returns
        -------
        str
            Absolute path of the output directory.

        Raises
        ------
        FileNotFoundError
            If ``ElmerGrid`` is not found on ``PATH``.
        """
        msh_file = Path(mesh_path)
        out_dir = Path(output_dir)

        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_file}")

        out_dir.mkdir(parents=True, exist_ok=True)

        # ElmerGrid: convert gmsh format (14) to Elmer format (2)
        cmd = [
            "ElmerGrid", "14", "2",
            str(msh_file),
            "-out", str(out_dir),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "ElmerGrid executable not found.  Install Elmer FEM or add "
                "ElmerGrid to PATH."
            ) from None

        if result.returncode != 0:
            raise RuntimeError(
                f"ElmerGrid failed (exit {result.returncode}):\n{result.stderr}"
            )

        logger.info("Elmer mesh written: %s", out_dir)
        return str(out_dir.resolve())

    # ------------------------------------------------------------------
    # .msh --> Abaqus .inp  (CalculiX)
    # ------------------------------------------------------------------

    def mesh_to_calculix(self, mesh_path: str, output_path: str) -> str:
        """Convert a gmsh ``.msh`` mesh to Abaqus ``.inp`` format for CalculiX.

        Parameters
        ----------
        mesh_path:
            Path to the input ``.msh`` file.
        output_path:
            Desired path for the output ``.inp`` file.

        Returns
        -------
        str
            Absolute path of the written ``.inp`` file.
        """
        meshio = _require_meshio()
        msh_file = Path(mesh_path)
        out_file = Path(output_path)

        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_file}")

        out_file.parent.mkdir(parents=True, exist_ok=True)

        mesh = meshio.read(str(msh_file))

        # meshio's Abaqus writer produces .inp directly
        meshio.abaqus.write(str(out_file), mesh)

        logger.info("CalculiX .inp written: %s", out_file)
        return str(out_file.resolve())

    # ------------------------------------------------------------------
    # Mesh info / statistics
    # ------------------------------------------------------------------

    def get_mesh_info(self, mesh_path: str) -> dict[str, Any]:
        """Get mesh statistics from an existing mesh file.

        Tries ``meshio`` first; falls back to basic ``.msh`` binary parsing
        for node / element counts.

        Returns
        -------
        dict
            Keys: ``n_nodes``, ``n_elements``, ``element_types``,
            ``bounding_box``, ``file_size_bytes``.
        """
        msh_file = Path(mesh_path)
        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_file}")

        file_size = msh_file.stat().st_size

        # Try meshio first
        try:
            meshio = _require_meshio()
            mesh = meshio.read(str(msh_file))

            n_nodes = len(mesh.points)
            n_elements = sum(len(cb.data) for cb in mesh.cells)
            element_types = list({cb.type for cb in mesh.cells})
            bb_min = mesh.points.min(axis=0).tolist()
            bb_max = mesh.points.max(axis=0).tolist()

            return {
                "n_nodes": n_nodes,
                "n_elements": n_elements,
                "element_types": element_types,
                "bounding_box": {"min": bb_min, "max": bb_max},
                "file_size_bytes": file_size,
            }
        except ImportError:
            pass

        # Fallback: parse .msh header for basic counts
        return self._parse_msh_basic(msh_file, file_size)

    # ------------------------------------------------------------------
    # Fallback .msh parser (no meshio required)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_msh_basic(msh_file: Path, file_size: int) -> dict[str, Any]:
        """Minimal parser for gmsh ``.msh`` v4 ASCII format.

        Extracts node and element counts from section headers.
        """
        n_nodes = 0
        n_elements = 0

        try:
            with open(msh_file, "r", errors="replace") as fh:
                section = None
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith("$"):
                        section = stripped
                        continue

                    if section == "$Nodes":
                        # First data line: numEntityBlocks numNodes ...
                        parts = stripped.split()
                        if len(parts) >= 2:
                            try:
                                n_nodes = int(parts[1])
                            except ValueError:
                                n_nodes = int(parts[0])
                        section = None  # only read first line
                        continue

                    if section == "$Elements":
                        parts = stripped.split()
                        if len(parts) >= 2:
                            try:
                                n_elements = int(parts[1])
                            except ValueError:
                                n_elements = int(parts[0])
                        section = None
                        continue
        except Exception as exc:
            logger.warning("Failed to parse .msh header: %s", exc)

        return {
            "n_nodes": n_nodes,
            "n_elements": n_elements,
            "element_types": ["unknown"],
            "bounding_box": {"min": [0, 0, 0], "max": [0, 0, 0]},
            "file_size_bytes": file_size,
        }
