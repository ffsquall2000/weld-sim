"""Mesh format converter for the Gmsh-to-dolfinx pipeline.

Converts ``.msh`` (Gmsh) meshes to XDMF format that FEniCSx/dolfinx can
read natively, and packages up complete solver input (mesh + material
properties + boundary conditions) for the ``FEniCSxRunner``.

**meshio** is an optional dependency (pip-installable).  All imports are
guarded so that this module can be imported even when meshio or h5py are
not available -- methods will raise ``ImportError`` with a helpful message.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guarded imports
# ---------------------------------------------------------------------------

_MESHIO_AVAILABLE = False
try:
    import meshio  # type: ignore[import-untyped]
    _MESHIO_AVAILABLE = True
except ImportError:
    meshio = None  # type: ignore[assignment]

_NUMPY_AVAILABLE = False
try:
    import numpy as np  # noqa: F401
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]


def _require_meshio() -> None:
    """Raise ``ImportError`` if meshio is not installed."""
    if not _MESHIO_AVAILABLE:
        raise ImportError(
            "meshio is required for mesh format conversion. "
            "Install it with: pip install meshio[all]"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MeshConverter:
    """Convert meshes between formats for the FEniCSx pipeline.

    All methods are ``@staticmethod`` so the class can be used without
    instantiation.
    """

    @staticmethod
    def gmsh_to_xdmf(msh_path: str, output_dir: str) -> dict[str, str]:
        """Convert a Gmsh ``.msh`` file to XDMF format for dolfinx.

        The conversion extracts:
        * **Volume (cell) mesh** -- tetrahedra or hexahedra.
        * **Facet (surface) mesh** -- triangles or quads tagged by Gmsh
          physical groups.

        Parameters
        ----------
        msh_path:
            Path to the input ``.msh`` file.
        output_dir:
            Directory where ``mesh.xdmf``, ``mesh.h5``, ``facets.xdmf``
            and ``facets.h5`` will be written.

        Returns
        -------
        dict[str, str]
            Mapping with keys ``"mesh"`` and ``"facets"`` pointing to the
            output XDMF file paths.

        Raises
        ------
        ImportError
            If meshio is not installed.
        FileNotFoundError
            If *msh_path* does not exist.
        ValueError
            If the mesh contains no supported cell types.
        """
        _require_meshio()

        msh_path = os.path.abspath(msh_path)
        if not os.path.isfile(msh_path):
            raise FileNotFoundError(f"Mesh file not found: {msh_path}")

        os.makedirs(output_dir, exist_ok=True)

        # Read the Gmsh mesh
        msh = meshio.read(msh_path)

        # ----- Volume mesh (3D cells) -----
        volume_types = ("tetra", "tetra10", "hexahedron", "hexahedron20")
        volume_cells = [
            (ctype, data)
            for ctype, data in _iter_cells(msh)
            if ctype in volume_types
        ]

        if not volume_cells:
            raise ValueError(
                f"No supported 3D cell types found in {msh_path}. "
                f"Expected one of {volume_types}."
            )

        # Build a meshio Mesh with only the volume cells
        volume_mesh = meshio.Mesh(
            points=msh.points,
            cells=volume_cells,
        )
        # Copy cell data if available (e.g. physical group tags)
        volume_cell_data = _extract_cell_data(msh, volume_types)
        if volume_cell_data:
            volume_mesh.cell_data = volume_cell_data

        mesh_xdmf = os.path.join(output_dir, "mesh.xdmf")
        meshio.write(mesh_xdmf, volume_mesh)
        logger.info("Wrote volume mesh: %s", mesh_xdmf)

        # ----- Facet mesh (2D cells) -----
        facet_types = ("triangle", "triangle6", "quad", "quad8")
        facet_cells = [
            (ctype, data)
            for ctype, data in _iter_cells(msh)
            if ctype in facet_types
        ]

        facets_xdmf = os.path.join(output_dir, "facets.xdmf")
        if facet_cells:
            facet_mesh = meshio.Mesh(
                points=msh.points,
                cells=facet_cells,
            )
            facet_cell_data = _extract_cell_data(msh, facet_types)
            if facet_cell_data:
                facet_mesh.cell_data = facet_cell_data
            meshio.write(facets_xdmf, facet_mesh)
            logger.info("Wrote facet mesh: %s", facets_xdmf)
        else:
            # Create an empty facets file as a placeholder
            facet_mesh = meshio.Mesh(
                points=msh.points,
                cells=[],
            )
            meshio.write(facets_xdmf, facet_mesh)
            logger.warning(
                "No facet cells found in %s; wrote empty facets.xdmf",
                msh_path,
            )

        return {
            "mesh": mesh_xdmf,
            "facets": facets_xdmf,
        }

    @staticmethod
    def prepare_dolfinx_input(
        mesh_path: str,
        material_props: dict[str, Any],
        boundary_conditions: dict[str, Any],
    ) -> str:
        """Prepare a complete input package for the FEniCSx solver.

        Creates a temporary directory containing:
        * The mesh file (or converted XDMF if ``.msh`` is provided).
        * A ``config.json`` with material properties and boundary
          conditions.

        Parameters
        ----------
        mesh_path:
            Path to the mesh file (``.msh`` or ``.xdmf``).
        material_props:
            Material properties dict (Young's modulus, density, Poisson's
            ratio, etc.).
        boundary_conditions:
            Boundary condition specification.

        Returns
        -------
        str
            Absolute path to the temporary directory containing all
            solver input files.

        Raises
        ------
        FileNotFoundError
            If *mesh_path* does not exist.
        """
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        # Create a temporary directory for the solver input
        input_dir = tempfile.mkdtemp(prefix="dolfinx-input-")

        mesh_ext = os.path.splitext(mesh_path)[1].lower()

        if mesh_ext == ".msh":
            # Convert .msh to XDMF
            xdmf_dir = os.path.join(input_dir, "xdmf")
            os.makedirs(xdmf_dir, exist_ok=True)
            xdmf_paths = MeshConverter.gmsh_to_xdmf(mesh_path, xdmf_dir)
            mesh_info = {
                "format": "xdmf",
                "mesh_file": os.path.basename(xdmf_paths["mesh"]),
                "facets_file": os.path.basename(xdmf_paths["facets"]),
                "mesh_dir": "xdmf",
            }
        else:
            # Copy the mesh file as-is (XDMF, VTK, etc.)
            import shutil
            dest = os.path.join(input_dir, os.path.basename(mesh_path))
            shutil.copy2(mesh_path, dest)
            mesh_info = {
                "format": mesh_ext.lstrip("."),
                "mesh_file": os.path.basename(mesh_path),
            }
            # Also copy the companion .h5 file if it exists
            h5_path = os.path.splitext(mesh_path)[0] + ".h5"
            if os.path.isfile(h5_path):
                shutil.copy2(h5_path, os.path.join(input_dir, os.path.basename(h5_path)))

        # Build the config dict
        config = {
            "mesh": mesh_info,
            "material": material_props,
            "boundary_conditions": boundary_conditions,
        }

        config_path = os.path.join(input_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as fp:
            json.dump(config, fp, indent=2, default=str)

        logger.info("Prepared dolfinx input in %s", input_dir)
        return input_dir

    @staticmethod
    def parse_dolfinx_output(output_dir: str) -> dict[str, Any]:
        """Parse FEniCSx solver output files.

        Looks for structured results in the output directory:
        * ``result.json`` -- main results file with displacement, stress,
          temperature fields and summary statistics.
        * ``*.xdmf`` / ``*.vtk`` -- field output files.

        Parameters
        ----------
        output_dir:
            Directory containing solver output files.

        Returns
        -------
        dict
            Parsed results with keys such as:
            ``displacement_max``, ``stress_max``, ``temperature_max``,
            ``eigenvalues``, ``mode_shapes``, ``field_files``, etc.

        Raises
        ------
        FileNotFoundError
            If *output_dir* does not exist.
        """
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(
                f"Output directory not found: {output_dir}"
            )

        results: dict[str, Any] = {
            "status": "parsed",
            "fields": {},
            "summary": {},
            "field_files": [],
        }

        # ----- Parse result.json if present -----
        result_json = os.path.join(output_dir, "result.json")
        if os.path.isfile(result_json):
            try:
                with open(result_json, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                results["summary"] = data
                # Promote common top-level keys
                for key in (
                    "displacement_max",
                    "stress_max",
                    "temperature_max",
                    "eigenvalues",
                    "natural_frequencies_hz",
                    "mode_shapes",
                ):
                    if key in data:
                        results[key] = data[key]
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to parse result.json: %s", exc)
                results["parse_error"] = str(exc)

        # ----- Collect field output files -----
        field_extensions = {".xdmf", ".vtk", ".vtu", ".pvd", ".h5"}
        for fname in sorted(os.listdir(output_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in field_extensions:
                fpath = os.path.join(output_dir, fname)
                results["field_files"].append(fpath)
                # Store by stem name for easy access
                stem = os.path.splitext(fname)[0]
                results["fields"][stem] = {
                    "path": fpath,
                    "format": ext.lstrip("."),
                }

        logger.info(
            "Parsed dolfinx output: %d summary keys, %d field files",
            len(results.get("summary", {})),
            len(results.get("field_files", [])),
        )

        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_cells(msh):
    """Yield ``(cell_type_name, cell_data_array)`` from a meshio Mesh.

    Handles both the old ``msh.cells`` dict format and the newer list-of-
    ``CellBlock`` format.
    """
    if isinstance(msh.cells, dict):
        # Old meshio (<5.0) format: {type_name: ndarray}
        for ctype, data in msh.cells.items():
            yield ctype, data
    else:
        # New meshio (>=5.0) format: list of CellBlock
        for block in msh.cells:
            yield block.type, block.data


def _extract_cell_data(
    msh,
    target_types: tuple[str, ...],
) -> dict[str, list]:
    """Extract cell_data arrays for the given cell types.

    Returns a dict suitable for ``meshio.Mesh.cell_data``.
    """
    if not msh.cell_data:
        return {}

    result: dict[str, list] = {}

    for key, arrays in msh.cell_data.items():
        filtered = []
        for i, (ctype, _) in enumerate(_iter_cells(msh)):
            if i < len(arrays) and ctype in target_types:
                filtered.append(arrays[i])
        if filtered:
            result[key] = filtered

    return result
