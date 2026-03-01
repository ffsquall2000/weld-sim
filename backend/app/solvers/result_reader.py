"""Read solver result files into the uniform :class:`FieldData` format.

Supports multiple output formats produced by different solver backends:

  - ``.vtu``  -- VTK unstructured grid (ParaView, FEniCS, Elmer, CalculiX)
  - ``.frd``  -- CalculiX native result format
  - ``.xdmf`` -- FEniCS / dolfinx HDF5-backed result format

The reader also provides :meth:`field_to_vtk_json` which serialises a single
field from :class:`FieldData` into a JSON-safe dict suitable for VTK.js
rendering in the browser frontend.

``meshio`` is the preferred parsing backend.  When it is not installed a
minimal VTU XML parser is used as a fallback for ``.vtu`` files.
"""

from __future__ import annotations

import logging
import math
import struct
import xml.etree.ElementTree as ET
from base64 import b64decode
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .base import FieldData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VTK cell-type constants  (subset used by ultrasonic welding meshes)
# ---------------------------------------------------------------------------
_VTK_CELL_NAMES: dict[int, str] = {
    3: "line2",
    5: "tri3",
    9: "quad4",
    10: "tet4",
    12: "hex8",
    13: "wedge6",
    22: "tri6",
    24: "tet10",
    25: "hex20",
}

_CELL_NAME_TO_VTK: dict[str, int] = {v: k for k, v in _VTK_CELL_NAMES.items()}

# Meshio type names differ slightly from our canonical names
_MESHIO_TO_CANONICAL: dict[str, str] = {
    "triangle": "tri3",
    "triangle6": "tri6",
    "quad": "quad4",
    "tetra": "tet4",
    "tetra10": "tet10",
    "hexahedron": "hex8",
    "hexahedron20": "hex20",
    "hexahedron27": "hex27",
    "wedge": "wedge6",
    "line": "line2",
}


# ---------------------------------------------------------------------------
# Lazy meshio import
# ---------------------------------------------------------------------------

def _try_meshio():
    """Return the meshio module or None."""
    try:
        import meshio  # type: ignore[import-untyped]
        return meshio
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# ResultReader
# ---------------------------------------------------------------------------

class ResultReader:
    """Parse solver output files into :class:`FieldData`."""

    # ------------------------------------------------------------------
    # VTU  (.vtu)
    # ------------------------------------------------------------------

    def read_vtu(self, path: str) -> FieldData:
        """Read a VTK Unstructured Grid (``.vtu``) file.

        Tries meshio first; falls back to a minimal XML parser.
        """
        vtu_path = Path(path)
        if not vtu_path.exists():
            raise FileNotFoundError(f"VTU file not found: {vtu_path}")

        meshio = _try_meshio()
        if meshio is not None:
            return self._read_vtu_meshio(meshio, vtu_path)

        logger.info("meshio not available; using fallback VTU XML parser")
        return self._read_vtu_xml(vtu_path)

    @staticmethod
    def _read_vtu_meshio(meshio, vtu_path: Path) -> FieldData:
        """Parse ``.vtu`` using meshio."""
        mesh = meshio.read(str(vtu_path))

        cells: list[np.ndarray] = []
        cell_types: list[str] = []
        for cb in mesh.cells:
            canonical = _MESHIO_TO_CANONICAL.get(cb.type, cb.type)
            cells.append(np.asarray(cb.data, dtype=np.int32))
            cell_types.append(canonical)

        point_data: dict[str, np.ndarray] = {}
        for name, arr in mesh.point_data.items():
            point_data[name] = np.asarray(arr)

        cell_data: dict[str, np.ndarray] = {}
        for name, arrays in mesh.cell_data.items():
            # meshio returns a list of arrays, one per cell block
            cell_data[name] = np.concatenate(
                [np.asarray(a) for a in arrays]
            )

        return FieldData(
            points=np.asarray(mesh.points, dtype=np.float64),
            cells=cells,
            cell_types=cell_types,
            point_data=point_data,
            cell_data=cell_data,
            metadata={"source_format": "vtu", "file": str(vtu_path)},
        )

    @staticmethod
    def _read_vtu_xml(vtu_path: Path) -> FieldData:
        """Minimal VTU XML parser (ASCII and base64-encoded binary).

        Handles the most common VTU layout produced by FEniCS, Elmer, and
        CalculiX post-processors.
        """
        tree = ET.parse(str(vtu_path))
        root = tree.getroot()

        piece = root.find(".//{http://www.vtk.org/XMLFileFormat}Piece")
        if piece is None:
            piece = root.find(".//Piece")
        if piece is None:
            raise ValueError(f"No <Piece> element found in {vtu_path}")

        n_points = int(piece.attrib.get("NumberOfPoints", 0))
        n_cells = int(piece.attrib.get("NumberOfCells", 0))

        # --- Parse points ---
        points_el = piece.find(".//Points/DataArray")
        points = _parse_data_array(points_el, n_points * 3, dtype=np.float64)
        points = points.reshape(-1, 3)

        # --- Parse cells ---
        cells_section = piece.find(".//Cells")
        connectivity_el = None
        offsets_el = None
        types_el = None
        if cells_section is not None:
            for da in cells_section.findall("DataArray"):
                name = da.attrib.get("Name", "")
                if name == "connectivity":
                    connectivity_el = da
                elif name == "offsets":
                    offsets_el = da
                elif name == "types":
                    types_el = da

        # Parse cell types
        vtk_types = np.zeros(n_cells, dtype=np.int32)
        if types_el is not None:
            vtk_types = _parse_data_array(types_el, n_cells, dtype=np.int32)

        # Parse offsets
        offsets = np.zeros(n_cells, dtype=np.int32)
        if offsets_el is not None:
            offsets = _parse_data_array(offsets_el, n_cells, dtype=np.int32)

        # Parse connectivity
        total_conn = int(offsets[-1]) if n_cells > 0 else 0
        connectivity = np.zeros(total_conn, dtype=np.int32)
        if connectivity_el is not None:
            connectivity = _parse_data_array(
                connectivity_el, total_conn, dtype=np.int32
            )

        # Group cells by VTK type
        cells_out: list[np.ndarray] = []
        cell_types_out: list[str] = []
        type_groups: dict[int, list[np.ndarray]] = {}

        prev_offset = 0
        for i in range(n_cells):
            cur_offset = int(offsets[i])
            cell_nodes = connectivity[prev_offset:cur_offset]
            vt = int(vtk_types[i])
            type_groups.setdefault(vt, []).append(cell_nodes)
            prev_offset = cur_offset

        for vt, cell_list in sorted(type_groups.items()):
            name = _VTK_CELL_NAMES.get(vt, f"vtk_{vt}")
            cells_out.append(np.array(cell_list, dtype=np.int32))
            cell_types_out.append(name)

        # --- Parse point data ---
        point_data: dict[str, np.ndarray] = {}
        pd_section = piece.find(".//PointData")
        if pd_section is not None:
            for da in pd_section.findall("DataArray"):
                name = da.attrib.get("Name", "unknown")
                n_comp = int(da.attrib.get("NumberOfComponents", 1))
                total = n_points * n_comp
                arr = _parse_data_array(da, total, dtype=np.float64)
                if n_comp > 1:
                    arr = arr.reshape(n_points, n_comp)
                point_data[name] = arr

        # --- Parse cell data ---
        cell_data: dict[str, np.ndarray] = {}
        cd_section = piece.find(".//CellData")
        if cd_section is not None:
            for da in cd_section.findall("DataArray"):
                name = da.attrib.get("Name", "unknown")
                n_comp = int(da.attrib.get("NumberOfComponents", 1))
                total = n_cells * n_comp
                arr = _parse_data_array(da, total, dtype=np.float64)
                if n_comp > 1:
                    arr = arr.reshape(n_cells, n_comp)
                cell_data[name] = arr

        return FieldData(
            points=points,
            cells=cells_out,
            cell_types=cell_types_out,
            point_data=point_data,
            cell_data=cell_data,
            metadata={"source_format": "vtu_xml", "file": str(vtu_path)},
        )

    # ------------------------------------------------------------------
    # FRD  (.frd)  --  CalculiX native format
    # ------------------------------------------------------------------

    def read_frd(self, path: str) -> FieldData:
        """Read a CalculiX ``.frd`` result file.

        The ``.frd`` format is a block-structured ASCII / binary format.
        This reader handles the ASCII variant, which is the default
        CalculiX output.
        """
        frd_path = Path(path)
        if not frd_path.exists():
            raise FileNotFoundError(f"FRD file not found: {frd_path}")

        # Try meshio first (it has an frd reader)
        meshio = _try_meshio()
        if meshio is not None:
            try:
                return self._read_vtu_meshio(meshio, frd_path)
            except Exception:
                logger.debug("meshio failed to read .frd; using built-in parser")

        return self._read_frd_ascii(frd_path)

    @staticmethod
    def _read_frd_ascii(frd_path: Path) -> FieldData:
        """Minimal ASCII ``.frd`` parser.

        Extracts node coordinates, element connectivity, and the first
        displacement / stress field found.
        """
        nodes: dict[int, list[float]] = {}
        elements: dict[int, list[int]] = {}
        displacement: dict[int, list[float]] = {}
        stress: dict[int, list[float]] = {}

        current_block: Optional[str] = None
        field_name: Optional[str] = None
        n_comp = 0

        with open(frd_path, "r") as fh:
            for line in fh:
                # Block markers
                if line.startswith("    2C"):
                    current_block = "nodes"
                    continue
                elif line.startswith("    3C"):
                    current_block = "elements"
                    continue
                elif line.startswith(" -4"):
                    # Field header: name is in columns 6-11
                    field_name = line[5:11].strip().lower()
                    n_comp_str = line[13:18].strip()
                    n_comp = int(n_comp_str) if n_comp_str else 0
                    current_block = "field"
                    continue
                elif line.startswith(" -3"):
                    current_block = None
                    continue

                if current_block == "nodes" and line.startswith(" -1"):
                    # Node line: -1 <id> <x> <y> <z>
                    try:
                        nid = int(line[3:13])
                        x = float(line[13:25])
                        y = float(line[25:37])
                        z = float(line[37:49])
                        nodes[nid] = [x, y, z]
                    except (ValueError, IndexError):
                        pass

                elif current_block == "elements" and line.startswith(" -1"):
                    # Element line (simplified -- reads first connectivity line)
                    try:
                        eid = int(line[3:13])
                        # Node IDs follow on the same or next lines
                        node_ids_str = line[13:].split()
                        node_ids = [int(n) for n in node_ids_str if n.strip()]
                        elements[eid] = node_ids
                    except (ValueError, IndexError):
                        pass

                elif current_block == "field" and line.startswith(" -1"):
                    # Field data line
                    try:
                        nid = int(line[3:13])
                        values = []
                        pos = 13
                        for _ in range(min(n_comp, 6)):
                            val = float(line[pos:pos + 12])
                            values.append(val)
                            pos += 12
                        if field_name and "disp" in field_name:
                            displacement[nid] = values
                        elif field_name and "stress" in field_name:
                            stress[nid] = values
                    except (ValueError, IndexError):
                        pass

        # Build arrays
        if not nodes:
            return FieldData(
                points=np.zeros((0, 3)),
                cells=[],
                cell_types=[],
                metadata={"source_format": "frd", "file": str(frd_path)},
            )

        # Sort by node id and create mapping
        sorted_nids = sorted(nodes.keys())
        nid_to_idx = {nid: i for i, nid in enumerate(sorted_nids)}

        points = np.array([nodes[nid] for nid in sorted_nids], dtype=np.float64)

        # Cells
        cells_list: list[np.ndarray] = []
        cell_types_list: list[str] = []
        if elements:
            # Group by number of nodes per element
            by_nnodes: dict[int, list[list[int]]] = {}
            for eid in sorted(elements.keys()):
                enodes = elements[eid]
                mapped = [nid_to_idx.get(n, 0) for n in enodes]
                by_nnodes.setdefault(len(enodes), []).append(mapped)

            nnodes_to_type = {4: "tet4", 8: "hex8", 10: "tet10", 20: "hex20", 6: "wedge6", 3: "tri3"}
            for nn, cell_list in sorted(by_nnodes.items()):
                ctype = nnodes_to_type.get(nn, f"cell{nn}")
                cells_list.append(np.array(cell_list, dtype=np.int32))
                cell_types_list.append(ctype)

        # Point data
        point_data: dict[str, np.ndarray] = {}
        if displacement:
            disp_arr = np.zeros((len(sorted_nids), 3), dtype=np.float64)
            for nid, vals in displacement.items():
                idx = nid_to_idx.get(nid)
                if idx is not None:
                    for j, v in enumerate(vals[:3]):
                        disp_arr[idx, j] = v
            point_data["displacement"] = disp_arr

        if stress:
            # Von Mises from 6-component stress tensor
            vm_arr = np.zeros(len(sorted_nids), dtype=np.float64)
            for nid, vals in stress.items():
                idx = nid_to_idx.get(nid)
                if idx is not None and len(vals) >= 6:
                    sx, sy, sz = vals[0], vals[1], vals[2]
                    txy, tyz, txz = vals[3], vals[4], vals[5]
                    vm = math.sqrt(
                        max(
                            0.5 * ((sx - sy)**2 + (sy - sz)**2 + (sz - sx)**2)
                            + 3.0 * (txy**2 + tyz**2 + txz**2),
                            0.0,
                        )
                    )
                    vm_arr[idx] = vm
            point_data["von_mises_stress"] = vm_arr

        return FieldData(
            points=points,
            cells=cells_list,
            cell_types=cell_types_list,
            point_data=point_data,
            metadata={"source_format": "frd", "file": str(frd_path)},
        )

    # ------------------------------------------------------------------
    # XDMF  (.xdmf)  --  FEniCS / dolfinx format
    # ------------------------------------------------------------------

    def read_xdmf(self, path: str) -> FieldData:
        """Read a FEniCS ``.xdmf`` result file.

        Requires ``meshio`` (the XDMF format uses HDF5 data arrays that
        cannot be parsed with a simple fallback).
        """
        xdmf_path = Path(path)
        if not xdmf_path.exists():
            raise FileNotFoundError(f"XDMF file not found: {xdmf_path}")

        meshio = _try_meshio()
        if meshio is None:
            raise ImportError(
                "meshio is required to read XDMF files.  "
                "Install it with:  pip install meshio h5py"
            )

        mesh = meshio.read(str(xdmf_path))

        cells: list[np.ndarray] = []
        cell_types: list[str] = []
        for cb in mesh.cells:
            canonical = _MESHIO_TO_CANONICAL.get(cb.type, cb.type)
            cells.append(np.asarray(cb.data, dtype=np.int32))
            cell_types.append(canonical)

        point_data: dict[str, np.ndarray] = {
            name: np.asarray(arr) for name, arr in mesh.point_data.items()
        }

        cell_data: dict[str, np.ndarray] = {}
        for name, arrays in mesh.cell_data.items():
            cell_data[name] = np.concatenate(
                [np.asarray(a) for a in arrays]
            )

        return FieldData(
            points=np.asarray(mesh.points, dtype=np.float64),
            cells=cells,
            cell_types=cell_types,
            point_data=point_data,
            cell_data=cell_data,
            metadata={"source_format": "xdmf", "file": str(xdmf_path)},
        )

    # ------------------------------------------------------------------
    # FieldData --> VTK.js JSON
    # ------------------------------------------------------------------

    def field_to_vtk_json(
        self,
        field_data: FieldData,
        field_name: str,
    ) -> dict[str, Any]:
        """Convert a single field from :class:`FieldData` to a JSON dict
        suitable for VTK.js rendering in the browser.

        Parameters
        ----------
        field_data:
            The :class:`FieldData` containing mesh geometry and fields.
        field_name:
            Name of the point-data or cell-data field to include.

        Returns
        -------
        dict
            ``{points, cells, cell_types, values, min_value, max_value,
            n_components, field_location, unit}``
        """
        # Flatten points to list
        points = field_data.points.ravel().tolist()

        # Build flat cell array for VTK.js:
        #   [n_verts, v0, v1, ..., n_verts, v0, v1, ...]
        vtk_cells: list[int] = []
        vtk_cell_types: list[int] = []
        for ct, ca in zip(field_data.cell_types, field_data.cells):
            vtk_type = _CELL_NAME_TO_VTK.get(ct, 0)
            for row in ca:
                vtk_cells.append(len(row))
                vtk_cells.extend(int(v) for v in row)
                vtk_cell_types.append(vtk_type)

        # Locate the requested field
        values: list[float] = []
        min_val = 0.0
        max_val = 0.0
        n_components = 1
        field_location = "point"
        unit = ""

        if field_name in field_data.point_data:
            arr = field_data.point_data[field_name]
            field_location = "point"
        elif field_name in field_data.cell_data:
            arr = field_data.cell_data[field_name]
            field_location = "cell"
        else:
            arr = None
            logger.warning(
                "Field %r not found in FieldData (available: %s)",
                field_name,
                list(field_data.point_data.keys()) + list(field_data.cell_data.keys()),
            )

        if arr is not None:
            if arr.ndim == 2:
                # Vector field -- compute magnitude for scalar coloring
                magnitudes = np.linalg.norm(arr, axis=1)
                values = magnitudes.tolist()
                n_components = int(arr.shape[1])
            else:
                values = arr.tolist()
                n_components = 1

            if len(values) > 0:
                min_val = float(min(values))
                max_val = float(max(values))

        # Guess unit from field name
        unit_map = {
            "displacement": "mm",
            "von_mises_stress": "MPa",
            "stress": "MPa",
            "temperature": "C",
            "amplitude": "um",
            "pressure": "Pa",
        }
        for key, u in unit_map.items():
            if key in field_name.lower():
                unit = u
                break

        return {
            "points": points,
            "cells": vtk_cells,
            "cell_types": vtk_cell_types,
            "values": values,
            "min_value": min_val,
            "max_value": max_val,
            "n_components": n_components,
            "field_location": field_location,
            "field_name": field_name,
            "unit": unit,
            "n_points": field_data.n_points,
            "n_cells": field_data.n_cells,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_data_array(
    element: Optional[ET.Element],
    expected_count: int,
    dtype: type = np.float64,
) -> np.ndarray:
    """Parse a VTK ``<DataArray>`` element (ASCII or base64 binary)."""
    if element is None:
        return np.zeros(expected_count, dtype=dtype)

    fmt = element.attrib.get("format", "ascii").lower()
    text = (element.text or "").strip()

    if fmt == "ascii":
        vals = text.split()
        arr = np.array([float(v) for v in vals], dtype=dtype)
    elif fmt == "binary":
        raw = b64decode(text)
        # First 8 bytes are the header (uint64 byte count)
        if len(raw) > 8:
            data = raw[8:]
        else:
            data = raw
        type_name = element.attrib.get("type", "Float64")
        struct_fmt = {"Float64": "d", "Float32": "f", "Int32": "i", "UInt8": "B", "Int64": "q"}
        sf = struct_fmt.get(type_name, "d")
        n_items = len(data) // struct.calcsize(sf)
        arr = np.array(struct.unpack(f"<{n_items}{sf}", data[:n_items * struct.calcsize(sf)]), dtype=dtype)
    else:
        # Appended format -- not supported in this fallback
        logger.warning("Unsupported DataArray format %r; returning zeros", fmt)
        arr = np.zeros(expected_count, dtype=dtype)

    return arr
