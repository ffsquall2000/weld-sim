"""Tests for the Gmsh-to-dolfinx mesh converter.

All tests use mocks for meshio so they run even when meshio/h5py are not
installed.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# Guard the import -- the module itself guards meshio, so import will work.
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter import (
    MeshConverter,
    _MESHIO_AVAILABLE,
    _iter_cells,
    _extract_cell_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: Path) -> str:
    """Return a temporary output directory."""
    d = tmp_path / "output"
    d.mkdir()
    return str(d)


@pytest.fixture()
def dummy_msh_file(tmp_path: Path) -> str:
    """Create a dummy .msh file and return its path."""
    msh = tmp_path / "test.msh"
    msh.write_bytes(b"$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
    return str(msh)


@pytest.fixture()
def dummy_xdmf_file(tmp_path: Path) -> str:
    """Create a dummy .xdmf file and return its path."""
    xdmf = tmp_path / "test.xdmf"
    xdmf.write_text("<Xdmf/>", encoding="utf-8")
    return str(xdmf)


@pytest.fixture()
def result_output_dir(tmp_path: Path) -> str:
    """Create an output directory with solver result files."""
    d = tmp_path / "solver_output"
    d.mkdir()

    # result.json
    results = {
        "displacement_max": 0.00123,
        "stress_max": 145.6,
        "temperature_max": 85.3,
        "eigenvalues": [1.5e9, 2.3e9, 3.1e9],
        "natural_frequencies_hz": [19500, 20100, 21200],
    }
    (d / "result.json").write_text(json.dumps(results), encoding="utf-8")

    # Field output files
    (d / "displacement.xdmf").write_text("<Xdmf/>", encoding="utf-8")
    (d / "displacement.h5").write_bytes(b"\x89HDF")
    (d / "stress.vtk").write_text("# vtk DataFile", encoding="utf-8")

    return str(d)


# ---------------------------------------------------------------------------
# Mock meshio helpers
# ---------------------------------------------------------------------------


class MockCellBlock:
    """Minimal CellBlock mock compatible with meshio >= 5.0."""

    def __init__(self, cell_type: str, data):
        self.type = cell_type
        self.data = data


class MockMesh:
    """Minimal meshio.Mesh mock."""

    def __init__(
        self,
        points=None,
        cells=None,
        cell_data=None,
    ):
        import numpy as _np
        self.points = points if points is not None else _np.zeros((10, 3))
        self.cells = cells or []
        self.cell_data = cell_data or {}


def _make_mock_msh_with_tets():
    """Create a mock meshio Mesh with tet and triangle cells."""
    import numpy as _np
    points = _np.random.rand(20, 3)
    cells = [
        MockCellBlock("tetra", _np.array([[0, 1, 2, 3], [4, 5, 6, 7]])),
        MockCellBlock("triangle", _np.array([[0, 1, 2], [3, 4, 5]])),
    ]
    cell_data = {
        "gmsh:physical": [
            _np.array([1, 1]),   # for tetra
            _np.array([2, 2]),   # for triangle
        ]
    }
    return MockMesh(points=points, cells=cells, cell_data=cell_data)


# ---------------------------------------------------------------------------
# Tests: MeshConverter.gmsh_to_xdmf
# ---------------------------------------------------------------------------


class TestGmshToXdmf:
    """Test Gmsh .msh -> XDMF conversion."""

    def test_file_not_found(self, output_dir: str):
        """Raise FileNotFoundError for missing input."""
        with _mock_meshio_available():
            with pytest.raises(FileNotFoundError, match="not found"):
                MeshConverter.gmsh_to_xdmf("/nonexistent/mesh.msh", output_dir)

    def test_meshio_not_installed(self, dummy_msh_file: str, output_dir: str):
        """Raise ImportError when meshio is not available."""
        with mock.patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter._MESHIO_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="meshio"):
                MeshConverter.gmsh_to_xdmf(dummy_msh_file, output_dir)

    def test_successful_conversion(self, dummy_msh_file: str, output_dir: str):
        """Successful conversion returns mesh and facets paths."""
        mock_msh = _make_mock_msh_with_tets()

        with _mock_meshio_available():
            with mock.patch(
                "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter.meshio"
            ) as mock_meshio_mod:
                mock_meshio_mod.read.return_value = mock_msh
                mock_meshio_mod.Mesh = MockMesh
                mock_meshio_mod.write = mock.Mock()

                result = MeshConverter.gmsh_to_xdmf(dummy_msh_file, output_dir)

        assert "mesh" in result
        assert "facets" in result
        assert result["mesh"].endswith("mesh.xdmf")
        assert result["facets"].endswith("facets.xdmf")

        # meshio.write should have been called twice (volume + facets)
        assert mock_meshio_mod.write.call_count == 2

    def test_no_volume_cells_raises(self, dummy_msh_file: str, output_dir: str):
        """Raise ValueError when no 3D cells are found."""
        import numpy as _np

        # A mesh with only triangles (no volume cells)
        mock_msh = MockMesh(
            points=_np.random.rand(10, 3),
            cells=[MockCellBlock("triangle", _np.array([[0, 1, 2]]))],
        )

        with _mock_meshio_available():
            with mock.patch(
                "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter.meshio"
            ) as mock_meshio_mod:
                mock_meshio_mod.read.return_value = mock_msh
                mock_meshio_mod.Mesh = MockMesh

                with pytest.raises(ValueError, match="No supported 3D cell types"):
                    MeshConverter.gmsh_to_xdmf(dummy_msh_file, output_dir)

    def test_creates_output_dir(self, dummy_msh_file: str, tmp_path: Path):
        """Output directory is created if it does not exist."""
        new_dir = str(tmp_path / "new" / "output")
        mock_msh = _make_mock_msh_with_tets()

        with _mock_meshio_available():
            with mock.patch(
                "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter.meshio"
            ) as mock_meshio_mod:
                mock_meshio_mod.read.return_value = mock_msh
                mock_meshio_mod.Mesh = MockMesh
                mock_meshio_mod.write = mock.Mock()

                result = MeshConverter.gmsh_to_xdmf(dummy_msh_file, new_dir)

        assert os.path.isdir(new_dir)

    def test_no_facets_writes_empty(self, dummy_msh_file: str, output_dir: str):
        """When no facet cells exist, an empty facets file is still created."""
        import numpy as _np

        # Mesh with only tetra cells, no triangles
        mock_msh = MockMesh(
            points=_np.random.rand(10, 3),
            cells=[MockCellBlock("tetra", _np.array([[0, 1, 2, 3]]))],
        )

        with _mock_meshio_available():
            with mock.patch(
                "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter.meshio"
            ) as mock_meshio_mod:
                mock_meshio_mod.read.return_value = mock_msh
                mock_meshio_mod.Mesh = MockMesh
                mock_meshio_mod.write = mock.Mock()

                result = MeshConverter.gmsh_to_xdmf(dummy_msh_file, output_dir)

        assert "facets" in result
        # meshio.write should still be called twice (volume + empty facets)
        assert mock_meshio_mod.write.call_count == 2


# ---------------------------------------------------------------------------
# Tests: MeshConverter.prepare_dolfinx_input
# ---------------------------------------------------------------------------


class TestPrepareDolfinxInput:
    """Test input package preparation for the FEniCSx solver."""

    def test_file_not_found(self):
        """Raise FileNotFoundError for missing mesh."""
        with pytest.raises(FileNotFoundError, match="not found"):
            MeshConverter.prepare_dolfinx_input(
                "/nonexistent/mesh.xdmf", {}, {}
            )

    def test_xdmf_input(self, dummy_xdmf_file: str):
        """Non-.msh input is copied as-is."""
        input_dir = MeshConverter.prepare_dolfinx_input(
            dummy_xdmf_file,
            material_props={"E": 2.1e11, "nu": 0.3, "rho": 7800},
            boundary_conditions={"type": "free-free"},
        )

        try:
            assert os.path.isdir(input_dir)

            # Config file should exist
            config_path = os.path.join(input_dir, "config.json")
            assert os.path.isfile(config_path)

            with open(config_path) as fp:
                config = json.load(fp)

            assert config["material"]["E"] == 2.1e11
            assert config["material"]["nu"] == 0.3
            assert config["boundary_conditions"]["type"] == "free-free"
            assert config["mesh"]["format"] == "xdmf"
            assert config["mesh"]["mesh_file"] == "test.xdmf"

            # The mesh file should be copied
            assert os.path.isfile(os.path.join(input_dir, "test.xdmf"))
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

    def test_msh_input_triggers_conversion(self, dummy_msh_file: str):
        """A .msh input triggers gmsh_to_xdmf conversion."""
        with mock.patch.object(
            MeshConverter,
            "gmsh_to_xdmf",
            return_value={"mesh": "mesh.xdmf", "facets": "facets.xdmf"},
        ) as mock_convert:
            input_dir = MeshConverter.prepare_dolfinx_input(
                dummy_msh_file,
                material_props={"E": 70e9},
                boundary_conditions={"type": "clamped"},
            )

        try:
            mock_convert.assert_called_once()
            config_path = os.path.join(input_dir, "config.json")
            with open(config_path) as fp:
                config = json.load(fp)
            assert config["mesh"]["format"] == "xdmf"
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)

    def test_config_json_structure(self, dummy_xdmf_file: str):
        """config.json has the expected structure."""
        mat = {"E": 2.1e11, "nu": 0.3, "rho": 7800}
        bc = {"type": "clamped", "nodes": "bottom_face"}

        input_dir = MeshConverter.prepare_dolfinx_input(
            dummy_xdmf_file,
            material_props=mat,
            boundary_conditions=bc,
        )

        try:
            with open(os.path.join(input_dir, "config.json")) as fp:
                config = json.load(fp)

            assert "mesh" in config
            assert "material" in config
            assert "boundary_conditions" in config
            assert config["material"] == mat
            assert config["boundary_conditions"] == bc
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: MeshConverter.parse_dolfinx_output
# ---------------------------------------------------------------------------


class TestParseDolfinxOutput:
    """Test parsing of FEniCSx solver output."""

    def test_dir_not_found(self):
        """Raise FileNotFoundError for missing output directory."""
        with pytest.raises(FileNotFoundError, match="not found"):
            MeshConverter.parse_dolfinx_output("/nonexistent/output")

    def test_parses_result_json(self, result_output_dir: str):
        """result.json is parsed and top-level keys promoted."""
        result = MeshConverter.parse_dolfinx_output(result_output_dir)

        assert result["status"] == "parsed"
        assert result["displacement_max"] == 0.00123
        assert result["stress_max"] == 145.6
        assert result["temperature_max"] == 85.3
        assert result["eigenvalues"] == [1.5e9, 2.3e9, 3.1e9]
        assert result["natural_frequencies_hz"] == [19500, 20100, 21200]

    def test_collects_field_files(self, result_output_dir: str):
        """Field output files (.xdmf, .vtk, .h5) are collected."""
        result = MeshConverter.parse_dolfinx_output(result_output_dir)

        field_files = result["field_files"]
        # Should find displacement.xdmf, displacement.h5, stress.vtk
        assert len(field_files) == 3

        # Check fields dict
        assert "displacement" in result["fields"]
        assert "stress" in result["fields"]
        assert result["fields"]["displacement"]["format"] == "xdmf"
        assert result["fields"]["stress"]["format"] == "vtk"

    def test_empty_output_dir(self, tmp_path: Path):
        """An empty output directory returns minimal results."""
        empty_dir = str(tmp_path / "empty")
        os.makedirs(empty_dir)

        result = MeshConverter.parse_dolfinx_output(empty_dir)

        assert result["status"] == "parsed"
        assert result["field_files"] == []
        assert result["summary"] == {}

    def test_corrupt_result_json(self, tmp_path: Path):
        """Corrupt result.json is handled gracefully."""
        d = str(tmp_path / "corrupt")
        os.makedirs(d)
        with open(os.path.join(d, "result.json"), "w") as fp:
            fp.write("{invalid json!!!")

        result = MeshConverter.parse_dolfinx_output(d)

        assert result["status"] == "parsed"
        assert "parse_error" in result

    def test_result_with_only_fields(self, tmp_path: Path):
        """Output with field files but no result.json works."""
        d = str(tmp_path / "fields_only")
        os.makedirs(d)
        Path(os.path.join(d, "output.vtu")).write_text("<VTK/>")

        result = MeshConverter.parse_dolfinx_output(d)

        assert result["status"] == "parsed"
        assert len(result["field_files"]) == 1
        assert "output" in result["fields"]


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestIterCells:
    """Test the _iter_cells helper for meshio compatibility."""

    def test_list_of_cellblocks(self):
        """Modern meshio format: list of CellBlock objects."""
        import numpy as _np
        blocks = [
            MockCellBlock("tetra", _np.array([[0, 1, 2, 3]])),
            MockCellBlock("triangle", _np.array([[0, 1, 2]])),
        ]
        msh = MockMesh(cells=blocks)

        pairs = list(_iter_cells(msh))
        assert len(pairs) == 2
        assert pairs[0][0] == "tetra"
        assert pairs[1][0] == "triangle"

    def test_dict_format(self):
        """Legacy meshio format: dict of {type: data}."""
        import numpy as _np
        cells = {
            "tetra": _np.array([[0, 1, 2, 3]]),
            "triangle": _np.array([[0, 1, 2]]),
        }
        msh = MockMesh(cells=cells)

        pairs = list(_iter_cells(msh))
        assert len(pairs) == 2
        types = {p[0] for p in pairs}
        assert "tetra" in types
        assert "triangle" in types


class TestExtractCellData:
    """Test the _extract_cell_data helper."""

    def test_extracts_matching_types(self):
        """Only cell data for matching types is extracted."""
        import numpy as _np

        cells = [
            MockCellBlock("tetra", _np.array([[0, 1, 2, 3]])),
            MockCellBlock("triangle", _np.array([[0, 1, 2]])),
        ]
        cell_data = {
            "gmsh:physical": [_np.array([1]), _np.array([2])],
        }
        msh = MockMesh(cells=cells, cell_data=cell_data)

        result = _extract_cell_data(msh, ("tetra",))
        assert "gmsh:physical" in result
        assert len(result["gmsh:physical"]) == 1
        assert result["gmsh:physical"][0].tolist() == [1]

    def test_empty_cell_data(self):
        """Empty cell_data returns empty dict."""
        import numpy as _np
        msh = MockMesh(
            cells=[MockCellBlock("tetra", _np.array([[0, 1, 2, 3]]))],
            cell_data={},
        )
        result = _extract_cell_data(msh, ("tetra",))
        assert result == {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_meshio_available():
    """Context manager that pretends meshio is available."""
    return mock.patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesh_converter._MESHIO_AVAILABLE",
        True,
    )
