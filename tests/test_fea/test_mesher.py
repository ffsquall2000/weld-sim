"""Tests for Gmsh-based mesh generation."""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh


class TestGmshMesher:
    def setup_method(self):
        self.mesher = GmshMesher()

    def test_mesh_cylinder_tet4(self):
        """Generate a TET4 mesh of a simple cylinder."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=1,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET4"
        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 4
        assert mesh.nodes.shape[0] > 50
        assert mesh.mesh_stats["num_nodes"] == mesh.nodes.shape[0]

    def test_mesh_cylinder_tet10(self):
        """Generate a TET10 mesh (quadratic elements)."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.element_type == "TET10"
        assert mesh.elements.shape[1] == 10

    def test_mesh_box_horn(self):
        """Generate mesh for a rectangular block horn."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions={"width_mm": 30.0, "depth_mm": 20.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.nodes.shape[0] > 100

    def test_node_sets_identified(self):
        """Top and bottom face node sets should be auto-detected."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=2,
        )
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0

    def test_bounding_box_correct(self):
        """Mesh bounding box should match input dimensions."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=2,
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length = bbox_max[1] - bbox_min[1]  # y-axis = longitudinal
        assert abs(length - 0.080) < 0.001  # 80mm = 0.08m, tolerance 1mm

    def test_no_inverted_elements_tet4(self):
        """All TET4 elements should have positive volume."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=1,
        )
        for elem in mesh.elements:
            v0, v1, v2, v3 = mesh.nodes[elem]
            vol = np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
            assert vol > 0, "Inverted element detected"

    def test_surface_tris_generated(self):
        """Surface triangulation should be present for visualization."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.surface_tris.shape[1] == 3
        assert mesh.surface_tris.shape[0] > 10

    def test_coordinates_in_meters(self):
        """All node coordinates should be in meters, not mm."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        # Max extent should be ~0.08m (80mm), not 80
        assert mesh.nodes.max() < 1.0  # less than 1 meter
        assert mesh.nodes.max() > 0.01  # more than 1cm

    # ------------------------------------------------------------------
    # Knurl-aware mesh refinement tests
    # ------------------------------------------------------------------

    def test_knurl_refinement_produces_more_elements(self):
        """Knurl refinement should produce more elements than uniform mesh."""
        dims = {"diameter_mm": 25.0, "length_mm": 80.0}
        mesh_uniform = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions=dims,
            mesh_size=5.0,
            order=2,
        )
        knurl = {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3}
        mesh_knurl = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions=dims,
            mesh_size=5.0,
            order=2,
            knurl_info=knurl,
        )
        assert mesh_knurl.elements.shape[0] > mesh_uniform.elements.shape[0], (
            "Knurl refinement should produce a finer mesh near the knurl face"
        )

    def test_knurl_refinement_sets_mesh_stats(self):
        """Knurl refinement should set knurl_refinement and knurl_info in mesh_stats."""
        knurl = {"type": "cross_hatch", "pitch_mm": 0.8, "depth_mm": 0.2}
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
            knurl_info=knurl,
        )
        assert mesh.mesh_stats.get("knurl_refinement") is True
        assert mesh.mesh_stats.get("knurl_info") == knurl

    def test_knurl_none_type_is_ignored(self):
        """Knurl with type 'none' should not trigger refinement."""
        knurl = {"type": "none", "pitch_mm": 1.0, "depth_mm": 0.3}
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
            knurl_info=knurl,
        )
        assert "knurl_refinement" not in mesh.mesh_stats

    def test_knurl_none_info_is_ignored(self):
        """Passing knurl_info=None should leave mesh unchanged."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
            knurl_info=None,
        )
        assert "knurl_refinement" not in mesh.mesh_stats

    def test_knurl_refinement_valid_mesh(self):
        """Knurl-refined mesh should still have valid node sets and surface tris."""
        knurl = {"type": "diamond", "pitch_mm": 1.2, "depth_mm": 0.4}
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions={"width_mm": 30.0, "depth_mm": 20.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
            knurl_info=knurl,
        )
        assert isinstance(mesh, FEAMesh)
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert mesh.surface_tris.shape[0] > 0
        assert mesh.surface_tris.shape[1] == 3


class TestKnurlMeshHelpers:
    """Tests for knurl mesh helper methods (no Gmsh needed)."""

    def test_has_knurl_none(self):
        assert GmshMesher._has_knurl(None) is False

    def test_has_knurl_none_type(self):
        assert GmshMesher._has_knurl({"type": "none"}) is False

    def test_has_knurl_empty_type(self):
        assert GmshMesher._has_knurl({"type": ""}) is False

    def test_has_knurl_linear(self):
        assert GmshMesher._has_knurl({"type": "linear"}) is True

    def test_has_knurl_cross_hatch(self):
        assert GmshMesher._has_knurl({"type": "cross_hatch"}) is True

    def test_knurl_mesh_sizes_default(self):
        fine, coarse = GmshMesher._knurl_mesh_sizes(None)
        assert fine == pytest.approx(0.0004, rel=0.01)  # 0.4mm
        assert coarse == pytest.approx(0.006, rel=0.01)  # 6mm

    def test_knurl_mesh_sizes_fine_pitch(self):
        """Fine pitch (0.6mm) should give smaller fine size."""
        fine, coarse = GmshMesher._knurl_mesh_sizes({"pitch_mm": 0.6})
        assert 0.0003 <= fine <= 0.0005  # 0.3-0.5mm range
        assert 0.005 <= coarse <= 0.008  # 5-8mm range

    def test_knurl_mesh_sizes_coarse_pitch(self):
        """Coarse pitch (2.0mm) should give larger fine size but still clamped."""
        fine, coarse = GmshMesher._knurl_mesh_sizes({"pitch_mm": 2.0})
        assert 0.0003 <= fine <= 0.0005  # still clamped to 0.3-0.5mm
        assert 0.005 <= coarse <= 0.008  # still clamped to 5-8mm
