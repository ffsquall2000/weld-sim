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
