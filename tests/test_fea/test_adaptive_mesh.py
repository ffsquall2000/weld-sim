"""Tests for adaptive mesh refinement strategy in GmshMesher."""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh


# Common geometry for all tests: a flat (box) horn that is easy to reason about.
_BOX_DIMS = {"width_mm": 30.0, "depth_mm": 20.0, "length_mm": 80.0}


class TestAdaptiveMeshBox:
    """Adaptive meshing tests on a flat (box) horn."""

    def setup_method(self):
        self.mesher = GmshMesher()

    # ------------------------------------------------------------------
    # Core behaviour
    # ------------------------------------------------------------------

    def test_adaptive_produces_valid_mesh(self):
        """Adaptive density should produce a valid TET10 mesh."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET10"
        assert mesh.elements.shape[1] == 10
        assert mesh.nodes.shape[0] > 0
        assert mesh.elements.shape[0] > 0

    def test_adaptive_fewer_nodes_than_fine(self):
        """Adaptive mesh should have significantly fewer nodes than uniform fine.

        The adaptive strategy uses coarse (8 mm) away from the weld face
        and fine (3 mm) only at the bottom, so the total node count
        should be substantially less than a uniform fine (3 mm) mesh
        that uses fine elements everywhere.
        """
        mesh_fine = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="fine",
        )
        mesh_adaptive = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        # Adaptive should have fewer nodes than uniform fine
        assert mesh_adaptive.nodes.shape[0] < mesh_fine.nodes.shape[0], (
            f"Adaptive ({mesh_adaptive.nodes.shape[0]} nodes) should have "
            f"fewer nodes than fine ({mesh_fine.nodes.shape[0]} nodes)"
        )

    def test_adaptive_mesh_stats_records_density(self):
        """mesh_stats should record mesh_density='adaptive'."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        assert mesh.mesh_stats.get("mesh_density") == "adaptive"

    def test_adaptive_has_node_sets(self):
        """Adaptive mesh should still detect top_face and bottom_face."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0

    def test_bottom_face_finer_than_top(self):
        """Elements near the bottom face should be smaller than those near the top.

        We compare the median edge length of elements whose centroid is
        in the bottom 10% of the Y-range against those in the top 10%.
        """
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        coords = mesh.nodes  # (N, 3) in meters
        # Use corner nodes only for edge-length computation (first 4 of TET10)
        corner_conn = mesh.elements[:, :4]

        # Element centroids (from corner nodes)
        centroids = coords[corner_conn].mean(axis=1)  # (E, 3)
        y_vals = centroids[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        y_range = y_max - y_min

        # Bottom 10% and top 10% element masks
        bottom_mask = y_vals < (y_min + 0.10 * y_range)
        top_mask = y_vals > (y_max - 0.10 * y_range)

        def median_edge_length(elem_indices):
            """Compute median edge length for a subset of elements."""
            edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
            lengths = []
            for idx in elem_indices:
                nodes_idx = corner_conn[idx]
                for a, b in edges:
                    diff = coords[nodes_idx[a]] - coords[nodes_idx[b]]
                    lengths.append(np.linalg.norm(diff))
            return np.median(lengths) if lengths else 0.0

        bottom_indices = np.where(bottom_mask)[0]
        top_indices = np.where(top_mask)[0]

        # Both regions must contain elements
        assert len(bottom_indices) > 0, "No elements in bottom region"
        assert len(top_indices) > 0, "No elements in top region"

        bottom_median = median_edge_length(bottom_indices)
        top_median = median_edge_length(top_indices)

        # Bottom face should have smaller elements
        assert bottom_median < top_median, (
            f"Bottom median edge ({bottom_median:.6f} m) should be smaller "
            f"than top median edge ({top_median:.6f} m)"
        )

    def test_surface_tris_valid(self):
        """Surface triangulation should be present and well-formed."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density="adaptive",
        )
        assert mesh.surface_tris.shape[1] == 3
        assert mesh.surface_tris.shape[0] > 10
        # All surface tri indices must be within valid range
        assert mesh.surface_tris.max() < mesh.nodes.shape[0]
        assert mesh.surface_tris.min() >= 0


class TestAdaptiveMeshCylinder:
    """Adaptive meshing tests on a cylindrical horn."""

    def setup_method(self):
        self.mesher = GmshMesher()

    def test_adaptive_cylinder_valid(self):
        """Adaptive mesh on a cylinder should produce a valid mesh."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            order=2,
            mesh_density="adaptive",
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET10"
        assert mesh.nodes.shape[0] > 0


class TestExistingDensitiesUnchanged:
    """Ensure that coarse/medium/fine density options still work."""

    def setup_method(self):
        self.mesher = GmshMesher()

    @pytest.mark.parametrize("density", ["coarse", "medium", "fine"])
    def test_named_density_works(self, density):
        """Named density '{density}' should produce a valid mesh."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            order=2,
            mesh_density=density,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET10"
        assert mesh.nodes.shape[0] > 0

    def test_none_density_uses_mesh_size(self):
        """When mesh_density is None, mesh_size parameter should be used."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions=_BOX_DIMS,
            mesh_size=5.0,
            order=2,
            mesh_density=None,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.mesh_stats["mesh_size_mm"] == 5.0
        assert "mesh_density" not in mesh.mesh_stats
