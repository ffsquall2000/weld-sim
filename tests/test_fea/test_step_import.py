"""Tests for STEP import, multi-body support, auto face detection, and defeaturing."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

try:
    import gmsh

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False

pytestmark = pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher


# ---------------------------------------------------------------------------
# Helpers -- create STEP files programmatically using Gmsh OCC
# ---------------------------------------------------------------------------


def _create_cylinder_step(path: str, radius_m: float = 0.01, length_m: float = 0.08):
    """Create a STEP file containing a single cylinder along Y-axis."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cyl")
        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, length_m, 0.0, radius_m)
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _create_box_step(
    path: str,
    width_m: float = 0.03,
    depth_m: float = 0.02,
    length_m: float = 0.08,
):
    """Create a STEP file containing a single box (flat horn shape)."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(
            -width_m / 2, 0.0, -depth_m / 2, width_m, length_m, depth_m
        )
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _create_stepped_shape_step(path: str):
    """Create a STEP file with a stepped shape (two cylinders of different radii).

    Bottom cylinder: radius 15mm, height 40mm
    Top cylinder:    radius 10mm, height 40mm
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stepped")
        r1 = 0.015  # 15mm
        r2 = 0.010  # 10mm
        h = 0.040  # 40mm each

        cyl1 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r1)
        cyl2 = gmsh.model.occ.addCylinder(0.0, h, 0.0, 0.0, h, 0.0, r2)
        gmsh.model.occ.fuse([(3, cyl1)], [(3, cyl2)])
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _create_multi_body_step(path: str):
    """Create a multi-body STEP file with two stacked cylinders (not fused).

    Body 1: cylinder from y=0 to y=0.04, radius 0.0125
    Body 2: cylinder from y=0.04 to y=0.08, radius 0.0125
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("multi_body")
        r = 0.0125  # 12.5mm
        h = 0.04  # 40mm

        gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r)
        gmsh.model.occ.addCylinder(0.0, h, 0.0, 0.0, h, 0.0, r)
        # Fragment to create shared interface
        gmsh.model.occ.fragment(
            gmsh.model.occ.getEntities(3), []
        )
        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


def _create_cylinder_with_fillet_step(path: str):
    """Create a STEP file with a cylinder that has a small fillet at the base.

    Uses a box fused with a cylinder and a small fillet added via chamfer.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("fillet_test")
        r = 0.0125  # 12.5mm
        h = 0.06  # 60mm

        # Main cylinder
        cyl = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r)

        # Small cylinder to fuse at bottom (creates edge to fillet)
        disc_r = r + 0.002  # 2mm wider
        disc_h = 0.003  # 3mm tall
        disc = gmsh.model.occ.addCylinder(
            0.0, 0.0, 0.0, 0.0, disc_h, 0.0, disc_r
        )
        gmsh.model.occ.fuse([(3, cyl)], [(3, disc)])
        gmsh.model.occ.synchronize()

        # Fillet the edges at the step junction
        edges = gmsh.model.getEntities(1)
        fillet_radius = 0.0003  # 0.3mm -- small feature
        for dim, tag in edges:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = (
                    gmsh.model.getBoundingBox(dim, tag)
                )
                edge_y_mid = (ymin + ymax) / 2.0
                # Target edges near the step at y = disc_h
                if abs(edge_y_mid - disc_h) < 0.001:
                    try:
                        gmsh.model.occ.fillet([tag], [fillet_radius])
                    except Exception:
                        pass  # Not all edges can be filleted
            except Exception:
                pass

        gmsh.model.occ.synchronize()
        gmsh.write(path)
    finally:
        gmsh.finalize()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMeshFromStep:
    """Tests for GmshMesher.mesh_from_step()."""

    def setup_method(self):
        self.mesher = GmshMesher()
        self._tmpdir = tempfile.mkdtemp()

    def _step_path(self, name: str) -> str:
        return os.path.join(self._tmpdir, name)

    def test_mesh_from_step_cylinder(self):
        """Import a programmatically created cylinder STEP and mesh it."""
        path = self._step_path("cylinder.step")
        _create_cylinder_step(path, radius_m=0.0125, length_m=0.08)

        mesh = self.mesher.mesh_from_step(path, mesh_size=5.0, order=2)

        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET10"
        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 10
        assert mesh.nodes.shape[0] > 50
        assert mesh.elements.shape[0] > 10
        assert mesh.surface_tris.shape[1] == 3
        assert mesh.mesh_stats["source"] == path
        assert mesh.mesh_stats["num_volumes"] >= 1

    def test_mesh_from_step_box(self):
        """Import a box STEP file."""
        path = self._step_path("box.step")
        _create_box_step(path)

        mesh = self.mesher.mesh_from_step(path, mesh_size=5.0, order=2)

        assert isinstance(mesh, FEAMesh)
        assert mesh.nodes.shape[0] > 50

    def test_mesh_from_step_tet4(self):
        """STEP import with TET4 (linear) elements."""
        path = self._step_path("cylinder_tet4.step")
        _create_cylinder_step(path)

        mesh = self.mesher.mesh_from_step(path, mesh_size=5.0, order=1)

        assert mesh.element_type == "TET4"
        assert mesh.elements.shape[1] == 4

    def test_mesh_from_step_bounding_box(self):
        """Bounding box of imported mesh should match original geometry."""
        path = self._step_path("cyl_bbox.step")
        _create_cylinder_step(path, radius_m=0.0125, length_m=0.08)

        mesh = self.mesher.mesh_from_step(path, mesh_size=4.0, order=2)

        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        y_extent = bbox_max[1] - bbox_min[1]
        assert abs(y_extent - 0.08) < 0.002  # 80mm +/- 2mm tolerance

    def test_mesh_from_step_node_sets(self):
        """Top and bottom face node sets should be auto-detected."""
        path = self._step_path("cyl_nodesets.step")
        _create_cylinder_step(path, radius_m=0.0125, length_m=0.08)

        mesh = self.mesher.mesh_from_step(path, mesh_size=4.0, order=2)

        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0

    def test_mesh_from_step_coordinates_in_meters(self):
        """Imported STEP mesh coordinates should be in meters."""
        path = self._step_path("cyl_meters.step")
        _create_cylinder_step(path, radius_m=0.0125, length_m=0.08)

        mesh = self.mesher.mesh_from_step(path, mesh_size=5.0, order=2)

        # Geometry was created in meters, so coords should be < 1m
        assert mesh.nodes.max() < 1.0
        assert mesh.nodes.max() > 0.001


class TestAutoIdentifyFaces:
    """Tests for GmshMesher._auto_identify_faces()."""

    def test_cylinder_faces(self):
        """A simple cylinder should have top, bottom, and cylindrical faces."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("test_cyl_faces")
            gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0125
            )
            gmsh.model.occ.synchronize()

            volumes = gmsh.model.getEntities(dim=3)
            face_sets = GmshMesher._auto_identify_faces(volumes)

            assert "top_faces" in face_sets
            assert "bottom_faces" in face_sets
            assert "cylindrical_faces" in face_sets
            assert len(face_sets["top_faces"]) >= 1
            assert len(face_sets["bottom_faces"]) >= 1
            # Cylinder should have at least one cylindrical face
            assert len(face_sets["cylindrical_faces"]) >= 1
        finally:
            gmsh.finalize()

    def test_box_faces(self):
        """A box should have top and bottom faces identified as flat."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("test_box_faces")
            gmsh.model.occ.addBox(-0.015, 0.0, -0.01, 0.03, 0.08, 0.02)
            gmsh.model.occ.synchronize()

            volumes = gmsh.model.getEntities(dim=3)
            face_sets = GmshMesher._auto_identify_faces(volumes)

            assert len(face_sets["top_faces"]) >= 1
            assert len(face_sets["bottom_faces"]) >= 1
            assert len(face_sets["flat_faces"]) >= 2  # top + bottom at minimum
        finally:
            gmsh.finalize()

    def test_stepped_shape_faces(self):
        """A stepped shape (two fused cylinders) should have top and bottom."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("test_stepped")
            r1 = 0.015
            r2 = 0.010
            h = 0.040
            cyl1 = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r1)
            cyl2 = gmsh.model.occ.addCylinder(0.0, h, 0.0, 0.0, h, 0.0, r2)
            gmsh.model.occ.fuse([(3, cyl1)], [(3, cyl2)])
            gmsh.model.occ.synchronize()

            volumes = gmsh.model.getEntities(dim=3)
            face_sets = GmshMesher._auto_identify_faces(volumes)

            assert len(face_sets["top_faces"]) >= 1
            assert len(face_sets["bottom_faces"]) >= 1
            # Stepped shape has more faces than a simple cylinder
            total_faces = (
                len(face_sets["top_faces"])
                + len(face_sets["bottom_faces"])
                + len(face_sets["cylindrical_faces"])
                + len(face_sets["flat_faces"])
            )
            assert total_faces >= 4
        finally:
            gmsh.finalize()


class TestDefeature:
    """Tests for GmshMesher._defeature()."""

    def test_defeature_heals_shapes(self):
        """Defeaturing should run without error on a simple shape."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("test_defeature")
            gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0125
            )
            # Do NOT synchronize before defeature -- it handles sync internally
            gmsh.model.occ.synchronize()

            # Count faces before
            faces_before = len(gmsh.model.getEntities(dim=2))

            GmshMesher._defeature(tolerance_mm=0.5)

            # Should still have faces (geometry not destroyed)
            faces_after = len(gmsh.model.getEntities(dim=2))
            assert faces_after > 0
            # Simple cylinder has no small features, so face count should be similar
            assert faces_after <= faces_before
        finally:
            gmsh.finalize()

    def test_defeature_on_complex_shape(self):
        """Defeaturing on a shape with a step should preserve the main geometry."""
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("test_defeature_complex")

            # Create a cylinder with a small protrusion
            r = 0.0125
            h = 0.06
            cyl = gmsh.model.occ.addCylinder(0.0, 0.0, 0.0, 0.0, h, 0.0, r)

            # Small disc at bottom
            disc_r = r + 0.002
            disc_h = 0.003
            disc = gmsh.model.occ.addCylinder(
                0.0, 0.0, 0.0, 0.0, disc_h, 0.0, disc_r
            )
            gmsh.model.occ.fuse([(3, cyl)], [(3, disc)])
            gmsh.model.occ.synchronize()

            faces_before = len(gmsh.model.getEntities(dim=2))

            GmshMesher._defeature(tolerance_mm=0.5)

            faces_after = len(gmsh.model.getEntities(dim=2))
            # Geometry should still be valid
            assert faces_after > 0
            volumes = gmsh.model.getEntities(dim=3)
            assert len(volumes) >= 1
        finally:
            gmsh.finalize()


class TestMultiBodyStep:
    """Tests for GmshMesher.mesh_multi_body_step()."""

    def setup_method(self):
        self.mesher = GmshMesher()
        self._tmpdir = tempfile.mkdtemp()

    def _step_path(self, name: str) -> str:
        return os.path.join(self._tmpdir, name)

    def test_multi_body_two_cylinders(self):
        """Two stacked cylinders should produce two separate meshes."""
        path = self._step_path("multi_body.step")
        _create_multi_body_step(path)

        meshes = self.mesher.mesh_multi_body_step(
            path, mesh_size=5.0, order=2
        )

        assert len(meshes) == 2
        for m in meshes:
            assert isinstance(m, FEAMesh)
            assert m.element_type == "TET10"
            assert m.nodes.shape[0] > 10
            assert m.elements.shape[0] > 5
            assert "top_face" in m.node_sets
            assert "bottom_face" in m.node_sets

    def test_multi_body_separate_node_sets(self):
        """Each body should have its own independent node indexing."""
        path = self._step_path("multi_separate.step")
        _create_multi_body_step(path)

        meshes = self.mesher.mesh_multi_body_step(
            path, mesh_size=5.0, order=2
        )

        for m in meshes:
            # Node indices should be 0-based
            assert m.elements.min() >= 0
            assert m.elements.max() < m.nodes.shape[0]

    def test_multi_body_interface_faces(self):
        """Interface faces between adjacent bodies should be detected."""
        path = self._step_path("multi_interface.step")
        _create_multi_body_step(path)

        meshes = self.mesher.mesh_multi_body_step(
            path, mesh_size=5.0, order=2
        )

        # At least one mesh should have interface_faces node set
        has_interface = any("interface_faces" in m.node_sets for m in meshes)
        assert has_interface, "No interface faces detected between bodies"

    def test_multi_body_tet4(self):
        """Multi-body meshing with TET4 elements."""
        path = self._step_path("multi_tet4.step")
        _create_multi_body_step(path)

        meshes = self.mesher.mesh_multi_body_step(
            path, mesh_size=5.0, order=1
        )

        assert len(meshes) == 2
        for m in meshes:
            assert m.element_type == "TET4"
            assert m.elements.shape[1] == 4

    def test_single_body_returns_one_mesh(self):
        """A single-body STEP file should return a list with one mesh."""
        path = self._step_path("single_body.step")
        _create_cylinder_step(path)

        meshes = self.mesher.mesh_multi_body_step(
            path, mesh_size=5.0, order=2
        )

        assert len(meshes) == 1
        assert isinstance(meshes[0], FEAMesh)


class TestInputValidation:
    """Tests for input validation on STEP import methods."""

    def setup_method(self):
        self.mesher = GmshMesher()

    def test_nonexistent_file(self):
        """Should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError, match="STEP file not found"):
            self.mesher.mesh_from_step("/nonexistent/path/file.step")

    def test_nonexistent_file_multi_body(self):
        """mesh_multi_body_step should also raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="STEP file not found"):
            self.mesher.mesh_multi_body_step("/nonexistent/path/file.step")

    def test_invalid_extension(self):
        """Should raise ValueError for non-STEP file extensions."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            f.write(b"not a step file")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Expected a STEP file"):
                self.mesher.mesh_from_step(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_invalid_order(self):
        """Should raise ValueError for unsupported element orders."""
        with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Element order must be"):
                self.mesher.mesh_from_step(tmp_path, order=3)
        finally:
            os.unlink(tmp_path)

    def test_invalid_step_content(self):
        """A file with .step extension but invalid content should raise ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".step", delete=False, mode="w"
        ) as f:
            f.write("this is not valid STEP data")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Failed to import STEP"):
                self.mesher.mesh_from_step(tmp_path)
        finally:
            os.unlink(tmp_path)
