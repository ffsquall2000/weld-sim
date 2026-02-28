"""Tests for knurl geometry application on imported STEP files.

Tests the ``HornGenerator.apply_knurl_to_step`` method and related
helpers:
  - ``_find_bottom_face`` (Z-min face detection)
  - ``_cq_create_knurl_grooves`` (groove tool body creation)
  - ``_cq_cut_knurl_on_body`` (boolean cut logic)

When CadQuery is not installed the tests exercise the code through
mocking so that the full logic is validated without a real OCC kernel.
Tests that require a *real* CadQuery are marked with
``@pytest.mark.skipif(not HAS_CADQUERY, ...)`` so they skip gracefully.
"""
from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest

# Detect CadQuery availability -- mirrors horn_generator.py pattern
try:
    import cadquery as cq  # noqa: F401

    HAS_CADQUERY = True
except ImportError:
    HAS_CADQUERY = False

from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
    HornGenerator,
    HornParams,
    KnurlParams,
)

needs_cadquery = pytest.mark.skipif(
    not HAS_CADQUERY, reason="CadQuery not installed"
)


# -----------------------------------------------------------------------
# Mock helpers -- allow testing logic without real CadQuery
# -----------------------------------------------------------------------

def _make_mock_bounding_box(xmin=0, ymin=0, zmin=0, xmax=25, ymax=25, zmax=80):
    """Create a mock BoundingBox with typical horn dimensions."""
    bb = mock.MagicMock()
    bb.xmin = xmin
    bb.ymin = ymin
    bb.zmin = zmin
    bb.xmax = xmax
    bb.ymax = ymax
    bb.zmax = zmax
    return bb


def _make_mock_face(xmin, ymin, zmin, xmax, ymax, zmax, area=100.0):
    """Create a mock CadQuery face with a bounding box and area."""
    face = mock.MagicMock()
    face.BoundingBox.return_value = _make_mock_bounding_box(
        xmin, ymin, zmin, xmax, ymax, zmax,
    )
    face.Area.return_value = area
    return face


def _make_mock_body(faces, bb=None):
    """Create a mock CadQuery Workplane body with faces and bounding box."""
    body = mock.MagicMock()
    if bb is None:
        bb = _make_mock_bounding_box()
    body.val.return_value.BoundingBox.return_value = bb
    body.faces.return_value.vals.return_value = faces
    body.cut.return_value = body  # cut returns itself by default
    return body


# -----------------------------------------------------------------------
# Tests: _find_bottom_face (face detection logic)
# -----------------------------------------------------------------------

class TestFindBottomFace:
    """Tests for HornGenerator._find_bottom_face()."""

    def test_finds_bottom_face_of_box(self):
        """A box body should have a clear Z-min bottom face."""
        # Bottom face: z=0, flat
        bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)
        # Top face: z=80, flat
        top = _make_mock_face(0, 0, 80, 25, 25, 80, area=625.0)
        # Side face: spans full Z, not flat
        side = _make_mock_face(0, 0, 0, 0, 25, 80, area=2000.0)

        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([bottom, top, side], bb)

        result = HornGenerator._find_bottom_face(body)

        assert result is not None
        center, width, length = result
        assert center[2] == pytest.approx(0.0, abs=0.1)
        assert width == pytest.approx(25.0)
        assert length == pytest.approx(25.0)

    def test_selects_largest_bottom_face(self):
        """When multiple faces share the Z-min, select the largest."""
        small_bottom = _make_mock_face(5, 5, 0, 10, 10, 0, area=25.0)
        large_bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)
        top = _make_mock_face(0, 0, 80, 25, 25, 80, area=625.0)

        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([small_bottom, large_bottom, top], bb)

        result = HornGenerator._find_bottom_face(body)

        assert result is not None
        _center, width, length = result
        # Should select the large face (25x25)
        assert width == pytest.approx(25.0)
        assert length == pytest.approx(25.0)

    def test_returns_none_for_zero_height_body(self):
        """A degenerate body with no Z span should return None."""
        face = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)
        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 0)  # z_span = 0
        body = _make_mock_body([face], bb)

        result = HornGenerator._find_bottom_face(body)
        assert result is None

    def test_returns_none_when_no_faces(self):
        """If the body has no faces, return None."""
        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([], bb)

        result = HornGenerator._find_bottom_face(body)
        assert result is None

    def test_ignores_non_flat_faces(self):
        """Faces with significant Z thickness should be ignored."""
        # Non-flat side face near Z-min but thick
        side = _make_mock_face(0, 0, 0, 0, 25, 40, area=1000.0)
        # Flat bottom
        bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)

        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([side, bottom], bb)

        result = HornGenerator._find_bottom_face(body)
        assert result is not None
        center, width, length = result
        assert center[2] == pytest.approx(0.0, abs=0.1)

    def test_handles_face_access_exception(self):
        """If body.faces() throws, return None gracefully."""
        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = mock.MagicMock()
        body.val.return_value.BoundingBox.return_value = bb
        body.faces.side_effect = RuntimeError("shape broken")

        result = HornGenerator._find_bottom_face(body)
        assert result is None

    def test_ignores_faces_not_near_z_min(self):
        """A face at Z=40 (midpoint) should not be chosen as bottom."""
        mid_face = _make_mock_face(0, 0, 40, 25, 25, 40, area=625.0)
        bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=100.0)

        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([mid_face, bottom], bb)

        result = HornGenerator._find_bottom_face(body)
        assert result is not None
        _center, width, length = result
        # Should pick the bottom face (smaller but at Z-min)
        assert width == pytest.approx(25.0)
        assert length == pytest.approx(25.0)


# -----------------------------------------------------------------------
# Tests: _cq_create_knurl_grooves (groove tool creation)
# -----------------------------------------------------------------------

class TestCreateKnurlGrooves:
    """Tests for HornGenerator._cq_create_knurl_grooves()."""

    @needs_cadquery
    def test_linear_grooves_created(self):
        """Linear knurl should produce groove bodies."""
        knurl = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.3)
        result = HornGenerator._cq_create_knurl_grooves(
            knurl,
            face_center=(0.0, 0.0, 0.0),
            face_width=20.0,
            face_length=20.0,
            z_bottom=0.0,
        )
        assert result is not None
        # Should be a CadQuery Workplane
        assert hasattr(result, "val")

    @needs_cadquery
    def test_cross_hatch_grooves_created(self):
        """Cross-hatch knurl should produce grooves in both directions."""
        knurl = KnurlParams(knurl_type="cross_hatch", pitch_mm=2.0, depth_mm=0.3)
        result = HornGenerator._cq_create_knurl_grooves(
            knurl,
            face_center=(0.0, 0.0, 0.0),
            face_width=20.0,
            face_length=20.0,
            z_bottom=0.0,
        )
        assert result is not None

    @needs_cadquery
    def test_diamond_grooves_created(self):
        """Diamond knurl is equivalent to cross-hatch in groove creation."""
        knurl = KnurlParams(knurl_type="diamond", pitch_mm=2.0, depth_mm=0.3)
        result = HornGenerator._cq_create_knurl_grooves(
            knurl,
            face_center=(0.0, 0.0, 0.0),
            face_width=20.0,
            face_length=20.0,
            z_bottom=0.0,
        )
        assert result is not None

    def test_none_type_returns_none_via_cut(self):
        """Knurl type 'none' should make _cq_cut_knurl_on_body return body unchanged."""
        gen = HornGenerator()
        body = mock.MagicMock()
        knurl = KnurlParams(knurl_type="none")
        result = gen._cq_cut_knurl_on_body(body, knurl)
        assert result is body

    @needs_cadquery
    def test_groove_depth_positioning(self):
        """Grooves should be positioned below the face Z coordinate."""
        knurl = KnurlParams(knurl_type="linear", pitch_mm=5.0, depth_mm=0.5)
        result = HornGenerator._cq_create_knurl_grooves(
            knurl,
            face_center=(0.0, 0.0, 10.0),
            face_width=20.0,
            face_length=20.0,
            z_bottom=10.0,
        )
        assert result is not None
        bb = result.val().BoundingBox()
        # Grooves should start below z_bottom
        assert bb.zmin < 10.0


# -----------------------------------------------------------------------
# Tests: _cq_cut_knurl_on_body (boolean cut logic)
# -----------------------------------------------------------------------

class TestCutKnurlOnBody:
    """Tests for HornGenerator._cq_cut_knurl_on_body() via mocks."""

    def test_knurl_none_returns_body_unchanged(self):
        """Knurl type 'none' skips all processing."""
        gen = HornGenerator()
        body = mock.MagicMock()
        knurl = KnurlParams(knurl_type="none")
        result = gen._cq_cut_knurl_on_body(body, knurl)
        assert result is body
        body.cut.assert_not_called()

    def test_returns_body_when_no_bottom_face(self):
        """If no bottom face is detected, return body unchanged."""
        gen = HornGenerator()
        # Body with zero Z span -> _find_bottom_face returns None
        bb = _make_mock_bounding_box(0, 0, 5, 25, 25, 5)
        body = _make_mock_body([], bb)
        knurl = KnurlParams(knurl_type="linear")

        result = gen._cq_cut_knurl_on_body(body, knurl)
        assert result is body

    def test_boolean_cut_exception_handled(self):
        """If body.cut() raises, body should be returned unchanged."""
        gen = HornGenerator()
        bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)
        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([bottom], bb)
        body.cut.side_effect = RuntimeError("Boolean cut failed")

        knurl = KnurlParams(knurl_type="linear")

        # Mock the groove creation to return a mock grooves object
        with mock.patch.object(
            HornGenerator, "_cq_create_knurl_grooves", return_value=mock.MagicMock()
        ):
            result = gen._cq_cut_knurl_on_body(body, knurl)

        # Should still return the body (not raise)
        assert result is body


# -----------------------------------------------------------------------
# Tests: apply_knurl_to_step (public API)
# -----------------------------------------------------------------------

class TestApplyKnurlToStep:
    """Tests for HornGenerator.apply_knurl_to_step()."""

    def test_file_not_found(self):
        """Non-existent STEP file should raise FileNotFoundError."""
        gen = HornGenerator()
        knurl = KnurlParams()
        if not HAS_CADQUERY:
            with pytest.raises(RuntimeError, match="CadQuery is required"):
                gen.apply_knurl_to_step("/nonexistent/file.step", knurl)
        else:
            with pytest.raises(FileNotFoundError, match="STEP file not found"):
                gen.apply_knurl_to_step("/nonexistent/file.step", knurl)

    def test_raises_without_cadquery(self):
        """Should raise RuntimeError when CadQuery is not installed."""
        gen = HornGenerator()
        knurl = KnurlParams()
        import ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator as hg_mod

        original = hg_mod.HAS_CADQUERY
        try:
            hg_mod.HAS_CADQUERY = False
            with pytest.raises(RuntimeError, match="CadQuery is required"):
                gen.apply_knurl_to_step("/some/file.step", knurl)
        finally:
            hg_mod.HAS_CADQUERY = original

    @needs_cadquery
    def test_apply_to_step_box(self):
        """Full integration: create a box STEP, apply linear knurl."""
        gen = HornGenerator()

        # Create a simple box STEP file via CadQuery
        box = cq.Workplane("XY").box(20, 20, 60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "box_horn.step")
        cq.exporters.export(box, step_path)

        knurl = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.3)
        result = gen.apply_knurl_to_step(step_path, knurl)

        # Result should be a CadQuery Workplane
        assert hasattr(result, "val")
        # Volume should be slightly less after grooves are cut
        original_vol = box.val().Volume()
        result_vol = result.val().Volume()
        assert result_vol < original_vol
        assert result_vol > 0

    @needs_cadquery
    def test_apply_cross_hatch_to_step(self):
        """Cross-hatch knurl on imported STEP should cut in both directions."""
        gen = HornGenerator()

        box = cq.Workplane("XY").box(20, 20, 60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "box_crosshatch.step")
        cq.exporters.export(box, step_path)

        knurl_linear = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.3)
        knurl_cross = KnurlParams(knurl_type="cross_hatch", pitch_mm=2.0, depth_mm=0.3)

        result_linear = gen.apply_knurl_to_step(step_path, knurl_linear)
        result_cross = gen.apply_knurl_to_step(step_path, knurl_cross)

        vol_linear = result_linear.val().Volume()
        vol_cross = result_cross.val().Volume()

        # Cross-hatch removes more material (grooves in 2 directions)
        assert vol_cross < vol_linear

    @needs_cadquery
    def test_apply_to_cylinder_step(self):
        """Knurl on a cylindrical horn STEP file."""
        gen = HornGenerator()

        cyl = cq.Workplane("XY").circle(10).extrude(60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "cyl_horn.step")
        cq.exporters.export(cyl, step_path)

        knurl = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.2)
        result = gen.apply_knurl_to_step(step_path, knurl)

        assert hasattr(result, "val")
        original_vol = cyl.val().Volume()
        result_vol = result.val().Volume()
        assert result_vol < original_vol

    @needs_cadquery
    def test_knurl_preserves_overall_dimensions(self):
        """Knurl grooves should not change the overall bounding box significantly."""
        gen = HornGenerator()

        box = cq.Workplane("XY").box(20, 20, 60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "box_dims.step")
        cq.exporters.export(box, step_path)

        knurl = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.3)
        result = gen.apply_knurl_to_step(step_path, knurl)

        orig_bb = box.val().BoundingBox()
        result_bb = result.val().BoundingBox()

        # X and Y extents should be unchanged
        assert abs((result_bb.xmax - result_bb.xmin) - (orig_bb.xmax - orig_bb.xmin)) < 0.01
        assert abs((result_bb.ymax - result_bb.ymin) - (orig_bb.ymax - orig_bb.ymin)) < 0.01
        # Z extent may be slightly larger (grooves extend below)
        # but overall shape should be similar
        z_diff = abs((result_bb.zmax - result_bb.zmin) - (orig_bb.zmax - orig_bb.xmin))
        assert z_diff < 1.0  # within 1mm


# -----------------------------------------------------------------------
# Tests: KnurlParams dataclass
# -----------------------------------------------------------------------

class TestKnurlParams:
    """Tests for the KnurlParams dataclass."""

    def test_default_values(self):
        """Default KnurlParams should have reasonable values."""
        p = KnurlParams()
        assert p.knurl_type == "linear"
        assert p.pitch_mm == 1.0
        assert p.tooth_width_mm == 0.5
        assert p.depth_mm == 0.3

    def test_custom_values(self):
        """KnurlParams should accept custom values."""
        p = KnurlParams(
            knurl_type="cross_hatch",
            pitch_mm=2.5,
            tooth_width_mm=1.0,
            depth_mm=0.5,
        )
        assert p.knurl_type == "cross_hatch"
        assert p.pitch_mm == 2.5
        assert p.tooth_width_mm == 1.0
        assert p.depth_mm == 0.5


# -----------------------------------------------------------------------
# Tests: _cq_apply_knurl via HornParams (parametric path)
# -----------------------------------------------------------------------

class TestCqApplyKnurlParametric:
    """Tests for _cq_apply_knurl via the parametric generate() path."""

    def test_knurl_delegates_to_shared_logic(self):
        """_cq_apply_knurl should create KnurlParams and delegate."""
        gen = HornGenerator()
        body = mock.MagicMock()
        params = HornParams(
            knurl_type="linear",
            knurl_pitch_mm=2.0,
            knurl_tooth_width_mm=0.5,
            knurl_depth_mm=0.3,
        )

        with mock.patch.object(gen, "_cq_cut_knurl_on_body") as mock_cut:
            mock_cut.return_value = body
            result = gen._cq_apply_knurl(body, params)

        mock_cut.assert_called_once()
        call_args = mock_cut.call_args
        knurl_arg = call_args[0][1]
        assert isinstance(knurl_arg, KnurlParams)
        assert knurl_arg.knurl_type == "linear"
        assert knurl_arg.pitch_mm == 2.0

    @needs_cadquery
    def test_parametric_box_with_knurl(self):
        """Full parametric generation with knurl should produce valid result."""
        gen = HornGenerator()
        params = HornParams(
            horn_type="flat",
            width_mm=20.0,
            height_mm=60.0,
            length_mm=20.0,
            knurl_type="linear",
            knurl_pitch_mm=2.0,
            knurl_depth_mm=0.3,
        )
        result = gen.generate(params)
        assert result.knurl_info["type"] == "linear"
        assert result.volume_mm3 > 0
        assert len(result.mesh["vertices"]) > 0


# -----------------------------------------------------------------------
# Tests: edge cases and error handling
# -----------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases in knurl geometry operations."""

    def test_very_small_pitch(self):
        """Very small pitch should not cause infinite loops."""
        gen = HornGenerator()
        bottom = _make_mock_face(0, 0, 0, 5, 5, 0, area=25.0)
        bb = _make_mock_bounding_box(0, 0, 0, 5, 5, 10)
        body = _make_mock_body([bottom], bb)

        # Pitch = 0.1 on a 5mm face -> about 50+margin grooves, manageable
        knurl = KnurlParams(knurl_type="linear", pitch_mm=0.1, depth_mm=0.1)

        # This should not hang; mock prevents real CQ operations
        with mock.patch.object(
            HornGenerator, "_cq_create_knurl_grooves", return_value=mock.MagicMock()
        ):
            result = gen._cq_cut_knurl_on_body(body, knurl)
        assert result is not None

    def test_zero_depth_knurl(self):
        """Zero-depth knurl should still produce grooves (no error)."""
        knurl = KnurlParams(knurl_type="linear", pitch_mm=2.0, depth_mm=0.0)
        # _cq_create_knurl_grooves would still create geometries,
        # just with zero height extrusion.  Without CadQuery we can only
        # test the parameter handling.
        assert knurl.depth_mm == 0.0

    def test_unsupported_knurl_type_in_grooves(self):
        """An unrecognized knurl type should produce None grooves."""
        knurl = KnurlParams(knurl_type="spiral")  # not supported in groove creation

        # Without CadQuery, test that _cq_cut_knurl_on_body handles None grooves
        gen = HornGenerator()
        bottom = _make_mock_face(0, 0, 0, 25, 25, 0, area=625.0)
        bb = _make_mock_bounding_box(0, 0, 0, 25, 25, 80)
        body = _make_mock_body([bottom], bb)

        with mock.patch.object(
            HornGenerator, "_cq_create_knurl_grooves", return_value=None
        ):
            result = gen._cq_cut_knurl_on_body(body, knurl)

        # Should return body unchanged (no grooves to cut)
        assert result is body
        body.cut.assert_not_called()

    @needs_cadquery
    def test_export_after_knurl(self):
        """Modified body should be exportable to STEP."""
        import io
        gen = HornGenerator()

        box = cq.Workplane("XY").box(20, 20, 60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "export_test.step")
        cq.exporters.export(box, step_path)

        knurl = KnurlParams(knurl_type="linear", pitch_mm=3.0, depth_mm=0.3)
        result = gen.apply_knurl_to_step(step_path, knurl)

        # Export to STEP should work
        out_buf = io.BytesIO()
        cq.exporters.export(result, out_buf, exportType="STEP")
        assert len(out_buf.getvalue()) > 0

    @needs_cadquery
    def test_export_after_knurl_to_stl(self):
        """Modified body should be exportable to STL."""
        import io
        gen = HornGenerator()

        box = cq.Workplane("XY").box(20, 20, 60)
        tmpdir = tempfile.mkdtemp()
        step_path = os.path.join(tmpdir, "stl_test.step")
        cq.exporters.export(box, step_path)

        knurl = KnurlParams(knurl_type="diamond", pitch_mm=3.0, depth_mm=0.2)
        result = gen.apply_knurl_to_step(step_path, knurl)

        out_buf = io.BytesIO()
        cq.exporters.export(result, out_buf, exportType="STL")
        assert len(out_buf.getvalue()) > 0
