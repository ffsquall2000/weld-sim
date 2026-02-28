"""Tests for the parametric anvil geometry generator.

CadQuery is NOT available in the test environment, so all tests exercise
the numpy fallback path.  CadQuery-specific methods are tested via mocks
where relevant.
"""
from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator import (
    AnvilGenerator,
    AnvilParams,
    HAS_CADQUERY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def generator() -> AnvilGenerator:
    """Return a fresh AnvilGenerator instance."""
    return AnvilGenerator()


@pytest.fixture()
def flat_params() -> AnvilParams:
    return AnvilParams(anvil_type="flat")


@pytest.fixture()
def groove_params() -> AnvilParams:
    return AnvilParams(anvil_type="groove", groove_count=3)


@pytest.fixture()
def knurled_params() -> AnvilParams:
    return AnvilParams(anvil_type="knurled")


@pytest.fixture()
def contour_params() -> AnvilParams:
    return AnvilParams(anvil_type="contour", contour_radius_mm=25.0)


# ---------------------------------------------------------------------------
# Tests: AnvilParams
# ---------------------------------------------------------------------------


class TestAnvilParams:
    """Test the AnvilParams dataclass."""

    def test_defaults(self):
        p = AnvilParams()
        assert p.anvil_type == "flat"
        assert p.width_mm == 50.0
        assert p.depth_mm == 30.0
        assert p.height_mm == 20.0
        assert p.groove_count == 3
        assert p.knurl_pitch_mm == 1.0
        assert p.contour_radius_mm == 25.0

    def test_custom_values(self):
        p = AnvilParams(
            anvil_type="groove",
            width_mm=80.0,
            depth_mm=40.0,
            height_mm=30.0,
            groove_width_mm=8.0,
            groove_depth_mm=5.0,
            groove_count=5,
        )
        assert p.anvil_type == "groove"
        assert p.width_mm == 80.0
        assert p.groove_count == 5

    def test_to_dict(self):
        p = AnvilParams(anvil_type="knurled", knurl_pitch_mm=2.0)
        d = p.to_dict()
        assert d["anvil_type"] == "knurled"
        assert d["knurl_pitch_mm"] == 2.0
        assert "width_mm" in d
        assert "height_mm" in d
        assert "groove_count" in d
        assert "contour_radius_mm" in d

    def test_to_dict_all_keys_present(self):
        p = AnvilParams()
        d = p.to_dict()
        expected_keys = {
            "anvil_type", "width_mm", "depth_mm", "height_mm",
            "groove_width_mm", "groove_depth_mm", "groove_count",
            "knurl_pitch_mm", "knurl_depth_mm", "contour_radius_mm",
        }
        assert expected_keys == set(d.keys())


# ---------------------------------------------------------------------------
# Tests: AnvilGenerator.generate() -- invalid input
# ---------------------------------------------------------------------------


class TestGenerateValidation:
    """Test input validation in generate()."""

    def test_invalid_type_raises(self, generator):
        with pytest.raises(ValueError, match="Invalid anvil_type"):
            generator.generate(AnvilParams(anvil_type="invalid"))

    def test_valid_types_accepted(self, generator):
        for atype in ("flat", "groove", "knurled", "contour"):
            result = generator.generate(AnvilParams(anvil_type=atype))
            assert result is not None
            assert "mesh_preview" in result


# ---------------------------------------------------------------------------
# Tests: Flat anvil
# ---------------------------------------------------------------------------


class TestFlatAnvil:
    """Test flat anvil generation (numpy path)."""

    def test_generates_mesh(self, generator, flat_params):
        result = generator.generate(flat_params)
        mesh = result["mesh_preview"]
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_has_8_vertices(self, generator, flat_params):
        """Flat box should have exactly 8 vertices."""
        result = generator.generate(flat_params)
        assert len(result["mesh_preview"]["vertices"]) == 8

    def test_has_12_faces(self, generator, flat_params):
        """Box with triangulated faces = 12 triangles."""
        result = generator.generate(flat_params)
        assert len(result["mesh_preview"]["faces"]) == 12

    def test_dimensions_correct(self, generator, flat_params):
        result = generator.generate(flat_params)
        verts = np.array(result["mesh_preview"]["vertices"])
        x_range = verts[:, 0].max() - verts[:, 0].min()
        y_range = verts[:, 1].max() - verts[:, 1].min()
        z_range = verts[:, 2].max() - verts[:, 2].min()
        assert abs(x_range - flat_params.width_mm) < 0.01
        assert abs(y_range - flat_params.depth_mm) < 0.01
        assert abs(z_range - flat_params.height_mm) < 0.01

    def test_contact_face_info(self, generator, flat_params):
        result = generator.generate(flat_params)
        cf = result["contact_face"]
        assert cf["width"] == flat_params.width_mm
        assert cf["length"] == flat_params.depth_mm
        assert cf["center"][2] == flat_params.height_mm

    def test_volume_positive(self, generator, flat_params):
        result = generator.generate(flat_params)
        assert result["volume_mm3"] > 0

    def test_surface_area_positive(self, generator, flat_params):
        result = generator.generate(flat_params)
        assert result["surface_area_mm2"] > 0

    def test_no_step_without_cadquery(self, generator, flat_params):
        """Without CadQuery, step_path should be None."""
        result = generator.generate(flat_params)
        if not HAS_CADQUERY:
            assert result["step_path"] is None
            assert result["solid"] is None

    def test_params_in_result(self, generator, flat_params):
        result = generator.generate(flat_params)
        assert result["params"]["anvil_type"] == "flat"
        assert result["params"]["width_mm"] == 50.0


# ---------------------------------------------------------------------------
# Tests: Groove anvil
# ---------------------------------------------------------------------------


class TestGrooveAnvil:
    """Test grooved anvil generation."""

    def test_generates_mesh(self, generator, groove_params):
        result = generator.generate(groove_params)
        mesh = result["mesh_preview"]
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_more_vertices_than_flat(self, generator, groove_params, flat_params):
        """Grooved anvil should have more vertices than flat."""
        flat = generator.generate(flat_params)
        grooved = generator.generate(groove_params)
        assert len(grooved["mesh_preview"]["vertices"]) > len(
            flat["mesh_preview"]["vertices"]
        )

    def test_groove_count_zero(self, generator):
        """Zero grooves should produce a flat-like result."""
        params = AnvilParams(anvil_type="groove", groove_count=0)
        result = generator.generate(params)
        # Should still generate valid mesh
        assert len(result["mesh_preview"]["vertices"]) > 0
        assert len(result["mesh_preview"]["faces"]) > 0

    def test_groove_depth_in_mesh(self, generator, groove_params):
        """Top face should have vertices at groove depth."""
        result = generator.generate(groove_params)
        verts = np.array(result["mesh_preview"]["vertices"])
        h = groove_params.height_mm
        g_depth = groove_params.groove_depth_mm
        # Should have some vertices at h and some at h - g_depth
        z_values = verts[:, 2]
        top_z = z_values[z_values > h * 0.5]  # upper half vertices
        assert np.any(np.abs(top_z - h) < 0.01), "Should have vertices at top"
        assert np.any(
            np.abs(top_z - (h - g_depth)) < 0.01
        ), "Should have vertices at groove bottom"

    def test_custom_groove_params(self, generator):
        params = AnvilParams(
            anvil_type="groove",
            groove_width_mm=10.0,
            groove_depth_mm=5.0,
            groove_count=2,
        )
        result = generator.generate(params)
        assert result["params"]["groove_width_mm"] == 10.0
        assert result["params"]["groove_count"] == 2

    def test_volume_positive(self, generator, groove_params):
        result = generator.generate(groove_params)
        assert result["volume_mm3"] > 0


# ---------------------------------------------------------------------------
# Tests: Knurled anvil
# ---------------------------------------------------------------------------


class TestKnurledAnvil:
    """Test knurled anvil generation."""

    def test_generates_mesh(self, generator, knurled_params):
        result = generator.generate(knurled_params)
        mesh = result["mesh_preview"]
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_dense_top_grid(self, generator, knurled_params):
        """Knurled anvil should have a dense grid on top."""
        result = generator.generate(knurled_params)
        verts = np.array(result["mesh_preview"]["vertices"])
        # Knurl creates a grid => many vertices
        assert len(verts) > 20

    def test_height_modulation(self, generator, knurled_params):
        """Top face vertices should show height variation from knurl."""
        result = generator.generate(knurled_params)
        verts = np.array(result["mesh_preview"]["vertices"])
        h = knurled_params.height_mm
        # Top vertices (z > h * 0.5)
        top_z = verts[verts[:, 2] > h * 0.5, 2]
        # Should have variation (not all the same height)
        assert top_z.std() > 0.0, "Knurl should create height variation"

    def test_custom_pitch(self, generator):
        params = AnvilParams(
            anvil_type="knurled",
            knurl_pitch_mm=2.0,
            knurl_depth_mm=0.5,
        )
        result = generator.generate(params)
        assert result["params"]["knurl_pitch_mm"] == 2.0

    def test_volume_positive(self, generator, knurled_params):
        result = generator.generate(knurled_params)
        assert result["volume_mm3"] > 0


# ---------------------------------------------------------------------------
# Tests: Contour anvil
# ---------------------------------------------------------------------------


class TestContourAnvil:
    """Test contour (concave) anvil generation."""

    def test_generates_mesh(self, generator, contour_params):
        result = generator.generate(contour_params)
        mesh = result["mesh_preview"]
        assert len(mesh["vertices"]) > 0
        assert len(mesh["faces"]) > 0

    def test_concave_top_face(self, generator, contour_params):
        """Top face should show concave curvature (z varies with x).

        The concave cylindrical cut removes material from the top face,
        with a deeper cut near the edges (large |x|) and a shallower
        cut near the centre (x ~ 0).  Therefore the centre of the top
        face should be *higher* than the edges.
        """
        result = generator.generate(contour_params)
        verts = np.array(result["mesh_preview"]["vertices"])
        h = contour_params.height_mm
        # Top vertices (z > h * 0.5)
        top_verts = verts[verts[:, 2] > h * 0.5]

        if len(top_verts) < 3:
            pytest.skip("Not enough top vertices to test concavity")

        # Center vertices (x ~ 0) should be higher than edge vertices
        center_mask = np.abs(top_verts[:, 0]) < contour_params.width_mm * 0.1
        edge_mask = np.abs(top_verts[:, 0]) > contour_params.width_mm * 0.3

        if np.any(center_mask) and np.any(edge_mask):
            center_z = top_verts[center_mask, 2].mean()
            edge_z = top_verts[edge_mask, 2].mean()
            assert center_z > edge_z, (
                "Centre should be higher than edges for concave cylindrical contour"
            )

    def test_custom_contour_radius(self, generator):
        params = AnvilParams(anvil_type="contour", contour_radius_mm=50.0)
        result = generator.generate(params)
        assert result["params"]["contour_radius_mm"] == 50.0

    def test_small_radius_valid(self, generator):
        """Small radius (tight curve) should still produce valid mesh."""
        params = AnvilParams(anvil_type="contour", contour_radius_mm=10.0)
        result = generator.generate(params)
        assert len(result["mesh_preview"]["vertices"]) > 0

    def test_large_radius_nearly_flat(self, generator):
        """Very large radius should produce nearly flat surface."""
        params = AnvilParams(
            anvil_type="contour",
            contour_radius_mm=10000.0,
            width_mm=50.0,
        )
        result = generator.generate(params)
        verts = np.array(result["mesh_preview"]["vertices"])
        top_verts = verts[verts[:, 2] > params.height_mm * 0.5]
        if len(top_verts) > 1:
            z_range = top_verts[:, 2].max() - top_verts[:, 2].min()
            # With r=10000 and w=50, sagitta ~ 0.03 mm
            assert z_range < 1.0, "Large radius should be nearly flat"

    def test_volume_positive(self, generator, contour_params):
        result = generator.generate(contour_params)
        assert result["volume_mm3"] > 0


# ---------------------------------------------------------------------------
# Tests: Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    """Test that all anvil types return the expected result structure."""

    @pytest.mark.parametrize("anvil_type", ["flat", "groove", "knurled", "contour"])
    def test_result_keys(self, generator, anvil_type):
        result = generator.generate(AnvilParams(anvil_type=anvil_type))
        expected_keys = {
            "solid", "mesh_preview", "step_path", "params",
            "contact_face", "volume_mm3", "surface_area_mm2",
        }
        assert expected_keys == set(result.keys())

    @pytest.mark.parametrize("anvil_type", ["flat", "groove", "knurled", "contour"])
    def test_mesh_preview_structure(self, generator, anvil_type):
        result = generator.generate(AnvilParams(anvil_type=anvil_type))
        mesh = result["mesh_preview"]
        assert "vertices" in mesh
        assert "faces" in mesh
        assert isinstance(mesh["vertices"], list)
        assert isinstance(mesh["faces"], list)
        # All vertices should be [x, y, z]
        for v in mesh["vertices"]:
            assert len(v) == 3
        # All faces should be [a, b, c]
        for f in mesh["faces"]:
            assert len(f) == 3

    @pytest.mark.parametrize("anvil_type", ["flat", "groove", "knurled", "contour"])
    def test_contact_face_structure(self, generator, anvil_type):
        result = generator.generate(AnvilParams(anvil_type=anvil_type))
        cf = result["contact_face"]
        assert "center" in cf
        assert "width" in cf
        assert "length" in cf
        assert len(cf["center"]) == 3

    @pytest.mark.parametrize("anvil_type", ["flat", "groove", "knurled", "contour"])
    def test_valid_face_indices(self, generator, anvil_type):
        """All face vertex indices should be within bounds."""
        result = generator.generate(AnvilParams(anvil_type=anvil_type))
        mesh = result["mesh_preview"]
        n_verts = len(mesh["vertices"])
        for face in mesh["faces"]:
            for idx in face:
                assert 0 <= idx < n_verts, (
                    f"Face index {idx} out of range [0, {n_verts})"
                )


# ---------------------------------------------------------------------------
# Tests: Different dimension configurations
# ---------------------------------------------------------------------------


class TestDimensionVariations:
    """Test various dimension combinations."""

    def test_wide_flat_anvil(self, generator):
        result = generator.generate(AnvilParams(
            anvil_type="flat", width_mm=100.0, depth_mm=20.0, height_mm=10.0
        ))
        verts = np.array(result["mesh_preview"]["vertices"])
        assert abs((verts[:, 0].max() - verts[:, 0].min()) - 100.0) < 0.01

    def test_tall_flat_anvil(self, generator):
        result = generator.generate(AnvilParams(
            anvil_type="flat", width_mm=30.0, depth_mm=30.0, height_mm=50.0
        ))
        verts = np.array(result["mesh_preview"]["vertices"])
        assert abs((verts[:, 2].max() - verts[:, 2].min()) - 50.0) < 0.01

    def test_many_grooves(self, generator):
        result = generator.generate(AnvilParams(
            anvil_type="groove",
            width_mm=100.0,
            groove_count=10,
            groove_width_mm=3.0,
        ))
        assert len(result["mesh_preview"]["vertices"]) > 8

    def test_fine_knurl(self, generator):
        result = generator.generate(AnvilParams(
            anvil_type="knurled",
            knurl_pitch_mm=0.5,
            knurl_depth_mm=0.1,
        ))
        # Fine pitch = more grid points
        assert len(result["mesh_preview"]["vertices"]) > 50


# ---------------------------------------------------------------------------
# Tests: CadQuery path (mocked)
# ---------------------------------------------------------------------------


class TestCadQueryPath:
    """Test CadQuery code path with mocks (since CadQuery is not installed)."""

    def test_cadquery_fallback_when_unavailable(self, generator, flat_params):
        """When CadQuery is not available, numpy fallback is used."""
        result = generator.generate(flat_params)
        if not HAS_CADQUERY:
            assert result["solid"] is None
            assert result["step_path"] is None

    def test_cadquery_exception_triggers_fallback(self, generator, flat_params):
        """If CadQuery raises, we fall back to numpy."""
        with mock.patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator.HAS_CADQUERY",
            True,
        ), mock.patch.object(
            generator, "_generate_cadquery", side_effect=RuntimeError("CQ failed")
        ):
            result = generator.generate(flat_params)
            # Should still succeed via numpy fallback
            assert result is not None
            assert len(result["mesh_preview"]["vertices"]) > 0
