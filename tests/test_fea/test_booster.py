"""Tests for booster parametric geometry generation.

Tests cover:
- Half-wavelength length computation for known materials
- Theoretical gain for all profile types
- Mesh generation for all profiles (requires Gmsh)
- Auto-length when length_mm=None
- Node sets (top_face, bottom_face) detection
- Input validation
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.booster_generator import (
    BoosterGenerator,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh

try:
    import gmsh

    HAS_GMSH = True
except ImportError:
    HAS_GMSH = False


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def gen():
    """Return a fresh BoosterGenerator instance."""
    return BoosterGenerator()


# ==========================================================================
# Computational tests (no Gmsh required)
# ==========================================================================


class TestHalfWavelengthLength:
    """Tests for half_wavelength_length()."""

    def test_titanium_20khz(self, gen):
        """Ti-6Al-4V at 20 kHz should give ~126.7 mm."""
        length = gen.half_wavelength_length("Titanium Ti-6Al-4V", 20000.0)
        # c = sqrt(113.8e9 / 4430) ~ 5068 m/s
        # L = c / (2*f) = 5068 / 40000 = 0.1267 m = 126.7 mm
        assert abs(length - 126.7) < 1.0  # within 1 mm

    def test_aluminum_20khz(self, gen):
        """Aluminum 7075-T6 at 20 kHz."""
        length = gen.half_wavelength_length("Aluminum 7075-T6", 20000.0)
        # c = sqrt(71.7e9 / 2810) ~ 5050 m/s
        # L = 5050 / 40000 ~ 126.3 mm
        assert abs(length - 126.3) < 1.0

    def test_steel_d2_20khz(self, gen):
        """Steel D2 at 20 kHz."""
        length = gen.half_wavelength_length("Steel D2", 20000.0)
        # c = sqrt(210e9 / 7700) ~ 5222 m/s
        # L = 5222 / 40000 ~ 130.6 mm
        assert abs(length - 130.6) < 1.0

    def test_different_frequency(self, gen):
        """Higher frequency => shorter wavelength."""
        l_20k = gen.half_wavelength_length("Titanium Ti-6Al-4V", 20000.0)
        l_40k = gen.half_wavelength_length("Titanium Ti-6Al-4V", 40000.0)
        assert abs(l_20k / l_40k - 2.0) < 0.01

    def test_unknown_material_raises(self, gen):
        """Unknown material should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nknown.*material"):
            gen.half_wavelength_length("Unobtainium", 20000.0)


class TestTheoreticalGain:
    """Tests for theoretical_gain()."""

    def test_uniform_gain_is_one(self, gen):
        """Uniform profile always has gain = 1.0."""
        assert gen.theoretical_gain("uniform", 50.0, 25.0) == pytest.approx(1.0)
        assert gen.theoretical_gain("uniform", 30.0, 30.0) == pytest.approx(1.0)

    def test_stepped_gain_area_ratio(self, gen):
        """Stepped gain = (D_in/D_out)^2."""
        gain = gen.theoretical_gain("stepped", 50.0, 25.0)
        expected = (50.0 / 25.0) ** 2  # = 4.0
        assert gain == pytest.approx(expected)

    def test_stepped_gain_same_diameter(self, gen):
        """Stepped with equal diameters => gain = 1."""
        gain = gen.theoretical_gain("stepped", 40.0, 40.0)
        assert gain == pytest.approx(1.0)

    def test_exponential_gain(self, gen):
        """Exponential gain = D_in / D_out (linear ratio)."""
        gain = gen.theoretical_gain("exponential", 50.0, 25.0)
        assert gain == pytest.approx(2.0)

    def test_catenoidal_gain(self, gen):
        """Catenoidal gain = D_in / D_out (same as exponential)."""
        gain = gen.theoretical_gain("catenoidal", 60.0, 20.0)
        assert gain == pytest.approx(3.0)

    def test_invalid_profile_raises(self, gen):
        """Unknown profile type should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported.*profile"):
            gen.theoretical_gain("conical", 50.0, 25.0)

    def test_zero_output_diameter_raises(self, gen):
        """Zero output diameter should raise ValueError."""
        with pytest.raises(ValueError):
            gen.theoretical_gain("stepped", 50.0, 0.0)

    def test_negative_diameter_raises(self, gen):
        """Negative diameter should raise ValueError."""
        with pytest.raises(ValueError):
            gen.theoretical_gain("stepped", -10.0, 25.0)


class TestProfileConstants:
    """Tests for class-level constants."""

    def test_profiles_tuple(self, gen):
        """PROFILES should contain all four profile types."""
        assert "uniform" in gen.PROFILES
        assert "stepped" in gen.PROFILES
        assert "exponential" in gen.PROFILES
        assert "catenoidal" in gen.PROFILES
        assert len(gen.PROFILES) == 4


# ==========================================================================
# Mesh generation tests (require Gmsh)
# ==========================================================================


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestGenerateMeshUniform:
    """Tests for uniform (cylindrical) profile mesh generation."""

    def test_uniform_returns_fea_mesh(self, gen):
        """generate_mesh with uniform profile returns FEAMesh."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=100.0,
            mesh_size=5.0,
            order=2,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET10"
        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 10

    def test_uniform_bounding_box(self, gen):
        """Bounding box should match input dimensions."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=100.0,
            mesh_size=5.0,
            order=2,
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length_m = bbox_max[1] - bbox_min[1]
        assert abs(length_m - 0.100) < 0.002  # 100 mm

    def test_uniform_node_sets(self, gen):
        """Top and bottom face node sets should be detected."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=100.0,
            mesh_size=5.0,
            order=2,
        )
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestGenerateMeshStepped:
    """Tests for stepped profile mesh generation."""

    def test_stepped_returns_fea_mesh(self, gen):
        """generate_mesh with stepped profile returns FEAMesh."""
        mesh = gen.generate_mesh(
            profile="stepped",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.nodes.shape[0] > 50

    def test_stepped_bounding_box(self, gen):
        """Bounding box Y extent should match length."""
        mesh = gen.generate_mesh(
            profile="stepped",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length_m = bbox_max[1] - bbox_min[1]
        assert abs(length_m - 0.120) < 0.002  # 120 mm

    def test_stepped_node_sets(self, gen):
        """Top and bottom face node sets detected."""
        mesh = gen.generate_mesh(
            profile="stepped",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestGenerateMeshExponential:
    """Tests for exponential profile mesh generation."""

    def test_exponential_returns_fea_mesh(self, gen):
        """generate_mesh with exponential profile returns FEAMesh."""
        mesh = gen.generate_mesh(
            profile="exponential",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.nodes.shape[0] > 50

    def test_exponential_bounding_box(self, gen):
        """Bounding box Y extent should match length."""
        mesh = gen.generate_mesh(
            profile="exponential",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length_m = bbox_max[1] - bbox_min[1]
        assert abs(length_m - 0.120) < 0.002

    def test_exponential_tet4(self, gen):
        """TET4 mesh generation works."""
        mesh = gen.generate_mesh(
            profile="exponential",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=1,
        )
        assert mesh.element_type == "TET4"
        assert mesh.elements.shape[1] == 4


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestGenerateMeshCatenoidal:
    """Tests for catenoidal profile mesh generation."""

    def test_catenoidal_returns_fea_mesh(self, gen):
        """generate_mesh with catenoidal profile returns FEAMesh."""
        mesh = gen.generate_mesh(
            profile="catenoidal",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.nodes.shape[0] > 50

    def test_catenoidal_bounding_box(self, gen):
        """Bounding box Y extent should match length."""
        mesh = gen.generate_mesh(
            profile="catenoidal",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length_m = bbox_max[1] - bbox_min[1]
        assert abs(length_m - 0.120) < 0.002


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestAutoLength:
    """Tests for automatic half-wavelength length computation."""

    def test_auto_length_uses_half_wavelength(self, gen):
        """When length_mm=None, auto-compute half-wavelength."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=None,
            material_name="Titanium Ti-6Al-4V",
            frequency_hz=20000.0,
            mesh_size=5.0,
            order=2,
        )
        expected_length_mm = gen.half_wavelength_length(
            "Titanium Ti-6Al-4V", 20000.0
        )
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        actual_length_m = bbox_max[1] - bbox_min[1]
        actual_length_mm = actual_length_m * 1000.0
        assert abs(actual_length_mm - expected_length_mm) < 2.0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not installed")
class TestMeshQuality:
    """Tests for mesh quality checks."""

    def test_coordinates_in_meters(self, gen):
        """All coordinates should be in meters, not mm."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=100.0,
            mesh_size=5.0,
            order=2,
        )
        assert mesh.nodes.max() < 1.0  # less than 1 meter
        assert mesh.nodes.max() > 0.01  # more than 1 cm

    def test_surface_tris_generated(self, gen):
        """Surface triangulation should be present."""
        mesh = gen.generate_mesh(
            profile="stepped",
            d_input_mm=50.0,
            d_output_mm=25.0,
            length_mm=120.0,
            mesh_size=5.0,
            order=2,
        )
        assert mesh.surface_tris.shape[1] == 3
        assert mesh.surface_tris.shape[0] > 10

    def test_mesh_stats_populated(self, gen):
        """mesh_stats should be populated with correct keys."""
        mesh = gen.generate_mesh(
            profile="uniform",
            d_input_mm=40.0,
            d_output_mm=40.0,
            length_mm=100.0,
            mesh_size=5.0,
            order=2,
        )
        assert "num_nodes" in mesh.mesh_stats
        assert "num_elements" in mesh.mesh_stats
        assert "element_type" in mesh.mesh_stats
        assert mesh.mesh_stats["num_nodes"] == mesh.nodes.shape[0]
        assert mesh.mesh_stats["num_elements"] == mesh.elements.shape[0]


class TestInputValidation:
    """Tests for input validation (no Gmsh required)."""

    def test_invalid_profile_raises(self, gen):
        """Unsupported profile type should raise ValueError."""
        with pytest.raises(ValueError, match="[Uu]nsupported.*profile"):
            gen.generate_mesh(
                profile="conical",
                d_input_mm=50.0,
                d_output_mm=25.0,
                length_mm=120.0,
            )

    def test_zero_input_diameter_raises(self, gen):
        """Zero input diameter should raise ValueError."""
        with pytest.raises(ValueError):
            gen.generate_mesh(
                profile="uniform",
                d_input_mm=0.0,
                d_output_mm=25.0,
                length_mm=100.0,
            )

    def test_negative_length_raises(self, gen):
        """Negative length should raise ValueError."""
        with pytest.raises(ValueError):
            gen.generate_mesh(
                profile="uniform",
                d_input_mm=40.0,
                d_output_mm=40.0,
                length_mm=-10.0,
            )

    def test_zero_output_diameter_raises(self, gen):
        """Zero output diameter should raise ValueError."""
        with pytest.raises(ValueError):
            gen.generate_mesh(
                profile="stepped",
                d_input_mm=50.0,
                d_output_mm=0.0,
                length_mm=100.0,
            )

    def test_invalid_order_raises(self, gen):
        """Invalid element order should raise ValueError."""
        with pytest.raises(ValueError):
            gen.generate_mesh(
                profile="uniform",
                d_input_mm=40.0,
                d_output_mm=40.0,
                length_mm=100.0,
                order=3,
            )
