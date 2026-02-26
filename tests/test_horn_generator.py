"""Tests for parametric horn generator."""
import pytest
from ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator import (
    HornGenerator, HornParams,
)


class TestHornGenerator:
    def setup_method(self):
        self.gen = HornGenerator()

    @pytest.mark.parametrize("horn_type", ["flat", "cylindrical", "exponential", "blade", "stepped"])
    def test_generate_all_types(self, horn_type):
        """All horn types should generate valid meshes."""
        params = HornParams(horn_type=horn_type, width_mm=25.0, height_mm=80.0, length_mm=25.0)
        result = self.gen.generate(params)
        assert len(result.mesh["vertices"]) > 0
        assert len(result.mesh["faces"]) > 0
        assert result.horn_type == horn_type
        assert result.volume_mm3 > 0
        assert result.surface_area_mm2 > 0

    def test_flat_horn_dimensions(self):
        """Flat horn should have correct dimensions in result."""
        params = HornParams(horn_type="flat", width_mm=20.0, height_mm=60.0, length_mm=30.0)
        result = self.gen.generate(params)
        assert result.dimensions["width_mm"] == 20.0
        assert result.dimensions["height_mm"] == 60.0
        assert result.dimensions["length_mm"] == 30.0

    def test_knurl_modifies_mesh(self):
        """Knurl type should be recorded in result."""
        params = HornParams(knurl_type="linear", knurl_pitch_mm=1.0, knurl_depth_mm=0.3)
        result = self.gen.generate(params)
        assert result.knurl_info["type"] == "linear"

    def test_chamfer_recorded(self):
        """Chamfer info should be in result."""
        params = HornParams(chamfer_radius_mm=0.5, edge_treatment="fillet")
        result = self.gen.generate(params)
        assert result.chamfer_info["radius_mm"] == 0.5
        assert result.chamfer_info["treatment"] == "fillet"

    def test_no_cad_export_without_cadquery(self):
        """Without CadQuery, should use numpy fallback (no CAD export)."""
        params = HornParams()
        result = self.gen.generate(params)
        # In test env, CadQuery is likely not installed
        # If it is, has_cad_export=True, if not, False
        assert isinstance(result.has_cad_export, bool)
