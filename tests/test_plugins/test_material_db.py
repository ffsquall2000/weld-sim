from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.material_db.plugin import MaterialDBPlugin

class TestMaterialDBPlugin:
    @pytest.fixture
    def plugin(self):
        p = MaterialDBPlugin()
        p.activate({})
        return p

    def test_get_info(self):
        p = MaterialDBPlugin()
        info = p.get_info()
        assert info.name == "material_db"

    def test_get_material_cu(self, plugin):
        cu = plugin.get_material("Cu")
        assert cu is not None
        assert cu["density_kg_m3"] > 0
        assert cu["yield_strength_mpa"] > 0
        assert cu["acoustic_impedance"] > 0

    def test_get_material_al(self, plugin):
        al = plugin.get_material("Al")
        assert al is not None
        assert al["thermal_conductivity"] > 0

    def test_get_material_ni(self, plugin):
        ni = plugin.get_material("Ni")
        assert ni is not None

    def test_get_material_unknown(self, plugin):
        result = plugin.get_material("Unobtanium")
        assert result is None

    def test_list_materials(self, plugin):
        materials = plugin.list_materials()
        assert len(materials) >= 3

    def test_get_combination_properties(self, plugin):
        props = plugin.get_combination_properties("Cu", "Al")
        assert "friction_coefficient" in props
