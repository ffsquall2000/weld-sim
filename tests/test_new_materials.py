"""Tests for new horn materials (M2, CPM10V, PM60, HAP40, HAP72)."""
import pytest
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material, list_materials,
)


class TestNewMaterials:
    NEW_MATERIALS = [
        ("M2 High Speed Steel", "m2"),
        ("CPM 10V", "cpm10v"),
        ("PM60 Powder Steel", "pm60"),
        ("HAP40 Powder HSS", "hap40"),
        ("HAP72 Powder HSS", "hap72"),
    ]

    def test_all_new_materials_exist(self):
        """All 5 new materials should be in the materials list."""
        mats = list_materials()
        for canonical, _ in self.NEW_MATERIALS:
            assert canonical in mats, f"{canonical} not found in materials"

    @pytest.mark.parametrize("canonical,alias", [
        ("M2 High Speed Steel", "m2"),
        ("CPM 10V", "cpm10v"),
        ("PM60 Powder Steel", "pm60"),
        ("HAP40 Powder HSS", "hap40"),
        ("HAP72 Powder HSS", "hap72"),
    ])
    def test_alias_lookup(self, canonical, alias):
        """Aliases should resolve to correct materials."""
        mat = get_material(alias)
        assert mat is not None, f"Alias '{alias}' returned None"
        assert mat["E_pa"] > 0

    @pytest.mark.parametrize("name", ["m2", "cpm10v", "pm60", "hap40", "hap72"])
    def test_required_properties(self, name):
        """Each material should have all required FEA properties."""
        mat = get_material(name)
        assert mat is not None
        for key in ["E_pa", "nu", "rho_kg_m3", "k_w_mk", "cp_j_kgk", "yield_mpa", "alpha_1_k"]:
            assert key in mat, f"Missing '{key}' in {name}"
            assert mat[key] > 0, f"{key}={mat[key]} should be > 0 for {name}"

    def test_yield_strength_ordering(self):
        """HAP72 should have highest yield, M2 should be lower."""
        m2 = get_material("m2")
        hap72 = get_material("hap72")
        assert hap72["yield_mpa"] > m2["yield_mpa"]
