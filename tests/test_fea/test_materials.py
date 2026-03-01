"""Tests for enhanced material database."""
from __future__ import annotations

import math
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
    list_materials,
    FEA_MATERIALS,
)


class TestExistingMaterials:
    """Ensure backward compatibility."""

    def test_titanium_exists(self):
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        assert mat["E_pa"] == 113.8e9

    def test_alias_lookup(self):
        mat = get_material("ti64")
        assert mat is not None
        assert mat["E_pa"] == 113.8e9

    def test_all_original_properties_preserved(self):
        """Original 7 properties must still be present for all original materials."""
        original_keys = {"E_pa", "nu", "rho_kg_m3", "k_w_mk", "cp_j_kgk", "yield_mpa", "alpha_1_k"}
        for name in ["Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
                      "Copper C11000", "Nickel 200", "M2 High Speed Steel",
                      "CPM 10V", "PM60 Powder Steel", "HAP40 Powder HSS", "HAP72 Powder HSS"]:
            mat = get_material(name)
            assert mat is not None, f"{name} not found"
            for key in original_keys:
                assert key in mat, f"{name} missing {key}"


class TestNewProperties:
    """Test new properties added to existing materials."""

    @pytest.mark.parametrize("name", [
        "Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
        "Copper C11000", "Nickel 200", "M2 High Speed Steel",
        "CPM 10V", "PM60 Powder Steel", "HAP40 Powder HSS", "HAP72 Powder HSS",
    ])
    def test_damping_ratio_present(self, name):
        mat = get_material(name)
        assert "damping_ratio" in mat
        assert 0 < mat["damping_ratio"] < 0.1

    @pytest.mark.parametrize("name", [
        "Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
        "Copper C11000", "Nickel 200", "M2 High Speed Steel",
        "CPM 10V", "PM60 Powder Steel", "HAP40 Powder HSS", "HAP72 Powder HSS",
    ])
    def test_acoustic_velocity_consistent(self, name):
        mat = get_material(name)
        assert "acoustic_velocity_m_s" in mat
        # Verify c = sqrt(E/rho) within 5%
        c_calc = math.sqrt(mat["E_pa"] / mat["rho_kg_m3"])
        assert abs(mat["acoustic_velocity_m_s"] - c_calc) / c_calc < 0.05

    @pytest.mark.parametrize("name", [
        "Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
        "Copper C11000", "Nickel 200", "M2 High Speed Steel",
        "CPM 10V", "PM60 Powder Steel", "HAP40 Powder HSS", "HAP72 Powder HSS",
    ])
    def test_fatigue_endurance_present(self, name):
        mat = get_material(name)
        assert "fatigue_endurance_mpa" in mat
        assert mat["fatigue_endurance_mpa"] > 0


class TestNewMaterials:
    """Test newly added materials."""

    def test_pzt4_exists(self):
        mat = get_material("PZT-4")
        assert mat is not None
        assert "d33" in mat
        assert "is_piezoelectric" in mat
        assert mat["is_piezoelectric"] is True

    def test_pzt8_exists(self):
        mat = get_material("PZT-8")
        assert mat is not None
        assert "d33" in mat
        assert mat["is_piezoelectric"] is True

    def test_steel_4140_exists(self):
        mat = get_material("Steel 4140")
        assert mat is not None
        assert mat["E_pa"] >= 190e9

    def test_pzt4_aliases(self):
        for alias in ["pzt-4", "pzt4", "PZT-4"]:
            mat = get_material(alias)
            assert mat is not None, f"Alias '{alias}' not found"

    def test_pzt8_aliases(self):
        for alias in ["pzt-8", "pzt8", "PZT-8"]:
            mat = get_material(alias)
            assert mat is not None, f"Alias '{alias}' not found"

    def test_steel_4140_aliases(self):
        for alias in ["steel 4140", "4140", "aisi 4140"]:
            mat = get_material(alias)
            assert mat is not None, f"Alias '{alias}' not found"
