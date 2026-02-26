"""Tests for chamfer/edge treatment physics."""
import pytest
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel


class TestChamferStressConcentration:
    def setup_method(self):
        self.physics = PhysicsModel()

    def test_sharp_edge_returns_max_kt(self):
        """No treatment -> Kt = 3.0"""
        kt = self.physics.chamfer_stress_concentration_factor(0.0, 25.0, "none")
        assert kt == 3.0

    def test_fillet_reduces_kt(self):
        """Fillet with radius should reduce Kt below 3.0"""
        kt = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "fillet")
        assert 1.0 < kt < 3.0

    def test_large_fillet_lower_than_small(self):
        """Larger fillet radius -> lower Kt than smaller radius"""
        kt_small = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "fillet")
        kt_large = self.physics.chamfer_stress_concentration_factor(5.0, 25.0, "fillet")
        assert kt_large < kt_small

    def test_chamfer_higher_than_fillet(self):
        """Pure chamfer has higher Kt than fillet at same radius"""
        kt_chamfer = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "chamfer")
        kt_fillet = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "fillet")
        assert kt_chamfer > kt_fillet

    def test_compound_best_treatment(self):
        """Compound treatment gives lowest Kt"""
        kt_compound = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "compound")
        kt_chamfer = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "chamfer")
        assert kt_compound < kt_chamfer

    def test_kt_clamped_to_range(self):
        """Kt should always be in [1.0, 3.0]"""
        for treatment in ["none", "chamfer", "fillet", "compound"]:
            for r in [0.0, 0.1, 0.5, 1.0, 5.0]:
                kt = self.physics.chamfer_stress_concentration_factor(r, 25.0, treatment)
                assert 1.0 <= kt <= 3.0

    def test_unknown_treatment_returns_sharp(self):
        """Unknown treatment type -> sharp edge behavior"""
        kt = self.physics.chamfer_stress_concentration_factor(0.5, 25.0, "unknown")
        assert kt == 3.0


class TestChamferDamageRisk:
    def setup_method(self):
        self.physics = PhysicsModel()

    def test_low_risk_for_small_kt(self):
        """Low Kt + low pressure -> low risk"""
        result = self.physics.chamfer_material_damage_risk(1.2, 1.0, 20.0, 200.0, 0.1)
        assert result["risk_level"] == "low"

    def test_high_risk_for_sharp_edge(self):
        """High Kt + high pressure + low yield -> high or critical risk"""
        # peak = 3.0*50 = 150, dynamic = 0.1*60*50/10 = 30, total = 180
        # damage_ratio = 180/40 = 4.5, time_factor = min(1+0.5*1.0, 2) = 1.5
        # effective = 4.5 * 1.5 = 6.75 > 2.0 -> critical
        result = self.physics.chamfer_material_damage_risk(3.0, 50.0, 60.0, 40.0, 1.0)
        assert result["risk_level"] in ("high", "critical")

    def test_returns_required_keys(self):
        """Result dict should contain all required keys"""
        result = self.physics.chamfer_material_damage_risk(2.0, 3.0, 30.0, 100.0, 0.2)
        assert "risk_level" in result
        assert "peak_stress_mpa" in result
        assert "contact_area_modification" in result
        assert "energy_redistribution_factor" in result
        assert "damage_ratio" in result

    def test_contact_area_mod_decreases_with_kt(self):
        """Higher Kt -> lower contact area modification"""
        r1 = self.physics.chamfer_material_damage_risk(1.0, 2.0, 30.0, 100.0, 0.2)
        r3 = self.physics.chamfer_material_damage_risk(3.0, 2.0, 30.0, 100.0, 0.2)
        assert r1["contact_area_modification"] >= r3["contact_area_modification"]


class TestChamferAreaCorrection:
    def setup_method(self):
        self.physics = PhysicsModel()

    def test_no_treatment_returns_nominal(self):
        """No edge treatment -> nominal area unchanged"""
        area = self.physics.chamfer_contact_area_correction(100.0, 0.5, 10.0, 10.0, "none")
        assert area == 100.0

    def test_zero_radius_returns_nominal(self):
        """Zero chamfer radius -> nominal area unchanged"""
        area = self.physics.chamfer_contact_area_correction(100.0, 0.0, 10.0, 10.0, "chamfer")
        assert area == 100.0

    def test_chamfer_reduces_area(self):
        """Chamfer treatment reduces contact area"""
        area = self.physics.chamfer_contact_area_correction(100.0, 0.5, 10.0, 10.0, "chamfer")
        assert area < 100.0

    def test_area_never_below_50_percent(self):
        """Area should never go below 50% of nominal"""
        area = self.physics.chamfer_contact_area_correction(100.0, 5.0, 10.0, 10.0, "chamfer")
        assert area >= 50.0

    def test_angle_affects_chamfer_area(self):
        """Different angles produce different area corrections"""
        area_45 = self.physics.chamfer_contact_area_correction(100.0, 0.5, 10.0, 10.0, "chamfer", 45.0)
        area_30 = self.physics.chamfer_contact_area_correction(100.0, 0.5, 10.0, 10.0, "chamfer", 30.0)
        assert area_45 != area_30  # different angles -> different areas
