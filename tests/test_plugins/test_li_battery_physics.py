from __future__ import annotations
import pytest
from ultrasonic_weld_master.plugins.li_battery.physics import PhysicsModel

class TestPhysicsModel:
    @pytest.fixture
    def model(self):
        return PhysicsModel()

    def test_acoustic_impedance_match(self, model):
        efficiency = model.acoustic_impedance_match(z1=41.7e6, z2=17.3e6)
        assert 0 < efficiency < 1

    def test_perfect_impedance_match(self, model):
        efficiency = model.acoustic_impedance_match(z1=41.7e6, z2=41.7e6)
        assert abs(efficiency - 1.0) < 1e-6

    def test_interface_power_density(self, model):
        pd = model.interface_power_density(
            frequency_hz=20000, amplitude_um=30.0, pressure_mpa=0.3,
            friction_coeff=0.3, contact_area_mm2=125.0)
        assert pd > 0

    def test_multilayer_energy_attenuation(self, model):
        ratios = model.multilayer_energy_attenuation(
            n_layers=40, material_impedance=17.3e6, layer_thickness_mm=0.012)
        assert len(ratios) == 40
        assert ratios[0] > ratios[-1]
        assert all(0 <= r <= 1 for r in ratios)

    def test_interface_temperature_rise(self, model):
        delta_t = model.interface_temperature_rise(
            power_density_w_mm2=2.0, weld_time_s=0.2,
            thermal_conductivity_1=401, thermal_conductivity_2=237,
            density_1=8960, density_2=2700,
            specific_heat_1=385, specific_heat_2=897)
        assert delta_t > 0
        assert delta_t < 1000
