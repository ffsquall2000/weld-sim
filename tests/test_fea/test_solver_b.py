"""Tests for SolverB (FEniCSx backend).

Unit tests for pure-Python helper functions run unconditionally.
Integration tests that require FEniCSx are guarded with
``@pytest.mark.skipif(not HAS_FENICSX, ...)``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    HarmonicConfig,
    ModalConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ContactResult,
    HarmonicResult,
    ImpedanceResult,
    ModalResult,
    StaticResult,
    ThermalResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_b import (
    HAS_FENICSX,
    ContactConfig,
    PiezoConfig,
    PZT4_PROPERTIES,
    PZT8_PROPERTIES,
    SolverB,
    ThermalConfig,
    classify_contact_status,
    compute_contact_gap,
    compute_contact_traction_augmented_lagrangian,
    compute_coulomb_friction_traction,
    compute_frequency_shift_thermal,
    compute_frictional_heat,
    compute_hysteretic_heat,
    compute_rayleigh_damping_coefficients,
    compute_time_averaged_frictional_heat,
    get_pzt_properties,
)


# =========================================================================
# Fixtures
# =========================================================================
@pytest.fixture
def simple_mesh() -> FEAMesh:
    """Minimal FEAMesh for unit-level tests (4 nodes, 1 tetrahedron)."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    elements = np.array([[0, 1, 2, 3]], dtype=np.int32)
    return FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET4",
        node_sets={
            "bottom_face": np.array([0, 1, 2], dtype=np.int32),
            "top_face": np.array([3], dtype=np.int32),
        },
        element_sets={"all": np.array([0], dtype=np.int32)},
        surface_tris=np.array([[0, 1, 2]], dtype=np.int32),
        mesh_stats={"n_nodes": 4, "n_elements": 1},
    )


@pytest.fixture
def contact_mesh() -> FEAMesh:
    """Mesh with contact surface definitions for contact analysis tests."""
    nodes = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.01],
        [1.0, 0.0, 1.01],
        [0.0, 1.0, 1.01],
        [0.0, 0.0, 2.0],
    ])
    elements = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ], dtype=np.int32)
    return FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET4",
        node_sets={
            "master_contact": np.array([3], dtype=np.int32),
            "slave_contact": np.array([4], dtype=np.int32),
            "bottom_face": np.array([0, 1, 2], dtype=np.int32),
            "top_face": np.array([7], dtype=np.int32),
            "bolt_bearing": np.array([0, 1, 2], dtype=np.int32),
        },
        element_sets={
            "body_lower": np.array([0], dtype=np.int32),
            "body_upper": np.array([1], dtype=np.int32),
        },
        surface_tris=np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32),
        mesh_stats={"n_nodes": 8, "n_elements": 2},
    )


# =========================================================================
# 1. Result dataclass tests (no FEniCSx)
# =========================================================================
class TestThermalResultDataclass:
    """Test ThermalResult dataclass construction and defaults."""

    def test_construction_minimal(self):
        """ThermalResult can be constructed with all required fields."""
        r = ThermalResult(
            time_steps=np.array([0.0, 0.01]),
            temperature_field=np.array([[25.0, 25.0], [30.0, 35.0]]),
            max_temperature_c=35.0,
            frequency_shift_hz=-10.0,
            thermal_stress_vm=np.array([0.0, 1.0]),
            thermal_expansion_strain=np.zeros((2, 6)),
        )
        assert r.max_temperature_c == 35.0
        assert r.frequency_shift_hz == -10.0
        assert r.solver_name == "fenicsx"
        assert r.solve_time_s == 0.0
        assert r.mesh is None
        assert r.metadata == {}

    def test_construction_full(self, simple_mesh):
        """ThermalResult with all fields explicitly set."""
        r = ThermalResult(
            time_steps=np.linspace(0, 0.5, 11),
            temperature_field=np.ones((11, 4)) * 25.0,
            max_temperature_c=100.0,
            frequency_shift_hz=-25.5,
            thermal_stress_vm=np.array([10.0, 20.0, 15.0, 5.0]),
            thermal_expansion_strain=np.zeros((4, 6)),
            mesh=simple_mesh,
            solve_time_s=1.23,
            solver_name="fenicsx",
            metadata={"time_scheme": "bdf2"},
        )
        assert r.mesh is not None
        assert r.metadata["time_scheme"] == "bdf2"
        assert len(r.time_steps) == 11

    def test_temperature_field_shape(self):
        """Temperature field has shape (n_steps, n_nodes)."""
        n_steps, n_nodes = 5, 10
        r = ThermalResult(
            time_steps=np.zeros(n_steps),
            temperature_field=np.zeros((n_steps, n_nodes)),
            max_temperature_c=25.0,
            frequency_shift_hz=0.0,
            thermal_stress_vm=np.zeros(n_nodes),
            thermal_expansion_strain=np.zeros((n_nodes, 6)),
        )
        assert r.temperature_field.shape == (n_steps, n_nodes)


class TestContactResultDataclass:
    """Test ContactResult dataclass construction and defaults."""

    def test_construction_minimal(self):
        """ContactResult can be constructed with required fields."""
        n = 5
        r = ContactResult(
            contact_pressure=np.zeros(n),
            gap=np.ones(n) * 0.01,
            slip=np.zeros(n),
            contact_status=np.zeros(n, dtype=np.int32),
            bolt_force_n=30000.0,
        )
        assert r.bolt_force_n == 30000.0
        assert r.n_newton_iterations == 0
        assert r.solver_name == "fenicsx"
        assert r.stressed_frequencies_hz is None
        assert r.stressed_mode_shapes is None

    def test_contact_status_values(self):
        """Contact status uses 0=open, 1=stick, 2=slip."""
        status = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        r = ContactResult(
            contact_pressure=np.zeros(5),
            gap=np.zeros(5),
            slip=np.zeros(5),
            contact_status=status,
            bolt_force_n=25000.0,
        )
        assert np.all(r.contact_status == status)


class TestImpedanceResultDataclass:
    """Test ImpedanceResult dataclass construction."""

    def test_construction(self):
        """ImpedanceResult can be constructed correctly."""
        n = 100
        r = ImpedanceResult(
            frequencies_hz=np.linspace(18000, 22000, n),
            impedance_magnitude=np.ones(n) * 50.0,
            impedance_phase_deg=np.zeros(n),
            admittance_magnitude=np.ones(n) * 0.02,
            resonant_freq_hz=19500.0,
            antiresonant_freq_hz=20500.0,
            k_eff=0.3,
        )
        assert r.resonant_freq_hz == 19500.0
        assert r.k_eff == pytest.approx(0.3)


# =========================================================================
# 2. SolverB class-level tests (no FEniCSx required)
# =========================================================================
class TestSolverBProperties:
    """Test SolverB properties and availability checks."""

    def test_solver_name(self):
        """SolverB reports correct solver name."""
        solver = SolverB()
        assert solver.solver_name == "fenicsx"

    def test_is_available(self):
        """is_available matches HAS_FENICSX flag."""
        solver = SolverB()
        assert solver.is_available == HAS_FENICSX

    def test_capabilities_without_fenicsx(self):
        """When FEniCSx is not available, capabilities is empty."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        assert solver.get_capabilities() == set()

    def test_capabilities_with_fenicsx(self):
        """When FEniCSx is available, capabilities includes all types."""
        if not HAS_FENICSX:
            pytest.skip("FEniCSx not installed")
        solver = SolverB()
        caps = solver.get_capabilities()
        assert "modal" in caps
        assert "harmonic" in caps
        assert "thermal" in caps
        assert "contact" in caps
        assert "piezoelectric" in caps

    def test_modal_raises_without_fenicsx(self, simple_mesh):
        """modal_analysis raises RuntimeError if FEniCSx not available."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = ModalConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
        )
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.modal_analysis(config)

    def test_harmonic_raises_without_fenicsx(self, simple_mesh):
        """harmonic_analysis raises RuntimeError if FEniCSx not available."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = HarmonicConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
        )
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.harmonic_analysis(config)

    def test_static_raises_without_fenicsx(self, simple_mesh):
        """static_analysis raises RuntimeError if FEniCSx not available."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = StaticConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
        )
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.static_analysis(config)

    def test_thermal_raises_without_fenicsx(self, simple_mesh):
        """thermal_analysis raises RuntimeError if FEniCSx not available."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = ThermalConfig(mesh=simple_mesh)
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.thermal_analysis(config)

    def test_contact_raises_without_fenicsx(self, simple_mesh):
        """contact_analysis raises RuntimeError if FEniCSx not available."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = ContactConfig(mesh=simple_mesh)
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.contact_analysis(config)

    def test_piezoelectric_raises_without_fenicsx(self, simple_mesh):
        """piezoelectric_analysis raises RuntimeError without FEniCSx."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = PiezoConfig(mesh=simple_mesh)
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.piezoelectric_analysis(config)

    def test_impedance_raises_without_fenicsx(self, simple_mesh):
        """impedance_spectrum raises RuntimeError without FEniCSx."""
        if HAS_FENICSX:
            pytest.skip("FEniCSx is installed")
        solver = SolverB()
        config = PiezoConfig(mesh=simple_mesh)
        with pytest.raises(RuntimeError, match="FEniCSx not available"):
            solver.impedance_spectrum(config)


# =========================================================================
# 3. Configuration validation tests (no FEniCSx)
# =========================================================================
class TestPiezoConfigValidation:
    """Test PiezoConfig validation logic."""

    def test_valid_config(self, simple_mesh):
        """Valid PiezoConfig produces no errors."""
        config = PiezoConfig(mesh=simple_mesh)
        assert config.validate() == []

    def test_negative_voltage(self, simple_mesh):
        """Negative driving_voltage is invalid."""
        config = PiezoConfig(mesh=simple_mesh, driving_voltage=-1.0)
        errors = config.validate()
        assert any("driving_voltage" in e for e in errors)

    def test_freq_range_reversed(self, simple_mesh):
        """freq_min_hz >= freq_max_hz is invalid."""
        config = PiezoConfig(
            mesh=simple_mesh, freq_min_hz=25000, freq_max_hz=15000,
        )
        errors = config.validate()
        assert any("freq_min_hz" in e for e in errors)

    def test_invalid_pzt_material(self, simple_mesh):
        """Unknown PZT material is invalid."""
        config = PiezoConfig(mesh=simple_mesh, pzt_material="PZT-99")
        errors = config.validate()
        assert any("PZT" in e for e in errors)

    def test_damping_ratio_out_of_range(self, simple_mesh):
        """Damping ratio outside (0, 1) is invalid."""
        config = PiezoConfig(mesh=simple_mesh, damping_ratio=1.5)
        errors = config.validate()
        assert any("damping_ratio" in e for e in errors)

    def test_too_few_freq_points(self, simple_mesh):
        """n_freq_points < 2 is invalid."""
        config = PiezoConfig(mesh=simple_mesh, n_freq_points=1)
        errors = config.validate()
        assert any("n_freq_points" in e for e in errors)


class TestThermalConfigValidation:
    """Test ThermalConfig validation logic."""

    def test_valid_config(self, simple_mesh):
        """Valid ThermalConfig produces no errors."""
        config = ThermalConfig(mesh=simple_mesh)
        assert config.validate() == []

    def test_negative_frequency(self, simple_mesh):
        """Negative frequency_hz is invalid."""
        config = ThermalConfig(mesh=simple_mesh, frequency_hz=-100)
        errors = config.validate()
        assert any("frequency_hz" in e for e in errors)

    def test_negative_dt(self, simple_mesh):
        """Negative dt is invalid."""
        config = ThermalConfig(mesh=simple_mesh, dt=-0.001)
        errors = config.validate()
        assert any("dt" in e for e in errors)

    def test_zero_steps(self, simple_mesh):
        """n_steps < 1 is invalid."""
        config = ThermalConfig(mesh=simple_mesh, n_steps=0)
        errors = config.validate()
        assert any("n_steps" in e for e in errors)

    def test_invalid_time_scheme(self, simple_mesh):
        """Unknown time_scheme is invalid."""
        config = ThermalConfig(mesh=simple_mesh, time_scheme="rk4")
        errors = config.validate()
        assert any("time_scheme" in e for e in errors)

    def test_invalid_material(self, simple_mesh):
        """Unknown material produces a validation error."""
        config = ThermalConfig(mesh=simple_mesh, material_name="unobtanium")
        errors = config.validate()
        assert any("material" in e.lower() for e in errors)

    def test_negative_friction(self, simple_mesh):
        """Negative friction coefficient is invalid."""
        config = ThermalConfig(mesh=simple_mesh, friction_coefficient=-0.1)
        errors = config.validate()
        assert any("friction" in e for e in errors)

    def test_bdf2_and_backward_euler_valid(self, simple_mesh):
        """Both time scheme options are valid."""
        for scheme in ("bdf2", "backward_euler"):
            config = ThermalConfig(mesh=simple_mesh, time_scheme=scheme)
            assert config.validate() == []


class TestContactConfigValidation:
    """Test ContactConfig validation logic."""

    def test_valid_config(self, simple_mesh):
        """Valid ContactConfig produces no errors."""
        config = ContactConfig(
            mesh=simple_mesh,
            contact_pairs=[{"master": "bottom_face", "slave": "top_face", "mu": 0.2}],
        )
        assert config.validate() == []

    def test_negative_bolt_preload(self, simple_mesh):
        """Negative bolt preload is invalid."""
        config = ContactConfig(mesh=simple_mesh, bolt_preload_n=-1000)
        errors = config.validate()
        assert any("bolt_preload" in e for e in errors)

    def test_zero_max_aug_iters(self, simple_mesh):
        """max_augmentation_iters < 1 is invalid."""
        config = ContactConfig(mesh=simple_mesh, max_augmentation_iters=0)
        errors = config.validate()
        assert any("max_augmentation" in e for e in errors)

    def test_negative_newton_rtol(self, simple_mesh):
        """Negative newton_rtol is invalid."""
        config = ContactConfig(mesh=simple_mesh, newton_rtol=-1e-8)
        errors = config.validate()
        assert any("newton_rtol" in e for e in errors)

    def test_missing_contact_pair_keys(self, simple_mesh):
        """Contact pairs missing master/slave keys is invalid."""
        config = ContactConfig(
            mesh=simple_mesh,
            contact_pairs=[{"mu": 0.3}],
        )
        errors = config.validate()
        assert any("master" in e and "slave" in e for e in errors)

    def test_negative_friction_coefficient(self, simple_mesh):
        """Negative friction mu in contact pair is invalid."""
        config = ContactConfig(
            mesh=simple_mesh,
            contact_pairs=[{"master": "a", "slave": "b", "mu": -0.1}],
        )
        errors = config.validate()
        assert any("friction" in e.lower() or "mu" in e.lower() for e in errors)

    def test_invalid_material(self, simple_mesh):
        """Unknown material produces a validation error."""
        config = ContactConfig(mesh=simple_mesh, material_name="unobtanium")
        errors = config.validate()
        assert any("material" in e.lower() for e in errors)


# =========================================================================
# 4. Hysteretic heat generation tests (no FEniCSx)
# =========================================================================
class TestHystereticHeat:
    """Test compute_hysteretic_heat pure-Python function."""

    def test_basic_formula(self):
        """Q_hyst = pi * f * eta * sigma . epsilon."""
        freq = 20000.0
        eta = 0.003
        stress = np.array([100e6, 0, 0, 0, 0, 0], dtype=np.float64)
        strain = np.array([1e-3, 0, 0, 0, 0, 0], dtype=np.float64)
        q = compute_hysteretic_heat(freq, eta, stress, strain)
        expected = math.pi * freq * eta * 100e6 * 1e-3
        assert q[0] == pytest.approx(expected, rel=1e-10)

    def test_multi_point(self):
        """Works for arrays of (n_points, 6)."""
        n = 10
        stress = np.ones((n, 6)) * 50e6
        strain = np.ones((n, 6)) * 5e-4
        q = compute_hysteretic_heat(20000.0, 0.005, stress, strain)
        assert q.shape == (n,)
        # Each point: pi * 20000 * 0.005 * (6 * 50e6 * 5e-4)
        expected = math.pi * 20000 * 0.005 * (6 * 50e6 * 5e-4)
        np.testing.assert_allclose(q, expected, rtol=1e-10)

    def test_zero_loss_factor(self):
        """Zero loss factor produces zero heat."""
        q = compute_hysteretic_heat(
            20000.0, 0.0,
            np.array([100e6, 0, 0, 0, 0, 0]),
            np.array([1e-3, 0, 0, 0, 0, 0]),
        )
        assert q[0] == pytest.approx(0.0)

    def test_negative_frequency_raises(self):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency_hz"):
            compute_hysteretic_heat(
                -1.0, 0.003,
                np.array([1.0]), np.array([1.0]),
            )

    def test_negative_loss_factor_raises(self):
        """Negative loss factor raises ValueError."""
        with pytest.raises(ValueError, match="loss_factor"):
            compute_hysteretic_heat(
                20000.0, -0.001,
                np.array([1.0]), np.array([1.0]),
            )

    def test_mismatched_shapes_raises(self):
        """Mismatched stress/strain shapes raises ValueError."""
        with pytest.raises(ValueError, match="shapes must match"):
            compute_hysteretic_heat(
                20000.0, 0.003,
                np.array([1.0, 2.0]),
                np.array([1.0]),
            )


# =========================================================================
# 5. Frictional heat generation tests (no FEniCSx)
# =========================================================================
class TestFrictionalHeat:
    """Test frictional heat computation functions."""

    def test_basic_formula(self):
        """Q_fric = mu * p * v_rel."""
        mu = 0.2
        p = 1e6  # Pa
        v = 1.0  # m/s
        q = compute_frictional_heat(mu, p, v)
        assert q[0] == pytest.approx(0.2 * 1e6 * 1.0)

    def test_array_inputs(self):
        """Works with array pressure and velocity."""
        mu = 0.3
        p = np.array([1e6, 2e6, 3e6])
        v = np.array([0.5, 1.0, 1.5])
        q = compute_frictional_heat(mu, p, v)
        expected = 0.3 * p * v
        np.testing.assert_allclose(q, expected)

    def test_negative_mu_raises(self):
        """Negative friction coefficient raises ValueError."""
        with pytest.raises(ValueError, match="friction_coefficient"):
            compute_frictional_heat(-0.1, 1e6, 1.0)

    def test_time_averaged_formula(self):
        """Q_avg = (2/pi) * mu * p * (2*pi*f*A)."""
        mu = 0.2
        p = 1e6
        f = 20000.0
        A = 30e-6
        q = compute_time_averaged_frictional_heat(mu, p, f, A)
        peak_v = 2 * math.pi * f * A
        expected = (2.0 / math.pi) * mu * p * peak_v
        assert q[0] == pytest.approx(expected, rel=1e-10)

    def test_time_averaged_zero_amplitude(self):
        """Zero amplitude produces zero heat."""
        q = compute_time_averaged_frictional_heat(0.2, 1e6, 20000.0, 0.0)
        assert q[0] == pytest.approx(0.0)

    def test_time_averaged_negative_freq_raises(self):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="frequency_hz"):
            compute_time_averaged_frictional_heat(0.2, 1e6, -20000.0, 30e-6)


# =========================================================================
# 6. Contact gap function tests (no FEniCSx)
# =========================================================================
class TestContactGap:
    """Test compute_contact_gap function."""

    def test_initial_gap_from_coordinates(self):
        """Gap computed from coordinate separation along normal."""
        master = np.array([[0.0, 0.0, 1.0]])
        slave = np.array([[0.0, 0.0, 1.1]])
        normal = np.array([0.0, 0.0, 1.0])
        gap = compute_contact_gap(master, slave, normal)
        assert gap[0] == pytest.approx(0.1, abs=1e-12)

    def test_gap_with_displacement(self):
        """Displacement changes the gap."""
        master = np.array([[0.0, 0.0, 1.0]])
        slave = np.array([[0.0, 0.0, 1.1]])
        normal = np.array([0.0, 0.0, 1.0])
        # Slave moves towards master by 0.05
        slave_disp = np.array([[0.0, 0.0, -0.05]])
        gap = compute_contact_gap(
            master, slave, normal, slave_disp=slave_disp,
        )
        assert gap[0] == pytest.approx(0.05, abs=1e-12)

    def test_penetration_negative_gap(self):
        """Penetration produces negative gap."""
        master = np.array([[0.0, 0.0, 1.0]])
        slave = np.array([[0.0, 0.0, 1.0]])  # initially touching
        normal = np.array([0.0, 0.0, 1.0])
        slave_disp = np.array([[0.0, 0.0, -0.02]])
        gap = compute_contact_gap(
            master, slave, normal, slave_disp=slave_disp,
        )
        assert gap[0] == pytest.approx(-0.02, abs=1e-12)

    def test_multiple_nodes(self):
        """Works with multiple contact node pairs."""
        n = 5
        master = np.zeros((n, 3))
        slave = np.zeros((n, 3))
        slave[:, 2] = np.linspace(0.05, 0.25, n)
        normal = np.array([0.0, 0.0, 1.0])
        gap = compute_contact_gap(master, slave, normal)
        np.testing.assert_allclose(gap, slave[:, 2], atol=1e-12)

    def test_explicit_initial_gap(self):
        """Explicit initial_gap overrides coordinate-based computation."""
        master = np.array([[0.0, 0.0, 0.0]])
        slave = np.array([[0.0, 0.0, 0.5]])
        normal = np.array([0.0, 0.0, 1.0])
        gap = compute_contact_gap(
            master, slave, normal, initial_gap=np.array([0.1]),
        )
        # 0.1 (initial) + 0.0 (no displacement)
        assert gap[0] == pytest.approx(0.1, abs=1e-12)

    def test_invalid_shape_raises(self):
        """Invalid coordinate shapes raise ValueError."""
        with pytest.raises(ValueError, match="must be"):
            compute_contact_gap(
                np.array([0.0, 0.0, 0.0]),  # 1D instead of 2D
                np.array([[0.0, 0.0, 1.0]]),
                np.array([0.0, 0.0, 1.0]),
            )


# =========================================================================
# 7. Augmented Lagrangian traction tests (no FEniCSx)
# =========================================================================
class TestAugmentedLagrangianTraction:
    """Test augmented Lagrangian contact traction computation."""

    def test_open_contact(self):
        """Open gap (positive) with zero multiplier -> zero traction."""
        gap = np.array([0.1, 0.2, 0.5])
        lam = np.zeros(3)
        r = 1e6
        p = compute_contact_traction_augmented_lagrangian(gap, lam, r)
        np.testing.assert_allclose(p, 0.0)

    def test_closed_contact(self):
        """Penetration (negative gap) -> positive traction."""
        gap = np.array([-0.01, -0.02])
        lam = np.zeros(2)
        r = 1e8
        p = compute_contact_traction_augmented_lagrangian(gap, lam, r)
        expected = np.array([1e8 * 0.01, 1e8 * 0.02])
        np.testing.assert_allclose(p, expected)

    def test_with_nonzero_multiplier(self):
        """Nonzero multiplier shifts the traction threshold."""
        gap = np.array([0.005])  # small positive gap
        lam = np.array([1e6])   # existing multiplier
        r = 1e8
        p = compute_contact_traction_augmented_lagrangian(gap, lam, r)
        # lambda + r * (-gap) = 1e6 + 1e8 * (-0.005) = 1e6 - 5e5 = 5e5 > 0
        assert p[0] == pytest.approx(5e5)

    def test_invalid_augmentation_param(self):
        """Non-positive augmentation parameter raises ValueError."""
        with pytest.raises(ValueError, match="augmentation_param"):
            compute_contact_traction_augmented_lagrangian(
                np.array([0.0]), np.array([0.0]), 0.0,
            )


# =========================================================================
# 8. Coulomb friction traction tests (no FEniCSx)
# =========================================================================
class TestCoulombFriction:
    """Test Coulomb friction traction computation."""

    def test_stick_regime(self):
        """Small slip stays in stick regime (trial < mu*pn)."""
        slip = np.array([0.0001])
        p_n = np.array([1e6])
        mu = 0.3
        r = 1e6
        t = compute_coulomb_friction_traction(slip, p_n, mu, r)
        # trial = r * slip = 1e6 * 0.0001 = 100
        # limit = 0.3 * 1e6 = 3e5
        # 100 < 3e5 -> stick
        assert t[0] == pytest.approx(r * slip[0])

    def test_slip_regime(self):
        """Large slip exceeds friction limit -> capped."""
        slip = np.array([1.0])
        p_n = np.array([1e6])
        mu = 0.2
        r = 1e6
        t = compute_coulomb_friction_traction(slip, p_n, mu, r)
        # trial = r * slip = 1e6
        # limit = 0.2 * 1e6 = 2e5
        # 1e6 > 2e5 -> slip: t = 2e5 * sign(trial)
        assert t[0] == pytest.approx(mu * p_n[0])

    def test_zero_normal_traction(self):
        """Zero normal traction -> zero friction traction (no contact)."""
        slip = np.array([0.5])
        p_n = np.array([0.0])
        mu = 0.3
        r = 1e6
        t = compute_coulomb_friction_traction(slip, p_n, mu, r)
        assert t[0] == pytest.approx(0.0)

    def test_negative_mu_raises(self):
        """Negative friction coefficient raises ValueError."""
        with pytest.raises(ValueError, match="friction_mu"):
            compute_coulomb_friction_traction(
                np.array([0.1]), np.array([1e6]), -0.1, 1e6,
            )

    def test_2d_slip_vector(self):
        """Works with 2D tangential slip vectors (n, 2)."""
        slip = np.array([[0.0001, 0.0002]])
        p_n = np.array([1e6])
        mu = 0.3
        r = 1e6
        t = compute_coulomb_friction_traction(slip, p_n, mu, r)
        assert t.shape == (1, 2)


# =========================================================================
# 9. Contact status classification tests (no FEniCSx)
# =========================================================================
class TestContactStatusClassification:
    """Test classify_contact_status function."""

    def test_open_contact(self):
        """Positive gap -> status 0 (open)."""
        gap = np.array([0.1, 0.2])
        slip = np.array([0.0, 0.0])
        p_n = np.array([0.0, 0.0])
        status = classify_contact_status(gap, slip, p_n, friction_mu=0.3)
        np.testing.assert_array_equal(status, [0, 0])

    def test_stick_contact(self):
        """Closed gap, no slip -> status 1 (stick)."""
        gap = np.array([-0.01, 0.0])
        slip = np.array([0.0, 0.0])
        p_n = np.array([1e6, 1e6])
        status = classify_contact_status(gap, slip, p_n, friction_mu=0.3)
        np.testing.assert_array_equal(status, [1, 1])

    def test_slip_contact(self):
        """Closed gap, significant slip -> status 2 (slip)."""
        gap = np.array([-0.01])
        slip = np.array([0.001])
        p_n = np.array([1e6])
        status = classify_contact_status(gap, slip, p_n, friction_mu=0.3, slip_tol=1e-6)
        assert status[0] == 2

    def test_mixed_status(self):
        """Mixed open/stick/slip nodes."""
        gap = np.array([0.1, -0.01, -0.01])
        slip = np.array([0.0, 0.0, 0.005])
        p_n = np.array([0.0, 1e6, 1e6])
        status = classify_contact_status(
            gap, slip, p_n, friction_mu=0.3, slip_tol=1e-6,
        )
        assert status[0] == 0  # open
        assert status[1] == 1  # stick
        assert status[2] == 2  # slip


# =========================================================================
# 10. Frequency shift from thermal effects (no FEniCSx)
# =========================================================================
class TestFrequencyShiftThermal:
    """Test compute_frequency_shift_thermal function."""

    def test_zero_temperature_rise(self):
        """No temperature rise -> no frequency shift."""
        df = compute_frequency_shift_thermal(
            f0_hz=20000.0, delta_t_avg=0.0, alpha_cte=8.6e-6,
        )
        assert df == pytest.approx(0.0)

    def test_positive_temperature_rise_decreases_frequency(self):
        """Temperature rise decreases frequency (negative shift)."""
        df = compute_frequency_shift_thermal(
            f0_hz=20000.0, delta_t_avg=50.0, alpha_cte=8.6e-6,
        )
        assert df < 0.0

    def test_formula_correctness(self):
        """Check the formula: df = -0.5*f0*(alpha_E + alpha)*dT."""
        f0 = 20000.0
        dT = 100.0
        alpha = 8.6e-6
        alpha_e = 3e-4
        df = compute_frequency_shift_thermal(f0, dT, alpha, alpha_e)
        expected = -0.5 * f0 * (alpha_e + alpha) * dT
        assert df == pytest.approx(expected, rel=1e-10)

    def test_typical_titanium_shift(self):
        """Typical Ti-6Al-4V: ~100 C rise -> ~300 Hz shift at 20 kHz."""
        df = compute_frequency_shift_thermal(
            f0_hz=20000.0,
            delta_t_avg=100.0,
            alpha_cte=8.6e-6,
            alpha_e=3e-4,
        )
        # df = -0.5 * 20000 * (3e-4 + 8.6e-6) * 100 = -0.5 * 20000 * 3.086e-4 * 100
        # df = -0.5 * 20000 * 0.03086 = -308.6 Hz
        assert -350 < df < -250


# =========================================================================
# 11. Rayleigh damping coefficients (no FEniCSx)
# =========================================================================
class TestRayleighDamping:
    """Test compute_rayleigh_damping_coefficients function."""

    def test_basic_coefficients(self):
        """Compute alpha and beta at 20 kHz with 0.5% damping."""
        alpha, beta = compute_rayleigh_damping_coefficients(20000.0, 0.005)
        omega = 2 * math.pi * 20000.0
        assert alpha == pytest.approx(2 * 0.005 * omega, rel=1e-10)
        assert beta == pytest.approx(2 * 0.005 / omega, rel=1e-10)

    def test_zero_damping(self):
        """Zero damping ratio gives zero coefficients."""
        alpha, beta = compute_rayleigh_damping_coefficients(20000.0, 0.0)
        assert alpha == pytest.approx(0.0)
        assert beta == pytest.approx(0.0)

    def test_negative_frequency_raises(self):
        """Negative frequency raises ValueError."""
        with pytest.raises(ValueError, match="f_center_hz"):
            compute_rayleigh_damping_coefficients(-20000.0, 0.005)

    def test_negative_damping_raises(self):
        """Negative damping ratio raises ValueError."""
        with pytest.raises(ValueError, match="damping_ratio"):
            compute_rayleigh_damping_coefficients(20000.0, -0.005)


# =========================================================================
# 12. PZT material properties (no FEniCSx)
# =========================================================================
class TestPZTProperties:
    """Test PZT material property retrieval."""

    def test_pzt4_properties(self):
        """PZT-4 properties are returned correctly."""
        props = get_pzt_properties("PZT-4")
        assert props["rho"] == 7500.0
        assert props["c_E"].shape == (6, 6)
        assert props["e"].shape == (3, 6)
        assert props["eps_S"].shape == (3, 3)

    def test_pzt8_properties(self):
        """PZT-8 properties are returned correctly."""
        props = get_pzt_properties("PZT-8")
        assert props["rho"] == 7600.0
        assert props["c_E"].shape == (6, 6)

    def test_unknown_pzt_raises(self):
        """Unknown PZT name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown PZT material"):
            get_pzt_properties("PZT-99")

    def test_pzt4_stiffness_symmetry(self):
        """PZT-4 elastic stiffness matrix is symmetric."""
        c_E = PZT4_PROPERTIES["c_E"]
        np.testing.assert_allclose(c_E, c_E.T, atol=1e-6)

    def test_pzt8_stiffness_symmetry(self):
        """PZT-8 elastic stiffness matrix is symmetric."""
        c_E = PZT8_PROPERTIES["c_E"]
        np.testing.assert_allclose(c_E, c_E.T, atol=1e-6)

    def test_pzt4_permittivity_positive_definite(self):
        """PZT-4 permittivity matrix is positive definite."""
        eps_S = PZT4_PROPERTIES["eps_S"]
        eigenvalues = np.linalg.eigvalsh(eps_S)
        assert np.all(eigenvalues > 0)

    def test_get_pzt_returns_copy(self):
        """get_pzt_properties returns a copy (not a reference)."""
        a = get_pzt_properties("PZT-4")
        b = get_pzt_properties("PZT-4")
        assert a is not b


# =========================================================================
# 13. Mode classification helper (no FEniCSx)
# =========================================================================
class TestModeClassification:
    """Test SolverB._classify_modes static method."""

    def test_longitudinal_mode(self):
        """Mode with dominant Z displacement is longitudinal."""
        mode = np.zeros((1, 10, 3))
        mode[0, :, 2] = np.linspace(0, 1, 10)  # Z dominant
        types = SolverB._classify_modes(mode)
        assert types[0] == "longitudinal"

    def test_flexural_mode(self):
        """Mode with dominant X displacement is flexural."""
        mode = np.zeros((1, 10, 3))
        mode[0, :, 0] = np.linspace(-1, 1, 10)  # X dominant
        types = SolverB._classify_modes(mode)
        assert types[0] == "flexural"

    def test_compound_mode(self):
        """Mode with similar X, Y, Z components is compound."""
        mode = np.zeros((1, 10, 3))
        mode[0, :, 0] = np.ones(10) * 0.35
        mode[0, :, 1] = np.ones(10) * 0.35
        mode[0, :, 2] = np.ones(10) * 0.30
        types = SolverB._classify_modes(mode)
        assert types[0] == "compound"

    def test_multiple_modes(self):
        """Classification works for multiple modes."""
        modes = np.zeros((3, 10, 3))
        modes[0, :, 2] = 1.0   # longitudinal
        modes[1, :, 0] = 1.0   # flexural
        modes[2, :, :] = 0.4   # compound
        types = SolverB._classify_modes(modes)
        assert len(types) == 3
        assert types[0] == "longitudinal"
        assert types[1] == "flexural"


# =========================================================================
# 14. Effective mass ratio computation (no FEniCSx)
# =========================================================================
class TestEffectiveMassRatios:
    """Test SolverB._compute_effective_mass_ratios static method."""

    def test_uniform_mode(self):
        """Uniform displacement -> high effective mass ratio."""
        mode = np.ones((1, 10, 3))
        ratios = SolverB._compute_effective_mass_ratios(mode)
        assert ratios.shape == (1,)
        # For uniform: sum(phi) = 30, sum(phi^2) = 30, n = 30
        # ratio = 30^2 / (30 * 30) = 1.0
        assert ratios[0] == pytest.approx(1.0)

    def test_antisymmetric_mode(self):
        """Antisymmetric mode -> low effective mass ratio."""
        mode = np.zeros((1, 10, 3))
        mode[0, :5, 2] = 1.0
        mode[0, 5:, 2] = -1.0
        ratios = SolverB._compute_effective_mass_ratios(mode)
        # Sum cancels out -> very low ratio
        assert ratios[0] < 0.1

    def test_zero_mode(self):
        """Zero mode shape -> zero ratio."""
        mode = np.zeros((1, 10, 3))
        ratios = SolverB._compute_effective_mass_ratios(mode)
        assert ratios[0] == pytest.approx(0.0)


# =========================================================================
# 15. Integration tests (require FEniCSx)
# =========================================================================
@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationModal:
    """Integration tests for SolverB modal analysis (requires FEniCSx)."""

    def test_modal_analysis_runs(self, simple_mesh):
        """Modal analysis completes without error on a simple mesh."""
        solver = SolverB()
        config = ModalConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
            n_modes=3,
            target_frequency_hz=20000.0,
        )
        result = solver.modal_analysis(config)
        assert isinstance(result, ModalResult)
        assert result.solver_name == "fenicsx"
        assert len(result.frequencies_hz) <= 3
        assert result.solve_time_s > 0

    def test_modal_result_shapes(self, simple_mesh):
        """Result arrays have consistent shapes."""
        solver = SolverB()
        config = ModalConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
            n_modes=5,
        )
        result = solver.modal_analysis(config)
        n_modes = len(result.frequencies_hz)
        assert result.mode_shapes.shape[0] == n_modes
        assert len(result.mode_types) == n_modes
        assert len(result.effective_mass_ratios) == n_modes


@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationHarmonic:
    """Integration tests for SolverB harmonic analysis."""

    def test_harmonic_analysis_runs(self, simple_mesh):
        """Harmonic analysis completes without error."""
        solver = SolverB()
        config = HarmonicConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
            freq_min_hz=18000.0,
            freq_max_hz=22000.0,
            n_freq_points=5,
        )
        result = solver.harmonic_analysis(config)
        assert isinstance(result, HarmonicResult)
        assert len(result.frequencies_hz) == 5


@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationStatic:
    """Integration tests for SolverB static analysis."""

    def test_static_analysis_runs(self, simple_mesh):
        """Static analysis completes without error."""
        solver = SolverB()
        config = StaticConfig(
            mesh=simple_mesh,
            material_name="Titanium Ti-6Al-4V",
        )
        result = solver.static_analysis(config)
        assert isinstance(result, StaticResult)
        assert result.solver_name == "fenicsx"


@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationThermal:
    """Integration tests for SolverB thermal analysis."""

    def test_thermal_analysis_runs(self, simple_mesh):
        """Thermal analysis completes without error."""
        solver = SolverB()
        config = ThermalConfig(
            mesh=simple_mesh,
            n_steps=5,
            dt=0.01,
        )
        result = solver.thermal_analysis(config)
        assert isinstance(result, ThermalResult)
        assert result.solver_name == "fenicsx"
        assert result.max_temperature_c >= 25.0  # at least ambient
        assert len(result.time_steps) == 6  # n_steps + 1


@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationContact:
    """Integration tests for SolverB contact analysis."""

    def test_contact_analysis_runs(self, contact_mesh):
        """Contact analysis completes without error."""
        solver = SolverB()
        config = ContactConfig(
            mesh=contact_mesh,
            contact_pairs=[{
                "master": "master_contact",
                "slave": "slave_contact",
                "mu": 0.2,
            }],
            bolt_preload_n=10000.0,
            max_augmentation_iters=3,
        )
        result = solver.contact_analysis(config)
        assert isinstance(result, ContactResult)
        assert result.bolt_force_n == 10000.0


@pytest.mark.skipif(not HAS_FENICSX, reason="FEniCSx not installed")
class TestSolverBIntegrationImpedance:
    """Integration tests for SolverB impedance spectrum."""

    def test_impedance_spectrum_runs(self, simple_mesh):
        """Impedance spectrum computation completes without error."""
        solver = SolverB()
        config = PiezoConfig(
            mesh=simple_mesh,
            n_freq_points=5,
        )
        result = solver.impedance_spectrum(config)
        assert isinstance(result, ImpedanceResult)
        assert len(result.frequencies_hz) == 5
