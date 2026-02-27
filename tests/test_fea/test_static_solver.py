"""Tests for SolverA static stress analysis.

Uses a Ti-6Al-4V uniform cylindrical bar, diameter 25 mm, length 80 mm,
mesh_size=6.0, order=2.  A 10 kN tensile force is applied in the Y
direction at the top face, with the bottom face fixed.

Analytical reference
--------------------
For a uniform bar under uniaxial tension:

    sigma_yy = F / A
    u_y(free end) = F * L / (E * A)

For Ti-6Al-4V with D=25 mm, L=80 mm, F=10 kN:
    A = pi * (0.0125)^2 = 4.9087e-4 m^2
    sigma = 10000 / 4.9087e-4 = 20.37 MPa
    u_y = 10000 * 0.08 / (113.8e9 * 4.9087e-4) = 1.433e-8 m
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import StaticResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA

# ---------------------------------------------------------------------------
# Material constants and analytical values
# ---------------------------------------------------------------------------
_TI_E = 113.8e9        # Young's modulus [Pa]
_TI_NU = 0.342         # Poisson's ratio
_TI_RHO = 4430.0       # Density [kg/m^3]

_DIAMETER_MM = 25.0
_LENGTH_MM = 80.0
_RADIUS_M = _DIAMETER_MM / 2.0 / 1000.0
_LENGTH_M = _LENGTH_MM / 1000.0
_AREA_M2 = np.pi * _RADIUS_M ** 2

_FORCE_N = 10000.0  # 10 kN

_SIGMA_ANALYTICAL = _FORCE_N / _AREA_M2  # ~20.37 MPa
_U_ANALYTICAL = _FORCE_N * _LENGTH_M / (_TI_E * _AREA_M2)  # ~1.433e-8 m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cylinder_mesh() -> FEAMesh:
    """TET10 mesh of a uniform cylinder: D=25mm, L=80mm."""
    mesher = GmshMesher()
    return mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={
            "diameter_mm": _DIAMETER_MM,
            "length_mm": _LENGTH_MM,
        },
        mesh_size=6.0,
        order=2,
    )


@pytest.fixture(scope="module")
def tension_config(cylinder_mesh: FEAMesh) -> StaticConfig:
    """StaticConfig for a tensile load test."""
    return StaticConfig(
        mesh=cylinder_mesh,
        material_name="Titanium Ti-6Al-4V",
        loads=[
            {
                "type": "force",
                "node_set": "top_face",
                "direction": [0, 1, 0],
                "magnitude": _FORCE_N,
            },
        ],
        boundary_conditions=[
            {
                "type": "fixed",
                "node_set": "bottom_face",
            },
        ],
    )


@pytest.fixture(scope="module")
def tension_result(tension_config: StaticConfig) -> StaticResult:
    """Run static analysis on the tensile load configuration.

    Cached at module scope so all tests share the same solve.
    """
    solver = SolverA()
    return solver.static_analysis(tension_config)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------
class TestStaticResultFields:
    """Test that all StaticResult fields are populated correctly."""

    def test_static_result_fields(self, tension_result: StaticResult):
        """All StaticResult fields should be populated correctly."""
        assert isinstance(tension_result, StaticResult)
        assert tension_result.displacement is not None
        assert tension_result.stress_vm is not None
        assert tension_result.stress_tensor is not None
        assert tension_result.max_stress_mpa > 0.0
        assert tension_result.mesh is not None
        assert tension_result.solve_time_s > 0.0
        assert tension_result.solver_name == "SolverA"


class TestDisplacementUnderTension:
    """Test displacement for a uniform bar under tension."""

    def test_displacement_under_tension(
        self, tension_result: StaticResult, cylinder_mesh: FEAMesh
    ):
        """Free-end Y displacement should match F*L/(E*A) within 5%.

        The penalty BC keeps fixed-face DOFs near zero but not exactly.
        We compare the mean Y-displacement at the top face with the
        analytical value.
        """
        top_nodes = cylinder_mesh.node_sets["top_face"]
        u = tension_result.displacement

        # Y-displacement at top face nodes
        u_y_top = u[3 * np.asarray(top_nodes, dtype=np.int64) + 1]
        mean_u_y = np.mean(u_y_top)

        rel_error = abs(mean_u_y - _U_ANALYTICAL) / _U_ANALYTICAL
        assert rel_error < 0.05, (
            f"Mean top-face Y displacement {mean_u_y:.6e} m differs from "
            f"analytical {_U_ANALYTICAL:.6e} m by {rel_error * 100:.2f}% "
            f"(limit 5%)"
        )


class TestStressUnderTension:
    """Test stress for a uniform bar under tension."""

    def test_stress_under_tension(self, tension_result: StaticResult):
        """Median Von Mises stress should approximate F/A within 5%.

        We use the median to avoid boundary effects near the fixed face.
        """
        stress_vm = tension_result.stress_vm
        median_vm_pa = np.median(stress_vm)

        rel_error = abs(median_vm_pa - _SIGMA_ANALYTICAL) / _SIGMA_ANALYTICAL
        assert rel_error < 0.05, (
            f"Median Von Mises stress {median_vm_pa / 1e6:.4f} MPa differs "
            f"from analytical {_SIGMA_ANALYTICAL / 1e6:.4f} MPa "
            f"by {rel_error * 100:.2f}% (limit 5%)"
        )


class TestFixedFaceDisplacement:
    """Test that fixed face nodes have near-zero displacement."""

    def test_zero_displacement_at_fixed_face(
        self, tension_result: StaticResult, cylinder_mesh: FEAMesh
    ):
        """Fixed face nodes should have ~0 displacement (penalty BC)."""
        bottom_nodes = cylinder_mesh.node_sets["bottom_face"]
        u = tension_result.displacement

        for node_idx in bottom_nodes:
            for d in range(3):
                dof = 3 * int(node_idx) + d
                assert abs(u[dof]) < 1e-12, (
                    f"Fixed node {node_idx} DOF {d} has displacement "
                    f"{u[dof]:.6e}, expected ~0"
                )


class TestVonMisesPositive:
    """Test Von Mises stress is non-negative."""

    def test_stress_vm_positive(self, tension_result: StaticResult):
        """Von Mises stress should be non-negative for all elements."""
        assert np.all(tension_result.stress_vm >= 0.0), (
            "Found negative Von Mises stress values"
        )


class TestStressTensorShape:
    """Test stress tensor shape."""

    def test_stress_tensor_shape(
        self, tension_result: StaticResult, cylinder_mesh: FEAMesh
    ):
        """stress_tensor should be (n_elements, 6)."""
        n_elements = cylinder_mesh.elements.shape[0]
        assert tension_result.stress_tensor.shape == (n_elements, 6), (
            f"Expected stress_tensor shape ({n_elements}, 6), "
            f"got {tension_result.stress_tensor.shape}"
        )


class TestGravityLoad:
    """Test gravity load produces nonzero displacement."""

    def test_gravity_load_produces_displacement(
        self, cylinder_mesh: FEAMesh
    ):
        """Gravity load should produce nonzero displacement."""
        config = StaticConfig(
            mesh=cylinder_mesh,
            material_name="Titanium Ti-6Al-4V",
            loads=[
                {
                    "type": "gravity",
                    "direction": [0, -1, 0],
                    "magnitude": 9.81,
                },
            ],
            boundary_conditions=[
                {
                    "type": "fixed",
                    "node_set": "bottom_face",
                },
            ],
        )
        solver = SolverA()
        result = solver.static_analysis(config)

        # Displacement should be nonzero
        max_disp = np.max(np.abs(result.displacement))
        assert max_disp > 0.0, (
            "Gravity load produced zero displacement"
        )

        # Y-displacement should be predominantly negative (gravity pulls down)
        u_y = result.displacement[1::3]
        # At least some nodes should have negative Y displacement
        assert np.any(u_y < 0.0), (
            "Expected negative Y-displacement under downward gravity"
        )


class TestEquilibriumCheck:
    """Test that reaction forces balance applied forces."""

    def test_equilibrium_check(
        self, tension_result: StaticResult, cylinder_mesh: FEAMesh,
        tension_config: StaticConfig,
    ):
        """Sum of reaction forces should approximately equal applied force.

        Reaction force = K * u at constrained DOFs. The sum of reaction
        forces in Y should equal -F_applied (Newton's third law).
        """
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler import (
            GlobalAssembler,
        )

        assembler = GlobalAssembler(cylinder_mesh, "Titanium Ti-6Al-4V")
        K, _M = assembler.assemble()

        # Compute internal forces: f_int = K * u
        f_int = K @ tension_result.displacement

        # Reaction forces at the fixed face
        bottom_nodes = cylinder_mesh.node_sets["bottom_face"]
        reaction_y = 0.0
        for node_idx in bottom_nodes:
            dof_y = 3 * int(node_idx) + 1
            reaction_y += f_int[dof_y]

        # The reaction force in Y at the fixed face should balance
        # the applied force: reaction_y ~ -F_applied
        # (Force applied upward at top, reaction pushes down at bottom)
        rel_error = abs(abs(reaction_y) - _FORCE_N) / _FORCE_N
        assert rel_error < 0.05, (
            f"Reaction force {reaction_y:.2f} N does not balance "
            f"applied force {_FORCE_N:.2f} N "
            f"(relative error {rel_error * 100:.2f}%, limit 5%)"
        )


class TestMaxStressLocation:
    """Test that max_stress_mpa matches max of stress_vm."""

    def test_max_stress_at_correct_location(
        self, tension_result: StaticResult
    ):
        """max_stress_mpa should match max(stress_vm) converted to MPa."""
        expected_max_mpa = np.max(tension_result.stress_vm) / 1e6
        assert abs(tension_result.max_stress_mpa - expected_max_mpa) < 1e-6, (
            f"max_stress_mpa={tension_result.max_stress_mpa:.6f} does not "
            f"match max(stress_vm)/1e6={expected_max_mpa:.6f}"
        )


class TestPressureLoad:
    """Test pressure load application."""

    def test_pressure_load_produces_displacement(
        self, cylinder_mesh: FEAMesh
    ):
        """Pressure load should produce nonzero displacement."""
        config = StaticConfig(
            mesh=cylinder_mesh,
            material_name="Titanium Ti-6Al-4V",
            loads=[
                {
                    "type": "pressure",
                    "node_set": "top_face",
                    "magnitude": 1e6,  # 1 MPa
                },
            ],
            boundary_conditions=[
                {
                    "type": "fixed",
                    "node_set": "bottom_face",
                },
            ],
        )
        solver = SolverA()
        result = solver.static_analysis(config)

        # Displacement should be nonzero
        max_disp = np.max(np.abs(result.displacement))
        assert max_disp > 0.0, "Pressure load produced zero displacement"

        # Y-displacement at the top face should be positive
        top_nodes = cylinder_mesh.node_sets["top_face"]
        u_y_top = result.displacement[
            3 * np.asarray(top_nodes, dtype=np.int64) + 1
        ]
        assert np.mean(u_y_top) > 0.0, (
            "Expected positive Y-displacement at top face under pressure"
        )
