"""Tests for FEA configuration and result dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
    HarmonicConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ModalResult,
    HarmonicResult,
    StaticResult,
    FatigueResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_interface import (
    SolverInterface,
)


class TestFEAMesh:
    def test_create_mesh(self):
        mesh = FEAMesh(
            nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="TET4",
            node_sets={"bottom": np.array([0, 1])},
            element_sets={},
            surface_tris=np.array([[0, 1, 2]]),
            mesh_stats={"num_nodes": 4, "num_elements": 1},
        )
        assert mesh.nodes.shape == (4, 3)
        assert mesh.element_type == "TET4"
        assert len(mesh.node_sets["bottom"]) == 2

    def test_mesh_n_dof(self):
        mesh = FEAMesh(
            nodes=np.zeros((100, 3)),
            elements=np.zeros((50, 10), dtype=int),
            element_type="TET10",
            node_sets={},
            element_sets={},
            surface_tris=np.zeros((20, 3), dtype=int),
            mesh_stats={},
        )
        assert mesh.n_dof == 300  # 100 nodes * 3 DOF


class TestModalConfig:
    def test_defaults(self):
        mesh = FEAMesh(
            nodes=np.zeros((4, 3)),
            elements=np.zeros((1, 4), dtype=int),
            element_type="TET4",
            node_sets={}, element_sets={},
            surface_tris=np.zeros((1, 3), dtype=int),
            mesh_stats={},
        )
        config = ModalConfig(mesh=mesh, material_name="Titanium Ti-6Al-4V")
        assert config.n_modes == 20
        assert config.target_frequency_hz == 20000.0
        assert config.boundary_conditions == "free-free"


class TestModalResult:
    def test_create_result(self):
        result = ModalResult(
            frequencies_hz=np.array([19500.0, 20100.0]),
            mode_shapes=np.zeros((2, 30)),
            mode_types=["longitudinal", "flexural"],
            effective_mass_ratios=np.array([0.85, 0.05]),
            mesh=None,
            solve_time_s=1.5,
            solver_name="SolverA",
        )
        assert len(result.frequencies_hz) == 2
        assert result.mode_types[0] == "longitudinal"


class TestSolverInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            SolverInterface()  # Cannot instantiate abstract class
