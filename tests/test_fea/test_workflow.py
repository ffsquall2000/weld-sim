"""Tests for the full-stack FEA analysis workflow.

Uses mock/synthetic data throughout -- actual FEA solves are too slow
for unit tests.  Tests cover:

1. ``compute_gain_chain`` standalone function (post_processing)
2. ``WorkflowConfig`` validation
3. ``AnalysisWorkflow`` orchestration with mocked solver
4. Gain chain computation within the workflow
5. Summary report generation
6. Error handling and edge cases
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import scipy.sparse as sp

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembly_builder import (
    AssemblyConfig,
    AssemblyResult,
    ComponentMesh,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.post_processing import (
    GainChainResult,
    compute_gain_chain,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    FatigueResult,
    HarmonicResult,
    ModalResult,
    StaticResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow import (
    AnalysisWorkflow,
    WorkflowConfig,
    WorkflowResult,
)


# ===================================================================
# Helpers: synthetic data factories
# ===================================================================


def _make_simple_mesh(n_nodes: int = 20, n_elements: int = 8) -> FEAMesh:
    """Create a synthetic FEAMesh for testing.

    Produces a small mesh with random node positions and fake
    TET10-like connectivity.  The mesh is not physically valid
    but has the correct data shape for the workflow.
    """
    nodes = np.random.rand(n_nodes, 3).astype(np.float64)
    # Spread Y coordinates to distinguish top from bottom
    nodes[:, 1] = np.linspace(0.0, 0.1, n_nodes)

    # TET10 connectivity: 10 nodes per element
    elements = np.zeros((n_elements, 10), dtype=int)
    for i in range(n_elements):
        elements[i] = np.random.choice(n_nodes, 10, replace=False)

    bottom_nodes = np.array([0, 1, 2], dtype=int)
    top_nodes = np.array([n_nodes - 3, n_nodes - 2, n_nodes - 1], dtype=int)

    return FEAMesh(
        nodes=nodes,
        elements=elements,
        element_type="TET10",
        node_sets={
            "bottom_face": bottom_nodes,
            "top_face": top_nodes,
        },
        element_sets={
            "all": np.arange(n_elements, dtype=int),
        },
        surface_tris=np.array([[0, 1, 2]], dtype=int),
        mesh_stats={"n_nodes": n_nodes, "n_elements": n_elements},
    )


def _make_component_dict(
    name: str,
    n_nodes: int = 20,
    n_elements: int = 8,
) -> dict:
    """Create a component dict suitable for WorkflowConfig."""
    mesh = _make_simple_mesh(n_nodes, n_elements)
    return {
        "name": name,
        "mesh": mesh,
        "material_name": "Ti-6Al-4V",
        "interface_nodes": {
            "bottom": np.array([0, 1, 2], dtype=int),
            "top": np.array([n_nodes - 3, n_nodes - 2, n_nodes - 1], dtype=int),
        },
    }


def _make_assembly_result(
    components: list[dict],
) -> AssemblyResult:
    """Create a synthetic AssemblyResult."""
    total_nodes = sum(c["mesh"].nodes.shape[0] for c in components)
    n_dof = total_nodes * 3

    K = sp.eye(n_dof, format="csr", dtype=np.float64) * 1e9
    M = sp.eye(n_dof, format="csr", dtype=np.float64) * 7800.0

    dof_map = {}
    node_offset_map = {}
    running_offset = 0
    running_node = 0
    for c in components:
        n = c["mesh"].nodes.shape[0]
        n_d = n * 3
        dof_map[c["name"]] = np.arange(running_offset, running_offset + n_d)
        node_offset_map[c["name"]] = running_node
        running_offset += n_d
        running_node += n

    return AssemblyResult(
        K_global=K,
        M_global=M,
        n_total_dof=n_dof,
        dof_map=dof_map,
        node_offset_map=node_offset_map,
        interface_pairs=[],
        impedance={c["name"]: 1e7 for c in components},
        transmission_coefficients={},
    )


def _make_modal_result(n_dof: int, n_modes: int = 5) -> ModalResult:
    """Create a synthetic ModalResult."""
    freqs = np.linspace(18000, 22000, n_modes)
    shapes = np.random.rand(n_modes, n_dof)
    return ModalResult(
        frequencies_hz=freqs,
        mode_shapes=shapes,
        mode_types=["longitudinal"] * n_modes,
        effective_mass_ratios=np.random.rand(n_modes, 3),
        mesh=None,
        solve_time_s=0.1,
        solver_name="SolverA",
    )


def _make_harmonic_result(n_dof: int, n_freq: int = 101) -> HarmonicResult:
    """Create a synthetic HarmonicResult with a clear resonance peak."""
    freqs = np.linspace(18000, 22000, n_freq)
    f0 = 20000.0
    zeta = 0.005

    # Build a Lorentzian FRF for each DOF
    r = freqs / f0
    H = 1.0 / (1.0 - r ** 2 + 2j * zeta * r)

    # Displacement amplitudes: all DOFs follow the same FRF, scaled
    disp = np.outer(H, np.random.rand(n_dof) * 1e-6 + 1e-7)

    return HarmonicResult(
        frequencies_hz=freqs,
        displacement_amplitudes=disp.astype(np.complex128),
        contact_face_uniformity=0.92,
        gain=2.5,
        q_factor=100.0,
        mesh=None,
        solve_time_s=0.5,
        solver_name="SolverA",
    )


def _make_static_result(n_dof: int, n_elements: int = 8) -> StaticResult:
    """Create a synthetic StaticResult."""
    return StaticResult(
        displacement=np.random.rand(n_dof) * 1e-6,
        stress_vm=np.random.rand(n_elements) * 1e8,
        stress_tensor=np.random.rand(n_elements, 6) * 1e8,
        max_stress_mpa=150.0,
        mesh=None,
        solve_time_s=0.2,
        solver_name="SolverA",
    )


def _make_fatigue_result(n_elements: int = 8) -> FatigueResult:
    """Create a synthetic FatigueResult."""
    sf = np.random.rand(n_elements) * 3.0 + 1.0
    crit_idx = int(np.argmin(sf))
    return FatigueResult(
        safety_factors=sf,
        min_safety_factor=float(sf[crit_idx]),
        critical_location=np.array([0.01, 0.05, 0.0]),
        estimated_life_cycles=1e12,
        sn_curve_name="Ti-6Al-4V",
    )


# ===================================================================
# Tests: compute_gain_chain (standalone post_processing function)
# ===================================================================


class TestComputeGainChain:
    """Tests for the standalone compute_gain_chain function."""

    def test_single_component_unity_gain(self):
        """When input and output have the same amplitude, gain = 1."""
        n_dof = 30
        displacement = np.ones(n_dof, dtype=np.complex128) * (1.0 + 0.5j)

        interfaces = [
            {
                "name": "horn",
                "input_dofs": np.array([1, 4, 7]),
                "output_dofs": np.array([10, 13, 16]),
            }
        ]

        result = compute_gain_chain(displacement, interfaces)

        assert isinstance(result, GainChainResult)
        assert len(result.components) == 1
        assert result.components[0]["name"] == "horn"
        assert result.components[0]["gain"] == pytest.approx(1.0, abs=1e-10)
        assert result.total_gain == pytest.approx(1.0, abs=1e-10)

    def test_amplification(self):
        """Output amplitude > input amplitude gives gain > 1."""
        displacement = np.zeros(30, dtype=np.complex128)
        displacement[1] = 1.0   # input
        displacement[4] = 1.0
        displacement[7] = 1.0
        displacement[10] = 2.0  # output (2x amplification)
        displacement[13] = 2.0
        displacement[16] = 2.0

        interfaces = [
            {
                "name": "booster",
                "input_dofs": np.array([1, 4, 7]),
                "output_dofs": np.array([10, 13, 16]),
            }
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["gain"] == pytest.approx(2.0)
        assert result.total_gain == pytest.approx(2.0)

    def test_attenuation(self):
        """Output amplitude < input amplitude gives gain < 1."""
        displacement = np.zeros(30, dtype=np.complex128)
        displacement[1] = 4.0
        displacement[4] = 4.0
        displacement[10] = 2.0
        displacement[13] = 2.0

        interfaces = [
            {
                "name": "adapter",
                "input_dofs": np.array([1, 4]),
                "output_dofs": np.array([10, 13]),
            }
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["gain"] == pytest.approx(0.5)
        assert result.total_gain == pytest.approx(0.5)

    def test_multi_component_chain(self):
        """Total gain is the product of individual component gains."""
        displacement = np.zeros(60, dtype=np.complex128)

        # Component A: input=1, output=2 -> gain=2
        displacement[0] = 1.0
        displacement[3] = 2.0

        # Component B: input=2, output=6 -> gain=3
        displacement[6] = 2.0
        displacement[9] = 6.0

        # Component C: input=6, output=3 -> gain=0.5
        displacement[12] = 6.0
        displacement[15] = 3.0

        interfaces = [
            {"name": "transducer", "input_dofs": [0], "output_dofs": [3]},
            {"name": "booster", "input_dofs": [6], "output_dofs": [9]},
            {"name": "horn", "input_dofs": [12], "output_dofs": [15]},
        ]

        result = compute_gain_chain(displacement, interfaces)

        assert len(result.components) == 3
        assert result.components[0]["gain"] == pytest.approx(2.0)
        assert result.components[1]["gain"] == pytest.approx(3.0)
        assert result.components[2]["gain"] == pytest.approx(0.5)
        assert result.total_gain == pytest.approx(3.0)  # 2 * 3 * 0.5

    def test_complex_displacement(self):
        """Gain chain works with complex displacement (uses |U|)."""
        displacement = np.zeros(20, dtype=np.complex128)
        displacement[1] = 3.0 + 4.0j  # |U| = 5
        displacement[4] = 3.0 + 4.0j
        displacement[10] = 6.0 + 8.0j  # |U| = 10
        displacement[13] = 6.0 + 8.0j

        interfaces = [
            {
                "name": "horn",
                "input_dofs": [1, 4],
                "output_dofs": [10, 13],
            }
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["input_amp"] == pytest.approx(5.0)
        assert result.components[0]["output_amp"] == pytest.approx(10.0)
        assert result.components[0]["gain"] == pytest.approx(2.0)

    def test_zero_input_gives_zero_gain(self):
        """Zero input amplitude results in gain=0 (not division error)."""
        displacement = np.zeros(20, dtype=np.complex128)
        displacement[10] = 1.0  # output only

        interfaces = [
            {
                "name": "comp",
                "input_dofs": [1, 4],
                "output_dofs": [10, 13],
            }
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["gain"] == 0.0
        assert result.total_gain == 0.0

    def test_empty_components_raises(self):
        """Empty component list raises ValueError."""
        displacement = np.ones(10, dtype=np.complex128)
        with pytest.raises(ValueError, match="at least one component"):
            compute_gain_chain(displacement, [])

    def test_empty_input_dofs_raises(self):
        """Empty input_dofs raises ValueError."""
        displacement = np.ones(10, dtype=np.complex128)
        interfaces = [
            {"name": "x", "input_dofs": [], "output_dofs": [1, 2]},
        ]
        with pytest.raises(ValueError, match="empty input_dofs"):
            compute_gain_chain(displacement, interfaces)

    def test_empty_output_dofs_raises(self):
        """Empty output_dofs raises ValueError."""
        displacement = np.ones(10, dtype=np.complex128)
        interfaces = [
            {"name": "x", "input_dofs": [0, 1], "output_dofs": []},
        ]
        with pytest.raises(ValueError, match="empty output_dofs"):
            compute_gain_chain(displacement, interfaces)

    def test_real_displacement(self):
        """Works with real-valued displacement arrays (not complex)."""
        displacement = np.array(
            [0.0, 2.0, 0.0, 0.0, 4.0, 0.0], dtype=np.float64
        )

        interfaces = [
            {"name": "bar", "input_dofs": [1], "output_dofs": [4]},
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["gain"] == pytest.approx(2.0)


# ===================================================================
# Tests: WorkflowConfig validation
# ===================================================================


class TestWorkflowConfigValidation:
    """Tests for WorkflowConfig and AnalysisWorkflow validation."""

    def test_empty_components_raises(self):
        """Empty components list triggers validation error."""
        config = WorkflowConfig(
            components=[],
            assembly_config={"coupling_method": "bonded"},
        )
        with pytest.raises(ValueError, match="must not be empty"):
            AnalysisWorkflow(config)

    def test_missing_component_keys_raises(self):
        """Component dict missing required keys raises ValueError."""
        config = WorkflowConfig(
            components=[{"name": "horn"}],  # missing mesh, material, etc.
            assembly_config={"coupling_method": "bonded"},
        )
        with pytest.raises(ValueError, match="missing keys"):
            AnalysisWorkflow(config)

    def test_unknown_analysis_name_raises(self):
        """Unknown analysis name raises ValueError."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal", "foobar"],
        )
        with pytest.raises(ValueError, match="Unknown analyses"):
            AnalysisWorkflow(config)

    def test_fatigue_without_harmonic_raises(self):
        """Fatigue without harmonic raises ValueError."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
            analyses=["fatigue"],
        )
        with pytest.raises(ValueError, match="requires harmonic"):
            AnalysisWorkflow(config)

    def test_valid_config_accepted(self):
        """A properly formed config passes validation."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal", "harmonic"],
        )
        wf = AnalysisWorkflow(config)
        assert wf is not None

    def test_all_analyses_valid(self):
        """All four analysis types are accepted together."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal", "harmonic", "static", "fatigue"],
        )
        wf = AnalysisWorkflow(config)
        assert wf is not None


# ===================================================================
# Tests: AnalysisWorkflow orchestration (mocked solver)
# ===================================================================


class TestAnalysisWorkflowOrchestration:
    """Tests for workflow orchestration logic using mocked solvers."""

    def _make_workflow(
        self,
        analyses: list[str] | None = None,
        n_components: int = 2,
    ) -> tuple[AnalysisWorkflow, list[dict]]:
        """Helper to create a workflow with n_components."""
        if analyses is None:
            analyses = ["modal", "harmonic"]

        comps = []
        for i, name in enumerate(
            ["transducer", "booster", "horn"][:n_components]
        ):
            comps.append(_make_component_dict(name))

        config = WorkflowConfig(
            components=comps,
            assembly_config={
                "coupling_method": "bonded",
                "penalty_factor": 1e3,
                "component_order": [c["name"] for c in comps],
            },
            analyses=analyses,
            frequency_hz=20000.0,
        )

        wf = AnalysisWorkflow(config)
        return wf, comps

    @patch.object(SolverA, "modal_analysis")
    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_modal_and_harmonic_run(
        self, mock_build, mock_harmonic, mock_modal
    ):
        """Modal and harmonic analyses are called when requested."""
        wf, comps = self._make_workflow(analyses=["modal", "harmonic"])

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_modal.return_value = _make_modal_result(n_dof)
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()

        assert isinstance(result, WorkflowResult)
        assert result.assembly is assembly
        assert result.modal is not None
        assert result.harmonic is not None
        assert result.static is None
        assert result.fatigue is None

        mock_build.assert_called_once()
        mock_modal.assert_called_once()
        mock_harmonic.assert_called_once()

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_harmonic_only(self, mock_build, mock_harmonic):
        """Workflow can run harmonic only (no modal)."""
        wf, comps = self._make_workflow(analyses=["harmonic"])

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()

        assert result.modal is None
        assert result.harmonic is not None
        assert result.gain_chain is not None  # gain chain computed from harmonic

    @patch.object(SolverA, "modal_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_modal_only(self, mock_build, mock_modal):
        """Workflow can run modal only (no harmonic, no gain chain)."""
        wf, comps = self._make_workflow(analyses=["modal"])

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_modal.return_value = _make_modal_result(n_dof)

        result = wf.run()

        assert result.modal is not None
        assert result.harmonic is None
        assert result.gain_chain is None  # no harmonic => no gain chain

    @patch.object(SolverA, "static_analysis")
    @patch.object(SolverA, "harmonic_analysis")
    @patch.object(SolverA, "modal_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_all_analyses(
        self, mock_build, mock_modal, mock_harmonic, mock_static
    ):
        """All analyses run when all four are requested."""
        wf, comps = self._make_workflow(
            analyses=["modal", "harmonic", "static", "fatigue"]
        )

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof
        n_nodes = n_dof // 3

        mock_build.return_value = assembly
        mock_modal.return_value = _make_modal_result(n_dof)
        mock_harmonic.return_value = _make_harmonic_result(n_dof)
        mock_static.return_value = _make_static_result(n_dof, n_elements=n_nodes)

        # Patch FatigueAssessor to avoid needing real mesh
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
            ".FatigueAssessor"
        ) as MockFA:
            mock_assessor = MagicMock()
            mock_assessor.assess.return_value = _make_fatigue_result(n_nodes)
            MockFA.return_value = mock_assessor

            result = wf.run()

        assert result.modal is not None
        assert result.harmonic is not None
        assert result.static is not None
        assert result.fatigue is not None
        assert result.gain_chain is not None

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_three_component_chain(self, mock_build, mock_harmonic):
        """Three-component workflow produces gain chain with 3 entries."""
        wf, comps = self._make_workflow(
            analyses=["harmonic"], n_components=3
        )

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()

        assert result.gain_chain is not None
        assert len(result.gain_chain["components"]) == 3
        assert "total_gain" in result.gain_chain
        assert result.gain_chain["total_gain"] > 0


# ===================================================================
# Tests: Summary report generation
# ===================================================================


class TestSummaryReport:
    """Tests for the summary report output."""

    @patch.object(SolverA, "harmonic_analysis")
    @patch.object(SolverA, "modal_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_summary_structure(self, mock_build, mock_modal, mock_harmonic):
        """Summary contains all expected keys."""
        comps = [_make_component_dict("horn")]
        config = WorkflowConfig(
            components=comps,
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal", "harmonic"],
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_modal.return_value = _make_modal_result(n_dof)
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()
        summary = result.summary

        assert summary["status"] == "completed"
        assert "total_time_s" in summary
        assert summary["n_components"] == 1
        assert summary["n_total_dof"] == n_dof
        assert summary["target_frequency_hz"] == 20000.0
        assert "modal" in summary
        assert "harmonic" in summary
        assert "impedance" in summary

    @patch.object(SolverA, "harmonic_analysis")
    @patch.object(SolverA, "modal_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_modal_summary_content(
        self, mock_build, mock_modal, mock_harmonic
    ):
        """Modal summary contains frequency info and mode count."""
        comps = [_make_component_dict("horn")]
        config = WorkflowConfig(
            components=comps,
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal", "harmonic"],
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_modal.return_value = _make_modal_result(n_dof, n_modes=5)
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()
        modal_summary = result.summary["modal"]

        assert modal_summary["n_modes_found"] == 5
        assert "frequency_range_hz" in modal_summary
        assert "closest_to_target_hz" in modal_summary
        assert "solve_time_s" in modal_summary

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_harmonic_summary_content(self, mock_build, mock_harmonic):
        """Harmonic summary contains gain, Q-factor, and uniformity."""
        comps = [_make_component_dict("horn")]
        config = WorkflowConfig(
            components=comps,
            assembly_config={"coupling_method": "bonded"},
            analyses=["harmonic"],
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()
        h_summary = result.summary["harmonic"]

        assert "gain" in h_summary
        assert "q_factor" in h_summary
        assert "uniformity" in h_summary
        assert "solve_time_s" in h_summary

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_gain_chain_summary(self, mock_build, mock_harmonic):
        """Gain chain appears in summary when harmonic is run."""
        comps = [
            _make_component_dict("transducer"),
            _make_component_dict("horn"),
        ]
        config = WorkflowConfig(
            components=comps,
            assembly_config={
                "coupling_method": "bonded",
                "component_order": ["transducer", "horn"],
            },
            analyses=["harmonic"],
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()
        gc_summary = result.summary.get("gain_chain")

        assert gc_summary is not None
        assert "total_gain" in gc_summary
        assert len(gc_summary["components"]) == 2


# ===================================================================
# Tests: WorkflowResult dataclass
# ===================================================================


class TestWorkflowResult:
    """Tests for the WorkflowResult dataclass."""

    def test_defaults(self):
        """Optional fields default to None."""
        assembly = _make_assembly_result(
            [_make_component_dict("horn")]
        )
        result = WorkflowResult(assembly=assembly)

        assert result.assembly is assembly
        assert result.modal is None
        assert result.harmonic is None
        assert result.static is None
        assert result.fatigue is None
        assert result.gain_chain is None
        assert result.summary == {}


# ===================================================================
# Tests: Edge cases and error handling
# ===================================================================


class TestEdgeCases:
    """Edge cases and error handling in the workflow."""

    def test_single_component_workflow(self):
        """Workflow works with a single component."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
            analyses=["modal"],
        )
        # Should pass validation
        wf = AnalysisWorkflow(config)
        assert wf is not None

    def test_workflow_config_default_analyses(self):
        """Default analyses are modal and harmonic."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={"coupling_method": "bonded"},
        )
        assert config.analyses == ["modal", "harmonic"]

    def test_workflow_config_default_frequency(self):
        """Default frequency is 20000 Hz."""
        comp = _make_component_dict("horn")
        config = WorkflowConfig(
            components=[comp],
            assembly_config={},
        )
        assert config.frequency_hz == 20000.0

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_fatigue_skipped_without_harmonic_result(
        self, mock_build, mock_harmonic
    ):
        """If harmonic result is somehow None, fatigue returns None."""
        comps = [_make_component_dict("horn")]
        config = WorkflowConfig(
            components=comps,
            assembly_config={"coupling_method": "bonded"},
            analyses=["harmonic", "fatigue"],
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        # Patch FatigueAssessor
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
            ".FatigueAssessor"
        ) as MockFA:
            mock_assessor = MagicMock()
            mock_assessor.assess.return_value = _make_fatigue_result()
            MockFA.return_value = mock_assessor

            result = wf.run()
            assert result.fatigue is not None

    def test_gain_chain_result_dataclass(self):
        """GainChainResult stores components and total_gain."""
        gc = GainChainResult(
            components=[
                {"name": "horn", "input_amp": 1.0, "output_amp": 2.0, "gain": 2.0}
            ],
            total_gain=2.0,
        )
        assert gc.total_gain == 2.0
        assert gc.components[0]["name"] == "horn"

    @patch.object(SolverA, "harmonic_analysis")
    @patch(
        "ultrasonic_weld_master.plugins.geometry_analyzer.fea.workflow"
        ".AssemblyBuilder.build"
    )
    def test_custom_frequency(self, mock_build, mock_harmonic):
        """Custom target frequency propagates to config."""
        comps = [_make_component_dict("horn")]
        config = WorkflowConfig(
            components=comps,
            assembly_config={"coupling_method": "bonded"},
            analyses=["harmonic"],
            frequency_hz=35000.0,
        )
        wf = AnalysisWorkflow(config)

        assembly = _make_assembly_result(comps)
        n_dof = assembly.n_total_dof

        mock_build.return_value = assembly
        mock_harmonic.return_value = _make_harmonic_result(n_dof)

        result = wf.run()
        assert result.summary["target_frequency_hz"] == 35000.0


# ===================================================================
# Tests: GainChainResult in post_processing
# ===================================================================


class TestGainChainIntegration:
    """Integration tests for gain chain with AmplitudeAnalyzer data."""

    def test_gain_chain_with_lorentzian_data(self):
        """Gain chain from Lorentzian FRF resonance displacement."""
        n_dof = 60
        f0 = 20000.0
        zeta = 0.005

        freqs = np.linspace(18000, 22000, 201)
        r = freqs / f0
        H = 1.0 / (1.0 - r ** 2 + 2j * zeta * r)

        # Resonance index
        idx_res = int(np.argmax(np.abs(H)))
        H_res = H[idx_res]

        # Build displacement: first component DOFs have amplitude 1*H,
        # second component DOFs have amplitude 2*H (gain=2)
        displacement = np.zeros(n_dof, dtype=np.complex128)
        # Component A input DOFs (Y-direction: 1, 4, 7)
        for d in [1, 4, 7]:
            displacement[d] = H_res * 1e-6
        # Component A output / B input DOFs
        for d in [10, 13, 16]:
            displacement[d] = H_res * 2e-6
        # Component B output DOFs
        for d in [25, 28, 31]:
            displacement[d] = H_res * 6e-6

        interfaces = [
            {
                "name": "booster",
                "input_dofs": [1, 4, 7],
                "output_dofs": [10, 13, 16],
            },
            {
                "name": "horn",
                "input_dofs": [25, 28, 31],   # Note: using same ref
                "output_dofs": [25, 28, 31],
            },
        ]

        # Component A gain = 2, Component B gain = 1 (same in/out)
        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["gain"] == pytest.approx(2.0)
        assert result.components[1]["gain"] == pytest.approx(1.0)
        assert result.total_gain == pytest.approx(2.0)

    def test_nonuniform_amplitudes(self):
        """Non-uniform amplitudes are correctly averaged."""
        displacement = np.zeros(20, dtype=np.complex128)
        displacement[0] = 1.0
        displacement[1] = 3.0   # input dofs: [0, 1] -> mean|U| = 2
        displacement[10] = 5.0
        displacement[11] = 7.0  # output dofs: [10, 11] -> mean|U| = 6

        interfaces = [
            {"name": "comp", "input_dofs": [0, 1], "output_dofs": [10, 11]},
        ]

        result = compute_gain_chain(displacement, interfaces)
        assert result.components[0]["input_amp"] == pytest.approx(2.0)
        assert result.components[0]["output_amp"] == pytest.approx(6.0)
        assert result.components[0]["gain"] == pytest.approx(3.0)
