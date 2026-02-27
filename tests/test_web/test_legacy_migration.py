"""Tests for the legacy HEX8 -> Gmsh TET10 migration.

Verifies that:
1. The default analysis path now uses the Gmsh TET10 + SolverA pipeline.
2. Explicitly requesting ``use_gmsh=False`` still works (legacy fallback).
3. The legacy code path emits a ``DeprecationWarning``.
4. API responses from both pipelines have compatible field names.
5. Assembly-level analysis (via the acoustic router) also defaults to Gmsh.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from web.services.fea_service import FEAService


# ---------------------------------------------------------------------------
# Helpers: build lightweight mock objects that satisfy the Gmsh pipeline
# ---------------------------------------------------------------------------

def _make_fake_fea_mesh():
    """Return a mock FEAMesh with the minimum attributes used by FEAService."""
    mesh = MagicMock()
    mesh.nodes = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.0, 0.08, 0.0],
        [0.0, 0.0, 0.01],
    ])
    mesh.elements = np.array([[0, 1, 2, 3]])
    mesh.surface_tris = np.array([[0, 1, 2], [0, 2, 3]])
    mesh.node_sets = {
        "top_face": np.array([2]),
        "bottom_face": np.array([0, 1, 3]),
    }
    mesh.element_sets = {}
    mesh.mesh_stats = {"n_nodes": 4, "n_elements": 1}
    return mesh


def _make_fake_modal_result(n_modes: int = 3):
    """Return a mock ModalResult."""
    result = MagicMock()
    result.frequencies_hz = np.array([18000.0, 20100.0, 22000.0][:n_modes])
    n_nodes = 4
    result.mode_shapes = np.random.randn(n_modes, n_nodes * 3)
    result.solve_time_s = 0.05
    return result


def _make_fake_classification(n_modes: int = 3):
    """Return a mock ClassificationResult."""
    classification = MagicMock()
    modes = []
    freqs = [18000.0, 20100.0, 22000.0][:n_modes]
    for i, f in enumerate(freqs):
        cm = MagicMock()
        cm.frequency_hz = f
        cm.mode_type = "longitudinal" if i == 1 else "flexural"
        cm.effective_mass = np.array([0.1, 0.8, 0.1])
        cm.displacement_ratios = np.array([0.15, 0.7, 0.15])
        modes.append(cm)
    classification.modes = modes
    classification.target_mode_index = 1
    return classification


# ---------------------------------------------------------------------------
# Patch targets -- these are imported inside the methods via lazy imports
# ---------------------------------------------------------------------------
_MESHER_PATH = "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
_SOLVER_PATH = "ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a.SolverA"
_ASSEMBLER_PATH = "ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembler.GlobalAssembler"
_CLASSIFIER_PATH = "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mode_classifier.ModeClassifier"


@pytest.fixture()
def mock_gmsh_pipeline():
    """Patch the entire Gmsh pipeline so tests run without Gmsh installed."""
    fake_mesh = _make_fake_fea_mesh()
    fake_modal = _make_fake_modal_result()
    fake_class = _make_fake_classification()

    # Build a minimal sparse matrix for M
    from scipy import sparse
    n_dof = fake_mesh.nodes.shape[0] * 3
    M = sparse.eye(n_dof, format="csr")
    K = sparse.eye(n_dof, format="csr")

    with (
        patch(_MESHER_PATH) as MockMesher,
        patch(_SOLVER_PATH) as MockSolver,
        patch(_ASSEMBLER_PATH) as MockAssembler,
        patch(_CLASSIFIER_PATH) as MockClassifier,
    ):
        MockMesher.return_value.mesh_parametric_horn.return_value = fake_mesh
        MockSolver.return_value.modal_analysis.return_value = fake_modal
        MockAssembler.return_value.assemble.return_value = (K, M)
        MockClassifier.return_value.classify.return_value = fake_class

        yield {
            "mesher": MockMesher,
            "solver": MockSolver,
            "assembler": MockAssembler,
            "classifier": MockClassifier,
            "mesh": fake_mesh,
            "modal_result": fake_modal,
            "classification": fake_class,
        }


# ===================================================================
# Test 1: Default analysis uses Gmsh pipeline
# ===================================================================

class TestDefaultUsesGmsh:
    """Verify that the default analysis path calls GmshMesher."""

    def test_dispatch_modal_defaults_to_gmsh(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.dispatch_modal_analysis()

        # GmshMesher must have been instantiated and called
        mock_gmsh_pipeline["mesher"].assert_called_once()
        mock_gmsh_pipeline["mesher"].return_value.mesh_parametric_horn.assert_called_once()
        mock_gmsh_pipeline["solver"].return_value.modal_analysis.assert_called_once()

    def test_dispatch_acoustic_defaults_to_gmsh(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.dispatch_acoustic_analysis()

        mock_gmsh_pipeline["mesher"].assert_called_once()
        mock_gmsh_pipeline["mesher"].return_value.mesh_parametric_horn.assert_called_once()

    def test_run_modal_analysis_gmsh_called_directly(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.run_modal_analysis_gmsh()

        mock_gmsh_pipeline["mesher"].assert_called_once()

    def test_run_acoustic_analysis_gmsh_called_directly(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.run_acoustic_analysis_gmsh()

        mock_gmsh_pipeline["mesher"].assert_called_once()


# ===================================================================
# Test 2: Legacy fallback still works
# ===================================================================

class TestLegacyFallback:
    """Verify that ``use_gmsh=False`` invokes the old HEX8 pipeline."""

    def test_dispatch_modal_legacy_fallback(self):
        svc = FEAService()
        # Legacy pipeline doesn't need Gmsh, runs with numpy/scipy only
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.dispatch_modal_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
                use_gmsh=False,
            )
        # Should get a valid legacy result
        assert "mode_shapes" in result
        assert "closest_mode_hz" in result
        assert "node_count" in result
        assert result["node_count"] > 0

    def test_dispatch_acoustic_legacy_fallback(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.dispatch_acoustic_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
                use_gmsh=False,
            )
        assert "modes" in result
        assert "harmonic_response" in result
        assert "stress_hotspots" in result

    def test_run_modal_analysis_directly_still_works(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.run_modal_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        assert "mode_shapes" in result
        assert result["node_count"] > 0

    def test_run_acoustic_analysis_directly_still_works(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.run_acoustic_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        assert "modes" in result
        assert "amplitude_distribution" in result


# ===================================================================
# Test 3: Legacy path emits DeprecationWarning
# ===================================================================

class TestDeprecationWarnings:
    """Verify that the legacy HEX8 code path emits a DeprecationWarning."""

    def test_modal_legacy_emits_deprecation(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svc.run_modal_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert "Legacy HEX8 pipeline is deprecated" in str(
            deprecation_warnings[0].message
        )

    def test_acoustic_legacy_emits_deprecation(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svc.run_acoustic_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1
        assert "Legacy HEX8 pipeline is deprecated" in str(
            deprecation_warnings[0].message
        )

    def test_gmsh_path_does_not_emit_deprecation(self, mock_gmsh_pipeline):
        svc = FEAService()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svc.run_modal_analysis_gmsh()
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

    def test_dispatch_with_gmsh_true_no_deprecation(self, mock_gmsh_pipeline):
        svc = FEAService()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svc.dispatch_modal_analysis(use_gmsh=True)
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 0

    def test_dispatch_with_gmsh_false_emits_deprecation(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            svc.dispatch_modal_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
                use_gmsh=False,
            )
        deprecation_warnings = [
            x for x in w if issubclass(x.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) >= 1


# ===================================================================
# Test 4: API response compatibility (field names present in both)
# ===================================================================

class TestResponseCompatibility:
    """Ensure both pipelines return the same top-level fields."""

    # Expected fields for modal analysis response
    MODAL_REQUIRED_FIELDS = {
        "mode_shapes",
        "closest_mode_hz",
        "target_frequency_hz",
        "frequency_deviation_percent",
        "node_count",
        "element_count",
        "solve_time_s",
        "mesh",
        "stress_max_mpa",
        "temperature_max_c",
    }

    # Expected fields for acoustic analysis response
    ACOUSTIC_REQUIRED_FIELDS = {
        "modes",
        "closest_mode_hz",
        "target_frequency_hz",
        "frequency_deviation_percent",
        "harmonic_response",
        "amplitude_distribution",
        "amplitude_uniformity",
        "stress_hotspots",
        "stress_max_mpa",
        "node_count",
        "element_count",
        "solve_time_s",
        "mesh",
    }

    def test_gmsh_modal_response_fields(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.run_modal_analysis_gmsh()
        for f in self.MODAL_REQUIRED_FIELDS:
            assert f in result, f"Missing field '{f}' in Gmsh modal response"

    def test_legacy_modal_response_fields(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.run_modal_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        for f in self.MODAL_REQUIRED_FIELDS:
            assert f in result, f"Missing field '{f}' in legacy modal response"

    def test_gmsh_acoustic_response_fields(self, mock_gmsh_pipeline):
        svc = FEAService()
        result = svc.run_acoustic_analysis_gmsh()
        for f in self.ACOUSTIC_REQUIRED_FIELDS:
            assert f in result, f"Missing field '{f}' in Gmsh acoustic response"

    def test_legacy_acoustic_response_fields(self):
        svc = FEAService()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = svc.run_acoustic_analysis(
                horn_type="flat",
                width_mm=25.0,
                height_mm=80.0,
                length_mm=25.0,
                material="Titanium Ti-6Al-4V",
                frequency_khz=20.0,
                mesh_density="coarse",
            )
        for f in self.ACOUSTIC_REQUIRED_FIELDS:
            assert f in result, f"Missing field '{f}' in legacy acoustic response"

    def test_modal_mode_shape_fields(self, mock_gmsh_pipeline):
        """Each mode shape dict should have consistent fields."""
        svc = FEAService()
        result = svc.run_modal_analysis_gmsh()
        required_mode_fields = {
            "frequency_hz",
            "mode_type",
            "participation_factor",
            "effective_mass_ratio",
            "displacement_max",
        }
        for mode in result["mode_shapes"]:
            for f in required_mode_fields:
                assert f in mode, f"Missing field '{f}' in mode shape dict"


# ===================================================================
# Test 5: Router defaults now use Gmsh
# ===================================================================

class TestRouterDefaults:
    """Verify that request model defaults have been switched to use_gmsh=True."""

    def test_fea_request_defaults_to_gmsh(self):
        from web.routers.geometry import FEARequest
        req = FEARequest()
        assert req.use_gmsh is True

    def test_acoustic_request_defaults_to_gmsh(self):
        from web.routers.acoustic import AcousticAnalysisRequest
        req = AcousticAnalysisRequest()
        assert req.use_gmsh is True

    def test_fea_request_can_opt_out(self):
        from web.routers.geometry import FEARequest
        req = FEARequest(use_gmsh=False)
        assert req.use_gmsh is False

    def test_acoustic_request_can_opt_out(self):
        from web.routers.acoustic import AcousticAnalysisRequest
        req = AcousticAnalysisRequest(use_gmsh=False)
        assert req.use_gmsh is False
