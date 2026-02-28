"""Tests for knurl FEA endpoints (generate, analyze, compare)."""
from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_mesh(n_nodes: int = 200, n_elements: int = 500, n_tris: int = 300):
    """Create a mock FEAMesh-like object."""
    mesh = MagicMock()
    mesh.nodes = np.random.rand(n_nodes, 3) * 0.08  # meters
    mesh.elements = np.random.randint(0, n_nodes, (n_elements, 10))
    mesh.surface_tris = np.random.randint(0, n_nodes, (n_tris, 3))
    mesh.node_sets = {
        "top_face": np.array([0, 1, 2]),
        "bottom_face": np.array([3, 4, 5]),
    }
    mesh.element_sets = {}
    mesh.element_type = "TET10"
    mesh.mesh_stats = {
        "num_nodes": n_nodes,
        "num_elements": n_elements,
        "num_surface_tris": n_tris,
        "element_type": "TET10",
        "order": 2,
        "mesh_size_mm": 5.0,
        "knurl_refinement": True,
        "knurl_info": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
    }
    return mesh


_MOCK_MODAL_RESULT = {
    "frequencies_hz": [18500.0, 19200.0, 20050.0, 21500.0, 22000.0],
    "mode_types": ["flexural", "flexural", "longitudinal", "torsional", "flexural"],
    "participation_factors": [0.1, 0.15, 0.85, 0.2, 0.05],
    "effective_mass_ratios": [0.01, 0.02, 0.75, 0.05, 0.01],
    "solve_time_s": 2.5,
    "stress_max_mpa": None,
    "amplitude_uniformity": None,
}


# ---------------------------------------------------------------------------
# Test: POST /api/v1/knurl-fea/generate
# ---------------------------------------------------------------------------


class TestKnurlFEAGenerate:
    """Tests for the generate endpoint."""

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_returns_200(self, mock_gen, client: TestClient) -> None:
        mock_gen.return_value = _make_mock_mesh()
        payload = {
            "horn": {
                "horn_type": "cylindrical",
                "width_mm": 25.0,
                "height_mm": 80.0,
                "length_mm": 25.0,
            },
            "knurl": {
                "type": "linear",
                "pitch_mm": 1.0,
                "depth_mm": 0.3,
            },
            "mesh_density": "medium",
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        assert response.status_code == 200

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_response_has_mesh(self, mock_gen, client: TestClient) -> None:
        mock_gen.return_value = _make_mock_mesh()
        payload = {
            "horn": {"horn_type": "cylindrical", "width_mm": 25.0, "height_mm": 80.0},
            "knurl": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        data = response.json()
        assert "mesh" in data
        assert "vertices" in data["mesh"]
        assert "faces" in data["mesh"]
        assert data["mesh"]["node_count"] > 0
        assert data["mesh"]["element_count"] > 0

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_response_has_knurl_info(
        self, mock_gen, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        payload = {
            "horn": {"horn_type": "flat", "width_mm": 30.0, "height_mm": 80.0},
            "knurl": {"type": "cross_hatch", "pitch_mm": 0.8, "depth_mm": 0.2},
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        data = response.json()
        assert data["knurl_info"]["type"] == "cross_hatch"
        assert data["knurl_info"]["pitch_mm"] == 0.8
        assert data["horn_type"] == "flat"

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_runtime_error_returns_400(
        self, mock_gen, client: TestClient
    ) -> None:
        mock_gen.side_effect = RuntimeError("Gmsh is not installed")
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        assert response.status_code == 400

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_file_not_found_returns_404(
        self, mock_gen, client: TestClient
    ) -> None:
        mock_gen.side_effect = FileNotFoundError("STEP file not found")
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
            "step_file_path": "/nonexistent/file.step",
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        assert response.status_code == 404

    def test_generate_default_params(self, client: TestClient) -> None:
        """Endpoint should accept request with all default params."""
        with patch("web.routers.knurl_fea._generate_knurl_mesh") as mock_gen:
            mock_gen.return_value = _make_mock_mesh()
            response = client.post("/api/v1/knurl-fea/generate", json={})
            assert response.status_code == 200

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_generate_vertices_in_mm(self, mock_gen, client: TestClient) -> None:
        """Vertices should be converted from meters to mm for display."""
        mesh = _make_mock_mesh()
        # Set a known node coordinate in meters
        mesh.nodes[0] = [0.01, 0.02, 0.03]
        mock_gen.return_value = mesh
        response = client.post("/api/v1/knurl-fea/generate", json={})
        data = response.json()
        v0 = data["mesh"]["vertices"][0]
        # Should be in mm (multiplied by 1000)
        assert abs(v0[0] - 10.0) < 0.01
        assert abs(v0[1] - 20.0) < 0.01
        assert abs(v0[2] - 30.0) < 0.01


# ---------------------------------------------------------------------------
# Test: POST /api/v1/knurl-fea/analyze
# ---------------------------------------------------------------------------


class TestKnurlFEAAnalyze:
    """Tests for the analyze endpoint."""

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_returns_200(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        mock_modal.return_value = _MOCK_MODAL_RESULT
        payload = {
            "horn": {"horn_type": "cylindrical", "width_mm": 25.0, "height_mm": 80.0},
            "knurl": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
            "material": "Titanium Ti-6Al-4V",
            "frequency_khz": 20.0,
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        assert response.status_code == 200

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_response_fields(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        mock_modal.return_value = _MOCK_MODAL_RESULT
        payload = {
            "horn": {"horn_type": "cylindrical", "width_mm": 25.0, "height_mm": 80.0},
            "knurl": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
            "frequency_khz": 20.0,
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        data = response.json()
        assert "mode_shapes" in data
        assert len(data["mode_shapes"]) == 5
        assert data["target_frequency_hz"] == 20000.0
        assert data["closest_mode_hz"] == 20050.0
        assert data["node_count"] > 0
        assert data["element_count"] > 0
        assert data["solve_time_s"] > 0
        assert data["knurl_info"]["type"] == "linear"

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_closest_mode_detection(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        mock_modal.return_value = _MOCK_MODAL_RESULT
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
            "frequency_khz": 20.0,
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        data = response.json()
        # 20050 Hz is closest to 20000 Hz target
        assert data["closest_mode_hz"] == 20050.0
        assert data["frequency_deviation_percent"] == pytest.approx(0.25, abs=0.01)

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_includes_mesh_preview(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        mock_modal.return_value = _MOCK_MODAL_RESULT
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        data = response.json()
        assert data["mesh"] is not None
        assert len(data["mesh"]["vertices"]) > 0
        assert len(data["mesh"]["faces"]) > 0

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_with_task_id(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.return_value = _make_mock_mesh()
        mock_modal.return_value = _MOCK_MODAL_RESULT
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
            "task_id": "test-task-123",
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        data = response.json()
        assert data["task_id"] == "test-task-123"

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_analyze_runtime_error_returns_400(
        self, mock_gen, mock_modal, client: TestClient
    ) -> None:
        mock_gen.side_effect = RuntimeError("Gmsh not installed")
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "linear"},
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test: POST /api/v1/knurl-fea/compare
# ---------------------------------------------------------------------------


class TestKnurlFEACompare:
    """Tests for the compare endpoint."""

    @patch("web.routers.knurl_fea._run_modal_analysis")
    @patch(
        "web.routers.knurl_fea.GmshMesher",
        create=True,
    )
    def test_compare_returns_200(
        self, mock_mesher_cls, mock_modal, client: TestClient
    ) -> None:
        # Mock the mesher used inside the compare endpoint
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
        ) as inner_mock:
            inner_mock.return_value.mesh_parametric_horn.return_value = (
                _make_mock_mesh()
            )
            mock_modal.return_value = _MOCK_MODAL_RESULT
            payload = {
                "horn": {
                    "horn_type": "cylindrical",
                    "width_mm": 25.0,
                    "height_mm": 80.0,
                },
                "knurl": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
                "frequency_khz": 20.0,
            }
            response = client.post("/api/v1/knurl-fea/compare", json=payload)
            assert response.status_code == 200

    @patch("web.routers.knurl_fea._run_modal_analysis")
    def test_compare_response_structure(
        self, mock_modal, client: TestClient
    ) -> None:
        """Compare should return with_knurl, without_knurl, and shift metrics."""
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
        ) as mock_mesher:
            mock_mesher.return_value.mesh_parametric_horn.return_value = (
                _make_mock_mesh()
            )
            mock_modal.return_value = _MOCK_MODAL_RESULT

            payload = {
                "horn": {"horn_type": "cylindrical"},
                "knurl": {"type": "linear"},
                "frequency_khz": 20.0,
            }
            response = client.post("/api/v1/knurl-fea/compare", json=payload)
            data = response.json()

            assert "with_knurl" in data
            assert "without_knurl" in data
            assert "frequency_shift_hz" in data
            assert "frequency_shift_percent" in data
            assert "target_frequency_hz" in data
            assert data["target_frequency_hz"] == 20000.0
            assert "knurl_info" in data

    @patch("web.routers.knurl_fea._run_modal_analysis")
    def test_compare_with_knurl_has_modes(
        self, mock_modal, client: TestClient
    ) -> None:
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
        ) as mock_mesher:
            mock_mesher.return_value.mesh_parametric_horn.return_value = (
                _make_mock_mesh()
            )
            mock_modal.return_value = _MOCK_MODAL_RESULT

            payload = {
                "horn": {"horn_type": "cylindrical"},
                "knurl": {"type": "diamond"},
            }
            response = client.post("/api/v1/knurl-fea/compare", json=payload)
            data = response.json()

            assert len(data["with_knurl"]["mode_shapes"]) == 5
            assert len(data["without_knurl"]["mode_shapes"]) == 5
            assert data["with_knurl"]["closest_mode_hz"] > 0
            assert data["without_knurl"]["closest_mode_hz"] > 0

    @patch("web.routers.knurl_fea._run_modal_analysis")
    def test_compare_frequency_shift(
        self, mock_modal, client: TestClient
    ) -> None:
        """When both results are the same, shift should be zero."""
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
        ) as mock_mesher:
            mock_mesher.return_value.mesh_parametric_horn.return_value = (
                _make_mock_mesh()
            )
            mock_modal.return_value = _MOCK_MODAL_RESULT

            payload = {
                "horn": {"horn_type": "cylindrical"},
                "knurl": {"type": "linear"},
                "frequency_khz": 20.0,
            }
            response = client.post("/api/v1/knurl-fea/compare", json=payload)
            data = response.json()

            # Both use same mock, so shift should be 0
            assert data["frequency_shift_hz"] == pytest.approx(0.0)
            assert data["frequency_shift_percent"] == pytest.approx(0.0)

    @patch("web.routers.knurl_fea._run_modal_analysis")
    def test_compare_runtime_error_returns_400(
        self, mock_modal, client: TestClient
    ) -> None:
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher.GmshMesher"
        ) as mock_mesher:
            mock_mesher.return_value.mesh_parametric_horn.side_effect = (
                RuntimeError("Gmsh not installed")
            )
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "knurl": {"type": "linear"},
            }
            response = client.post("/api/v1/knurl-fea/compare", json=payload)
            assert response.status_code == 400


# ---------------------------------------------------------------------------
# Test: Request validation
# ---------------------------------------------------------------------------


class TestKnurlFEAValidation:
    """Tests for request validation."""

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_invalid_pitch_rejected(self, mock_gen, client: TestClient) -> None:
        payload = {
            "knurl": {"type": "linear", "pitch_mm": -1.0},
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        assert response.status_code == 422  # Pydantic validation error

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_invalid_depth_rejected(self, mock_gen, client: TestClient) -> None:
        payload = {
            "knurl": {"type": "linear", "depth_mm": 0},
        }
        response = client.post("/api/v1/knurl-fea/generate", json=payload)
        assert response.status_code == 422

    @patch("web.routers.knurl_fea._generate_knurl_mesh")
    def test_invalid_frequency_rejected(self, mock_gen, client: TestClient) -> None:
        payload = {
            "frequency_khz": -5.0,
        }
        response = client.post("/api/v1/knurl-fea/analyze", json=payload)
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Test: Helper functions (no HTTP needed)
# ---------------------------------------------------------------------------


class TestKnurlFEAHelpers:
    """Tests for helper functions in knurl_fea module."""

    def test_build_knurl_dict(self) -> None:
        from web.routers.knurl_fea import KnurlParams, _build_knurl_dict

        params = KnurlParams(type="diamond", pitch_mm=1.5, depth_mm=0.4)
        result = _build_knurl_dict(params)
        assert result["type"] == "diamond"
        assert result["pitch_mm"] == 1.5
        assert result["depth_mm"] == 0.4

    def test_build_dimensions_cylindrical(self) -> None:
        from web.routers.knurl_fea import HornDimensions, _build_dimensions

        horn = HornDimensions(
            horn_type="cylindrical", width_mm=25.0, height_mm=80.0
        )
        dims = _build_dimensions(horn)
        assert dims["diameter_mm"] == 25.0
        assert dims["length_mm"] == 80.0

    def test_build_dimensions_flat(self) -> None:
        from web.routers.knurl_fea import HornDimensions, _build_dimensions

        horn = HornDimensions(
            horn_type="flat", width_mm=30.0, height_mm=80.0, length_mm=20.0
        )
        dims = _build_dimensions(horn)
        assert dims["width_mm"] == 30.0
        assert dims["depth_mm"] == 20.0
        assert dims["length_mm"] == 80.0

    def test_mesh_to_preview(self) -> None:
        from web.routers.knurl_fea import _mesh_to_preview

        mesh = _make_mock_mesh(n_nodes=10, n_elements=5, n_tris=8)
        preview = _mesh_to_preview(mesh)
        assert preview.node_count == 10
        assert preview.element_count == 5
        assert len(preview.vertices) == 10
        assert len(preview.faces) == 8
        # Vertices should be in mm (converted from meters)
        for v in preview.vertices:
            assert len(v) == 3


# ---------------------------------------------------------------------------
# Test: App registration
# ---------------------------------------------------------------------------


class TestKnurlFEARegistered:
    """Verify that the knurl-fea router is properly registered in the app."""

    def test_routes_registered(self, client: TestClient) -> None:
        """The OpenAPI schema should contain knurl-fea paths."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/api/v1/knurl-fea/generate" in paths
        assert "/api/v1/knurl-fea/analyze" in paths
        assert "/api/v1/knurl-fea/compare" in paths
