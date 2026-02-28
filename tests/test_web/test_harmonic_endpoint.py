"""Tests for the harmonic response analysis API endpoints."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


# A realistic result dict that _run_fea_subprocess would return
MOCK_HARMONIC_RESULT = {
    "task_id": "test-task-id-123",
    "frequencies_hz": [16000.0, 18000.0, 20000.0, 22000.0, 24000.0],
    "gain": 2.35,
    "q_factor": 150.0,
    "contact_face_uniformity": 0.92,
    "resonance_hz": 20000.0,
    "node_count": 1200,
    "element_count": 600,
    "solve_time_s": 3.14,
}


class TestHarmonicRequestModel:
    """Tests for the HarmonicRequest pydantic model."""

    def test_default_values(self):
        from web.routers.geometry import HarmonicRequest

        req = HarmonicRequest()
        assert req.horn_type == "cylindrical"
        assert req.width_mm == 25.0
        assert req.height_mm == 80.0
        assert req.length_mm == 25.0
        assert req.material == "Titanium Ti-6Al-4V"
        assert req.frequency_khz == 20.0
        assert req.mesh_density == "medium"
        assert req.freq_range_percent == 20.0
        assert req.n_freq_points == 201
        assert req.damping_model == "hysteretic"
        assert req.damping_ratio == 0.005
        assert req.task_id is None

    def test_custom_values(self):
        from web.routers.geometry import HarmonicRequest

        req = HarmonicRequest(
            horn_type="flat",
            width_mm=30.0,
            height_mm=100.0,
            length_mm=30.0,
            material="Steel AISI 4340",
            frequency_khz=15.0,
            mesh_density="fine",
            freq_range_percent=10.0,
            n_freq_points=101,
            damping_model="rayleigh",
            damping_ratio=0.01,
            task_id="my-task",
        )
        assert req.horn_type == "flat"
        assert req.n_freq_points == 101
        assert req.damping_model == "rayleigh"
        assert req.task_id == "my-task"

    def test_invalid_width_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import HarmonicRequest

        with pytest.raises(ValidationError):
            HarmonicRequest(width_mm=-1.0)

    def test_invalid_damping_ratio_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import HarmonicRequest

        with pytest.raises(ValidationError):
            HarmonicRequest(damping_ratio=-0.1)


class TestHarmonicRunResponse:
    """Tests for the HarmonicRunResponse pydantic model."""

    def test_from_dict(self):
        from web.routers.geometry import HarmonicRunResponse

        resp = HarmonicRunResponse(**MOCK_HARMONIC_RESULT)
        assert resp.task_id == "test-task-id-123"
        assert resp.gain == 2.35
        assert resp.q_factor == 150.0
        assert resp.contact_face_uniformity == 0.92
        assert resp.resonance_hz == 20000.0
        assert resp.node_count == 1200
        assert resp.element_count == 600
        assert resp.solve_time_s == 3.14
        assert len(resp.frequencies_hz) == 5

    def test_defaults(self):
        from web.routers.geometry import HarmonicRunResponse

        resp = HarmonicRunResponse()
        assert resp.task_id is None
        assert resp.frequencies_hz == []
        assert resp.gain == 0.0


class TestRunHarmonicEndpoint:
    """Tests for POST /api/v1/geometry/fea/run-harmonic."""

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_parametric_harmonic_success(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_HARMONIC_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-harmonic",
            json={
                "horn_type": "cylindrical",
                "width_mm": 25.0,
                "height_mm": 80.0,
                "length_mm": 25.0,
                "material": "Titanium Ti-6Al-4V",
                "frequency_khz": 20.0,
                "mesh_density": "medium",
                "freq_range_percent": 20.0,
                "n_freq_points": 201,
                "damping_model": "hysteretic",
                "damping_ratio": 0.005,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["gain"] == 2.35
        assert data["q_factor"] == 150.0
        assert data["resonance_hz"] == 20000.0
        assert data["node_count"] == 1200
        assert len(data["frequencies_hz"]) == 5

        # Verify subprocess was called with correct task_type and params
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "harmonic"
        params = call_args[0][1]
        assert params["horn_type"] == "cylindrical"
        assert params["freq_min_hz"] == 16000.0  # 20000 - 20%*20000
        assert params["freq_max_hz"] == 24000.0  # 20000 + 20%*20000
        assert params["n_freq_points"] == 201
        assert params["damping_model"] == "hysteretic"
        assert params["damping_ratio"] == 0.005

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_parametric_harmonic_with_defaults(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_HARMONIC_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-harmonic",
            json={},
        )

        assert response.status_code == 200
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "harmonic"

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_parametric_harmonic_with_task_id(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_HARMONIC_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-harmonic",
            json={"task_id": "client-uuid-456"},
        )

        assert response.status_code == 200
        call_args = mock_subprocess.call_args
        assert call_args[1]["client_task_id"] == "client-uuid-456"

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_parametric_harmonic_fea_failure(self, mock_subprocess, client):
        mock_subprocess.side_effect = Exception("Solver diverged")

        response = client.post(
            "/api/v1/geometry/fea/run-harmonic",
            json={},
        )

        assert response.status_code == 500
        assert "Harmonic analysis failed" in response.json()["detail"]

    def test_invalid_request_rejected(self, client):
        """Negative width should be rejected by pydantic validation."""
        response = client.post(
            "/api/v1/geometry/fea/run-harmonic",
            json={"width_mm": -5.0},
        )
        assert response.status_code == 422


class TestRunHarmonicStepEndpoint:
    """Tests for POST /api/v1/geometry/fea/run-harmonic-step."""

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_step_harmonic_success(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_HARMONIC_RESULT.copy()

        # Create a minimal fake STEP file
        step_content = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        response = client.post(
            "/api/v1/geometry/fea/run-harmonic-step",
            files={"file": ("test_horn.step", step_content, "application/octet-stream")},
            data={
                "material": "Titanium Ti-6Al-4V",
                "frequency_khz": "20.0",
                "mesh_density": "medium",
                "freq_range_percent": "20.0",
                "n_freq_points": "201",
                "damping_model": "hysteretic",
                "damping_ratio": "0.005",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["gain"] == 2.35
        assert data["resonance_hz"] == 20000.0

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "harmonic_step"
        params = call_args[0][1]
        assert "step_file_path" in params
        assert params["freq_min_hz"] == 16000.0
        assert params["freq_max_hz"] == 24000.0

    def test_step_harmonic_wrong_file_type(self, client):
        response = client.post(
            "/api/v1/geometry/fea/run-harmonic-step",
            files={"file": ("model.stl", b"solid foo\nendsolid foo\n", "application/octet-stream")},
            data={"material": "Titanium Ti-6Al-4V"},
        )
        assert response.status_code == 400
        assert "Only STEP files supported" in response.json()["detail"]

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_step_harmonic_with_task_id(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_HARMONIC_RESULT.copy()

        step_content = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        response = client.post(
            "/api/v1/geometry/fea/run-harmonic-step",
            files={"file": ("horn.stp", step_content, "application/octet-stream")},
            data={"task_id": "step-task-789"},
        )

        assert response.status_code == 200
        call_args = mock_subprocess.call_args
        assert call_args[1]["client_task_id"] == "step-task-789"


class TestFEAProcessRunnerHarmonicConfig:
    """Tests for harmonic-related configuration in fea_process_runner."""

    def test_harmonic_steps_defined(self):
        from web.services.fea_process_runner import HARMONIC_STEPS, HARMONIC_STEP_STEPS

        assert "init" in HARMONIC_STEPS
        assert "meshing" in HARMONIC_STEPS
        assert "assembly" in HARMONIC_STEPS
        assert "solving" in HARMONIC_STEPS
        assert "packaging" in HARMONIC_STEPS
        # Harmonic does NOT have "classifying" step
        assert "classifying" not in HARMONIC_STEPS

        assert "import_step" in HARMONIC_STEP_STEPS
        assert "classifying" not in HARMONIC_STEP_STEPS

    def test_worker_map_has_harmonic(self):
        from web.services.fea_process_runner import _WORKER_MAP

        assert "harmonic" in _WORKER_MAP
        assert "harmonic_step" in _WORKER_MAP

    def test_phase_weights_cover_harmonic_phases(self):
        from web.services.fea_process_runner import PHASE_WEIGHTS, HARMONIC_STEPS

        for step in HARMONIC_STEPS:
            assert step in PHASE_WEIGHTS, f"Phase weight missing for step: {step}"
