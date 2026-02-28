"""Tests for the fatigue life assessment API endpoint."""
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
MOCK_FATIGUE_RESULT = {
    "task_id": "fatigue-task-001",
    "min_safety_factor": 1.85,
    "estimated_life_cycles": 5.4e9,
    "estimated_hours_at_20khz": 75.0,
    "critical_locations": [
        {
            "element_id": 42,
            "safety_factor": 1.85,
            "x": 0.012,
            "y": 0.003,
            "z": 0.045,
        }
    ],
    "sn_curve_name": "Ti-6Al-4V",
    "corrected_endurance_mpa": 210.5,
    "max_stress_mpa": 113.8,
    "safety_factor_distribution": [2.1, 1.85, 3.4, 2.8],
    "node_count": 1500,
    "element_count": 800,
    "solve_time_s": 5.23,
}


class TestFatigueRequestModel:
    """Tests for the FatigueRequest pydantic model."""

    def test_default_values(self):
        from web.routers.geometry import FatigueRequest

        req = FatigueRequest()
        assert req.material == "Titanium Ti-6Al-4V"
        assert req.frequency_khz == 20.0
        assert req.mesh_density == "medium"
        assert req.surface_finish == "machined"
        assert req.characteristic_diameter_mm == 25.0
        assert req.reliability_pct == 90.0
        assert req.temperature_c == 25.0
        assert req.Kt_global == 1.5
        assert req.horn_type == "cylindrical"
        assert req.width_mm == 25.0
        assert req.height_mm == 80.0
        assert req.length_mm == 25.0
        assert req.damping_model == "hysteretic"
        assert req.damping_ratio == 0.005
        assert req.task_id is None

    def test_custom_values(self):
        from web.routers.geometry import FatigueRequest

        req = FatigueRequest(
            material="Aluminum 7075-T6",
            frequency_khz=15.0,
            mesh_density="fine",
            surface_finish="polished",
            characteristic_diameter_mm=10.0,
            reliability_pct=99.0,
            temperature_c=100.0,
            Kt_global=2.0,
            horn_type="flat",
            width_mm=30.0,
            height_mm=100.0,
            length_mm=30.0,
            damping_model="rayleigh",
            damping_ratio=0.01,
            task_id="my-fatigue-task",
        )
        assert req.material == "Aluminum 7075-T6"
        assert req.surface_finish == "polished"
        assert req.Kt_global == 2.0
        assert req.reliability_pct == 99.0
        assert req.task_id == "my-fatigue-task"

    def test_invalid_width_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import FatigueRequest

        with pytest.raises(ValidationError):
            FatigueRequest(width_mm=-1.0)

    def test_invalid_damping_ratio_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import FatigueRequest

        with pytest.raises(ValidationError):
            FatigueRequest(damping_ratio=-0.1)

    def test_invalid_reliability_too_low_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import FatigueRequest

        with pytest.raises(ValidationError):
            FatigueRequest(reliability_pct=10.0)

    def test_invalid_Kt_below_one_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import FatigueRequest

        with pytest.raises(ValidationError):
            FatigueRequest(Kt_global=0.5)


class TestFatigueResponseModel:
    """Tests for the FatigueResponse pydantic model."""

    def test_from_dict(self):
        from web.routers.geometry import FatigueResponse

        resp = FatigueResponse(**MOCK_FATIGUE_RESULT)
        assert resp.task_id == "fatigue-task-001"
        assert resp.min_safety_factor == 1.85
        assert resp.estimated_life_cycles == 5.4e9
        assert resp.estimated_hours_at_20khz == 75.0
        assert resp.sn_curve_name == "Ti-6Al-4V"
        assert resp.corrected_endurance_mpa == 210.5
        assert resp.max_stress_mpa == 113.8
        assert len(resp.safety_factor_distribution) == 4
        assert len(resp.critical_locations) == 1
        assert resp.node_count == 1500
        assert resp.element_count == 800
        assert resp.solve_time_s == 5.23

    def test_defaults(self):
        from web.routers.geometry import FatigueResponse

        resp = FatigueResponse()
        assert resp.task_id is None
        assert resp.min_safety_factor == 0.0
        assert resp.estimated_life_cycles == 0.0
        assert resp.estimated_hours_at_20khz == 0.0
        assert resp.critical_locations == []
        assert resp.sn_curve_name == ""
        assert resp.safety_factor_distribution == []

    def test_estimated_hours_calculation(self):
        """Verify the expected conversion: hours = cycles / (20000 * 3600)."""
        from web.routers.geometry import FatigueResponse

        cycles = 72_000_000.0  # exactly 1 hour at 20 kHz
        hours = cycles / (20000 * 3600)
        assert hours == pytest.approx(1.0, rel=1e-10)

        resp = FatigueResponse(
            estimated_life_cycles=cycles,
            estimated_hours_at_20khz=hours,
        )
        assert resp.estimated_hours_at_20khz == pytest.approx(1.0, rel=1e-10)


class TestRunFatigueEndpoint:
    """Tests for POST /api/v1/geometry/fea/run-fatigue."""

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_fatigue_endpoint_success(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_FATIGUE_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={
                "material": "Titanium Ti-6Al-4V",
                "frequency_khz": 20.0,
                "mesh_density": "medium",
                "surface_finish": "machined",
                "characteristic_diameter_mm": 25.0,
                "reliability_pct": 90.0,
                "temperature_c": 25.0,
                "Kt_global": 1.5,
                "horn_type": "cylindrical",
                "width_mm": 25.0,
                "height_mm": 80.0,
                "length_mm": 25.0,
                "damping_model": "hysteretic",
                "damping_ratio": 0.005,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["min_safety_factor"] == 1.85
        assert data["estimated_life_cycles"] == 5.4e9
        assert data["estimated_hours_at_20khz"] == 75.0
        assert data["sn_curve_name"] == "Ti-6Al-4V"
        assert data["corrected_endurance_mpa"] == 210.5
        assert data["max_stress_mpa"] == 113.8
        assert data["node_count"] == 1500
        assert data["element_count"] == 800
        assert len(data["critical_locations"]) == 1

        # Verify subprocess was called with correct task_type and params
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "fatigue"
        params = call_args[0][1]
        assert params["horn_type"] == "cylindrical"
        assert params["surface_finish"] == "machined"
        assert params["Kt_global"] == 1.5
        assert params["reliability_pct"] == 90.0

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_fatigue_endpoint_with_defaults(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_FATIGUE_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={},
        )

        assert response.status_code == 200
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "fatigue"

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_fatigue_endpoint_with_task_id(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_FATIGUE_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={"task_id": "fatigue-client-uuid-789"},
        )

        assert response.status_code == 200
        call_args = mock_subprocess.call_args
        assert call_args[1]["client_task_id"] == "fatigue-client-uuid-789"

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_fatigue_endpoint_fea_failure(self, mock_subprocess, client):
        mock_subprocess.side_effect = Exception("Solver diverged in fatigue")

        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={},
        )

        assert response.status_code == 500
        assert "Fatigue analysis failed" in response.json()["detail"]

    def test_invalid_request_rejected(self, client):
        """Negative width should be rejected by pydantic validation."""
        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={"width_mm": -5.0},
        )
        assert response.status_code == 422

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_fatigue_params_passed_correctly(self, mock_subprocess, client):
        """Verify fatigue-specific parameters are passed to subprocess."""
        mock_subprocess.return_value = MOCK_FATIGUE_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-fatigue",
            json={
                "surface_finish": "polished",
                "characteristic_diameter_mm": 10.0,
                "reliability_pct": 99.0,
                "temperature_c": 100.0,
                "Kt_global": 2.5,
            },
        )

        assert response.status_code == 200
        params = mock_subprocess.call_args[0][1]
        assert params["surface_finish"] == "polished"
        assert params["characteristic_diameter_mm"] == 10.0
        assert params["reliability_pct"] == 99.0
        assert params["temperature_c"] == 100.0
        assert params["Kt_global"] == 2.5


class TestFEAProcessRunnerFatigueConfig:
    """Tests for fatigue-related configuration in fea_process_runner."""

    def test_fatigue_steps_defined(self):
        from web.services.fea_process_runner import FATIGUE_STEPS

        assert "init" in FATIGUE_STEPS
        assert "meshing" in FATIGUE_STEPS
        assert "assembly" in FATIGUE_STEPS
        assert "modal_solve" in FATIGUE_STEPS
        assert "harmonic_solve" in FATIGUE_STEPS
        assert "stress_compute" in FATIGUE_STEPS
        assert "fatigue_assess" in FATIGUE_STEPS
        assert "packaging" in FATIGUE_STEPS

    def test_worker_map_has_fatigue(self):
        from web.services.fea_process_runner import _WORKER_MAP

        assert "fatigue" in _WORKER_MAP

    def test_phase_weights_cover_fatigue_phases(self):
        from web.services.fea_process_runner import PHASE_WEIGHTS, FATIGUE_STEPS

        for step in FATIGUE_STEPS:
            assert step in PHASE_WEIGHTS, f"Phase weight missing for step: {step}"

    def test_hours_conversion_formula(self):
        """Verify the conversion: hours = cycles / (frequency_hz * 3600)."""
        # At 20 kHz, 72_000_000 cycles = 1 hour
        cycles = 72_000_000.0
        frequency_hz = 20000.0
        hours = cycles / (frequency_hz * 3600.0)
        assert hours == pytest.approx(1.0, rel=1e-10)

        # At 20 kHz, 1e9 cycles = ~13.89 hours
        cycles2 = 1e9
        hours2 = cycles2 / (frequency_hz * 3600.0)
        assert hours2 == pytest.approx(1e9 / 72_000_000, rel=1e-10)
