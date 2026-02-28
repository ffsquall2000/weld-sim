"""Tests for the contact analysis API endpoints.

All tests use mocks for the ContactSolver and AnvilGenerator so that
Docker/FEniCSx are not required locally.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_MOCK_CONTACT_RESULT = {
    "status": "success",
    "run_id": "mock-run-123",
    "contact_pressure": {
        "max_MPa": 15.0,
        "mean_MPa": 7.5,
        "distribution": [1.0, 2.0, 3.0],
    },
    "slip_distance": {
        "max_um": 10.0,
        "mean_um": 4.0,
        "distribution": [0.5, 1.0],
    },
    "deformation": {
        "max_um": 28.0,
        "field": [0.1, 0.2, 0.3],
    },
    "stress": {
        "von_mises_max_MPa": 42.0,
        "field": [10.0, 20.0],
    },
    "summary": {
        "contact_area_mm2": 150.0,
        "total_force_N": 2000.0,
        "newton_iterations": 5,
        "converged": True,
        "solve_time_s": 8.5,
        "contact_type": "penalty",
    },
}

_MOCK_ANVIL_RESULT = {
    "solid": None,
    "mesh_preview": {
        "vertices": [
            [-25.0, -15.0, 0.0],
            [25.0, -15.0, 0.0],
            [25.0, 15.0, 0.0],
            [-25.0, 15.0, 0.0],
            [-25.0, -15.0, 20.0],
            [25.0, -15.0, 20.0],
            [25.0, 15.0, 20.0],
            [-25.0, 15.0, 20.0],
        ],
        "faces": [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
        ],
    },
    "step_path": None,
    "params": {"anvil_type": "flat"},
    "contact_face": {"center": [0.0, 0.0, 20.0], "width": 50.0, "length": 30.0},
    "volume_mm3": 30000.0,
    "surface_area_mm2": 6200.0,
}


# Patch targets -- the lazy imports resolve inside the endpoint functions,
# so we patch at the actual module paths used in the import statements.
_CONTACT_SOLVER_PATH = (
    "ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_fenicsx.ContactSolver"
)
_ANVIL_GENERATOR_PATH = (
    "ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator.AnvilGenerator"
)
_ANVIL_PARAMS_PATH = (
    "ultrasonic_weld_master.plugins.geometry_analyzer.anvil_generator.AnvilParams"
)
_FENICSX_RUNNER_PATH = "web.services.fenicsx_runner.FEniCSxRunner"
_THERMAL_SOLVER_PATH = (
    "ultrasonic_weld_master.plugins.geometry_analyzer.fea.thermal_solver.ThermalSolver"
)

# Mock thermal result
_MOCK_THERMAL_RESULT = {
    "status": "success",
    "run_id": "thermal-mock-456",
    "max_temperature_c": 185.5,
    "mean_temperature_c": 95.3,
    "min_temperature_c": 25.0,
    "initial_temperature_c": 25.0,
    "temperature_distribution": [25.0, 50.0, 100.0, 150.0, 185.5],
    "melt_zone": {
        "melt_temp_c": 200.0,
        "melt_fraction": 0.0,
        "melt_volume_mm3": 0.0,
        "n_melt_nodes": 0,
        "n_total_nodes": 5000,
    },
    "thermal_history": [
        {"time_s": 0.0, "max_temp_c": 25.0, "mean_temp_c": 25.0, "min_temp_c": 25.0},
        {"time_s": 0.25, "max_temp_c": 120.0, "mean_temp_c": 60.0, "min_temp_c": 25.0},
        {"time_s": 0.5, "max_temp_c": 185.5, "mean_temp_c": 95.3, "min_temp_c": 25.0},
    ],
    "max_temp_history": [25.0, 80.0, 120.0, 150.0, 185.5],
    "mean_temp_history": [25.0, 40.0, 60.0, 80.0, 95.3],
    "heat_generation_rate_w_m3": 1.13e9,
    "surface_heat_flux_w_m2": 1.13e6,
    "solve_time_s": 5.2,
    "weld_time_s": 0.5,
    "n_time_steps": 50,
}


# ---------------------------------------------------------------------------
# Tests: POST /contact/analyze
# ---------------------------------------------------------------------------


class TestContactAnalyze:
    """Tests for POST /api/v1/contact/analyze."""

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_returns_200(self, mock_solver_cls, client: TestClient) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_solver_cls.return_value = mock_instance

        payload = {
            "horn": {"horn_type": "cylindrical", "width_mm": 25, "height_mm": 80},
            "workpiece": {"material": "ABS", "thickness_mm": 3.0},
            "contact_type": "penalty",
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
        }
        response = client.post("/api/v1/contact/analyze", json=payload)
        assert response.status_code == 200

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_response_fields(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_solver_cls.return_value = mock_instance

        payload = {"contact_type": "penalty", "frequency_hz": 20000}
        response = client.post("/api/v1/contact/analyze", json=payload)
        data = response.json()

        assert data["status"] == "success"
        assert data["contact_pressure"]["max_MPa"] == 15.0
        assert data["slip_distance"]["max_um"] == 10.0
        assert data["deformation"]["max_um"] == 28.0
        assert data["stress"]["von_mises_max_MPa"] == 42.0
        assert data["summary"]["converged"] is True
        assert data["summary"]["contact_area_mm2"] == 150.0

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_weld_quality_included(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post(
            "/api/v1/contact/analyze",
            json={"contact_type": "penalty"},
        )
        data = response.json()

        assert "weld_quality" in data
        assert "score" in data["weld_quality"]
        assert "rating" in data["weld_quality"]
        assert "notes" in data["weld_quality"]
        assert data["weld_quality"]["score"] > 0

    def test_analyze_invalid_contact_type_returns_400(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/analyze",
            json={"contact_type": "invalid_method"},
        )
        assert response.status_code == 400

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_nitsche_type(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        nitsche_result = dict(_MOCK_CONTACT_RESULT)
        nitsche_result["summary"] = dict(_MOCK_CONTACT_RESULT["summary"])
        nitsche_result["summary"]["contact_type"] = "nitsche"

        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=nitsche_result)
        mock_solver_cls.return_value = mock_instance

        response = client.post(
            "/api/v1/contact/analyze",
            json={"contact_type": "nitsche"},
        )
        assert response.status_code == 200
        assert response.json()["summary"]["contact_type"] == "nitsche"

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_solver_error_returns_500(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(
            return_value={
                "status": "error",
                "error": "Docker not available",
            }
        )
        mock_solver_cls.return_value = mock_instance

        response = client.post(
            "/api/v1/contact/analyze",
            json={"contact_type": "penalty"},
        )
        assert response.status_code == 500

    @patch(_CONTACT_SOLVER_PATH)
    def test_analyze_defaults(self, mock_solver_cls, client: TestClient) -> None:
        """Endpoint accepts empty body (all defaults)."""
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/analyze", json={})
        assert response.status_code == 200

    def test_analyze_validation_negative_frequency(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/analyze",
            json={"frequency_hz": -100},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: POST /contact/check-docker
# ---------------------------------------------------------------------------


class TestCheckDocker:
    """Tests for POST /api/v1/contact/check-docker."""

    @patch(_FENICSX_RUNNER_PATH)
    def test_docker_available(
        self, mock_runner_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.check_available = AsyncMock(return_value=True)
        mock_runner_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/check-docker")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is True
        assert "dolfinx" in data["image"]

    @patch(_FENICSX_RUNNER_PATH)
    def test_docker_not_available(
        self, mock_runner_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.check_available = AsyncMock(return_value=False)
        mock_runner_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/check-docker")
        assert response.status_code == 200
        data = response.json()
        assert data["available"] is False


# ---------------------------------------------------------------------------
# Tests: POST /contact/anvil-preview
# ---------------------------------------------------------------------------


class TestAnvilPreview:
    """Tests for POST /api/v1/contact/anvil-preview."""

    @patch(_ANVIL_GENERATOR_PATH)
    def test_flat_anvil_preview(
        self, mock_gen_cls, client: TestClient
    ) -> None:
        mock_gen = MagicMock()
        mock_gen.generate.return_value = _MOCK_ANVIL_RESULT
        mock_gen_cls.return_value = mock_gen

        response = client.post(
            "/api/v1/contact/anvil-preview",
            json={"anvil_type": "flat", "width_mm": 50, "depth_mm": 30},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["anvil_type"] == "flat"
        assert len(data["vertices"]) == 8
        assert len(data["faces"]) == 4
        assert data["volume_mm3"] == 30000.0

    @patch(_ANVIL_GENERATOR_PATH)
    def test_groove_anvil_preview(
        self, mock_gen_cls, client: TestClient
    ) -> None:
        mock_gen = MagicMock()
        groove_result = dict(_MOCK_ANVIL_RESULT)
        groove_result["params"] = {"anvil_type": "groove"}
        mock_gen.generate.return_value = groove_result
        mock_gen_cls.return_value = mock_gen

        response = client.post(
            "/api/v1/contact/anvil-preview",
            json={"anvil_type": "groove", "groove_count": 5},
        )
        assert response.status_code == 200

    @patch(_ANVIL_GENERATOR_PATH)
    def test_anvil_preview_invalid_type_returns_400(
        self, mock_gen_cls, client: TestClient
    ) -> None:
        mock_gen = MagicMock()
        mock_gen.generate.side_effect = ValueError(
            "Invalid anvil_type: 'invalid'"
        )
        mock_gen_cls.return_value = mock_gen

        response = client.post(
            "/api/v1/contact/anvil-preview",
            json={"anvil_type": "invalid"},
        )
        assert response.status_code == 400

    @patch(_ANVIL_GENERATOR_PATH)
    def test_anvil_preview_defaults(
        self, mock_gen_cls, client: TestClient
    ) -> None:
        """Endpoint accepts empty body with all defaults."""
        mock_gen = MagicMock()
        mock_gen.generate.return_value = _MOCK_ANVIL_RESULT
        mock_gen_cls.return_value = mock_gen

        response = client.post("/api/v1/contact/anvil-preview", json={})
        assert response.status_code == 200

    def test_anvil_preview_negative_width_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/anvil-preview",
            json={"width_mm": -10},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Weld quality estimation
# ---------------------------------------------------------------------------


class TestWeldQualityEstimation:
    """Tests for the _estimate_weld_quality helper."""

    def test_excellent_quality(self) -> None:
        from web.routers.contact import _estimate_weld_quality

        result = _estimate_weld_quality({
            "summary": {
                "converged": True,
                "contact_area_mm2": 300.0,
                "total_force_N": 5000.0,
            },
            "contact_pressure": {
                "max_MPa": 20.0,
                "mean_MPa": 18.0,
            },
        })
        assert result["score"] > 0
        assert result["rating"] in ("excellent", "good", "fair", "poor")
        assert len(result["notes"]) > 0

    def test_poor_quality_no_convergence(self) -> None:
        from web.routers.contact import _estimate_weld_quality

        result = _estimate_weld_quality({
            "summary": {
                "converged": False,
                "contact_area_mm2": 0.0,
                "total_force_N": 0.0,
            },
            "contact_pressure": {
                "max_MPa": 0.0,
                "mean_MPa": 0.0,
            },
        })
        assert result["score"] == 0.0
        assert result["rating"] == "poor"

    def test_empty_results(self) -> None:
        from web.routers.contact import _estimate_weld_quality

        result = _estimate_weld_quality({})
        assert result["score"] == 0.0
        assert result["rating"] == "poor"


# ---------------------------------------------------------------------------
# Tests: Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Tests for request model validation."""

    def test_invalid_friction_coefficient(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/contact/analyze",
            json={"friction_coefficient": 1.5},
        )
        assert response.status_code == 422

    def test_zero_amplitude_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/contact/analyze",
            json={"amplitude_um": 0},
        )
        assert response.status_code == 422

    def test_negative_thickness_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/contact/analyze",
            json={"workpiece": {"material": "ABS", "thickness_mm": -1}},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: POST /contact/thermal
# ---------------------------------------------------------------------------


class TestThermalAnalyze:
    """Tests for POST /api/v1/contact/thermal."""

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_returns_200(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        payload = {
            "workpiece_material": "ABS",
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "weld_time_s": 0.5,
            "contact_pressure_mpa": 1.0,
        }
        response = client.post("/api/v1/contact/thermal", json=payload)
        assert response.status_code == 200

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_response_fields(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post(
            "/api/v1/contact/thermal",
            json={"workpiece_material": "ABS"},
        )
        data = response.json()

        assert data["status"] == "success"
        assert data["max_temperature_c"] == 185.5
        assert data["mean_temperature_c"] == 95.3
        assert isinstance(data["temperature_distribution"], list)
        assert len(data["temperature_distribution"]) > 0

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_melt_zone(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/thermal", json={})
        data = response.json()

        assert "melt_zone" in data
        assert data["melt_zone"]["melt_temp_c"] == 200.0
        assert data["melt_zone"]["n_total_nodes"] == 5000

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_history(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/thermal", json={})
        data = response.json()

        assert "thermal_history" in data
        assert len(data["thermal_history"]) == 3
        assert data["thermal_history"][0]["time_s"] == 0.0
        assert data["thermal_history"][-1]["max_temp_c"] == 185.5

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_heat_source_info(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/thermal", json={})
        data = response.json()

        assert data["heat_generation_rate_w_m3"] == 1.13e9
        assert data["surface_heat_flux_w_m2"] == 1.13e6
        assert data["weld_time_s"] == 0.5
        assert data["n_time_steps"] == 50

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_solver_error_returns_500(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(
            return_value={
                "status": "error",
                "error": "Docker not available",
            }
        )
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/thermal", json={})
        assert response.status_code == 500

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_defaults(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        """Endpoint accepts empty body with all defaults."""
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post("/api/v1/contact/thermal", json={})
        assert response.status_code == 200

    def test_thermal_negative_weld_time_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/thermal",
            json={"weld_time_s": -0.5},
        )
        assert response.status_code == 422

    def test_thermal_zero_pressure_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/thermal",
            json={"contact_pressure_mpa": 0},
        )
        assert response.status_code == 422

    def test_thermal_invalid_friction_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/thermal",
            json={"friction_coefficient": 2.0},
        )
        assert response.status_code == 422

    @patch(_THERMAL_SOLVER_PATH)
    def test_thermal_pressure_conversion(
        self, mock_solver_cls, client: TestClient
    ) -> None:
        """Verify MPa -> Pa conversion in config."""
        mock_instance = MagicMock()
        mock_instance.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_solver_cls.return_value = mock_instance

        response = client.post(
            "/api/v1/contact/thermal",
            json={"contact_pressure_mpa": 2.0},
        )
        assert response.status_code == 200

        # Verify the analyze was called with Pa (not MPa)
        call_args = mock_instance.analyze.call_args
        config = call_args[0][0]
        assert config["contact_pressure_pa"] == 2.0e6


# ---------------------------------------------------------------------------
# Tests: POST /contact/full-analysis
# ---------------------------------------------------------------------------


class TestFullAnalysis:
    """Tests for POST /api/v1/contact/full-analysis."""

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_returns_200(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        # Mock contact solver
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_contact_cls.return_value = mock_contact

        # Mock thermal solver
        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        payload = {
            "horn": {"horn_type": "cylindrical", "width_mm": 25},
            "workpiece": {"material": "ABS"},
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "weld_time_s": 0.5,
        }
        response = client.post("/api/v1/contact/full-analysis", json=payload)
        assert response.status_code == 200

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_has_both_results(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_contact_cls.return_value = mock_contact

        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        response = client.post("/api/v1/contact/full-analysis", json={})
        data = response.json()

        assert data["status"] == "success"
        assert data["contact"] is not None
        assert data["thermal"] is not None
        assert data["contact"]["status"] == "success"
        assert data["thermal"]["status"] == "success"
        assert data["total_solve_time_s"] >= 0

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_weld_quality(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_contact_cls.return_value = mock_contact

        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        response = client.post("/api/v1/contact/full-analysis", json={})
        data = response.json()

        assert "weld_quality" in data
        assert data["weld_quality"]["score"] > 0
        assert data["weld_quality"]["rating"] in (
            "excellent", "good", "fair", "poor"
        )

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_contact_feeds_thermal(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        """Contact pressure from contact analysis is passed to thermal solver."""
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_contact_cls.return_value = mock_contact

        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        response = client.post("/api/v1/contact/full-analysis", json={})
        assert response.status_code == 200

        # Verify thermal solver received contact pressure
        thermal_call = mock_thermal.analyze.call_args
        thermal_config = thermal_call[0][0]
        # Contact result has mean_MPa = 7.5, so Pa = 7.5e6
        assert thermal_config["contact_pressure_pa"] == 7.5e6

    def test_full_analysis_invalid_contact_type_returns_400(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/full-analysis",
            json={"contact_type": "invalid"},
        )
        assert response.status_code == 400

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_contact_failure_still_runs_thermal(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        """If contact analysis fails, thermal should still run."""
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(
            return_value={"status": "error", "error": "Docker down"}
        )
        mock_contact_cls.return_value = mock_contact

        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        response = client.post("/api/v1/contact/full-analysis", json={})
        assert response.status_code == 200
        data = response.json()

        # Contact should be None (failed), thermal should succeed
        assert data["contact"] is None
        assert data["thermal"] is not None
        assert data["thermal"]["status"] == "success"

    @patch(_THERMAL_SOLVER_PATH)
    @patch(_CONTACT_SOLVER_PATH)
    def test_full_analysis_defaults(
        self, mock_contact_cls, mock_thermal_cls, client: TestClient
    ) -> None:
        mock_contact = MagicMock()
        mock_contact.analyze = AsyncMock(return_value=_MOCK_CONTACT_RESULT)
        mock_contact_cls.return_value = mock_contact

        mock_thermal = MagicMock()
        mock_thermal.analyze = AsyncMock(return_value=_MOCK_THERMAL_RESULT)
        mock_thermal_cls.return_value = mock_thermal

        response = client.post("/api/v1/contact/full-analysis", json={})
        assert response.status_code == 200

    def test_full_analysis_negative_frequency_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/full-analysis",
            json={"frequency_hz": -100},
        )
        assert response.status_code == 422

    def test_full_analysis_zero_weld_time_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/full-analysis",
            json={"weld_time_s": 0},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: Thermal request validation
# ---------------------------------------------------------------------------


class TestThermalRequestValidation:
    """Tests for thermal and full-analysis request validation."""

    def test_thermal_excessive_time_steps_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/thermal",
            json={"n_time_steps": 2000},
        )
        assert response.status_code == 422

    def test_full_analysis_excessive_time_steps_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post(
            "/api/v1/contact/full-analysis",
            json={"n_time_steps": 2000},
        )
        assert response.status_code == 422
