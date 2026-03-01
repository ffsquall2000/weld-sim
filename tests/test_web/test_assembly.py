"""Tests for assembly analysis endpoints."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Mock data for FEAService.run_assembly_analysis
# ---------------------------------------------------------------------------

_MOCK_ASSEMBLY_RESULT = {
    "success": True,
    "message": "Assembly analysis completed: 2 components, 600 total DOF",
    "n_total_dof": 600,
    "n_components": 2,
    "frequencies_hz": [18500.0, 19200.0, 20100.0, 21500.0],
    "mode_types": ["flexural", "longitudinal", "longitudinal", "torsional"],
    "resonance_frequency_hz": 20100.0,
    "gain": 1.5,
    "q_factor": 100.0,
    "uniformity": 0.9,
    "gain_chain": {"horn": 1.0, "booster": 1.5},
    "impedance": {
        "horn": {
            "acoustic_impedance": 12345.67,
            "acoustic_velocity_m_s": 5068.0,
            "area_m2": 0.000491,
        },
        "booster": {
            "acoustic_impedance": 11000.0,
            "acoustic_velocity_m_s": 5068.0,
            "area_m2": 0.000400,
        },
    },
    "transmission_coefficients": {"horn->booster": 0.9975},
    "solve_time_s": 1.234,
}


# ---------------------------------------------------------------------------
# Tests: GET endpoints (no mocking needed for static data)
# ---------------------------------------------------------------------------


class TestAssemblyMaterials:
    """Tests for GET /api/v1/assembly/materials."""

    def test_list_materials_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/assembly/materials")
        assert response.status_code == 200

    def test_list_materials_returns_list(self, client: TestClient) -> None:
        response = client.get("/api/v1/assembly/materials")
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_materials_has_required_fields(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/assembly/materials")
        data = response.json()
        first = data[0]
        assert "name" in first
        assert "E_gpa" in first
        assert "density_kg_m3" in first
        assert "poisson_ratio" in first

    def test_list_materials_includes_titanium(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/assembly/materials")
        data = response.json()
        names = [m["name"] for m in data]
        assert "Titanium Ti-6Al-4V" in names


class TestAssemblyProfiles:
    """Tests for GET /api/v1/assembly/profiles."""

    def test_list_profiles_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/assembly/profiles")
        assert response.status_code == 200

    def test_list_profiles_returns_four(self, client: TestClient) -> None:
        response = client.get("/api/v1/assembly/profiles")
        data = response.json()
        assert len(data) == 4

    def test_list_profiles_has_required_fields(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/assembly/profiles")
        data = response.json()
        for item in data:
            assert "profile" in item
            assert "description" in item

    def test_list_profiles_includes_all_types(
        self, client: TestClient
    ) -> None:
        response = client.get("/api/v1/assembly/profiles")
        data = response.json()
        profile_names = {item["profile"] for item in data}
        assert profile_names == {
            "uniform",
            "stepped",
            "exponential",
            "catenoidal",
        }


# ---------------------------------------------------------------------------
# Tests: POST endpoints (mocked service layer)
# ---------------------------------------------------------------------------


class TestAssemblyAnalyze:
    """Tests for POST /api/v1/assembly/analyze."""

    @patch("web.services.fea_service.FEAService")
    def test_analyze_returns_200(self, mock_cls, client: TestClient) -> None:
        mock_cls.return_value.run_assembly_analysis.return_value = (
            _MOCK_ASSEMBLY_RESULT
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "horn_type": "cylindrical",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                    "material_name": "Titanium Ti-6Al-4V",
                    "mesh_size": 5.0,
                },
                {
                    "name": "booster",
                    "horn_type": "cylindrical",
                    "dimensions": {"diameter_mm": 40, "length_mm": 100},
                    "material_name": "Titanium Ti-6Al-4V",
                    "mesh_size": 5.0,
                },
            ],
            "analyses": ["modal", "harmonic"],
            "frequency_hz": 20000.0,
        }
        response = client.post("/api/v1/assembly/analyze", json=payload)
        assert response.status_code == 200

    @patch("web.services.fea_service.FEAService")
    def test_analyze_response_fields(
        self, mock_cls, client: TestClient
    ) -> None:
        mock_cls.return_value.run_assembly_analysis.return_value = (
            _MOCK_ASSEMBLY_RESULT
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "horn_type": "cylindrical",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                },
            ],
        }
        response = client.post("/api/v1/assembly/analyze", json=payload)
        data = response.json()
        assert data["success"] is True
        assert data["n_components"] == 2
        assert data["n_total_dof"] == 600
        assert isinstance(data["frequencies_hz"], list)
        assert isinstance(data["mode_types"], list)
        assert isinstance(data["gain_chain"], dict)
        assert isinstance(data["impedance"], dict)
        assert isinstance(data["transmission_coefficients"], dict)
        assert data["solve_time_s"] > 0

    @patch("web.services.fea_service.FEAService")
    def test_analyze_value_error_returns_400(
        self, mock_cls, client: TestClient
    ) -> None:
        mock_cls.return_value.run_assembly_analysis.side_effect = ValueError(
            "At least one component is required."
        )
        payload = {"components": []}
        response = client.post("/api/v1/assembly/analyze", json=payload)
        assert response.status_code == 400

    @patch("web.services.fea_service.FEAService")
    def test_analyze_runtime_error_returns_500(
        self, mock_cls, client: TestClient
    ) -> None:
        mock_cls.return_value.run_assembly_analysis.side_effect = RuntimeError(
            "Mesh generation failed"
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                },
            ],
        }
        response = client.post("/api/v1/assembly/analyze", json=payload)
        assert response.status_code == 500


class TestAssemblyModal:
    """Tests for POST /api/v1/assembly/modal."""

    @patch("web.services.fea_service.FEAService")
    def test_modal_returns_200(self, mock_cls, client: TestClient) -> None:
        mock_cls.return_value.run_assembly_analysis.return_value = (
            _MOCK_ASSEMBLY_RESULT
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "horn_type": "cylindrical",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                },
            ],
        }
        response = client.post("/api/v1/assembly/modal", json=payload)
        assert response.status_code == 200

    @patch("web.services.fea_service.FEAService")
    def test_modal_forces_modal_only(
        self, mock_cls, client: TestClient
    ) -> None:
        mock_cls.return_value.run_assembly_analysis.return_value = (
            _MOCK_ASSEMBLY_RESULT
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                },
            ],
            "analyses": ["modal", "harmonic"],  # should be overridden
        }
        response = client.post("/api/v1/assembly/modal", json=payload)
        assert response.status_code == 200

        # Verify the service was called with analyses=["modal"]
        call_kwargs = (
            mock_cls.return_value.run_assembly_analysis.call_args
        )
        assert call_kwargs.kwargs["analyses"] == ["modal"]

    @patch("web.services.fea_service.FEAService")
    def test_modal_value_error_returns_400(
        self, mock_cls, client: TestClient
    ) -> None:
        mock_cls.return_value.run_assembly_analysis.side_effect = ValueError(
            "Unknown material"
        )
        payload = {
            "components": [
                {
                    "name": "horn",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                    "material_name": "UnknownMetal",
                },
            ],
        }
        response = client.post("/api/v1/assembly/modal", json=payload)
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tests: Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    """Tests for request model validation."""

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/api/v1/assembly/analyze", json={})
        assert response.status_code == 422

    def test_missing_components_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/assembly/analyze",
            json={"frequency_hz": 20000},
        )
        assert response.status_code == 422

    def test_invalid_frequency_returns_422(self, client: TestClient) -> None:
        payload = {
            "components": [
                {
                    "name": "horn",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                },
            ],
            "frequency_hz": -1000,
        }
        response = client.post("/api/v1/assembly/analyze", json=payload)
        assert response.status_code == 422

    def test_invalid_mesh_size_returns_422(self, client: TestClient) -> None:
        payload = {
            "components": [
                {
                    "name": "horn",
                    "dimensions": {"diameter_mm": 25, "length_mm": 80},
                    "mesh_size": -1.0,
                },
            ],
        }
        response = client.post("/api/v1/assembly/analyze", json=payload)
        assert response.status_code == 422
