"""Tests for the analysis chain orchestrator and API endpoint."""
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


# ---------------------------------------------------------------------------
# Mock result returned by _run_fea_subprocess for chain calls
# ---------------------------------------------------------------------------
MOCK_CHAIN_RESULT = {
    "task_id": "chain-task-001",
    "modules_executed": ["modal", "harmonic", "stress", "fatigue"],
    "modal": {
        "mode_shapes": [
            {
                "frequency_hz": 19850.0,
                "mode_type": "longitudinal",
                "participation_factor": 0.85,
                "effective_mass_ratio": 0.72,
                "displacement_max": 1.0,
            }
        ],
        "closest_mode_hz": 19850.0,
        "target_frequency_hz": 20000.0,
        "frequency_deviation_percent": 0.75,
        "solve_time_s": 1.2,
    },
    "harmonic": {
        "frequencies_hz": [16000.0, 18000.0, 20000.0, 22000.0, 24000.0],
        "gain": 2.35,
        "q_factor": 150.0,
        "contact_face_uniformity": 0.92,
        "resonance_hz": 20000.0,
        "solve_time_s": 2.5,
    },
    "stress": {
        "max_stress_mpa": 113.8,
        "safety_factor": 1.85,
        "max_displacement_mm": 0.015,
        "contact_face_uniformity": 0.92,
        "resonance_hz": 20000.0,
        "solve_time_s": 0.8,
    },
    "fatigue": {
        "min_safety_factor": 1.85,
        "estimated_life_cycles": 5.4e9,
        "estimated_hours_at_20khz": 75.0,
        "critical_locations": [],
        "sn_curve_name": "Ti-6Al-4V",
        "corrected_endurance_mpa": 210.5,
        "max_stress_mpa": 113.8,
        "safety_factor_distribution": [2.1, 1.85, 3.4],
    },
    "total_solve_time_s": 4.5,
    "node_count": 1500,
    "element_count": 800,
}


# ---------------------------------------------------------------------------
# Tests for resolve_dependencies()
# ---------------------------------------------------------------------------


class TestResolveDependencies:
    """Tests for the dependency resolution function."""

    def test_fatigue_resolves_to_full_chain(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["fatigue"])
        assert result == ["modal", "harmonic", "stress", "fatigue"]

    def test_modal_alone_stays_as_modal(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["modal"])
        assert result == ["modal"]

    def test_harmonic_adds_modal(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["harmonic"])
        assert result == ["modal", "harmonic"]

    def test_stress_adds_modal_and_harmonic(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["stress"])
        assert result == ["modal", "harmonic", "stress"]

    def test_uniformity_adds_modal_and_harmonic(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["uniformity"])
        assert result == ["modal", "harmonic", "uniformity"]

    def test_static_alone_stays_as_static(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["static"])
        assert result == ["static"]

    def test_multiple_modules_dedup_and_order(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["stress", "modal"])
        assert result == ["modal", "harmonic", "stress"]

    def test_fatigue_and_uniformity_combined(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(["fatigue", "uniformity"])
        assert result == ["modal", "harmonic", "uniformity", "stress", "fatigue"]

    def test_empty_input_returns_empty(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies([])
        assert result == []

    def test_all_modules(self):
        from web.services.chain_runner import resolve_dependencies

        result = resolve_dependencies(
            ["modal", "static", "harmonic", "uniformity", "stress", "fatigue"]
        )
        assert result == ["modal", "static", "harmonic", "uniformity", "stress", "fatigue"]


# ---------------------------------------------------------------------------
# Tests for build_chain_steps()
# ---------------------------------------------------------------------------


class TestBuildChainSteps:
    """Tests for the chain step builder."""

    def test_full_chain_steps(self):
        from web.services.chain_runner import build_chain_steps

        steps = build_chain_steps(["modal", "harmonic", "stress", "fatigue"])
        assert steps == [
            "init", "modal_run", "harmonic_run", "stress_run", "fatigue_run", "packaging"
        ]

    def test_single_module_steps(self):
        from web.services.chain_runner import build_chain_steps

        steps = build_chain_steps(["modal"])
        assert steps == ["init", "modal_run", "packaging"]

    def test_empty_modules(self):
        from web.services.chain_runner import build_chain_steps

        steps = build_chain_steps([])
        assert steps == ["init", "packaging"]


# ---------------------------------------------------------------------------
# Tests for ChainRequest model
# ---------------------------------------------------------------------------


class TestChainRequestModel:
    """Tests for the ChainRequest pydantic model."""

    def test_default_values(self):
        from web.routers.geometry import ChainRequest

        req = ChainRequest(modules=["modal"])
        assert req.modules == ["modal"]
        assert req.material == "Titanium Ti-6Al-4V"
        assert req.frequency_khz == 20.0
        assert req.mesh_density == "medium"
        assert req.horn_type == "cylindrical"
        assert req.width_mm == 25.0
        assert req.height_mm == 80.0
        assert req.length_mm == 25.0
        assert req.damping_model == "hysteretic"
        assert req.damping_ratio == 0.005
        assert req.freq_range_percent == 20.0
        assert req.n_freq_points == 201
        assert req.surface_finish == "machined"
        assert req.Kt_global == 1.5
        assert req.reliability_pct == 90.0
        assert req.task_id is None

    def test_custom_values(self):
        from web.routers.geometry import ChainRequest

        req = ChainRequest(
            modules=["modal", "harmonic", "stress"],
            material="Aluminum 7075-T6",
            frequency_khz=15.0,
            mesh_density="fine",
            horn_type="flat",
            width_mm=30.0,
            height_mm=100.0,
            length_mm=30.0,
            damping_model="rayleigh",
            damping_ratio=0.01,
            freq_range_percent=10.0,
            n_freq_points=101,
            surface_finish="polished",
            Kt_global=2.0,
            reliability_pct=99.0,
            task_id="my-chain-task",
        )
        assert req.modules == ["modal", "harmonic", "stress"]
        assert req.material == "Aluminum 7075-T6"
        assert req.task_id == "my-chain-task"

    def test_invalid_width_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], width_mm=-1.0)

    def test_invalid_damping_ratio_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], damping_ratio=0)

    def test_invalid_freq_range_percent_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], freq_range_percent=60.0)

    def test_invalid_n_freq_points_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], n_freq_points=5)

    def test_invalid_Kt_below_one_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], Kt_global=0.5)

    def test_invalid_reliability_too_low_rejected(self):
        from pydantic import ValidationError
        from web.routers.geometry import ChainRequest

        with pytest.raises(ValidationError):
            ChainRequest(modules=["modal"], reliability_pct=10.0)


# ---------------------------------------------------------------------------
# Tests for ChainResponse model
# ---------------------------------------------------------------------------


class TestChainResponseModel:
    """Tests for the ChainResponse pydantic model."""

    def test_from_dict(self):
        from web.routers.geometry import ChainResponse

        resp = ChainResponse(**MOCK_CHAIN_RESULT)
        assert resp.task_id == "chain-task-001"
        assert resp.modules_executed == ["modal", "harmonic", "stress", "fatigue"]
        assert resp.modal is not None
        assert resp.harmonic is not None
        assert resp.stress is not None
        assert resp.fatigue is not None
        assert resp.total_solve_time_s == 4.5
        assert resp.node_count == 1500
        assert resp.element_count == 800

    def test_defaults(self):
        from web.routers.geometry import ChainResponse

        resp = ChainResponse()
        assert resp.task_id is None
        assert resp.modules_executed == []
        assert resp.modal is None
        assert resp.harmonic is None
        assert resp.stress is None
        assert resp.fatigue is None
        assert resp.uniformity is None
        assert resp.static is None
        assert resp.total_solve_time_s == 0.0
        assert resp.node_count == 0
        assert resp.element_count == 0

    def test_partial_modules(self):
        from web.routers.geometry import ChainResponse

        resp = ChainResponse(
            modules_executed=["modal"],
            modal={"mode_shapes": [], "closest_mode_hz": 20000.0},
            total_solve_time_s=1.2,
            node_count=500,
            element_count=200,
        )
        assert resp.modal is not None
        assert resp.harmonic is None
        assert resp.stress is None


# ---------------------------------------------------------------------------
# Tests for POST /api/v1/geometry/fea/run-chain endpoint
# ---------------------------------------------------------------------------


class TestRunChainEndpoint:
    """Tests for POST /api/v1/geometry/fea/run-chain."""

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_success(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_CHAIN_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={
                "modules": ["modal", "harmonic", "stress", "fatigue"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["modules_executed"] == ["modal", "harmonic", "stress", "fatigue"]
        assert data["modal"] is not None
        assert data["harmonic"] is not None
        assert data["stress"] is not None
        assert data["fatigue"] is not None
        assert data["total_solve_time_s"] == 4.5
        assert data["node_count"] == 1500

        # Verify subprocess was called with correct task_type
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "chain"
        params = call_args[0][1]
        assert params["chain_modules"] == ["modal", "harmonic", "stress", "fatigue"]

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_with_defaults(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_CHAIN_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["modal"]},
        )

        assert response.status_code == 200
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[0][0] == "chain"
        params = call_args[0][1]
        # modal alone should resolve to ["modal"]
        assert params["chain_modules"] == ["modal"]

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_resolves_dependencies(self, mock_subprocess, client):
        """Requesting only fatigue should auto-resolve to full chain."""
        mock_subprocess.return_value = MOCK_CHAIN_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["fatigue"]},
        )

        assert response.status_code == 200
        params = mock_subprocess.call_args[0][1]
        # fatigue depends on stress -> harmonic -> modal
        assert params["chain_modules"] == ["modal", "harmonic", "stress", "fatigue"]

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_with_task_id(self, mock_subprocess, client):
        mock_subprocess.return_value = MOCK_CHAIN_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["modal"], "task_id": "chain-client-uuid-123"},
        )

        assert response.status_code == 200
        call_args = mock_subprocess.call_args
        assert call_args[1]["client_task_id"] == "chain-client-uuid-123"

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_passes_params(self, mock_subprocess, client):
        """Verify all parameters are forwarded to the subprocess."""
        mock_subprocess.return_value = MOCK_CHAIN_RESULT.copy()

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={
                "modules": ["fatigue"],
                "material": "Aluminum 7075-T6",
                "frequency_khz": 15.0,
                "horn_type": "flat",
                "width_mm": 30.0,
                "surface_finish": "polished",
                "Kt_global": 2.5,
                "reliability_pct": 99.0,
                "damping_model": "rayleigh",
                "damping_ratio": 0.01,
                "freq_range_percent": 10.0,
                "n_freq_points": 101,
            },
        )

        assert response.status_code == 200
        params = mock_subprocess.call_args[0][1]
        assert params["material"] == "Aluminum 7075-T6"
        assert params["frequency_khz"] == 15.0
        assert params["horn_type"] == "flat"
        assert params["surface_finish"] == "polished"
        assert params["Kt_global"] == 2.5
        assert params["reliability_pct"] == 99.0
        assert params["damping_model"] == "rayleigh"
        assert params["damping_ratio"] == 0.01
        assert params["n_freq_points"] == 101
        # freq range: 15kHz +/- 10% = 13500..16500
        assert params["freq_min_hz"] == pytest.approx(13500.0)
        assert params["freq_max_hz"] == pytest.approx(16500.0)

    @patch("web.routers.geometry._run_fea_subprocess", new_callable=AsyncMock)
    def test_chain_endpoint_fea_failure(self, mock_subprocess, client):
        mock_subprocess.side_effect = Exception("Solver diverged in chain")

        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["modal"]},
        )

        assert response.status_code == 500
        assert "Chain analysis failed" in response.json()["detail"]

    def test_chain_endpoint_empty_modules_rejected(self, client):
        """Empty modules list should be rejected."""
        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": []},
        )
        assert response.status_code == 400
        assert "No analysis modules specified" in response.json()["detail"]

    def test_chain_endpoint_unknown_module_rejected(self, client):
        """Unknown module names should be rejected."""
        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["nonexistent_module"]},
        )
        assert response.status_code == 400
        assert "Unknown analysis modules" in response.json()["detail"]

    def test_chain_endpoint_invalid_width_rejected(self, client):
        """Negative width should be rejected by pydantic validation."""
        response = client.post(
            "/api/v1/geometry/fea/run-chain",
            json={"modules": ["modal"], "width_mm": -5.0},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests for chain-related configuration in fea_process_runner
# ---------------------------------------------------------------------------


class TestFEAProcessRunnerChainConfig:
    """Tests for chain-related configuration in fea_process_runner."""

    def test_worker_map_has_chain(self):
        from web.services.fea_process_runner import _WORKER_MAP

        assert "chain" in _WORKER_MAP

    def test_phase_weights_cover_chain_phases(self):
        from web.services.fea_process_runner import PHASE_WEIGHTS

        chain_phases = [
            "init", "modal_run", "harmonic_run", "stress_run",
            "fatigue_run", "static_run", "uniformity_run", "packaging",
        ]
        for phase in chain_phases:
            assert phase in PHASE_WEIGHTS, f"Phase weight missing for: {phase}"
