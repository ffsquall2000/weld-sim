"""Tests for the KnurlFEAOptimizer service and optimize endpoint."""
from __future__ import annotations

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
# KnurlFEAOptimizer unit tests
# ---------------------------------------------------------------------------


class TestKnurlFEAOptimizerGrid:
    """Tests for grid generation and analytical prescreening."""

    def test_generate_grid_produces_candidates(self):
        """Grid should contain candidates for all types x pitches x depths."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        grid = optimizer._generate_grid()

        assert len(grid) > 0
        # 3 types x 6 pitches x 4 depths = 72 max, minus invalid combos
        assert len(grid) <= 72

    def test_grid_candidate_fields(self):
        """Each candidate should have required fields."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        grid = optimizer._generate_grid()

        for c in grid:
            assert c.knurl_type in ("linear", "cross_hatch", "diamond")
            assert c.pitch_mm > 0
            assert c.depth_mm > 0
            assert c.tooth_width_mm > 0
            assert c.tooth_width_mm < c.pitch_mm

    def test_grid_excludes_invalid_combos(self):
        """Grid should not contain configs where depth > pitch."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        grid = optimizer._generate_grid()

        for c in grid:
            assert c.depth_mm <= c.pitch_mm
            assert c.tooth_width_mm < c.pitch_mm

    def test_analytical_prescreening_sorts_by_score(self):
        """Prescreening should sort candidates by descending analytical score."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        grid = optimizer._generate_grid()
        scored = optimizer._analytical_prescreening(grid, 20.0)

        scores = [c.analytical_score for c in scored]
        assert scores == sorted(scores, reverse=True)

    def test_analytical_scores_positive(self):
        """All analytical scores should be positive."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        grid = optimizer._generate_grid()
        scored = optimizer._analytical_prescreening(grid, 20.0)

        for c in scored:
            assert c.analytical_score >= 0.0

    def test_compute_analytical_score_types(self):
        """Diamond and cross_hatch should generally score higher than linear."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
        )

        optimizer = KnurlFEAOptimizer()

        # Same pitch and depth, different types
        linear = CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2, tooth_width_mm=0.5
        )
        diamond = CandidateConfig(
            knurl_type="diamond", pitch_mm=1.0, depth_mm=0.2, tooth_width_mm=0.5
        )

        s_linear = optimizer._compute_analytical_score(linear, 20.0)
        s_diamond = optimizer._compute_analytical_score(diamond, 20.0)

        # Diamond gets a type bonus
        assert s_diamond > s_linear


class TestKnurlFEAOptimizerFallback:
    """Tests for the analytical fallback when FEA is unavailable."""

    def test_analytical_fallback_produces_result(self):
        """Fallback should return a valid FEAResult."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        optimizer = KnurlFEAOptimizer()
        candidate = CandidateConfig(
            knurl_type="linear",
            pitch_mm=1.0,
            depth_mm=0.2,
            tooth_width_mm=0.5,
            analytical_score=0.75,
        )

        result = optimizer._analytical_fallback(candidate, 20000.0, 0.01)

        assert isinstance(result, FEAResult)
        assert result.closest_mode_hz > 0
        assert result.frequency_deviation_percent >= 0
        assert 0 <= result.amplitude_uniformity <= 1.0
        assert result.fea_score >= 0
        assert result.node_count == 0  # analytical fallback has no mesh
        assert result.element_count == 0

    def test_analytical_fallback_uniformity_by_type(self):
        """Diamond should have higher estimated uniformity than linear."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
        )

        optimizer = KnurlFEAOptimizer()

        linear = CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2, tooth_width_mm=0.5
        )
        diamond = CandidateConfig(
            knurl_type="diamond", pitch_mm=1.0, depth_mm=0.2, tooth_width_mm=0.5
        )

        r_linear = optimizer._analytical_fallback(linear, 20000.0, 0.01)
        r_diamond = optimizer._analytical_fallback(diamond, 20000.0, 0.01)

        assert r_diamond.amplitude_uniformity > r_linear.amplitude_uniformity

    def test_analytical_fallback_deeper_reduces_uniformity(self):
        """Deeper knurl should reduce estimated uniformity."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
        )

        optimizer = KnurlFEAOptimizer()

        shallow = CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.1, tooth_width_mm=0.5
        )
        deep = CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.5, tooth_width_mm=0.5
        )

        r_shallow = optimizer._analytical_fallback(shallow, 20000.0, 0.01)
        r_deep = optimizer._analytical_fallback(deep, 20000.0, 0.01)

        assert r_shallow.amplitude_uniformity > r_deep.amplitude_uniformity


class TestKnurlFEAOptimizerPareto:
    """Tests for the Pareto front computation."""

    def test_pareto_front_single_result(self):
        """Single result should always be on the Pareto front."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        optimizer = KnurlFEAOptimizer()
        config = CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2
        )
        result = FEAResult(
            config=config,
            frequency_deviation_percent=1.0,
            amplitude_uniformity=0.8,
            fea_score=0.7,
        )

        pareto = optimizer._compute_pareto_front([result])
        assert len(pareto) == 1

    def test_pareto_front_dominated_excluded(self):
        """Dominated candidates should not be on the Pareto front."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        optimizer = KnurlFEAOptimizer()
        cfg = lambda: CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2
        )

        # A dominates B (lower deviation AND higher uniformity)
        a = FEAResult(
            config=cfg(),
            frequency_deviation_percent=0.5,
            amplitude_uniformity=0.9,
            fea_score=0.9,
        )
        b = FEAResult(
            config=cfg(),
            frequency_deviation_percent=1.0,
            amplitude_uniformity=0.8,
            fea_score=0.7,
        )

        pareto = optimizer._compute_pareto_front([a, b])
        assert len(pareto) == 1
        assert pareto[0] is a

    def test_pareto_front_tradeoff_both_included(self):
        """Non-dominated candidates (trade-off) should both be on the front."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        optimizer = KnurlFEAOptimizer()
        cfg = lambda: CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2
        )

        # A: better frequency match
        a = FEAResult(
            config=cfg(),
            frequency_deviation_percent=0.1,
            amplitude_uniformity=0.7,
            fea_score=0.8,
        )
        # B: better uniformity
        b = FEAResult(
            config=cfg(),
            frequency_deviation_percent=2.0,
            amplitude_uniformity=0.95,
            fea_score=0.75,
        )

        pareto = optimizer._compute_pareto_front([a, b])
        assert len(pareto) == 2

    def test_pareto_front_empty_input(self):
        """Empty input should produce empty Pareto front."""
        from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

        optimizer = KnurlFEAOptimizer()
        pareto = optimizer._compute_pareto_front([])
        assert pareto == []

    def test_pareto_front_sorted_by_fea_score(self):
        """Pareto front should be sorted by fea_score descending."""
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        optimizer = KnurlFEAOptimizer()
        cfg = lambda: CandidateConfig(
            knurl_type="linear", pitch_mm=1.0, depth_mm=0.2
        )

        results = [
            FEAResult(
                config=cfg(),
                frequency_deviation_percent=0.1 * (i + 1),
                amplitude_uniformity=0.9 - 0.05 * i,
                fea_score=0.9 - 0.1 * i,
            )
            for i in range(5)
        ]

        pareto = optimizer._compute_pareto_front(results)
        scores = [r.fea_score for r in pareto]
        assert scores == sorted(scores, reverse=True)


def _mock_fea_fallback():
    """Context manager that forces the optimizer to use analytical fallback.

    Replaces _run_fea_sync with a version that always uses the
    analytical fallback, avoiding actual Gmsh/FEA solver calls.
    This makes tests fast and deterministic.
    """
    from web.services.knurl_fea_optimizer import KnurlFEAOptimizer

    original_fallback = KnurlFEAOptimizer._analytical_fallback

    def _fake_fea_sync(self, candidate, horn_config, material, target_hz):
        return original_fallback(self, candidate, target_hz, 0.001)

    return patch.object(
        KnurlFEAOptimizer,
        "_run_fea_sync",
        _fake_fea_sync,
    )


class TestKnurlFEAOptimizerOptimize:
    """Tests for the full optimize() method via the HTTP endpoint.

    All endpoint tests use the analytical fallback to avoid slow real
    FEA runs.  The _run_fea_sync method is patched to call the
    analytical fallback directly.
    """

    def test_optimize_returns_expected_keys(self, client: TestClient):
        """optimize() should return dict with all expected keys."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical", "width_mm": 25.0, "height_mm": 80.0},
                "material": "Titanium Ti-6Al-4V",
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            result = response.json()

            assert "candidates" in result
            assert "pareto_front" in result
            assert "best_frequency_match" in result
            assert "best_uniformity" in result
            assert "summary" in result

    def test_optimize_candidates_count(self, client: TestClient):
        """optimize() should return the requested number of candidates."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical", "width_mm": 25.0, "height_mm": 80.0},
                "frequency_khz": 20.0,
                "n_candidates": 5,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            assert len(response.json()["candidates"]) == 5

    def test_optimize_summary_fields(self, client: TestClient):
        """Summary should contain all expected fields."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "flat", "width_mm": 30.0, "height_mm": 80.0},
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

            summary = response.json()["summary"]
            assert summary["total_grid_size"] > 0
            assert summary["candidates_evaluated"] == 3
            assert summary["pareto_front_size"] > 0
            assert summary["total_time_s"] >= 0
            assert summary["target_frequency_khz"] == 20.0
            assert summary["material"] == "Titanium Ti-6Al-4V"

    def test_optimize_pareto_front_nonempty(self, client: TestClient):
        """Pareto front should not be empty if candidates are evaluated."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 5,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            assert len(response.json()["pareto_front"]) > 0

    def test_optimize_best_candidates_present(self, client: TestClient):
        """best_frequency_match and best_uniformity should be present."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            result = response.json()
            assert result["best_frequency_match"] is not None
            assert result["best_uniformity"] is not None

    def test_optimize_candidates_sorted_by_fea_score(self, client: TestClient):
        """Candidates should be sorted by fea_score descending."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 5,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

            scores = [c["fea_score"] for c in response.json()["candidates"]]
            assert scores == sorted(scores, reverse=True)

    def test_optimize_candidate_dict_fields(self, client: TestClient):
        """Each candidate dict should have all expected fields."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

            expected_keys = {
                "knurl_type",
                "pitch_mm",
                "depth_mm",
                "tooth_width_mm",
                "closest_mode_hz",
                "frequency_deviation_percent",
                "amplitude_uniformity",
                "node_count",
                "element_count",
                "solve_time_s",
                "mode_count",
                "analytical_score",
                "fea_score",
            }

            for candidate in response.json()["candidates"]:
                assert set(candidate.keys()) == expected_keys


class TestResultToDict:
    """Tests for the _result_to_dict serializer."""

    def test_result_to_dict_serialization(self):
        """_result_to_dict should produce JSON-serializable output."""
        import json
        from web.services.knurl_fea_optimizer import (
            KnurlFEAOptimizer,
            CandidateConfig,
            FEAResult,
        )

        config = CandidateConfig(
            knurl_type="diamond",
            pitch_mm=1.0,
            depth_mm=0.3,
            tooth_width_mm=0.5,
            analytical_score=0.8,
        )
        result = FEAResult(
            config=config,
            closest_mode_hz=20050.0,
            frequency_deviation_percent=0.25,
            amplitude_uniformity=0.85,
            node_count=500,
            element_count=1200,
            solve_time_s=2.5,
            mode_count=20,
            analytical_score=0.8,
            fea_score=0.82,
        )

        d = KnurlFEAOptimizer._result_to_dict(result)

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert json_str

        assert d["knurl_type"] == "diamond"
        assert d["pitch_mm"] == 1.0
        assert d["closest_mode_hz"] == 20050.0


# ---------------------------------------------------------------------------
# POST /api/v1/knurl-fea/optimize endpoint tests
# ---------------------------------------------------------------------------


class TestOptimizeEndpoint:
    """Tests for the optimize API endpoint.

    Uses _mock_fea_fallback to ensure fast test execution without
    requiring actual Gmsh/FEA solver availability.
    """

    def test_optimize_returns_200(self, client: TestClient):
        """Endpoint should return 200 with analytical fallback results."""
        with _mock_fea_fallback():
            payload = {
                "horn": {
                    "horn_type": "cylindrical",
                    "width_mm": 25.0,
                    "height_mm": 80.0,
                },
                "material": "Titanium Ti-6Al-4V",
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

    def test_optimize_response_structure(self, client: TestClient):
        """Response should have all expected fields."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

            data = response.json()
            assert "candidates" in data
            assert "pareto_front" in data
            assert "best_frequency_match" in data
            assert "best_uniformity" in data
            assert "summary" in data
            assert len(data["candidates"]) == 3
            assert data["summary"]["candidates_evaluated"] == 3

    def test_optimize_with_task_id(self, client: TestClient):
        """Endpoint should echo back the task_id."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 3,
                "task_id": "opt-test-123",
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            assert response.json()["task_id"] == "opt-test-123"

    def test_optimize_default_params(self, client: TestClient):
        """Endpoint should work with default params."""
        with _mock_fea_fallback():
            response = client.post("/api/v1/knurl-fea/optimize", json={})
            assert response.status_code == 200
            data = response.json()
            assert len(data["candidates"]) == 10  # default n_candidates

    def test_optimize_pareto_front_nonempty_endpoint(self, client: TestClient):
        """Pareto front should not be empty."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 20.0,
                "n_candidates": 5,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert len(data["pareto_front"]) > 0

    def test_optimize_invalid_frequency_rejected(self, client: TestClient):
        """Negative frequency should be rejected."""
        payload = {"frequency_khz": -5.0}
        response = client.post("/api/v1/knurl-fea/optimize", json=payload)
        assert response.status_code == 422

    def test_optimize_invalid_n_candidates_rejected(self, client: TestClient):
        """n_candidates=0 should be rejected."""
        payload = {"n_candidates": 0}
        response = client.post("/api/v1/knurl-fea/optimize", json=payload)
        assert response.status_code == 422

    def test_optimize_n_candidates_max_rejected(self, client: TestClient):
        """n_candidates > 50 should be rejected."""
        payload = {"n_candidates": 100}
        response = client.post("/api/v1/knurl-fea/optimize", json=payload)
        assert response.status_code == 422

    def test_optimize_flat_horn(self, client: TestClient):
        """Endpoint should work with flat horn type."""
        with _mock_fea_fallback():
            payload = {
                "horn": {
                    "horn_type": "flat",
                    "width_mm": 30.0,
                    "height_mm": 80.0,
                    "length_mm": 20.0,
                },
                "frequency_khz": 15.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200

    def test_optimize_summary_has_target_freq(self, client: TestClient):
        """Summary should reflect the requested target frequency."""
        with _mock_fea_fallback():
            payload = {
                "horn": {"horn_type": "cylindrical"},
                "frequency_khz": 35.0,
                "n_candidates": 3,
            }
            response = client.post("/api/v1/knurl-fea/optimize", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["summary"]["target_frequency_khz"] == 35.0


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestOptimizeRegistered:
    """Verify that the optimize endpoint is registered."""

    def test_optimize_route_registered(self, client: TestClient):
        """The OpenAPI schema should include the optimize path."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/api/v1/knurl-fea/optimize" in paths
