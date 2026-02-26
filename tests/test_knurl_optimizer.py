"""Tests for knurl pattern optimizer."""
import pytest
from web.services.knurl_service import KnurlOptimizer


class TestKnurlOptimizer:
    def setup_method(self):
        self.optimizer = KnurlOptimizer()

    def test_basic_optimization(self):
        """Basic optimization should return results."""
        result = self.optimizer.optimize(
            upper_material="Cu", lower_material="Al",
            upper_hardness_hv=50, lower_hardness_hv=23,
        )
        assert "recommendations" in result
        assert "pareto_front" in result
        assert "analysis_summary" in result
        assert len(result["recommendations"]) > 0

    def test_recommendations_sorted_by_score(self):
        """Recommendations should be sorted by overall_score descending."""
        result = self.optimizer.optimize(upper_material="Cu", lower_material="Al")
        scores = [r["overall_score"] for r in result["recommendations"]]
        assert scores == sorted(scores, reverse=True)

    def test_pareto_front_not_empty(self):
        """Pareto front should have at least 1 entry."""
        result = self.optimizer.optimize(upper_material="Cu", lower_material="Al")
        assert len(result["pareto_front"]) > 0

    def test_max_results_limits_output(self):
        """max_results should limit returned recommendations."""
        result = self.optimizer.optimize(
            upper_material="Cu", lower_material="Al", max_results=3
        )
        assert len(result["recommendations"]) <= 3

    def test_score_range(self):
        """All scores should be in [0, 1+bonus] range."""
        result = self.optimizer.optimize(upper_material="Cu", lower_material="Al")
        for rec in result["recommendations"]:
            assert 0 <= rec["energy_coupling_efficiency"] <= 1.0
            assert 0 <= rec["material_damage_index"] <= 1.0
            assert rec["overall_score"] >= 0
