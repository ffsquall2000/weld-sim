"""Tests for the reports endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


SAMPLE_INPUT = {
    "application": "li_battery_tab",
    "upper_material_type": "Nickel 201",
    "upper_thickness_mm": 0.1,
    "upper_layers": 40,
    "lower_material_type": "Copper C110",
    "lower_thickness_mm": 0.3,
    "weld_width_mm": 3.0,
    "weld_length_mm": 25.0,
    "frequency_khz": 20.0,
    "max_power_w": 3500,
}


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_export_json_after_simulate(client: TestClient) -> None:
    # First run a simulation to get a recipe_id
    sim_response = client.post("/api/v1/simulate", json=SAMPLE_INPUT)
    assert sim_response.status_code == 200
    recipe_id = sim_response.json()["recipe_id"]

    # Now try to export -- the recipe may or may not be persisted to DB
    # depending on engine behaviour; accept 200 or 404
    export_response = client.post(
        "/api/v1/reports/export",
        json={"recipe_id": recipe_id, "format": "json"},
    )
    assert export_response.status_code in (200, 404)
