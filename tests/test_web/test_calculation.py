"""Tests for the calculation / simulation endpoints."""
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


def test_simulate_returns_recipe(client: TestClient) -> None:
    response = client.post("/api/v1/simulate", json=SAMPLE_INPUT)
    assert response.status_code == 200
    data = response.json()
    assert "recipe_id" in data
    assert "parameters" in data
    assert "validation" in data
    assert data["application"] == "li_battery_tab"


def test_simulate_with_horn_params(client: TestClient) -> None:
    payload = {
        **SAMPLE_INPUT,
        "horn_type": "stepped",
        "horn_gain": 1.5,
        "knurl_type": "linear",
        "knurl_pitch_mm": 1.0,
        "knurl_tooth_width_mm": 0.5,
        "knurl_depth_mm": 0.3,
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recipe_id" in data


def test_simulate_invalid_application(client: TestClient) -> None:
    payload = {**SAMPLE_INPUT, "application": "nonexistent"}
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code in (400, 500)


def test_simulate_general_metal(client: TestClient) -> None:
    payload = {
        "application": "general_metal",
        "upper_material_type": "Nickel 201",
        "upper_thickness_mm": 0.5,
        "upper_layers": 1,
        "lower_material_type": "Copper C110",
        "lower_thickness_mm": 1.0,
        "weld_width_mm": 5.0,
        "weld_length_mm": 20.0,
        "frequency_khz": 20.0,
        "max_power_w": 3500,
    }
    response = client.post("/api/v1/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == "general_metal"


def test_batch_simulate(client: TestClient) -> None:
    payload = {"items": [SAMPLE_INPUT, SAMPLE_INPUT]}
    response = client.post("/api/v1/simulate/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_get_schema(client: TestClient) -> None:
    response = client.get("/api/v1/simulate/schema/li_battery_tab")
    assert response.status_code == 200
    data = response.json()
    assert "properties" in data
