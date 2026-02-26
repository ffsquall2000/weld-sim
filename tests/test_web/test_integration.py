"""End-to-end integration test: full workflow."""
import pytest
from fastapi.testclient import TestClient
from web.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_full_calculation_workflow(client):
    # 1. List materials
    resp = client.get("/api/v1/materials")
    assert resp.status_code == 200
    materials = resp.json()["materials"]
    assert len(materials) > 0
    assert "Ni" in materials

    # 2. Get material details
    resp = client.get("/api/v1/materials/Ni")
    assert resp.status_code == 200
    assert "properties" in resp.json()

    # 3. Get material combination
    resp = client.get("/api/v1/materials/combination/Ni/Cu")
    assert resp.status_code == 200

    # 4. Run simulation
    sim_resp = client.post("/api/v1/simulate", json={
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
    })
    assert sim_resp.status_code == 200
    result = sim_resp.json()
    assert "recipe_id" in result
    assert result["application"] == "li_battery_tab"
    assert len(result["parameters"]) > 0
    assert result["validation"]["status"] in ("pass", "warning", "fail")
    assert len(result["safety_window"]) > 0
    assert len(result["risk_assessment"]) > 0

    # 5. Run with horn/knurl parameters
    sim2 = client.post("/api/v1/simulate", json={
        "application": "li_battery_tab",
        "upper_material_type": "Nickel 201",
        "upper_thickness_mm": 0.1,
        "upper_layers": 40,
        "lower_material_type": "Copper C110",
        "lower_thickness_mm": 0.3,
        "weld_width_mm": 3.0,
        "weld_length_mm": 25.0,
        "horn_type": "flat",
        "knurl_type": "cross_hatch",
        "knurl_pitch_mm": 1.2,
        "knurl_tooth_width_mm": 0.6,
        "knurl_depth_mm": 0.35,
        "anvil_type": "knurled",
        "booster_gain": 1.5,
    })
    assert sim2.status_code == 200

    # 6. Run general metal simulation
    sim3 = client.post("/api/v1/simulate", json={
        "application": "general_metal",
        "upper_material_type": "Aluminum 1100",
        "upper_thickness_mm": 0.5,
        "upper_layers": 1,
        "lower_material_type": "Copper C110",
        "lower_thickness_mm": 1.0,
        "weld_width_mm": 5.0,
        "weld_length_mm": 30.0,
    })
    assert sim3.status_code == 200

    # 7. Batch simulation
    batch_resp = client.post("/api/v1/simulate/batch", json={
        "items": [
            {"application": "li_battery_tab", "upper_material_type": "Nickel 201",
             "upper_thickness_mm": 0.1, "upper_layers": 20,
             "lower_material_type": "Copper C110", "lower_thickness_mm": 0.3,
             "weld_width_mm": 3.0, "weld_length_mm": 25.0},
            {"application": "li_battery_tab", "upper_material_type": "Nickel 201",
             "upper_thickness_mm": 0.1, "upper_layers": 60,
             "lower_material_type": "Copper C110", "lower_thickness_mm": 0.3,
             "weld_width_mm": 3.0, "weld_length_mm": 25.0},
        ]
    })
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert len(batch_data["results"]) == 2

    # 8. Get input schema
    schema_resp = client.get("/api/v1/simulate/schema/li_battery_tab")
    assert schema_resp.status_code == 200
    assert "properties" in schema_resp.json()

    # 9. List recipes
    recipes_resp = client.get("/api/v1/recipes")
    assert recipes_resp.status_code == 200


def test_error_handling(client):
    # Invalid application
    resp = client.post("/api/v1/simulate", json={
        "application": "nonexistent",
        "upper_material_type": "Nickel 201",
        "upper_thickness_mm": 0.1,
        "upper_layers": 40,
        "lower_material_type": "Copper C110",
        "lower_thickness_mm": 0.3,
        "weld_width_mm": 3.0,
        "weld_length_mm": 25.0,
    })
    assert resp.status_code in (400, 500)

    # Missing required field
    resp = client.post("/api/v1/simulate", json={
        "application": "li_battery_tab",
    })
    assert resp.status_code == 422  # Pydantic validation error

    # Material not found
    resp = client.get("/api/v1/materials/FakeMaterial999")
    assert resp.status_code == 404

    # Schema for unknown app
    resp = client.get("/api/v1/simulate/schema/nonexistent")
    assert resp.status_code in (400, 404)
