"""Tests for the materials endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_list_materials(client: TestClient) -> None:
    response = client.get("/api/v1/materials")
    assert response.status_code == 200
    data = response.json()
    assert "materials" in data
    assert len(data["materials"]) > 0


def test_get_material(client: TestClient) -> None:
    # Material keys in the database are short codes like "Cu", "Al", "Ni"
    response = client.get("/api/v1/materials/Ni")
    assert response.status_code == 200
    data = response.json()
    assert "properties" in data


def test_get_material_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/materials/FakeMaterial999")
    assert response.status_code == 404


def test_get_combination(client: TestClient) -> None:
    response = client.get("/api/v1/materials/combination/Cu/Al")
    assert response.status_code == 200
