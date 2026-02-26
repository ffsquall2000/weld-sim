"""Tests for the recipes endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_list_recipes(client: TestClient) -> None:
    response = client.get("/api/v1/recipes")
    assert response.status_code == 200
    data = response.json()
    assert "recipes" in data
    assert "count" in data
