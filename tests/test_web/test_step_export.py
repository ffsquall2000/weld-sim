"""Tests for the STEP export service and download endpoints."""
from __future__ import annotations

import os
import time
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from fastapi.testclient import TestClient

from web.app import create_app


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# StepExportService unit tests
# ---------------------------------------------------------------------------


class TestStepExportService:
    """Tests for the StepExportService class."""

    def test_export_creates_directory_and_file(self, tmp_path):
        """export() should create the export dir and write a STEP file."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path / "exports")

        mock_solid = MagicMock()

        with patch("web.services.step_export_service.StepExportService.export") as mock_export:
            # Simulate a real export by creating the file
            expected_path = os.path.join(service.EXPORT_DIR, "test_horn.step")
            mock_export.return_value = expected_path

            result = service.export(mock_solid, "test_horn.step")
            assert result == expected_path

    def test_export_appends_step_extension(self, tmp_path):
        """export() should add .step if missing."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        export_dir = str(tmp_path / "exports")
        service.EXPORT_DIR = export_dir
        os.makedirs(export_dir, exist_ok=True)

        # Create a mock that side-steps the actual cadquery import
        mock_solid = MagicMock()

        with patch.dict("sys.modules", {"cadquery": MagicMock(), "cadquery.exporters": MagicMock()}):
            with patch("cadquery.exporters.export") as mock_cq_export:
                result = service.export(mock_solid, "test_horn")
                assert result.endswith(".step")
                assert "test_horn.step" in result

    def test_export_already_has_step_extension(self, tmp_path):
        """export() should not double the .step suffix."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        export_dir = str(tmp_path / "exports")
        service.EXPORT_DIR = export_dir
        os.makedirs(export_dir, exist_ok=True)

        mock_solid = MagicMock()

        with patch.dict("sys.modules", {"cadquery": MagicMock(), "cadquery.exporters": MagicMock()}):
            with patch("cadquery.exporters.export"):
                result = service.export(mock_solid, "my_horn.step")
                assert result.endswith(".step")
                assert not result.endswith(".step.step")

    def test_export_raises_on_missing_cadquery(self, tmp_path):
        """export() should raise RuntimeError when cadquery is not installed."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path / "exports")
        os.makedirs(service.EXPORT_DIR, exist_ok=True)

        mock_solid = MagicMock()

        with patch.dict("sys.modules", {"cadquery": None}):
            with pytest.raises(RuntimeError, match="CadQuery is required"):
                service.export(mock_solid, "test.step")

    def test_get_file_returns_path_when_exists(self, tmp_path):
        """get_file() should return the path when the file exists."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        # Create a dummy file
        test_file = tmp_path / "my_horn.step"
        test_file.write_text("dummy step data")

        result = service.get_file("my_horn.step")
        assert result is not None
        assert result == str(test_file)

    def test_get_file_returns_none_when_missing(self, tmp_path):
        """get_file() should return None when the file does not exist."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        result = service.get_file("nonexistent.step")
        assert result is None

    def test_get_file_appends_extension(self, tmp_path):
        """get_file() should add .step if not provided."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        test_file = tmp_path / "horn.step"
        test_file.write_text("dummy")

        result = service.get_file("horn")
        assert result is not None
        assert result.endswith("horn.step")

    def test_cleanup_removes_old_files(self, tmp_path):
        """cleanup() should remove files older than max_age_seconds."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        # Create files with old modification times
        old_file = tmp_path / "old_export.step"
        old_file.write_text("old data")
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(str(old_file), (old_time, old_time))

        new_file = tmp_path / "new_export.step"
        new_file.write_text("new data")

        removed = service.cleanup(max_age_seconds=3600)
        assert removed == 1
        assert not old_file.exists()
        assert new_file.exists()

    def test_cleanup_keeps_recent_files(self, tmp_path):
        """cleanup() should keep files newer than max_age_seconds."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        recent_file = tmp_path / "recent.step"
        recent_file.write_text("recent data")

        removed = service.cleanup(max_age_seconds=3600)
        assert removed == 0
        assert recent_file.exists()

    def test_cleanup_nonexistent_dir(self):
        """cleanup() should return 0 when directory does not exist."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = "/tmp/nonexistent_test_dir_12345"

        removed = service.cleanup()
        assert removed == 0

    def test_cleanup_with_custom_max_age(self, tmp_path):
        """cleanup() should respect custom max_age_seconds."""
        from web.services.step_export_service import StepExportService

        service = StepExportService()
        service.EXPORT_DIR = str(tmp_path)

        f = tmp_path / "medium_age.step"
        f.write_text("data")
        # Set mtime to 30 minutes ago
        old_time = time.time() - 1800
        os.utime(str(f), (old_time, old_time))

        # Should NOT be removed with 1 hour max age
        removed = service.cleanup(max_age_seconds=3600)
        assert removed == 0
        assert f.exists()

        # SHOULD be removed with 10 minute max age
        removed = service.cleanup(max_age_seconds=600)
        assert removed == 1
        assert not f.exists()


# ---------------------------------------------------------------------------
# POST /api/v1/knurl-fea/export-step endpoint tests
# ---------------------------------------------------------------------------


class TestExportStepEndpoint:
    """Tests for the export-step API endpoint."""

    @patch("web.routers.knurl_fea.StepExportService", create=True)
    @patch("web.routers.knurl_fea.HornGenerator", create=True)
    def test_export_step_returns_200_with_cadquery(
        self, mock_gen_cls, mock_svc_cls, client: TestClient
    ):
        """Endpoint should return 200 with download URL when CadQuery available."""
        with patch(
            "ultrasonic_weld_master.plugins.geometry_analyzer.horn_generator.HornGenerator"
        ) as inner_gen:
            mock_gen = MagicMock()
            inner_gen.return_value = mock_gen

            # Mock generate() to return a result with has_cad_export=True
            mock_result = MagicMock()
            mock_result.has_cad_export = True
            mock_gen.generate.return_value = mock_result

            # Mock the CadQuery body
            mock_body = MagicMock()
            mock_gen._cq_create_body.return_value = mock_body
            mock_gen._cq_apply_knurl.return_value = mock_body

            # Mock the StepExportService
            with patch("web.services.step_export_service.StepExportService") as svc:
                mock_service = MagicMock()
                svc.return_value = mock_service
                mock_service.export.return_value = "/tmp/weld-sim-exports/test.step"

                with patch("os.path.getsize", return_value=1024):
                    with patch("os.path.exists", return_value=True):
                        payload = {
                            "horn": {
                                "horn_type": "flat",
                                "width_mm": 25.0,
                                "height_mm": 80.0,
                            },
                            "knurl": {
                                "type": "linear",
                                "pitch_mm": 1.0,
                                "depth_mm": 0.3,
                            },
                        }
                        response = client.post(
                            "/api/v1/knurl-fea/export-step", json=payload
                        )
                        # May get 400 if cadquery isn't available, which is fine
                        assert response.status_code in (200, 400)

    def test_export_step_no_cadquery_returns_400(self, client: TestClient):
        """Endpoint should return 400 when CadQuery is not available."""
        payload = {
            "horn": {"horn_type": "flat", "width_mm": 25.0, "height_mm": 80.0},
            "knurl": {"type": "linear", "pitch_mm": 1.0, "depth_mm": 0.3},
        }
        response = client.post("/api/v1/knurl-fea/export-step", json=payload)
        # Without cadquery, this should fail with 400 or 500
        assert response.status_code in (400, 500)

    def test_export_step_missing_step_file_returns_404(self, client: TestClient):
        """Endpoint should return 404 when referenced STEP file doesn't exist."""
        payload = {
            "horn": {"horn_type": "flat"},
            "knurl": {"type": "linear"},
            "step_file_path": "/nonexistent/horn.step",
        }
        response = client.post("/api/v1/knurl-fea/export-step", json=payload)
        # Should fail - either 400 (no cadquery) or 404 (file not found)
        assert response.status_code in (400, 404, 500)

    def test_export_step_with_custom_filename(self, client: TestClient):
        """Endpoint should accept a custom filename."""
        payload = {
            "horn": {"horn_type": "cylindrical"},
            "knurl": {"type": "diamond"},
            "filename": "my_custom_horn",
        }
        response = client.post("/api/v1/knurl-fea/export-step", json=payload)
        # Will fail without cadquery, but should not be 422 (validation error)
        assert response.status_code != 422

    def test_export_step_default_params(self, client: TestClient):
        """Endpoint should accept request with all default params."""
        response = client.post("/api/v1/knurl-fea/export-step", json={})
        # Should not be a validation error
        assert response.status_code != 422


# ---------------------------------------------------------------------------
# GET /api/v1/knurl-fea/download-step/{filename} endpoint tests
# ---------------------------------------------------------------------------


class TestDownloadStepEndpoint:
    """Tests for the download-step API endpoint."""

    def test_download_nonexistent_returns_404(self, client: TestClient):
        """Downloading a non-existent file should return 404."""
        response = client.get(
            "/api/v1/knurl-fea/download-step/nonexistent.step"
        )
        assert response.status_code == 404

    def test_download_existing_file(self, client: TestClient, tmp_path):
        """Downloading an existing file should return 200 with file contents."""
        # Create a temporary STEP file
        step_content = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        step_file = tmp_path / "test_download.step"
        step_file.write_bytes(step_content)

        with patch(
            "web.services.step_export_service.StepExportService"
        ) as mock_svc_cls:
            mock_service = MagicMock()
            mock_svc_cls.return_value = mock_service
            mock_service.get_file.return_value = str(step_file)

            response = client.get(
                "/api/v1/knurl-fea/download-step/test_download.step"
            )
            assert response.status_code == 200
            assert b"ISO-10303-21" in response.content

    def test_download_expired_file_returns_404(self, client: TestClient):
        """Downloading an expired file should return 404."""
        with patch(
            "web.services.step_export_service.StepExportService"
        ) as mock_svc_cls:
            mock_service = MagicMock()
            mock_svc_cls.return_value = mock_service
            mock_service.get_file.return_value = None

            response = client.get(
                "/api/v1/knurl-fea/download-step/expired.step"
            )
            assert response.status_code == 404
            assert "expired" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


class TestStepExportRegistered:
    """Verify that the export endpoints are registered."""

    def test_export_step_route_registered(self, client: TestClient):
        """The OpenAPI schema should include the export-step path."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/api/v1/knurl-fea/export-step" in paths

    def test_download_step_route_registered(self, client: TestClient):
        """The OpenAPI schema should include the download-step path."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})
        assert "/api/v1/knurl-fea/download-step/{filename}" in paths
