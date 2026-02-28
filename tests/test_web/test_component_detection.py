"""Tests for STEP assembly component auto-detection."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from web.app import create_app
from web.services.component_detector import ComponentDetector, COMPONENT_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Create a TestClient for each test."""
    app = create_app()
    return TestClient(app)


@pytest.fixture()
def detector() -> ComponentDetector:
    """Create a ComponentDetector instance."""
    return ComponentDetector()


@pytest.fixture()
def minimal_step_file() -> str:
    """Create a minimal STEP file with CARTESIAN_POINT data for testing."""
    step_content = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('STEP AP214'),'2;1');
FILE_NAME('test.step','2024-01-01',('author'),('org'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(-12.5,-40.0,-12.5));
#2=CARTESIAN_POINT('',(12.5,-40.0,-12.5));
#3=CARTESIAN_POINT('',(12.5,40.0,-12.5));
#4=CARTESIAN_POINT('',(-12.5,40.0,-12.5));
#5=CARTESIAN_POINT('',(-12.5,-40.0,12.5));
#6=CARTESIAN_POINT('',(12.5,-40.0,12.5));
#7=CARTESIAN_POINT('',(12.5,40.0,12.5));
#8=CARTESIAN_POINT('',(-12.5,40.0,12.5));
ENDSEC;
END-ISO-10303-21;
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".step", delete=False
    ) as f:
        f.write(step_content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture()
def horn_step_file() -> str:
    """Create a STEP file that looks like a horn (tall, tapered)."""
    # Tapered: wider at bottom, narrower at top (aspect ratio > 2)
    step_content = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('STEP AP214'),'2;1');
FILE_NAME('horn.step','2024-01-01',('author'),('org'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(-20.0,0.0,-20.0));
#2=CARTESIAN_POINT('',(20.0,0.0,-20.0));
#3=CARTESIAN_POINT('',(20.0,0.0,20.0));
#4=CARTESIAN_POINT('',(-20.0,0.0,20.0));
#5=CARTESIAN_POINT('',(-10.0,80.0,-10.0));
#6=CARTESIAN_POINT('',(10.0,80.0,-10.0));
#7=CARTESIAN_POINT('',(10.0,80.0,10.0));
#8=CARTESIAN_POINT('',(-10.0,80.0,10.0));
ENDSEC;
END-ISO-10303-21;
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".step", delete=False
    ) as f:
        f.write(step_content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture()
def booster_step_file() -> str:
    """Create a STEP file that looks like a booster (long cylinder, aspect > 3)."""
    step_content = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('STEP AP214'),'2;1');
FILE_NAME('booster.step','2024-01-01',('author'),('org'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(-15.0,0.0,-15.0));
#2=CARTESIAN_POINT('',(15.0,0.0,-15.0));
#3=CARTESIAN_POINT('',(15.0,0.0,15.0));
#4=CARTESIAN_POINT('',(-15.0,0.0,15.0));
#5=CARTESIAN_POINT('',(-15.0,120.0,-15.0));
#6=CARTESIAN_POINT('',(15.0,120.0,-15.0));
#7=CARTESIAN_POINT('',(15.0,120.0,15.0));
#8=CARTESIAN_POINT('',(-15.0,120.0,15.0));
ENDSEC;
END-ISO-10303-21;
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".step", delete=False
    ) as f:
        f.write(step_content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture()
def empty_step_file() -> str:
    """Create a STEP file with no geometry points."""
    step_content = """\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('STEP AP214'),'2;1');
FILE_NAME('empty.step','2024-01-01',('author'),('org'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".step", delete=False
    ) as f:
        f.write(step_content)
        path = f.name
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# Unit tests: ComponentDetector._classify_component
# ---------------------------------------------------------------------------


class TestClassifyComponent:
    """Tests for the static classification heuristics."""

    def test_booster_long_cylinder(self, detector: ComponentDetector) -> None:
        """Long cylinder (aspect > 3, round cross-section) -> booster."""
        result = detector._classify_component(
            width=30.0, height=120.0, length=30.0, volume=84823.0
        )
        assert result == "booster"

    def test_horn_tapered(self, detector: ComponentDetector) -> None:
        """Tapered shape (aspect > 2, low fill) -> horn."""
        # bbox volume = 40 * 80 * 40 = 128000, actual 64000 => fill = 0.5
        result = detector._classify_component(
            width=40.0, height=80.0, length=40.0, volume=64000.0
        )
        assert result == "horn"

    def test_transducer_disc(self, detector: ComponentDetector) -> None:
        """Short wide disc (low aspect ratio, round cross) -> transducer."""
        result = detector._classify_component(
            width=50.0, height=40.0, length=50.0, volume=78540.0
        )
        assert result == "transducer"

    def test_anvil_flat_base(self, detector: ComponentDetector) -> None:
        """Flat base shape (width >> height) -> anvil."""
        result = detector._classify_component(
            width=100.0, height=10.0, length=80.0, volume=60000.0
        )
        assert result == "anvil"

    def test_workpiece_small_thin(self, detector: ComponentDetector) -> None:
        """Small thin shape -> workpiece."""
        result = detector._classify_component(
            width=30.0, height=3.0, length=20.0, volume=1500.0
        )
        assert result == "workpiece"

    def test_unknown_zero_dims(self, detector: ComponentDetector) -> None:
        """Zero dimensions -> unknown."""
        result = detector._classify_component(
            width=0.0, height=0.0, length=0.0, volume=0.0
        )
        assert result == "unknown"

    def test_all_types_are_valid(self) -> None:
        """Verify COMPONENT_TYPES contains expected types."""
        expected = {"horn", "booster", "transducer", "anvil", "workpiece", "unknown"}
        assert set(COMPONENT_TYPES) == expected


# ---------------------------------------------------------------------------
# Unit tests: ComponentDetector.detect (text fallback path)
# ---------------------------------------------------------------------------


class TestDetectTextFallback:
    """Tests for detection using the STEP text parser fallback."""

    def test_detect_minimal_step(
        self, detector: ComponentDetector, minimal_step_file: str
    ) -> None:
        """Detection on a minimal STEP file returns at least one component."""
        results = detector.detect(minimal_step_file)
        assert isinstance(results, list)
        assert len(results) >= 1

        comp = results[0]
        assert "type" in comp
        assert comp["type"] in COMPONENT_TYPES
        assert "name" in comp
        assert "volume_mm3" in comp
        assert "bbox" in comp
        assert len(comp["bbox"]) == 6
        assert "centroid" in comp
        assert len(comp["centroid"]) == 3
        assert "dimensions" in comp
        assert "width_mm" in comp["dimensions"]
        assert "height_mm" in comp["dimensions"]
        assert "length_mm" in comp["dimensions"]

    def test_detect_horn_geometry(
        self, detector: ComponentDetector, horn_step_file: str
    ) -> None:
        """A horn-shaped STEP file should be classified as horn."""
        results = detector.detect(horn_step_file)
        assert len(results) == 1
        assert results[0]["type"] == "horn"
        # Dimensions should reflect the tapered shape
        dims = results[0]["dimensions"]
        assert dims["height_mm"] == 80.0
        assert dims["width_mm"] == 40.0

    def test_detect_booster_geometry(
        self, detector: ComponentDetector, booster_step_file: str
    ) -> None:
        """A booster-shaped STEP file should be classified as booster."""
        results = detector.detect(booster_step_file)
        assert len(results) == 1
        assert results[0]["type"] == "booster"
        dims = results[0]["dimensions"]
        assert dims["height_mm"] == 120.0

    def test_detect_empty_step_returns_fallback(
        self, detector: ComponentDetector, empty_step_file: str
    ) -> None:
        """An empty STEP file returns a graceful fallback result."""
        results = detector.detect(empty_step_file)
        assert isinstance(results, list)
        assert len(results) >= 1
        # Should be unknown or parse_error
        comp = results[0]
        assert comp["type"] == "unknown"

    def test_detect_nonexistent_file(
        self, detector: ComponentDetector
    ) -> None:
        """Detection on a nonexistent file returns graceful error."""
        results = detector.detect("/nonexistent/path/file.step")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert results[0]["type"] == "unknown"
        assert results[0]["name"] == "parse_error"

    def test_volume_is_positive(
        self, detector: ComponentDetector, minimal_step_file: str
    ) -> None:
        """Volume should be positive for valid geometry."""
        results = detector.detect(minimal_step_file)
        assert results[0]["volume_mm3"] > 0


# ---------------------------------------------------------------------------
# Unit tests: ComponentDetector._refine_names
# ---------------------------------------------------------------------------


class TestRefineNames:
    """Tests for name refinement logic."""

    def test_single_type_no_suffix(self) -> None:
        """Single component of a type gets no numeric suffix."""
        components = [
            {"type": "horn", "name": "horn_1"},
            {"type": "booster", "name": "booster_1"},
        ]
        ComponentDetector._refine_names(components)
        assert components[0]["name"] == "horn"
        assert components[1]["name"] == "booster"

    def test_multiple_same_type_gets_suffix(self) -> None:
        """Multiple components of the same type keep numeric suffixes."""
        components = [
            {"type": "horn", "name": "horn_1"},
            {"type": "horn", "name": "horn_2"},
        ]
        ComponentDetector._refine_names(components)
        assert components[0]["name"] == "horn_1"
        assert components[1]["name"] == "horn_2"


# ---------------------------------------------------------------------------
# Unit tests: ComponentDetector with mocked cadquery
# ---------------------------------------------------------------------------


class TestDetectWithMockedCadquery:
    """Tests for the cadquery detection path using mocks."""

    def _build_mock_solid(
        self,
        volume: float,
        centroid: tuple[float, float, float],
        bbox: tuple[float, float, float, float, float, float],
    ) -> MagicMock:
        """Build a mock OCC solid with GProp and Bnd behavior."""
        solid = MagicMock()
        return solid

    @patch("web.services.component_detector.ComponentDetector._detect_with_cadquery")
    def test_detect_uses_cadquery_when_available(
        self, mock_cq_detect: MagicMock, detector: ComponentDetector
    ) -> None:
        """When cadquery is available, detect() uses that path."""
        mock_cq_detect.return_value = [
            {
                "type": "horn",
                "name": "horn",
                "volume_mm3": 50000.0,
                "bbox": [-12.5, 0.0, -12.5, 12.5, 80.0, 12.5],
                "centroid": [0.0, 40.0, 0.0],
                "dimensions": {"width_mm": 25.0, "height_mm": 80.0, "length_mm": 25.0},
            }
        ]
        results = detector.detect("some_file.step")
        assert len(results) == 1
        assert results[0]["type"] == "horn"
        mock_cq_detect.assert_called_once_with("some_file.step")

    @patch("web.services.component_detector.ComponentDetector._detect_with_cadquery")
    @patch("web.services.component_detector.ComponentDetector._detect_from_step_text")
    def test_falls_back_to_text_on_cadquery_error(
        self,
        mock_text_detect: MagicMock,
        mock_cq_detect: MagicMock,
        detector: ComponentDetector,
    ) -> None:
        """When cadquery fails, falls back to text-based parser."""
        mock_cq_detect.side_effect = ImportError("No module named 'cadquery'")
        mock_text_detect.return_value = [
            {
                "type": "unknown",
                "name": "unknown_1",
                "volume_mm3": 1000.0,
                "bbox": [0, 0, 0, 10, 10, 10],
                "centroid": [5, 5, 5],
                "dimensions": {"width_mm": 10, "height_mm": 10, "length_mm": 10},
            }
        ]
        results = detector.detect("some_file.step")
        assert len(results) == 1
        mock_text_detect.assert_called_once_with("some_file.step")

    @patch("web.services.component_detector.ComponentDetector._detect_with_cadquery")
    @patch("web.services.component_detector.ComponentDetector._detect_from_step_text")
    def test_returns_error_when_both_fail(
        self,
        mock_text_detect: MagicMock,
        mock_cq_detect: MagicMock,
        detector: ComponentDetector,
    ) -> None:
        """When both paths fail, returns a graceful error result."""
        mock_cq_detect.side_effect = ImportError("no cadquery")
        mock_text_detect.side_effect = RuntimeError("no points")
        results = detector.detect("bad_file.step")
        assert len(results) == 1
        assert results[0]["type"] == "unknown"
        assert results[0]["name"] == "parse_error"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestDetectComponentsEndpoint:
    """Tests for POST /api/v1/geometry/detect-components."""

    def test_no_file_returns_422(self, client: TestClient) -> None:
        """Missing file returns 422."""
        response = client.post("/api/v1/geometry/detect-components")
        assert response.status_code == 422

    def test_wrong_extension_returns_400(self, client: TestClient) -> None:
        """Non-STEP file returns 400."""
        response = client.post(
            "/api/v1/geometry/detect-components",
            files={"file": ("model.stl", b"solid empty endsolid", "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Only STEP files" in response.json()["detail"]

    @patch("web.services.component_detector.ComponentDetector.detect")
    def test_valid_step_returns_components(
        self, mock_detect: MagicMock, client: TestClient
    ) -> None:
        """Valid STEP file returns detected components."""
        mock_detect.return_value = [
            {
                "type": "horn",
                "name": "horn",
                "volume_mm3": 50000.0,
                "bbox": [-12.5, 0, -12.5, 12.5, 80, 12.5],
                "centroid": [0, 40, 0],
                "dimensions": {"width_mm": 25, "height_mm": 80, "length_mm": 25},
            },
            {
                "type": "booster",
                "name": "booster",
                "volume_mm3": 80000.0,
                "bbox": [-20, -100, -20, 20, 0, 20],
                "centroid": [0, -50, 0],
                "dimensions": {"width_mm": 40, "height_mm": 100, "length_mm": 40},
            },
        ]

        step_content = b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        response = client.post(
            "/api/v1/geometry/detect-components",
            files={"file": ("assembly.step", step_content, "application/octet-stream")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["filename"] == "assembly.step"
        assert len(data["components"]) == 2
        assert data["components"][0]["type"] == "horn"
        assert data["components"][1]["type"] == "booster"

    @patch("web.services.component_detector.ComponentDetector.detect")
    def test_response_component_fields(
        self, mock_detect: MagicMock, client: TestClient
    ) -> None:
        """Response components have all required fields."""
        mock_detect.return_value = [
            {
                "type": "transducer",
                "name": "transducer",
                "volume_mm3": 30000.0,
                "bbox": [-25, -20, -25, 25, 20, 25],
                "centroid": [0, 0, 0],
                "dimensions": {"width_mm": 50, "height_mm": 40, "length_mm": 50},
            }
        ]

        step_content = b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        response = client.post(
            "/api/v1/geometry/detect-components",
            files={"file": ("part.stp", step_content, "application/octet-stream")},
        )
        assert response.status_code == 200
        comp = response.json()["components"][0]
        assert comp["type"] == "transducer"
        assert comp["name"] == "transducer"
        assert comp["volume_mm3"] == 30000.0
        assert len(comp["bbox"]) == 6
        assert len(comp["centroid"]) == 3
        assert "width_mm" in comp["dimensions"]
        assert "height_mm" in comp["dimensions"]
        assert "length_mm" in comp["dimensions"]

    @patch("web.services.component_detector.ComponentDetector.detect")
    def test_detect_runtime_error_returns_400(
        self, mock_detect: MagicMock, client: TestClient
    ) -> None:
        """RuntimeError from detector returns 400."""
        mock_detect.side_effect = RuntimeError("Corrupt STEP file")
        step_content = b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        response = client.post(
            "/api/v1/geometry/detect-components",
            files={"file": ("bad.step", step_content, "application/octet-stream")},
        )
        assert response.status_code == 400
        assert "Corrupt STEP file" in response.json()["detail"]

    @patch("web.services.component_detector.ComponentDetector.detect")
    def test_detect_unexpected_error_returns_500(
        self, mock_detect: MagicMock, client: TestClient
    ) -> None:
        """Unexpected exception returns 500."""
        mock_detect.side_effect = ValueError("unexpected")
        step_content = b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        response = client.post(
            "/api/v1/geometry/detect-components",
            files={"file": ("bad.step", step_content, "application/octet-stream")},
        )
        assert response.status_code == 500
        assert "Component detection failed" in response.json()["detail"]

    def test_stp_extension_accepted(self, client: TestClient) -> None:
        """Both .step and .stp extensions should be accepted."""
        with patch("web.services.component_detector.ComponentDetector.detect") as mock_detect:
            mock_detect.return_value = [
                {
                    "type": "unknown",
                    "name": "unknown",
                    "volume_mm3": 0,
                    "bbox": [0, 0, 0, 0, 0, 0],
                    "centroid": [0, 0, 0],
                    "dimensions": {"width_mm": 0, "height_mm": 0, "length_mm": 0},
                }
            ]
            step_content = b"ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
            response = client.post(
                "/api/v1/geometry/detect-components",
                files={"file": ("part.stp", step_content, "application/octet-stream")},
            )
            assert response.status_code == 200
