"""Tests for the FEniCSx contact mechanics solver.

All tests use mocks so that neither Docker nor FEniCSx are required.
Async methods are tested via ``asyncio.run()`` to avoid a dependency on
``pytest-asyncio``.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_fenicsx import (
    ContactConfig,
    ContactSolver,
    _DEFAULT_MATERIALS,
    _lookup_material,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Convenience wrapper for ``asyncio.run``."""
    return asyncio.run(coro)


def _make_mock_runner(result: dict | None = None):
    """Return a mock FEniCSxRunner whose ``run_script`` returns *result*."""
    if result is None:
        result = {
            "status": "success",
            "run_id": "abc123",
            "stdout": "done",
            "stderr": "",
            "output_files": {"result.json": "/tmp/result.json"},
            "results": {
                "contact_pressure_max_mpa": 12.5,
                "contact_pressure_mean_mpa": 6.3,
                "contact_pressure_distribution": [1.0, 2.0, 3.0],
                "slip_distance_max_um": 8.4,
                "slip_distance_mean_um": 3.1,
                "slip_distance_distribution": [0.5, 1.0, 1.5],
                "deformation_max_um": 25.6,
                "deformation_field": [0.1, 0.2, 0.3],
                "stress_von_mises_max_mpa": 45.2,
                "stress_field": [],
                "contact_area_mm2": 120.5,
                "total_force_n": 1500.0,
                "newton_iterations": 5,
                "converged": True,
                "solve_time_s": 12.3,
                "contact_type": "penalty",
            },
        }
    runner = mock.AsyncMock()
    runner.run_script = mock.AsyncMock(return_value=result)
    return runner


# ---------------------------------------------------------------------------
# Tests: ContactConfig
# ---------------------------------------------------------------------------


class TestContactConfig:
    """Test the ContactConfig dataclass."""

    def test_defaults(self):
        cfg = ContactConfig()
        assert cfg.contact_type == "penalty"
        assert cfg.frequency_hz == 20_000.0
        assert cfg.amplitude_um == 30.0
        assert cfg.friction_coefficient == 0.3
        assert cfg.max_newton_iterations == 50

    def test_to_dict(self):
        cfg = ContactConfig(contact_type="nitsche", frequency_hz=35000)
        d = cfg.to_dict()
        assert d["contact_type"] == "nitsche"
        assert d["frequency_hz"] == 35000
        assert isinstance(d["output_fields"], list)
        assert "displacement" in d["output_fields"]

    def test_custom_values(self):
        cfg = ContactConfig(
            mesh_path="/some/mesh.msh",
            horn_material="Aluminum 7075-T6",
            penalty_stiffness=5e11,
            time_steps=20,
        )
        assert cfg.mesh_path == "/some/mesh.msh"
        assert cfg.horn_material == "Aluminum 7075-T6"
        assert cfg.penalty_stiffness == 5e11
        assert cfg.time_steps == 20


# ---------------------------------------------------------------------------
# Tests: _lookup_material
# ---------------------------------------------------------------------------


class TestLookupMaterial:
    """Test material lookup helper."""

    def test_known_material(self):
        props = _lookup_material("Titanium Ti-6Al-4V")
        assert props["E_Pa"] == 113.8e9
        assert props["nu"] == 0.342
        assert props["rho_kg_m3"] == 4430.0

    def test_abs_material(self):
        props = _lookup_material("ABS")
        assert props["E_Pa"] == 2.3e9
        assert props["rho_kg_m3"] == 1050.0

    def test_tool_steel(self):
        props = _lookup_material("Tool Steel")
        assert props["E_Pa"] == 210.0e9

    def test_unknown_material_fallback(self):
        """Unknown material names should fall back to Tool Steel."""
        props = _lookup_material("Unknown Material XYZ")
        assert props["E_Pa"] == _DEFAULT_MATERIALS["Tool Steel"]["E_Pa"]

    def test_all_default_materials_have_required_keys(self):
        for name, props in _DEFAULT_MATERIALS.items():
            assert "E_Pa" in props, f"{name} missing E_Pa"
            assert "nu" in props, f"{name} missing nu"
            assert "rho_kg_m3" in props, f"{name} missing rho_kg_m3"
            assert "yield_MPa" in props, f"{name} missing yield_MPa"


# ---------------------------------------------------------------------------
# Tests: ContactSolver construction
# ---------------------------------------------------------------------------


class TestContactSolverInit:
    """Test solver construction."""

    def test_with_mock_runner(self):
        runner = _make_mock_runner()
        solver = ContactSolver(runner=runner)
        assert solver.runner is runner

    def test_without_runner(self):
        solver = ContactSolver(runner=None)
        # runner may or may not be available depending on env
        # but construction should not crash

    def test_custom_script_path(self):
        solver = ContactSolver(
            runner=_make_mock_runner(),
            docker_script="/custom/path/solver.py",
        )
        assert solver.docker_script == "/custom/path/solver.py"

    def test_get_script_path(self):
        solver = ContactSolver(runner=_make_mock_runner())
        path = solver.get_script_path()
        assert path.endswith("contact_solver.py")


# ---------------------------------------------------------------------------
# Tests: prepare_config
# ---------------------------------------------------------------------------


class TestPrepareConfig:
    """Test config preparation."""

    def test_basic_config(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({
            "mesh_path": "/tmp/mesh.msh",
            "horn_material": "Titanium Ti-6Al-4V",
            "workpiece_material": "ABS",
            "anvil_material": "Tool Steel",
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_type": "penalty",
        })
        assert cfg["materials"]["horn"]["name"] == "Titanium Ti-6Al-4V"
        assert cfg["materials"]["workpiece"]["name"] == "ABS"
        assert cfg["materials"]["anvil"]["name"] == "Tool Steel"
        assert cfg["excitation"]["frequency_hz"] == 20000
        assert cfg["contact"]["type"] == "penalty"

    def test_nitsche_contact_type(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({"contact_type": "nitsche"})
        assert cfg["contact"]["type"] == "nitsche"

    def test_invalid_contact_type_raises(self):
        solver = ContactSolver(runner=_make_mock_runner())
        with pytest.raises(ValueError, match="Invalid contact_type"):
            solver.prepare_config({"contact_type": "invalid"})

    def test_material_properties_resolved(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({"horn_material": "Titanium Ti-6Al-4V"})
        horn = cfg["materials"]["horn"]
        assert horn["E_Pa"] == 113.8e9
        assert horn["nu"] == 0.342

    def test_default_solver_params(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({})
        assert cfg["solver"]["max_newton_iterations"] == 50
        assert cfg["solver"]["newton_tolerance"] == 1e-8
        assert cfg["solver"]["time_steps"] == 10

    def test_custom_solver_params(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({
            "max_newton_iterations": 100,
            "time_steps": 20,
        })
        assert cfg["solver"]["max_newton_iterations"] == 100
        assert cfg["solver"]["time_steps"] == 20

    def test_output_fields_default(self):
        solver = ContactSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({})
        assert "displacement" in cfg["output_fields"]
        assert "contact_pressure" in cfg["output_fields"]


# ---------------------------------------------------------------------------
# Tests: analyze (full pipeline with mocked Docker)
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test the full analyze pipeline with mocked runner."""

    def test_successful_analysis(self, tmp_path):
        """Successful run returns parsed contact results."""
        runner = _make_mock_runner()
        # Create dummy mesh file and Docker script
        mesh_file = tmp_path / "mesh.msh"
        mesh_file.write_bytes(b"\x00mesh-data")
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({
            "mesh_path": str(mesh_file),
            "horn_material": "Titanium Ti-6Al-4V",
            "workpiece_material": "ABS",
            "anvil_material": "Tool Steel",
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_type": "penalty",
        }))

        assert result["status"] == "success"
        assert result["contact_pressure"]["max_MPa"] == 12.5
        assert result["slip_distance"]["max_um"] == 8.4
        assert result["deformation"]["max_um"] == 25.6
        assert result["stress"]["von_mises_max_MPa"] == 45.2
        assert result["summary"]["contact_area_mm2"] == 120.5
        assert result["summary"]["converged"] is True

    def test_no_runner_returns_error(self):
        """Without a runner, analyze returns an error dict."""
        solver = ContactSolver(runner=None)
        # Force runner to None
        solver.runner = None
        result = _run(solver.analyze({"mesh_path": "/tmp/mesh.msh"}))
        assert result["status"] == "error"
        assert "FEniCSxRunner" in result["error"]

    def test_docker_failure(self, tmp_path):
        """Docker execution failure is reported."""
        runner = _make_mock_runner(result={
            "status": "error",
            "error": "Container exited with code 1",
            "stdout": "",
            "stderr": "segfault",
        })
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({
            "contact_type": "penalty",
        }))

        assert result["status"] == "error"
        assert "exited with code 1" in result["error"]

    def test_missing_docker_script(self):
        """Missing Docker script returns error."""
        runner = _make_mock_runner()
        solver = ContactSolver(
            runner=runner,
            docker_script="/nonexistent/path/solver.py",
        )
        result = _run(solver.analyze({"contact_type": "penalty"}))
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_invalid_contact_type_in_analyze(self, tmp_path):
        """Invalid contact type in analyze returns error."""
        runner = _make_mock_runner()
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({"contact_type": "invalid_method"}))
        assert result["status"] == "error"
        assert "Invalid contact_type" in result["error"]

    def test_runner_exception(self, tmp_path):
        """Exception from runner.run_script is caught."""
        runner = mock.AsyncMock()
        runner.run_script = mock.AsyncMock(
            side_effect=RuntimeError("Docker crashed")
        )
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({"contact_type": "penalty"}))
        assert result["status"] == "error"
        assert "Docker crashed" in result["error"]


# ---------------------------------------------------------------------------
# Tests: _parse_results
# ---------------------------------------------------------------------------


class TestParseResults:
    """Test result parsing from Docker output."""

    def test_parse_full_results(self):
        docker_result = {
            "status": "success",
            "run_id": "test123",
            "results": {
                "contact_pressure_max_mpa": 15.0,
                "contact_pressure_mean_mpa": 7.0,
                "contact_pressure_distribution": [1, 2, 3],
                "slip_distance_max_um": 10.0,
                "slip_distance_mean_um": 4.0,
                "slip_distance_distribution": [0.5, 1.0],
                "deformation_max_um": 30.0,
                "deformation_field": [0.1, 0.2],
                "stress_von_mises_max_mpa": 50.0,
                "stress_field": [10.0, 20.0],
                "contact_area_mm2": 100.0,
                "total_force_n": 1200.0,
                "newton_iterations": 8,
                "converged": True,
                "solve_time_s": 15.0,
                "contact_type": "penalty",
            },
        }
        parsed = ContactSolver._parse_results(docker_result)

        assert parsed["status"] == "success"
        assert parsed["run_id"] == "test123"
        assert parsed["contact_pressure"]["max_MPa"] == 15.0
        assert parsed["slip_distance"]["max_um"] == 10.0
        assert parsed["deformation"]["max_um"] == 30.0
        assert parsed["stress"]["von_mises_max_MPa"] == 50.0
        assert parsed["summary"]["contact_area_mm2"] == 100.0
        assert parsed["summary"]["converged"] is True

    def test_parse_empty_results(self):
        """Missing keys should default to zero/empty."""
        docker_result = {
            "status": "success",
            "run_id": "empty",
            "results": {},
        }
        parsed = ContactSolver._parse_results(docker_result)
        assert parsed["status"] == "success"
        assert parsed["contact_pressure"]["max_MPa"] == 0.0
        assert parsed["slip_distance"]["max_um"] == 0.0
        assert parsed["deformation"]["max_um"] == 0.0
        assert parsed["summary"]["converged"] is False

    def test_parse_preserves_distributions(self):
        docker_result = {
            "status": "success",
            "run_id": "dist",
            "results": {
                "contact_pressure_distribution": [1.1, 2.2, 3.3, 4.4],
                "slip_distance_distribution": [0.1, 0.2],
            },
        }
        parsed = ContactSolver._parse_results(docker_result)
        assert len(parsed["contact_pressure"]["distribution"]) == 4
        assert parsed["contact_pressure"]["distribution"][0] == 1.1
        assert len(parsed["slip_distance"]["distribution"]) == 2


# ---------------------------------------------------------------------------
# Tests: Docker script existence
# ---------------------------------------------------------------------------


class TestDockerScript:
    """Verify the Docker contact solver script is shipped."""

    def test_docker_script_exists(self):
        """The contact_solver.py script should exist in docker/scripts/."""
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "contact_solver.py"
        assert script.exists(), f"Docker script not found at {script}"

    def test_docker_script_has_main(self):
        """The script should have a main() function."""
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "contact_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def main()" in content
        assert 'if __name__ == "__main__"' in content

    def test_docker_script_has_penalty_solver(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "contact_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def setup_penalty_contact" in content

    def test_docker_script_has_nitsche_solver(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "contact_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def setup_nitsche_contact" in content

    def test_docker_script_reads_config(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "contact_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "config.json" in content
        assert "result.json" in content


# ---------------------------------------------------------------------------
# Tests: Integration-style (mocked Docker, full pipeline)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests with mocked Docker."""

    def test_penalty_analysis_flow(self, tmp_path):
        """Full penalty analysis flow from config to parsed result."""
        runner = _make_mock_runner()
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({
            "horn_material": "Titanium Ti-6Al-4V",
            "workpiece_material": "ABS",
            "anvil_material": "Tool Steel",
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_type": "penalty",
        }))

        assert result["status"] == "success"
        # Verify runner was called
        runner.run_script.assert_called_once()
        call_args = runner.run_script.call_args
        assert call_args.kwargs.get("timeout_s") == 600 or (
            len(call_args.args) >= 3 and call_args.args[2] == 600
        )

    def test_nitsche_analysis_flow(self, tmp_path):
        """Full Nitsche analysis flow."""
        mock_result = {
            "status": "success",
            "run_id": "nitsche_run",
            "stdout": "done",
            "stderr": "",
            "output_files": {},
            "results": {
                "contact_pressure_max_mpa": 18.0,
                "contact_type": "nitsche",
                "converged": True,
                "solve_time_s": 20.0,
            },
        }
        runner = _make_mock_runner(result=mock_result)
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({"contact_type": "nitsche"}))

        assert result["status"] == "success"
        assert result["contact_pressure"]["max_MPa"] == 18.0
        assert result["summary"]["contact_type"] == "nitsche"

    def test_config_written_to_temp_dir(self, tmp_path):
        """Verify config.json is passed to the runner."""
        runner = _make_mock_runner()
        script_file = tmp_path / "contact_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ContactSolver(runner=runner, docker_script=str(script_file))
        _run(solver.analyze({"contact_type": "penalty"}))

        # Check the input_files argument to run_script
        call_kwargs = runner.run_script.call_args.kwargs
        input_files = call_kwargs.get("input_files", {})
        assert "config.json" in input_files
        config_path = input_files["config.json"]
        assert os.path.isfile(config_path)

        # Verify the config content
        with open(config_path, "r") as fp:
            config_data = json.load(fp)
        assert "materials" in config_data
        assert "contact" in config_data
        assert config_data["contact"]["type"] == "penalty"
