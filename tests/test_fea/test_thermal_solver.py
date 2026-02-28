"""Tests for the thermal analysis solver.

All tests use mocks so that neither Docker nor FEniCSx are required.
Async methods are tested via ``asyncio.run()`` to avoid a dependency on
``pytest-asyncio``.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from unittest import mock

import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.thermal_solver import (
    ThermalConfig,
    ThermalSolver,
    THERMAL_MATERIALS,
    _lookup_thermal_properties,
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
            "run_id": "thermal-abc123",
            "stdout": "done",
            "stderr": "",
            "output_files": {"result.json": "/tmp/result.json"},
            "results": {
                "max_temperature_c": 185.5,
                "mean_temperature_c": 95.3,
                "min_temperature_c": 25.0,
                "initial_temperature_c": 25.0,
                "temperature_distribution": [25.0, 50.0, 100.0, 150.0, 185.5],
                "melt_zone": {
                    "melt_temp_c": 200.0,
                    "melt_fraction": 0.0,
                    "melt_volume_mm3": 0.0,
                    "n_melt_nodes": 0,
                    "n_total_nodes": 5000,
                },
                "thermal_history": [
                    {"time_s": 0.0, "max_temp_c": 25.0, "mean_temp_c": 25.0, "min_temp_c": 25.0},
                    {"time_s": 0.25, "max_temp_c": 120.0, "mean_temp_c": 60.0, "min_temp_c": 25.0},
                    {"time_s": 0.5, "max_temp_c": 185.5, "mean_temp_c": 95.3, "min_temp_c": 25.0},
                ],
                "max_temp_history": [25.0, 80.0, 120.0, 150.0, 185.5],
                "mean_temp_history": [25.0, 40.0, 60.0, 80.0, 95.3],
                "heat_generation_rate_w_m3": 1.13e9,
                "surface_heat_flux_w_m2": 1.13e6,
                "solve_time_s": 5.2,
                "weld_time_s": 0.5,
                "n_time_steps": 50,
            },
        }
    runner = mock.AsyncMock()
    runner.run_script = mock.AsyncMock(return_value=result)
    return runner


# ---------------------------------------------------------------------------
# Tests: ThermalConfig
# ---------------------------------------------------------------------------


class TestThermalConfig:
    """Test the ThermalConfig dataclass."""

    def test_defaults(self):
        cfg = ThermalConfig()
        assert cfg.workpiece_material == "ABS"
        assert cfg.frequency_hz == 20_000.0
        assert cfg.amplitude_um == 30.0
        assert cfg.friction_coefficient == 0.3
        assert cfg.weld_time_s == 0.5
        assert cfg.n_time_steps == 50
        assert cfg.initial_temp_c == 25.0

    def test_to_dict(self):
        cfg = ThermalConfig(
            workpiece_material="Nylon 6",
            weld_time_s=1.0,
        )
        d = cfg.to_dict()
        assert d["workpiece_material"] == "Nylon 6"
        assert d["weld_time_s"] == 1.0
        assert "frequency_hz" in d
        assert "amplitude_um" in d

    def test_custom_values(self):
        cfg = ThermalConfig(
            contact_pressure_pa=2e6,
            weld_time_s=1.0,
            n_time_steps=100,
            initial_temp_c=30.0,
        )
        assert cfg.contact_pressure_pa == 2e6
        assert cfg.weld_time_s == 1.0
        assert cfg.n_time_steps == 100
        assert cfg.initial_temp_c == 30.0


# ---------------------------------------------------------------------------
# Tests: Thermal material properties
# ---------------------------------------------------------------------------


class TestThermalMaterials:
    """Test thermal material property lookup."""

    def test_abs_properties(self):
        props = _lookup_thermal_properties("ABS")
        assert props["k"] == 0.17
        assert props["Cp"] == 1400.0
        assert props["rho"] == 1050.0
        assert "melt_temp_c" in props

    def test_titanium_properties(self):
        props = _lookup_thermal_properties("Titanium Ti-6Al-4V")
        assert props["k"] == 6.7
        assert props["rho"] == 4430.0

    def test_unknown_material_fallback(self):
        """Unknown material falls back to ABS."""
        props = _lookup_thermal_properties("Unknown Material")
        assert props["k"] == THERMAL_MATERIALS["ABS"]["k"]
        assert props["Cp"] == THERMAL_MATERIALS["ABS"]["Cp"]

    def test_all_materials_have_required_keys(self):
        for name, props in THERMAL_MATERIALS.items():
            assert "k" in props, f"{name} missing k"
            assert "Cp" in props, f"{name} missing Cp"
            assert "rho" in props, f"{name} missing rho"
            assert "melt_temp_c" in props, f"{name} missing melt_temp_c"

    def test_polymer_lower_conductivity(self):
        """Polymers should have much lower conductivity than metals."""
        abs_k = THERMAL_MATERIALS["ABS"]["k"]
        ti_k = THERMAL_MATERIALS["Titanium Ti-6Al-4V"]["k"]
        assert abs_k < ti_k

    def test_nylon_properties(self):
        props = _lookup_thermal_properties("Nylon 6")
        assert props["k"] == 0.25
        assert props["melt_temp_c"] == 220.0


# ---------------------------------------------------------------------------
# Tests: Heat generation computation
# ---------------------------------------------------------------------------


class TestHeatGeneration:
    """Test the compute_heat_generation method."""

    def test_basic_calculation(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        q = solver.compute_heat_generation(
            contact_pressure=1e6,  # 1 MPa
            frequency_hz=20000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        # Q = mu * P * 2*pi*f*A
        # = 0.3 * 1e6 * 2*pi*20000 * 30e-6
        expected = 0.3 * 1e6 * 2 * math.pi * 20000 * 30e-6
        assert abs(q - expected) < 1.0  # W/m^2

    def test_zero_pressure_gives_zero(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        q = solver.compute_heat_generation(
            contact_pressure=0.0,
            frequency_hz=20000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        assert q == 0.0

    def test_zero_frequency_gives_zero(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        q = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=0,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        assert q == 0.0

    def test_higher_frequency_more_heat(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        q_20k = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=20000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        q_40k = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=40000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        assert q_40k > q_20k
        assert abs(q_40k / q_20k - 2.0) < 0.001

    def test_higher_amplitude_more_heat(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        q_30 = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=20000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        q_60 = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=20000,
            amplitude_um=60.0,
            friction_coeff=0.3,
        )
        assert q_60 > q_30
        assert abs(q_60 / q_30 - 2.0) < 0.001

    def test_heat_generation_units(self):
        """Result should be in W/m^2 (surface heat flux)."""
        solver = ThermalSolver(runner=_make_mock_runner())
        q = solver.compute_heat_generation(
            contact_pressure=1e6,
            frequency_hz=20000,
            amplitude_um=30.0,
            friction_coeff=0.3,
        )
        # Should be order of 1e6 W/m^2 for typical welding params
        assert 1e4 < q < 1e8


# ---------------------------------------------------------------------------
# Tests: ThermalSolver construction
# ---------------------------------------------------------------------------


class TestThermalSolverInit:
    """Test solver construction."""

    def test_with_mock_runner(self):
        runner = _make_mock_runner()
        solver = ThermalSolver(runner=runner)
        assert solver.runner is runner

    def test_without_runner(self):
        solver = ThermalSolver(runner=None)
        # Construction should not crash

    def test_custom_script_path(self):
        solver = ThermalSolver(
            runner=_make_mock_runner(),
            docker_script="/custom/path/thermal_solver.py",
        )
        assert solver.docker_script == "/custom/path/thermal_solver.py"

    def test_get_script_path(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        path = solver.get_script_path()
        assert path.endswith("thermal_solver.py")


# ---------------------------------------------------------------------------
# Tests: prepare_config
# ---------------------------------------------------------------------------


class TestPrepareConfig:
    """Test config preparation."""

    def test_basic_config(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({
            "materials": {"workpiece": {"name": "ABS"}},
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_pressure_pa": 1e6,
            "weld_time_s": 0.5,
            "initial_temp_c": 25.0,
        })
        assert cfg["materials"]["workpiece"]["name"] == "ABS"
        assert cfg["materials"]["workpiece"]["k"] == 0.17
        assert cfg["excitation"]["frequency_hz"] == 20000
        assert cfg["contact"]["contact_pressure_pa"] == 1e6
        assert cfg["thermal"]["weld_time_s"] == 0.5

    def test_default_values(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({})
        assert cfg["materials"]["workpiece"]["name"] == "ABS"
        assert cfg["thermal"]["initial_temp_c"] == 25.0
        assert cfg["thermal"]["n_time_steps"] == 50

    def test_titanium_material(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({
            "materials": {"workpiece": {"name": "Titanium Ti-6Al-4V"}},
        })
        wp = cfg["materials"]["workpiece"]
        assert wp["k"] == 6.7
        assert wp["Cp"] == 526.3
        assert wp["melt_temp_c"] == 1660.0

    def test_custom_time_steps(self):
        solver = ThermalSolver(runner=_make_mock_runner())
        cfg = solver.prepare_config({
            "weld_time_s": 1.0,
            "n_time_steps": 100,
        })
        assert cfg["thermal"]["weld_time_s"] == 1.0
        assert cfg["thermal"]["n_time_steps"] == 100


# ---------------------------------------------------------------------------
# Tests: analyze (full pipeline with mocked Docker)
# ---------------------------------------------------------------------------


class TestAnalyze:
    """Test the full analyze pipeline with mocked runner."""

    def test_successful_analysis(self, tmp_path):
        """Successful run returns parsed thermal results."""
        runner = _make_mock_runner()
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({
            "materials": {"workpiece": {"name": "ABS"}},
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_pressure_pa": 1e6,
            "weld_time_s": 0.5,
        }))

        assert result["status"] == "success"
        assert result["max_temperature_c"] == 185.5
        assert result["mean_temperature_c"] == 95.3
        assert isinstance(result["temperature_distribution"], list)
        assert len(result["temperature_distribution"]) > 0
        assert isinstance(result["melt_zone"], dict)
        assert isinstance(result["thermal_history"], list)

    def test_no_runner_returns_error(self):
        """Without a runner, analyze returns an error dict."""
        solver = ThermalSolver(runner=None)
        solver.runner = None
        result = _run(solver.analyze({}))
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
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({}))
        assert result["status"] == "error"
        assert "exited with code 1" in result["error"]

    def test_missing_docker_script(self):
        """Missing Docker script returns error."""
        runner = _make_mock_runner()
        solver = ThermalSolver(
            runner=runner,
            docker_script="/nonexistent/path/thermal_solver.py",
        )
        result = _run(solver.analyze({}))
        assert result["status"] == "error"
        assert "not found" in result["error"]

    def test_runner_exception(self, tmp_path):
        """Exception from runner.run_script is caught."""
        runner = mock.AsyncMock()
        runner.run_script = mock.AsyncMock(
            side_effect=RuntimeError("Docker crashed")
        )
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({}))
        assert result["status"] == "error"
        assert "Docker crashed" in result["error"]

    def test_melt_zone_result(self, tmp_path):
        """Check that melt zone data is parsed correctly."""
        mock_result = {
            "status": "success",
            "run_id": "melt-test",
            "stdout": "",
            "stderr": "",
            "output_files": {},
            "results": {
                "max_temperature_c": 250.0,
                "melt_zone": {
                    "melt_temp_c": 200.0,
                    "melt_fraction": 0.15,
                    "melt_volume_mm3": 45.0,
                    "n_melt_nodes": 750,
                    "n_total_nodes": 5000,
                },
                "thermal_history": [],
            },
        }
        runner = _make_mock_runner(result=mock_result)
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({}))

        assert result["status"] == "success"
        assert result["max_temperature_c"] == 250.0
        assert result["melt_zone"]["melt_fraction"] == 0.15
        assert result["melt_zone"]["melt_volume_mm3"] == 45.0


# ---------------------------------------------------------------------------
# Tests: _parse_results
# ---------------------------------------------------------------------------


class TestParseResults:
    """Test result parsing from Docker output."""

    def test_parse_full_results(self):
        docker_result = {
            "status": "success",
            "run_id": "test-thermal",
            "results": {
                "max_temperature_c": 200.0,
                "mean_temperature_c": 100.0,
                "min_temperature_c": 25.0,
                "initial_temperature_c": 25.0,
                "temperature_distribution": [25.0, 100.0, 200.0],
                "melt_zone": {
                    "melt_temp_c": 200.0,
                    "melt_fraction": 0.05,
                    "melt_volume_mm3": 10.0,
                    "n_melt_nodes": 250,
                    "n_total_nodes": 5000,
                },
                "thermal_history": [
                    {"time_s": 0.0, "max_temp_c": 25.0},
                    {"time_s": 0.5, "max_temp_c": 200.0},
                ],
                "max_temp_history": [25.0, 200.0],
                "mean_temp_history": [25.0, 100.0],
                "heat_generation_rate_w_m3": 1e9,
                "surface_heat_flux_w_m2": 1e6,
                "solve_time_s": 3.5,
                "weld_time_s": 0.5,
                "n_time_steps": 50,
            },
        }
        parsed = ThermalSolver._parse_results(docker_result)

        assert parsed["status"] == "success"
        assert parsed["max_temperature_c"] == 200.0
        assert parsed["mean_temperature_c"] == 100.0
        assert len(parsed["temperature_distribution"]) == 3
        assert parsed["melt_zone"]["melt_fraction"] == 0.05
        assert len(parsed["thermal_history"]) == 2
        assert parsed["solve_time_s"] == 3.5

    def test_parse_empty_results(self):
        """Missing keys should default to zero/empty."""
        docker_result = {
            "status": "success",
            "run_id": "empty",
            "results": {},
        }
        parsed = ThermalSolver._parse_results(docker_result)
        assert parsed["status"] == "success"
        assert parsed["max_temperature_c"] == 0.0
        assert parsed["temperature_distribution"] == []
        assert parsed["melt_zone"]["melt_fraction"] == 0.0
        assert parsed["thermal_history"] == []


# ---------------------------------------------------------------------------
# Tests: Docker script existence
# ---------------------------------------------------------------------------


class TestDockerScript:
    """Verify the Docker thermal solver script is shipped."""

    def test_docker_script_exists(self):
        """The thermal_solver.py script should exist in docker/scripts/."""
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "thermal_solver.py"
        assert script.exists(), f"Docker script not found at {script}"

    def test_docker_script_has_main(self):
        """The script should have a main() function."""
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "thermal_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def main()" in content
        assert 'if __name__ == "__main__"' in content

    def test_docker_script_has_solve_thermal(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "thermal_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def solve_thermal" in content

    def test_docker_script_has_heat_generation(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "thermal_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "def compute_heat_generation" in content

    def test_docker_script_reads_config(self):
        project_root = Path(__file__).resolve().parents[2]
        script = project_root / "docker" / "scripts" / "thermal_solver.py"
        content = script.read_text(encoding="utf-8")
        assert "config.json" in content
        assert "result.json" in content


# ---------------------------------------------------------------------------
# Tests: Integration-style (mocked Docker, full pipeline)
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests with mocked Docker."""

    def test_abs_thermal_analysis(self, tmp_path):
        """Full thermal analysis flow for ABS workpiece."""
        runner = _make_mock_runner()
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        result = _run(solver.analyze({
            "materials": {"workpiece": {"name": "ABS"}},
            "frequency_hz": 20000,
            "amplitude_um": 30.0,
            "contact_pressure_pa": 1e6,
            "weld_time_s": 0.5,
            "initial_temp_c": 25.0,
        }))

        assert result["status"] == "success"
        runner.run_script.assert_called_once()

    def test_config_passed_to_runner(self, tmp_path):
        """Verify config.json is passed to the runner."""
        runner = _make_mock_runner()
        script_file = tmp_path / "thermal_solver.py"
        script_file.write_text("# solver script", encoding="utf-8")

        solver = ThermalSolver(runner=runner, docker_script=str(script_file))
        _run(solver.analyze({
            "materials": {"workpiece": {"name": "ABS"}},
            "weld_time_s": 1.0,
        }))

        call_kwargs = runner.run_script.call_args.kwargs
        input_files = call_kwargs.get("input_files", {})
        assert "config.json" in input_files
        config_path = input_files["config.json"]
        assert os.path.isfile(config_path)

        with open(config_path, "r") as fp:
            config_data = json.load(fp)
        assert "materials" in config_data
        assert "thermal" in config_data
        assert config_data["thermal"]["weld_time_s"] == 1.0
