"""Tests for the FEniCSx Docker runner service.

All tests use mocks so that Docker is never actually invoked.
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

from web.services.fenicsx_runner import FEniCSxRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_exchange(tmp_path: Path) -> str:
    """Return a temporary exchange directory path."""
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    return str(exchange)


@pytest.fixture()
def runner(tmp_exchange: str) -> FEniCSxRunner:
    """Return a FEniCSxRunner configured with a temp exchange dir."""
    return FEniCSxRunner(
        exchange_dir=tmp_exchange,
        docker_image="dolfinx/dolfinx:test",
    )


@pytest.fixture()
def dummy_script(tmp_path: Path) -> str:
    """Create a minimal Python script file and return its path."""
    script = tmp_path / "solver.py"
    script.write_text("print('hello from solver')\n", encoding="utf-8")
    return str(script)


@pytest.fixture()
def dummy_input_files(tmp_path: Path) -> dict[str, str]:
    """Create dummy input files and return the mapping."""
    mesh_file = tmp_path / "mesh.msh"
    mesh_file.write_bytes(b"\x00mesh-data")

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"material": "steel"}), encoding="utf-8")

    return {
        "mesh.msh": str(mesh_file),
        "config.json": str(config_file),
    }


# ---------------------------------------------------------------------------
# Helper: create a mock async subprocess
# ---------------------------------------------------------------------------


def _make_mock_process(
    returncode: int = 0, stdout: bytes = b"", stderr: bytes = b""
):
    """Return an object that mimics asyncio.subprocess.Process."""
    proc = mock.AsyncMock()
    proc.returncode = returncode
    proc.communicate = mock.AsyncMock(return_value=(stdout, stderr))
    proc.wait = mock.AsyncMock(return_value=returncode)
    return proc


def _run(coro):
    """Convenience wrapper for ``asyncio.run``."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests: construction and defaults
# ---------------------------------------------------------------------------


class TestFEniCSxRunnerInit:
    """Test runner construction and default values."""

    def test_default_exchange_dir(self):
        r = FEniCSxRunner()
        assert r.exchange_dir  # non-empty string
        assert r.container_exchange == "/exchange"

    def test_custom_exchange_dir(self, tmp_exchange: str):
        r = FEniCSxRunner(exchange_dir=tmp_exchange)
        assert r.exchange_dir == tmp_exchange

    def test_custom_docker_image(self):
        r = FEniCSxRunner(docker_image="my/image:v1")
        assert r.docker_image == "my/image:v1"

    def test_default_constants(self):
        r = FEniCSxRunner()
        assert r.container_exchange == "/exchange"
        assert isinstance(r.docker_image, str)
        assert isinstance(r.exchange_dir, str)


# ---------------------------------------------------------------------------
# Tests: check_available
# ---------------------------------------------------------------------------


class TestCheckAvailable:
    """Test the Docker availability check."""

    def test_docker_available(self, runner: FEniCSxRunner):
        """When both docker info and image inspect succeed, return True."""
        proc = _make_mock_process(returncode=0)
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(runner.check_available())
        assert result is True

    def test_docker_info_fails(self, runner: FEniCSxRunner):
        """If docker info returns non-zero, return False."""
        proc = _make_mock_process(returncode=1)
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(runner.check_available())
        assert result is False

    def test_docker_not_installed(self, runner: FEniCSxRunner):
        """If docker binary is not found, return False."""
        with mock.patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("docker not found"),
        ):
            result = _run(runner.check_available())
        assert result is False

    def test_docker_general_exception(self, runner: FEniCSxRunner):
        """Unexpected exceptions return False without crashing."""
        with mock.patch(
            "asyncio.create_subprocess_exec",
            side_effect=RuntimeError("unexpected"),
        ):
            result = _run(runner.check_available())
        assert result is False


# ---------------------------------------------------------------------------
# Tests: _prepare_exchange
# ---------------------------------------------------------------------------


class TestPrepareExchange:
    """Test exchange directory preparation."""

    def test_creates_run_dir(self, runner: FEniCSxRunner, dummy_input_files):
        run_id = runner._prepare_exchange(dummy_input_files)
        run_dir = os.path.join(runner.exchange_dir, run_id)
        assert os.path.isdir(run_dir)
        assert os.path.isdir(os.path.join(run_dir, "output"))

    def test_copies_input_files(self, runner: FEniCSxRunner, dummy_input_files):
        run_id = runner._prepare_exchange(dummy_input_files)
        run_dir = os.path.join(runner.exchange_dir, run_id)
        assert os.path.isfile(os.path.join(run_dir, "mesh.msh"))
        assert os.path.isfile(os.path.join(run_dir, "config.json"))

    def test_unique_run_ids(self, runner: FEniCSxRunner, dummy_input_files):
        ids = {runner._prepare_exchange(dummy_input_files) for _ in range(10)}
        assert len(ids) == 10, "Run IDs must be unique"

    def test_empty_input_files(self, runner: FEniCSxRunner):
        run_id = runner._prepare_exchange({})
        run_dir = os.path.join(runner.exchange_dir, run_id)
        assert os.path.isdir(run_dir)
        # Only the output sub-dir should exist (no input files)
        contents = os.listdir(run_dir)
        assert "output" in contents


# ---------------------------------------------------------------------------
# Tests: _cleanup_exchange
# ---------------------------------------------------------------------------


class TestCleanupExchange:
    """Test exchange directory cleanup."""

    def test_removes_run_dir(self, runner: FEniCSxRunner, dummy_input_files):
        run_id = runner._prepare_exchange(dummy_input_files)
        run_dir = os.path.join(runner.exchange_dir, run_id)
        assert os.path.isdir(run_dir)
        runner._cleanup_exchange(run_id)
        assert not os.path.exists(run_dir)

    def test_handles_missing_dir(self, runner: FEniCSxRunner):
        """Cleaning up a non-existent run_id should not raise."""
        runner._cleanup_exchange("nonexistent-id")


# ---------------------------------------------------------------------------
# Tests: run_script
# ---------------------------------------------------------------------------


class TestRunScript:
    """Test script running inside Docker container."""

    def test_success_run(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """Successful run returns status=success and collects output."""
        proc = _make_mock_process(
            returncode=0, stdout=b"solver output\n", stderr=b""
        )
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files, timeout_s=60)
            )

        assert result["status"] == "success"
        assert "run_id" in result
        assert result["stdout"] == "solver output\n"
        assert result["stderr"] == ""
        assert isinstance(result["output_files"], dict)
        assert isinstance(result["results"], dict)

    def test_script_not_found(self, runner: FEniCSxRunner, dummy_input_files):
        """If the script file does not exist, return error."""
        result = _run(
            runner.run_script("/nonexistent/script.py", dummy_input_files)
        )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_input_file_not_found(self, runner: FEniCSxRunner, dummy_script):
        """If an input file does not exist, return error."""
        result = _run(
            runner.run_script(dummy_script, {"bad.msh": "/nonexistent/mesh.msh"})
        )
        assert result["status"] == "error"
        assert "not found" in result["error"].lower()

    def test_container_failure(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """Non-zero exit code returns error status."""
        proc = _make_mock_process(
            returncode=1,
            stdout=b"",
            stderr=b"ImportError: no module named foo\n",
        )
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files, timeout_s=60)
            )

        assert result["status"] == "error"
        assert "exited with code 1" in result["error"]
        assert "ImportError" in result["stderr"]

    def test_docker_not_installed_during_run(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """If Docker is not found at runtime, return a descriptive error."""
        with mock.patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("docker not found"),
        ):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files)
            )

        assert result["status"] == "error"
        assert "docker" in result["error"].lower()

    def test_result_json_parsed(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """If the script produces result.json in output/, it is parsed."""
        original_prepare = runner._prepare_exchange

        def patched_prepare(input_files):
            run_id = original_prepare(input_files)
            output_dir = os.path.join(runner.exchange_dir, run_id, "output")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "result.json"), "w") as f:
                json.dump({"displacement_max": 0.0012, "stress_max": 45.6}, f)
            return run_id

        runner._prepare_exchange = patched_prepare

        proc = _make_mock_process(returncode=0, stdout=b"done\n", stderr=b"")
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files, timeout_s=60)
            )

        assert result["status"] == "success"
        assert result["results"]["displacement_max"] == 0.0012
        assert result["results"]["stress_max"] == 45.6
        assert "result.json" in result["output_files"]

    def test_exchange_dir_contains_script(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """The solver script should be copied into the run directory."""
        proc = _make_mock_process(returncode=0, stdout=b"ok\n", stderr=b"")

        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files, timeout_s=60)
            )

        assert result["status"] == "success"
        run_id = result["run_id"]
        script_copy = os.path.join(runner.exchange_dir, run_id, "script.py")
        assert os.path.isfile(script_copy)


# ---------------------------------------------------------------------------
# Tests: pull_image
# ---------------------------------------------------------------------------


class TestPullImage:
    """Test Docker image pull convenience method."""

    def test_pull_success(self, runner: FEniCSxRunner):
        proc = _make_mock_process(returncode=0, stdout=b"Pulled\n", stderr=b"")
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(runner.pull_image())
        assert result is True

    def test_pull_failure(self, runner: FEniCSxRunner):
        proc = _make_mock_process(
            returncode=1, stdout=b"", stderr=b"not found\n"
        )
        with mock.patch("asyncio.create_subprocess_exec", return_value=proc):
            result = _run(runner.pull_image())
        assert result is False

    def test_pull_docker_not_found(self, runner: FEniCSxRunner):
        with mock.patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError,
        ):
            result = _run(runner.pull_image())
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Docker command construction
# ---------------------------------------------------------------------------


class TestDockerCommand:
    """Verify the docker run command is correctly assembled."""

    def test_docker_run_command_structure(
        self, runner: FEniCSxRunner, dummy_script, dummy_input_files
    ):
        """Validate the docker run CLI arguments."""
        captured_args: list[str] = []

        async def capture_subprocess(*args, **kwargs):
            captured_args.extend(args)
            return _make_mock_process(returncode=0, stdout=b"ok\n", stderr=b"")

        with mock.patch(
            "asyncio.create_subprocess_exec", side_effect=capture_subprocess
        ):
            result = _run(
                runner.run_script(dummy_script, dummy_input_files, timeout_s=60)
            )

        assert result["status"] == "success"

        # Verify key parts of the command
        assert "docker" in captured_args
        assert "run" in captured_args
        assert "--rm" in captured_args
        assert "-v" in captured_args

        # Volume mount should reference the exchange dir
        v_idx = captured_args.index("-v")
        volume_arg = captured_args[v_idx + 1]
        assert runner.container_exchange in volume_arg

        # Image name should be present
        assert runner.docker_image in captured_args

        # The script path inside the container should reference /exchange
        script_arg = captured_args[-1]
        assert script_arg.startswith(runner.container_exchange)
        assert script_arg.endswith("script.py")


# ---------------------------------------------------------------------------
# Tests: updated Docker config files
# ---------------------------------------------------------------------------


class TestUpdatedDockerConfig:
    """Verify enhancements to Docker config files."""

    def test_dockerfile_has_exchange_dir(self):
        """Dockerfile should reference the exchange directory."""
        dockerfile = (
            Path(__file__).resolve().parents[2] / "docker" / "Dockerfile.fenics"
        )
        content = dockerfile.read_text(encoding="utf-8")
        assert "EXCHANGE_DIR" in content
        assert "/exchange" in content
        assert "meshio" in content

    def test_compose_has_exchange_volume(self):
        """docker-compose.yml should mount the exchange directory."""
        compose = (
            Path(__file__).resolve().parents[2] / "docker" / "docker-compose.yml"
        )
        content = compose.read_text(encoding="utf-8")
        assert "/exchange" in content
        assert "EXCHANGE_DIR" in content
