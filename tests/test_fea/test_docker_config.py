"""Tests for Docker deployment configuration files.

Validates that the Docker files for the FEniCSx solver worker exist,
have valid syntax, and contain the expected configuration directives.
"""
from __future__ import annotations

import pathlib
import re
import textwrap

import pytest
import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCKER_DIR = PROJECT_ROOT / "docker"


# ---------------------------------------------------------------------------
# Dockerfile tests
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Validate docker/Dockerfile.fenics."""

    @pytest.fixture()
    def dockerfile(self) -> str:
        path = DOCKER_DIR / "Dockerfile.fenics"
        assert path.exists(), f"Dockerfile not found at {path}"
        return path.read_text(encoding="utf-8")

    def test_file_exists(self):
        assert (DOCKER_DIR / "Dockerfile.fenics").exists()

    def test_base_image(self, dockerfile: str):
        """Must be based on dolfinx/dolfinx:v0.8.0."""
        assert re.search(
            r"^FROM\s+dolfinx/dolfinx:v0\.8\.0", dockerfile, re.MULTILINE
        ), "Dockerfile must use dolfinx/dolfinx:v0.8.0 as base image"

    def test_exposes_port_8002(self, dockerfile: str):
        assert re.search(
            r"^EXPOSE\s+8002", dockerfile, re.MULTILINE
        ), "Dockerfile must EXPOSE 8002"

    def test_healthcheck(self, dockerfile: str):
        assert "HEALTHCHECK" in dockerfile, "Dockerfile must contain a HEALTHCHECK"
        assert "8002/health" in dockerfile, "HEALTHCHECK must probe /health on 8002"

    def test_workdir(self, dockerfile: str):
        assert re.search(
            r"^WORKDIR\s+/app", dockerfile, re.MULTILINE
        ), "WORKDIR should be /app"

    def test_cmd_uvicorn(self, dockerfile: str):
        assert "uvicorn" in dockerfile, "CMD must launch uvicorn"
        assert "fenics_worker.main:app" in dockerfile, (
            "CMD must reference fenics_worker.main:app"
        )

    def test_environment_variables(self, dockerfile: str):
        required_vars = [
            "FENICS_WORKER_PORT",
            "FENICS_LOG_LEVEL",
            "MESH_DATA_DIR",
        ]
        for var in required_vars:
            assert var in dockerfile, f"ENV {var} must be set in Dockerfile"

    def test_mesh_data_dir(self, dockerfile: str):
        assert "/data/meshes" in dockerfile, (
            "Dockerfile must reference /data/meshes for shared mesh data"
        )

    def test_pip_install(self, dockerfile: str):
        assert "pip install" in dockerfile, "Dockerfile must pip install requirements"
        assert "requirements.txt" in dockerfile, (
            "Dockerfile must reference requirements.txt"
        )

    def test_valid_dockerfile_syntax(self, dockerfile: str):
        """Basic syntax check: every non-comment, non-blank line that is not
        a continuation should start with a valid Dockerfile instruction."""
        valid_instructions = {
            "FROM", "RUN", "CMD", "LABEL", "EXPOSE", "ENV", "ADD",
            "COPY", "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG",
            "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
        }
        continuation = False
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if continuation:
                continuation = stripped.endswith("\\")
                continue
            first_word = stripped.split()[0].upper()
            assert first_word in valid_instructions, (
                f"Invalid Dockerfile instruction: {first_word!r}"
            )
            continuation = stripped.endswith("\\")


# ---------------------------------------------------------------------------
# docker-compose.yml tests
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """Validate docker/docker-compose.yml."""

    @pytest.fixture()
    def compose_path(self) -> pathlib.Path:
        path = DOCKER_DIR / "docker-compose.yml"
        assert path.exists(), f"docker-compose.yml not found at {path}"
        return path

    @pytest.fixture()
    def compose(self, compose_path: pathlib.Path) -> dict:
        return yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    def test_file_exists(self):
        assert (DOCKER_DIR / "docker-compose.yml").exists()

    def test_valid_yaml(self, compose_path: pathlib.Path):
        """File must be valid YAML."""
        data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "docker-compose.yml must be a YAML mapping"

    def test_solver_b_service(self, compose: dict):
        services = compose.get("services", {})
        assert "solver-b" in services, "Must define a 'solver-b' service"

    def test_solver_b_build_context(self, compose: dict):
        solver = compose["services"]["solver-b"]
        build = solver.get("build", {})
        assert "dockerfile" in build or "Dockerfile" in str(build), (
            "solver-b must reference Dockerfile.fenics"
        )

    def test_port_mapping(self, compose: dict):
        solver = compose["services"]["solver-b"]
        ports = solver.get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("8002" in p for p in port_strs), (
            "solver-b must expose port 8002"
        )

    def test_restart_policy(self, compose: dict):
        solver = compose["services"]["solver-b"]
        assert "restart" in solver, "solver-b must have a restart policy"

    def test_volumes_defined(self, compose: dict):
        assert "volumes" in compose, "Top-level volumes must be defined"
        solver = compose["services"]["solver-b"]
        assert "volumes" in solver, "solver-b must mount volumes"

    def test_networks_defined(self, compose: dict):
        assert "networks" in compose, "Top-level networks must be defined"
        solver = compose["services"]["solver-b"]
        assert "networks" in solver, "solver-b must be on a network"

    def test_environment_vars(self, compose: dict):
        solver = compose["services"]["solver-b"]
        env = solver.get("environment", [])
        env_str = str(env)
        assert "FENICS_WORKER_PORT" in env_str, "Must set FENICS_WORKER_PORT"
        assert "MESH_DATA_DIR" in env_str, "Must set MESH_DATA_DIR"

    def test_healthcheck(self, compose: dict):
        solver = compose["services"]["solver-b"]
        assert "healthcheck" in solver, "solver-b must have a healthcheck"
        hc = solver["healthcheck"]
        assert "test" in hc, "healthcheck must have a test command"


# ---------------------------------------------------------------------------
# Worker package tests
# ---------------------------------------------------------------------------


class TestFenicsWorkerPackage:
    """Validate the fenics_worker Python package structure."""

    def test_init_exists(self):
        assert (DOCKER_DIR / "fenics_worker" / "__init__.py").exists()

    def test_main_exists(self):
        assert (DOCKER_DIR / "fenics_worker" / "main.py").exists()

    def test_requirements_exists(self):
        path = DOCKER_DIR / "fenics_worker" / "requirements.txt"
        assert path.exists()

    def test_requirements_content(self):
        path = DOCKER_DIR / "fenics_worker" / "requirements.txt"
        content = path.read_text(encoding="utf-8")
        assert "fastapi" in content.lower(), "requirements must include fastapi"
        assert "uvicorn" in content.lower(), "requirements must include uvicorn"
        assert "numpy" in content.lower(), "requirements must include numpy"
        assert "scipy" in content.lower(), "requirements must include scipy"

    def test_main_has_app(self):
        """main.py must define a FastAPI app object."""
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert "app = FastAPI(" in content, "main.py must create a FastAPI app"

    def test_main_has_health_endpoint(self):
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert '"/health"' in content, "main.py must define /health endpoint"

    def test_main_has_modal_endpoint(self):
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert "/api/v1/modal" in content, "main.py must define /api/v1/modal endpoint"

    def test_main_has_harmonic_endpoint(self):
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert "/api/v1/harmonic" in content, (
            "main.py must define /api/v1/harmonic endpoint"
        )

    def test_main_has_static_endpoint(self):
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert "/api/v1/static" in content, (
            "main.py must define /api/v1/static endpoint"
        )

    def test_stubs_return_501(self):
        """Stub endpoints must raise 501 Not Implemented."""
        content = (DOCKER_DIR / "fenics_worker" / "main.py").read_text(encoding="utf-8")
        assert "501" in content, "Stub endpoints must reference HTTP 501"
