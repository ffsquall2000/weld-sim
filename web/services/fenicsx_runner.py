"""FEniCSx Docker runner -- execute solver scripts inside a Docker container.

This service provides a high-level async interface for running FEniCSx Python
scripts inside the ``dolfinx/dolfinx:stable`` Docker image.  Communication
between the host and the container happens through a shared *exchange directory*
that is bind-mounted into the container at ``/exchange``.

Workflow:
    1. Caller prepares input files (mesh ``.msh``, config ``.json``, etc.).
    2. ``run_script`` copies them into a unique run sub-directory under
       ``EXCHANGE_DIR``, along with the solver script.
    3. The script is executed inside the container via ``docker run``.
    4. Output files written by the script to ``/exchange/<run_id>/output/``
       are read back and returned as a dict.

If Docker is not installed or the image is not pulled, all methods degrade
gracefully (return error dicts / ``False``) rather than raising exceptions.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Local directory shared with the Docker container via bind-mount.
EXCHANGE_DIR: str = os.environ.get(
    "WELD_SIM_EXCHANGE", "/tmp/weld-sim-fea-exchange"
)

#: Mount point inside the Docker container.
CONTAINER_EXCHANGE: str = "/exchange"

#: Default Docker image for FEniCSx computations.
DOCKER_IMAGE: str = os.environ.get(
    "FENICS_DOCKER_IMAGE", "dolfinx/dolfinx:stable"
)

#: Default script execution timeout in seconds.
DEFAULT_TIMEOUT_S: int = 300


class FEniCSxRunner:
    """Execute FEniCSx solver scripts inside a Docker container.

    Parameters
    ----------
    exchange_dir:
        Host-side directory for exchanging files with the container.
        Defaults to ``/tmp/weld-sim-fea-exchange``.
    docker_image:
        Docker image name/tag. Defaults to ``dolfinx/dolfinx:stable``.
    """

    def __init__(
        self,
        exchange_dir: str = EXCHANGE_DIR,
        docker_image: str = DOCKER_IMAGE,
    ) -> None:
        self.exchange_dir = exchange_dir
        self.docker_image = docker_image
        self.container_exchange = CONTAINER_EXCHANGE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def check_available(self) -> bool:
        """Return *True* if Docker is installed and the solver image exists.

        This does **not** start a container -- it only runs lightweight
        ``docker info`` and ``docker image inspect`` commands.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await asyncio.wait_for(proc.wait(), timeout=10)
            if rc != 0:
                logger.warning("Docker daemon not reachable (exit code %d)", rc)
                return False

            proc = await asyncio.create_subprocess_exec(
                "docker", "image", "inspect", self.docker_image,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            rc = await asyncio.wait_for(proc.wait(), timeout=10)
            if rc != 0:
                logger.warning(
                    "Docker image %s not found locally", self.docker_image
                )
                return False

            return True
        except FileNotFoundError:
            logger.warning("Docker CLI not found on PATH")
            return False
        except asyncio.TimeoutError:
            logger.warning("Docker availability check timed out")
            return False
        except Exception:
            logger.exception("Unexpected error checking Docker availability")
            return False

    async def run_script(
        self,
        script_path: str,
        input_files: dict[str, str],
        timeout_s: int = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        """Run a FEniCSx Python script inside the Docker container.

        Parameters
        ----------
        script_path:
            Path to the Python solver script on the host.
        input_files:
            Mapping of ``{filename: host_path}`` for files that must be
            available inside the container (e.g. mesh ``.msh``, config
            ``.json``).
        timeout_s:
            Maximum wall-clock time in seconds before the container is
            killed.

        Returns
        -------
        dict
            On success::

                {
                    "status": "success",
                    "run_id": "<uuid>",
                    "stdout": "...",
                    "stderr": "...",
                    "output_files": {"result.json": "<host_path>", ...},
                    "results": { ... }   # parsed from result.json if present
                }

            On failure::

                {
                    "status": "error",
                    "error": "<description>",
                    "stdout": "...",
                    "stderr": "..."
                }
        """
        # ------------------------------------------------------------------
        # 0.  Validate inputs
        # ------------------------------------------------------------------
        if not os.path.isfile(script_path):
            return {
                "status": "error",
                "error": f"Script not found: {script_path}",
                "stdout": "",
                "stderr": "",
            }

        for name, path in input_files.items():
            if not os.path.isfile(path):
                return {
                    "status": "error",
                    "error": f"Input file not found: {name} -> {path}",
                    "stdout": "",
                    "stderr": "",
                }

        # ------------------------------------------------------------------
        # 1.  Prepare exchange directory
        # ------------------------------------------------------------------
        run_id = self._prepare_exchange(input_files)
        run_dir = os.path.join(self.exchange_dir, run_id)

        try:
            # Copy the solver script into the run directory
            script_dest = os.path.join(run_dir, "script.py")
            shutil.copy2(script_path, script_dest)

            # Create an output sub-directory
            output_dir = os.path.join(run_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            # ------------------------------------------------------------------
            # 2.  Build docker run command
            # ------------------------------------------------------------------
            container_run = f"{self.container_exchange}/{run_id}"
            cmd = [
                "docker", "run",
                "--rm",
                "--name", f"fenics-run-{run_id[:12]}",
                "-v", f"{os.path.abspath(self.exchange_dir)}:{self.container_exchange}",
                self.docker_image,
                "python", f"{container_run}/script.py",
            ]

            logger.info(
                "Starting FEniCSx container run_id=%s timeout=%ds",
                run_id, timeout_s,
            )

            # ------------------------------------------------------------------
            # 3.  Execute
            # ------------------------------------------------------------------
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                # Kill the container if it exceeds the timeout
                logger.error(
                    "FEniCSx run timed out after %ds (run_id=%s)",
                    timeout_s, run_id,
                )
                await self._kill_container(f"fenics-run-{run_id[:12]}")
                return {
                    "status": "error",
                    "error": f"Execution timed out after {timeout_s}s",
                    "run_id": run_id,
                    "stdout": "",
                    "stderr": "",
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                logger.error(
                    "FEniCSx run failed (rc=%d, run_id=%s): %s",
                    proc.returncode, run_id, stderr[:500],
                )
                return {
                    "status": "error",
                    "error": f"Container exited with code {proc.returncode}",
                    "run_id": run_id,
                    "stdout": stdout,
                    "stderr": stderr,
                }

            # ------------------------------------------------------------------
            # 4.  Collect output files
            # ------------------------------------------------------------------
            output_files: dict[str, str] = {}
            if os.path.isdir(output_dir):
                for fname in os.listdir(output_dir):
                    output_files[fname] = os.path.join(output_dir, fname)

            # Try to parse result.json if present
            results: dict[str, Any] = {}
            result_json = os.path.join(output_dir, "result.json")
            if os.path.isfile(result_json):
                try:
                    with open(result_json, "r", encoding="utf-8") as fp:
                        results = json.load(fp)
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(
                        "Failed to parse result.json (run_id=%s): %s",
                        run_id, exc,
                    )

            logger.info(
                "FEniCSx run completed (run_id=%s, outputs=%d)",
                run_id, len(output_files),
            )

            return {
                "status": "success",
                "run_id": run_id,
                "stdout": stdout,
                "stderr": stderr,
                "output_files": output_files,
                "results": results,
            }

        except FileNotFoundError:
            return {
                "status": "error",
                "error": "Docker CLI not found on PATH. Is Docker installed?",
                "stdout": "",
                "stderr": "",
            }
        except Exception as exc:
            logger.exception("Unexpected error in FEniCSx run (run_id=%s)", run_id)
            return {
                "status": "error",
                "error": f"Unexpected error: {exc}",
                "run_id": run_id,
                "stdout": "",
                "stderr": "",
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_exchange(self, input_files: dict[str, str]) -> str:
        """Create a unique run directory and copy input files into it.

        Parameters
        ----------
        input_files:
            ``{filename: host_path}`` mapping.

        Returns
        -------
        str
            The run ID (also the sub-directory name inside exchange_dir).
        """
        run_id = uuid.uuid4().hex[:16]
        run_dir = os.path.join(self.exchange_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "output"), exist_ok=True)

        for filename, host_path in input_files.items():
            dest = os.path.join(run_dir, filename)
            shutil.copy2(host_path, dest)
            logger.debug("Copied %s -> %s", host_path, dest)

        return run_id

    def _cleanup_exchange(self, run_id: str) -> None:
        """Remove the temporary run directory for *run_id*.

        Silently ignores errors (e.g. directory already removed).
        """
        run_dir = os.path.join(self.exchange_dir, run_id)
        try:
            shutil.rmtree(run_dir)
            logger.debug("Cleaned up exchange dir for run_id=%s", run_id)
        except OSError as exc:
            logger.warning(
                "Failed to clean up exchange dir %s: %s", run_dir, exc
            )

    @staticmethod
    async def _kill_container(name: str) -> None:
        """Send ``docker kill`` to a named container, ignoring errors."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "kill", name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10)
        except Exception:
            logger.debug("Could not kill container %s (may already be gone)", name)

    # ------------------------------------------------------------------
    # Convenience: pull the Docker image
    # ------------------------------------------------------------------

    async def pull_image(self) -> bool:
        """Pull the configured Docker image.  Returns *True* on success."""
        try:
            logger.info("Pulling Docker image %s ...", self.docker_image)
            proc = await asyncio.create_subprocess_exec(
                "docker", "pull", self.docker_image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=600
            )
            if proc.returncode == 0:
                logger.info("Successfully pulled %s", self.docker_image)
                return True
            else:
                logger.error(
                    "Failed to pull %s: %s",
                    self.docker_image,
                    stderr.decode("utf-8", errors="replace")[:500],
                )
                return False
        except FileNotFoundError:
            logger.warning("Docker CLI not found on PATH")
            return False
        except asyncio.TimeoutError:
            logger.error("Docker pull timed out")
            return False
        except Exception:
            logger.exception("Unexpected error pulling Docker image")
            return False
