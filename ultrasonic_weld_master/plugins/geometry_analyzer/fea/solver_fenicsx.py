"""FEniCSx contact mechanics solver for horn-workpiece-anvil analysis.

Provides a high-level ``ContactSolver`` class that orchestrates:

1. Input preparation -- builds a ``config.json`` describing the contact
   problem (materials, boundary conditions, contact surfaces, excitation
   parameters).
2. Docker execution -- delegates the actual FEniCSx computation to a
   standalone Python script (``docker/scripts/contact_solver.py``) running
   inside the ``dolfinx/dolfinx`` container via :class:`FEniCSxRunner`.
3. Result parsing -- reads the solver output (``result.json``) and returns
   a structured dict with contact pressures, slip distances, deformation
   fields, and stress data.

Two contact formulations are supported:
  - **Penalty method** -- simple and robust; adds a large penalty
    stiffness to enforce the non-penetration constraint.  Best for
    moderate contact pressures.
  - **Nitsche method** -- variationally consistent; enforces contact
    weakly via boundary integrals.  Better accuracy for thin
    workpieces and high-frequency loading.

Usage
-----
::

    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_fenicsx import (
        ContactSolver,
    )

    solver = ContactSolver()  # uses default FEniCSxRunner

    result = await solver.analyze({
        "mesh_path": "/path/to/assembly.msh",
        "horn_material": "Titanium Ti-6Al-4V",
        "workpiece_material": "ABS",
        "anvil_material": "Tool Steel",
        "frequency_hz": 20000,
        "amplitude_um": 30.0,
        "contact_type": "penalty",
    })

Dependencies
------------
- **FEniCSx / dolfinx** are NOT required on the host; the solver runs
  inside Docker.
- The ``FEniCSxRunner`` from ``web.services.fenicsx_runner`` handles all
  Docker orchestration.
- ``MeshConverter`` from the same FEA package handles mesh format
  conversion.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the Docker contact solver script (shipped with the project)
# ---------------------------------------------------------------------------

#: Path to the Docker-side solver script.
_DOCKER_SCRIPT_PATH: str = str(
    Path(__file__).resolve().parents[4] / "docker" / "scripts" / "contact_solver.py"
)

# ---------------------------------------------------------------------------
# Guarded import: FEniCSxRunner
# ---------------------------------------------------------------------------

_RUNNER_AVAILABLE = False
try:
    from web.services.fenicsx_runner import FEniCSxRunner  # type: ignore[import-untyped]
    _RUNNER_AVAILABLE = True
except ImportError:
    FEniCSxRunner = None  # type: ignore[assignment,misc]
    logger.debug("FEniCSxRunner not importable (web.services not on path)")


# ---------------------------------------------------------------------------
# Dataclass for contact configuration
# ---------------------------------------------------------------------------


@dataclass
class ContactConfig:
    """Structured configuration for a contact analysis job.

    All lengths are in millimetres; the Docker script converts to SI
    internally.
    """

    # Mesh
    mesh_path: str = ""

    # Materials (names matching ``material_properties.FEA_MATERIALS``)
    horn_material: str = "Titanium Ti-6Al-4V"
    workpiece_material: str = "ABS"
    anvil_material: str = "Tool Steel"

    # Excitation
    frequency_hz: float = 20_000.0
    amplitude_um: float = 30.0

    # Contact formulation
    contact_type: str = "penalty"  # "penalty" | "nitsche"
    penalty_stiffness: float = 1e12  # N/m^3 (penalty method)
    nitsche_parameter: float = 100.0  # dimensionless (Nitsche method)
    friction_coefficient: float = 0.3

    # Solver parameters
    max_newton_iterations: int = 50
    newton_tolerance: float = 1e-8
    time_steps: int = 10
    total_time_s: float = 0.001  # 1 ms default simulation window

    # Output control
    output_fields: list[str] = field(
        default_factory=lambda: [
            "displacement",
            "stress_von_mises",
            "contact_pressure",
            "slip_distance",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "mesh_path": self.mesh_path,
            "horn_material": self.horn_material,
            "workpiece_material": self.workpiece_material,
            "anvil_material": self.anvil_material,
            "frequency_hz": self.frequency_hz,
            "amplitude_um": self.amplitude_um,
            "contact_type": self.contact_type,
            "penalty_stiffness": self.penalty_stiffness,
            "nitsche_parameter": self.nitsche_parameter,
            "friction_coefficient": self.friction_coefficient,
            "max_newton_iterations": self.max_newton_iterations,
            "newton_tolerance": self.newton_tolerance,
            "time_steps": self.time_steps,
            "total_time_s": self.total_time_s,
            "output_fields": list(self.output_fields),
        }


# ---------------------------------------------------------------------------
# Material helper
# ---------------------------------------------------------------------------

# Minimal built-in material table so the module works standalone.
# When the full material_properties module is available we prefer that.
_DEFAULT_MATERIALS: dict[str, dict[str, float]] = {
    "Titanium Ti-6Al-4V": {
        "E_Pa": 113.8e9,
        "nu": 0.342,
        "rho_kg_m3": 4430.0,
        "yield_MPa": 880.0,
    },
    "Tool Steel": {
        "E_Pa": 210.0e9,
        "nu": 0.30,
        "rho_kg_m3": 7800.0,
        "yield_MPa": 1500.0,
    },
    "ABS": {
        "E_Pa": 2.3e9,
        "nu": 0.394,
        "rho_kg_m3": 1050.0,
        "yield_MPa": 43.0,
    },
    "Steel AISI 4140": {
        "E_Pa": 210.0e9,
        "nu": 0.29,
        "rho_kg_m3": 7850.0,
        "yield_MPa": 655.0,
    },
    "Aluminum 7075-T6": {
        "E_Pa": 71.7e9,
        "nu": 0.33,
        "rho_kg_m3": 2810.0,
        "yield_MPa": 503.0,
    },
}


def _lookup_material(name: str) -> dict[str, float]:
    """Return material properties dict, falling back to built-in table.

    The authoritative material database uses lowercase keys (``E_pa``,
    ``yield_mpa``).  We normalise to the keys expected by the Docker
    solver script: ``E_Pa``, ``nu``, ``rho_kg_m3``, ``yield_MPa``.
    """
    try:
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
            get_material,
        )
        mat = get_material(name)
        if mat is not None and isinstance(mat, dict):
            return {
                "E_Pa": mat.get("E_pa", mat.get("E_Pa", 0.0)),
                "nu": mat.get("nu", 0.3),
                "rho_kg_m3": mat.get("rho_kg_m3", 0.0),
                "yield_MPa": mat.get("yield_mpa", mat.get("yield_MPa", 0.0)),
            }
    except Exception:
        pass

    if name in _DEFAULT_MATERIALS:
        return dict(_DEFAULT_MATERIALS[name])

    logger.warning("Material '%s' not found; using generic steel defaults", name)
    return dict(_DEFAULT_MATERIALS["Tool Steel"])


# ---------------------------------------------------------------------------
# ContactSolver
# ---------------------------------------------------------------------------


class ContactSolver:
    """Contact mechanics solver using FEniCSx in Docker.

    Parameters
    ----------
    runner:
        An optional :class:`FEniCSxRunner` instance.  If ``None`` a
        default runner is created (requires ``web.services`` on the
        import path).
    docker_script:
        Path to the contact solver Python script that executes inside
        the container.  Defaults to ``docker/scripts/contact_solver.py``
        relative to the project root.
    """

    def __init__(
        self,
        runner: Any = None,
        docker_script: Optional[str] = None,
    ) -> None:
        if runner is not None:
            self.runner = runner
        elif _RUNNER_AVAILABLE and FEniCSxRunner is not None:
            self.runner = FEniCSxRunner()
        else:
            self.runner = None

        self.docker_script = docker_script or _DOCKER_SCRIPT_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run a contact analysis.

        Parameters
        ----------
        config:
            Dictionary with keys:
            - ``mesh_path`` -- path to the assembly mesh (``.msh`` or
              ``.xdmf``).
            - ``horn_material``, ``workpiece_material``,
              ``anvil_material`` -- material names.
            - ``frequency_hz`` -- ultrasonic frequency.
            - ``amplitude_um`` -- vibration amplitude in micrometres.
            - ``contact_type`` -- ``"penalty"`` or ``"nitsche"``.

            Additional optional keys mirror :class:`ContactConfig`
            attributes.

        Returns
        -------
        dict
            On success::

                {
                    "status": "success",
                    "contact_pressure": {...},
                    "slip_distance": {...},
                    "deformation": {...},
                    "stress": {...},
                    "summary": {...},
                }

            On failure::

                {
                    "status": "error",
                    "error": "<description>",
                }
        """
        # 0. Validate runner
        if self.runner is None:
            return {
                "status": "error",
                "error": (
                    "FEniCSxRunner is not available. "
                    "Ensure web.services is on the Python path and Docker is installed."
                ),
            }

        # 1. Build the full config
        try:
            full_config = self.prepare_config(config)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to prepare config: {exc}",
            }

        # 2. Write config to a temporary file
        tmp_dir = tempfile.mkdtemp(prefix="contact-solver-")
        config_path = os.path.join(tmp_dir, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as fp:
                json.dump(full_config, fp, indent=2, default=str)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to write config file: {exc}",
            }

        # 3. Prepare input files mapping
        input_files: dict[str, str] = {
            "config.json": config_path,
        }
        mesh_path = config.get("mesh_path", "")
        if mesh_path and os.path.isfile(mesh_path):
            input_files["mesh.msh"] = mesh_path

        # 4. Validate the Docker script exists
        if not os.path.isfile(self.docker_script):
            return {
                "status": "error",
                "error": f"Docker solver script not found: {self.docker_script}",
            }

        # 5. Run in Docker
        try:
            result = await self.runner.run_script(
                script_path=self.docker_script,
                input_files=input_files,
                timeout_s=600,
            )
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Docker execution failed: {exc}",
            }

        # 6. Parse results
        if result.get("status") != "success":
            return {
                "status": "error",
                "error": result.get("error", "Unknown Docker error"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }

        return self._parse_results(result)

    def prepare_config(self, user_config: dict[str, Any]) -> dict[str, Any]:
        """Build a complete solver config from user-supplied parameters.

        Resolves material names to property dicts and fills in defaults
        for any unspecified parameters.

        Parameters
        ----------
        user_config:
            Partial configuration dict from the caller.

        Returns
        -------
        dict
            Complete config ready for JSON serialisation and the Docker
            script.
        """
        # Start with a ContactConfig to get defaults
        cc = ContactConfig()

        # Override from user config
        contact_type = user_config.get("contact_type", cc.contact_type)
        if contact_type not in ("penalty", "nitsche"):
            raise ValueError(
                f"Invalid contact_type: {contact_type!r}. "
                f"Must be 'penalty' or 'nitsche'."
            )

        # Resolve materials
        horn_name = user_config.get("horn_material", cc.horn_material)
        workpiece_name = user_config.get("workpiece_material", cc.workpiece_material)
        anvil_name = user_config.get("anvil_material", cc.anvil_material)

        horn_props = _lookup_material(horn_name)
        workpiece_props = _lookup_material(workpiece_name)
        anvil_props = _lookup_material(anvil_name)

        config = {
            "mesh_path": user_config.get("mesh_path", cc.mesh_path),
            "materials": {
                "horn": {"name": horn_name, **horn_props},
                "workpiece": {"name": workpiece_name, **workpiece_props},
                "anvil": {"name": anvil_name, **anvil_props},
            },
            "excitation": {
                "frequency_hz": user_config.get("frequency_hz", cc.frequency_hz),
                "amplitude_um": user_config.get("amplitude_um", cc.amplitude_um),
            },
            "contact": {
                "type": contact_type,
                "penalty_stiffness": user_config.get(
                    "penalty_stiffness", cc.penalty_stiffness
                ),
                "nitsche_parameter": user_config.get(
                    "nitsche_parameter", cc.nitsche_parameter
                ),
                "friction_coefficient": user_config.get(
                    "friction_coefficient", cc.friction_coefficient
                ),
            },
            "solver": {
                "max_newton_iterations": user_config.get(
                    "max_newton_iterations", cc.max_newton_iterations
                ),
                "newton_tolerance": user_config.get(
                    "newton_tolerance", cc.newton_tolerance
                ),
                "time_steps": user_config.get("time_steps", cc.time_steps),
                "total_time_s": user_config.get("total_time_s", cc.total_time_s),
            },
            "output_fields": user_config.get("output_fields", cc.output_fields),
        }

        return config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(docker_result: dict[str, Any]) -> dict[str, Any]:
        """Extract structured contact results from Docker output.

        Parameters
        ----------
        docker_result:
            The dict returned by ``FEniCSxRunner.run_script`` with
            ``status == "success"``.

        Returns
        -------
        dict
            Normalised result dict with top-level keys:
            ``status``, ``contact_pressure``, ``slip_distance``,
            ``deformation``, ``stress``, ``summary``.
        """
        raw = docker_result.get("results", {})

        # The Docker script writes result.json with a known structure.
        # We normalise it here for the rest of the application.
        parsed: dict[str, Any] = {
            "status": "success",
            "run_id": docker_result.get("run_id", ""),
        }

        # Contact pressure
        parsed["contact_pressure"] = {
            "max_MPa": raw.get("contact_pressure_max_mpa", 0.0),
            "mean_MPa": raw.get("contact_pressure_mean_mpa", 0.0),
            "distribution": raw.get("contact_pressure_distribution", []),
        }

        # Slip distance
        parsed["slip_distance"] = {
            "max_um": raw.get("slip_distance_max_um", 0.0),
            "mean_um": raw.get("slip_distance_mean_um", 0.0),
            "distribution": raw.get("slip_distance_distribution", []),
        }

        # Deformation
        parsed["deformation"] = {
            "max_um": raw.get("deformation_max_um", 0.0),
            "field": raw.get("deformation_field", []),
        }

        # Stress
        parsed["stress"] = {
            "von_mises_max_MPa": raw.get("stress_von_mises_max_mpa", 0.0),
            "field": raw.get("stress_field", []),
        }

        # Summary / metadata
        parsed["summary"] = {
            "contact_area_mm2": raw.get("contact_area_mm2", 0.0),
            "total_force_N": raw.get("total_force_n", 0.0),
            "newton_iterations": raw.get("newton_iterations", 0),
            "converged": raw.get("converged", False),
            "solve_time_s": raw.get("solve_time_s", 0.0),
            "contact_type": raw.get("contact_type", "unknown"),
        }

        return parsed

    def get_script_path(self) -> str:
        """Return the path to the Docker solver script."""
        return self.docker_script
