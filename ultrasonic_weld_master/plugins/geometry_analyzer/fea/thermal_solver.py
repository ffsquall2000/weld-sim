"""Thermal analysis solver for ultrasonic welding process.

Provides a high-level ``ThermalSolver`` class that orchestrates:

1. Input preparation -- builds a ``config.json`` describing the thermal
   problem (material thermal properties, contact pressure, excitation
   parameters, weld duration).
2. Docker execution -- delegates the actual FEniCSx computation to a
   standalone Python script (``docker/scripts/thermal_solver.py``) running
   inside the ``dolfinx/dolfinx`` container via :class:`FEniCSxRunner`.
3. Result parsing -- reads the solver output (``result.json``) and returns
   a structured dict with temperature fields, melt zone, and thermal history.

The friction heating model computes the volumetric heat generation rate:

    Q = mu * P * v / d_interface

where:
    - mu: friction coefficient
    - P: contact pressure (Pa)
    - v = 2 * pi * f * A: peak sliding velocity
    - f: ultrasonic frequency (Hz)
    - A: vibration amplitude (m)
    - d_interface: interface layer thickness for volumetric conversion

Usage
-----
::

    from ultrasonic_weld_master.plugins.geometry_analyzer.fea.thermal_solver import (
        ThermalSolver,
    )

    solver = ThermalSolver()

    result = await solver.analyze({
        "materials": {"workpiece": {"name": "ABS"}},
        "contact_pressure_pa": 1e6,
        "frequency_hz": 20000,
        "amplitude_um": 30.0,
        "weld_time_s": 0.5,
        "initial_temp_c": 25.0,
    })

Dependencies
------------
- **FEniCSx / dolfinx** are NOT required on the host; the solver runs
  inside Docker.
- The ``FEniCSxRunner`` from ``web.services.fenicsx_runner`` handles all
  Docker orchestration.
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the Docker thermal solver script (shipped with the project)
# ---------------------------------------------------------------------------

#: Path to the Docker-side thermal solver script.
_DOCKER_SCRIPT_PATH: str = str(
    Path(__file__).resolve().parents[4] / "docker" / "scripts" / "thermal_solver.py"
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
# Thermal material properties
# ---------------------------------------------------------------------------

#: Default thermal properties for common materials.
THERMAL_MATERIALS: dict[str, dict[str, float]] = {
    "Titanium Ti-6Al-4V": {
        "k": 6.7,        # W/m.K  (thermal conductivity)
        "Cp": 526.3,     # J/kg.K (specific heat)
        "rho": 4430.0,   # kg/m^3 (density)
        "melt_temp_c": 1660.0,
    },
    "Tool Steel": {
        "k": 44.5,
        "Cp": 475.0,
        "rho": 7800.0,
        "melt_temp_c": 1500.0,
    },
    "ABS": {
        "k": 0.17,
        "Cp": 1400.0,
        "rho": 1050.0,
        "melt_temp_c": 200.0,
    },
    "Aluminum 7075-T6": {
        "k": 130.0,
        "Cp": 960.0,
        "rho": 2810.0,
        "melt_temp_c": 635.0,
    },
    "Steel AISI 4140": {
        "k": 42.7,
        "Cp": 473.0,
        "rho": 7850.0,
        "melt_temp_c": 1416.0,
    },
    "Nylon 6": {
        "k": 0.25,
        "Cp": 1700.0,
        "rho": 1140.0,
        "melt_temp_c": 220.0,
    },
    "Polycarbonate": {
        "k": 0.20,
        "Cp": 1200.0,
        "rho": 1200.0,
        "melt_temp_c": 267.0,
    },
    "PEEK": {
        "k": 0.25,
        "Cp": 1340.0,
        "rho": 1320.0,
        "melt_temp_c": 343.0,
    },
}


def _lookup_thermal_properties(name: str) -> dict[str, float]:
    """Return thermal properties for a material name.

    Falls back to ABS properties if the material is unknown.
    """
    if name in THERMAL_MATERIALS:
        return dict(THERMAL_MATERIALS[name])
    logger.warning(
        "Thermal properties for '%s' not found; using ABS defaults", name
    )
    return dict(THERMAL_MATERIALS["ABS"])


# ---------------------------------------------------------------------------
# ThermalConfig
# ---------------------------------------------------------------------------


@dataclass
class ThermalConfig:
    """Configuration for a thermal analysis job."""

    # Mesh
    mesh_path: str = ""

    # Materials
    workpiece_material: str = "ABS"

    # Contact (from contact analysis or user input)
    contact_pressure_pa: float = 1e6  # 1 MPa default

    # Excitation
    frequency_hz: float = 20_000.0
    amplitude_um: float = 30.0

    # Friction
    friction_coefficient: float = 0.3

    # Thermal parameters
    weld_time_s: float = 0.5
    n_time_steps: int = 50
    initial_temp_c: float = 25.0
    interface_thickness_m: float = 0.001

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON."""
        return {
            "mesh_path": self.mesh_path,
            "workpiece_material": self.workpiece_material,
            "contact_pressure_pa": self.contact_pressure_pa,
            "frequency_hz": self.frequency_hz,
            "amplitude_um": self.amplitude_um,
            "friction_coefficient": self.friction_coefficient,
            "weld_time_s": self.weld_time_s,
            "n_time_steps": self.n_time_steps,
            "initial_temp_c": self.initial_temp_c,
            "interface_thickness_m": self.interface_thickness_m,
        }


# ---------------------------------------------------------------------------
# ThermalSolver
# ---------------------------------------------------------------------------


class ThermalSolver:
    """Thermal analysis for ultrasonic welding process.

    Parameters
    ----------
    runner:
        An optional :class:`FEniCSxRunner` instance.  If ``None`` a
        default runner is created (requires ``web.services`` on the
        import path).
    docker_script:
        Path to the thermal solver Python script that executes inside
        the container.
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
        """Run thermal analysis.

        Parameters
        ----------
        config:
            Dictionary with keys:
            - ``mesh_path`` -- path to mesh file (optional).
            - ``materials`` -- ``{"workpiece": {"name": "ABS"}}``
            - ``contact_pressure_pa`` -- contact pressure from contact solve.
            - ``frequency_hz`` -- ultrasonic frequency.
            - ``amplitude_um`` -- vibration amplitude in micrometres.
            - ``weld_time_s`` -- total weld duration.
            - ``initial_temp_c`` -- initial temperature in Celsius.

        Returns
        -------
        dict
            On success::

                {
                    "status": "success",
                    "max_temperature_c": ...,
                    "temperature_distribution": [...],
                    "melt_zone": {...},
                    "thermal_history": [...],
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

        # 1. Build full config
        try:
            full_config = self.prepare_config(config)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to prepare thermal config: {exc}",
            }

        # 2. Write config to temp file
        tmp_dir = tempfile.mkdtemp(prefix="thermal-solver-")
        config_path = os.path.join(tmp_dir, "config.json")
        try:
            with open(config_path, "w", encoding="utf-8") as fp:
                json.dump(full_config, fp, indent=2, default=str)
        except Exception as exc:
            return {
                "status": "error",
                "error": f"Failed to write config file: {exc}",
            }

        # 3. Input files
        input_files: dict[str, str] = {
            "config.json": config_path,
        }
        mesh_path = config.get("mesh_path", "")
        if mesh_path and os.path.isfile(mesh_path):
            input_files["mesh.msh"] = mesh_path

        # 4. Validate Docker script
        if not os.path.isfile(self.docker_script):
            return {
                "status": "error",
                "error": f"Docker thermal solver script not found: {self.docker_script}",
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
        """Build a complete thermal solver config from user parameters.

        Parameters
        ----------
        user_config:
            Partial configuration dict.

        Returns
        -------
        dict
            Complete config ready for the Docker script.
        """
        # Extract workpiece material
        materials = user_config.get("materials", {})
        workpiece_name = materials.get("workpiece", {}).get("name", "ABS")
        thermal_props = _lookup_thermal_properties(workpiece_name)

        # Excitation
        frequency_hz = user_config.get("frequency_hz", 20000.0)
        amplitude_um = user_config.get("amplitude_um", 30.0)

        # Contact
        friction_coeff = user_config.get("friction_coefficient", 0.3)
        contact_pressure_pa = user_config.get("contact_pressure_pa", 1e6)

        # Thermal time parameters
        weld_time_s = user_config.get("weld_time_s", 0.5)
        n_time_steps = user_config.get("n_time_steps", 50)
        initial_temp_c = user_config.get("initial_temp_c", 25.0)
        interface_thickness_m = user_config.get("interface_thickness_m", 0.001)

        config = {
            "mesh_path": user_config.get("mesh_path", ""),
            "materials": {
                "workpiece": {
                    "name": workpiece_name,
                    **thermal_props,
                },
            },
            "excitation": {
                "frequency_hz": frequency_hz,
                "amplitude_um": amplitude_um,
            },
            "contact": {
                "friction_coefficient": friction_coeff,
                "contact_pressure_pa": contact_pressure_pa,
            },
            "thermal": {
                "weld_time_s": weld_time_s,
                "n_time_steps": n_time_steps,
                "initial_temp_c": initial_temp_c,
                "melt_temp_c": thermal_props.get("melt_temp_c", 200.0),
                "interface_thickness_m": interface_thickness_m,
            },
        }

        return config

    def compute_heat_generation(
        self,
        contact_pressure: float,
        frequency_hz: float,
        amplitude_um: float,
        friction_coeff: float,
    ) -> float:
        """Compute volumetric heat generation rate.

        Q = mu * P * v = mu * P * 2 * pi * f * A

        This returns the surface heat flux in W/m^2.  To convert to
        volumetric (W/m^3), divide by the interface thickness.

        Parameters
        ----------
        contact_pressure : float
            Contact pressure in Pascals.
        frequency_hz : float
            Ultrasonic frequency in Hz.
        amplitude_um : float
            Vibration amplitude in micrometres.
        friction_coeff : float
            Coulomb friction coefficient.

        Returns
        -------
        float
            Surface heat flux in W/m^2.
        """
        amplitude_m = amplitude_um * 1e-6
        v_peak = 2.0 * math.pi * frequency_hz * amplitude_m
        q_surface = friction_coeff * contact_pressure * v_peak
        return q_surface

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(docker_result: dict[str, Any]) -> dict[str, Any]:
        """Extract structured thermal results from Docker output.

        Parameters
        ----------
        docker_result:
            The dict returned by ``FEniCSxRunner.run_script`` with
            ``status == "success"``.

        Returns
        -------
        dict
            Normalised result dict with top-level keys:
            ``status``, ``max_temperature_c``, ``temperature_distribution``,
            ``melt_zone``, ``thermal_history``.
        """
        raw = docker_result.get("results", {})

        parsed: dict[str, Any] = {
            "status": "success",
            "run_id": docker_result.get("run_id", ""),
        }

        # Temperature results
        parsed["max_temperature_c"] = raw.get("max_temperature_c", 0.0)
        parsed["mean_temperature_c"] = raw.get("mean_temperature_c", 0.0)
        parsed["min_temperature_c"] = raw.get("min_temperature_c", 0.0)
        parsed["initial_temperature_c"] = raw.get("initial_temperature_c", 25.0)

        # Temperature distribution
        parsed["temperature_distribution"] = raw.get("temperature_distribution", [])

        # Melt zone
        parsed["melt_zone"] = {
            "melt_temp_c": raw.get("melt_zone", {}).get("melt_temp_c", 0.0),
            "melt_fraction": raw.get("melt_zone", {}).get("melt_fraction", 0.0),
            "melt_volume_mm3": raw.get("melt_zone", {}).get("melt_volume_mm3", 0.0),
            "n_melt_nodes": raw.get("melt_zone", {}).get("n_melt_nodes", 0),
            "n_total_nodes": raw.get("melt_zone", {}).get("n_total_nodes", 0),
        }

        # Thermal history
        parsed["thermal_history"] = raw.get("thermal_history", [])
        parsed["max_temp_history"] = raw.get("max_temp_history", [])
        parsed["mean_temp_history"] = raw.get("mean_temp_history", [])

        # Heat source info
        parsed["heat_generation_rate_w_m3"] = raw.get(
            "heat_generation_rate_w_m3", 0.0
        )
        parsed["surface_heat_flux_w_m2"] = raw.get("surface_heat_flux_w_m2", 0.0)

        # Timing
        parsed["solve_time_s"] = raw.get("solve_time_s", 0.0)
        parsed["weld_time_s"] = raw.get("weld_time_s", 0.0)
        parsed["n_time_steps"] = raw.get("n_time_steps", 0)

        return parsed

    def get_script_path(self) -> str:
        """Return the path to the Docker thermal solver script."""
        return self.docker_script
