#!/usr/bin/env python3
"""Thermal analysis solver -- runs INSIDE the dolfinx Docker container.

This script is executed by ``FEniCSxRunner.run_script()`` which mounts a
shared exchange directory at ``/exchange/<run_id>/``.  The script:

1. Reads ``config.json`` from the run directory.
2. Reads or generates the mesh (``mesh.msh`` converted via gmshio, or a
   built-in test geometry if no mesh is supplied).
3. Sets up a transient heat equation in FEniCSx:
       rho * Cp * dT/dt = k * nabla^2(T) + Q
   where Q is the friction heating source term at the contact interface.
4. Time-steps over the weld duration using an implicit backward-Euler scheme.
5. Writes ``result.json`` to ``/exchange/<run_id>/output/``.

The script is entirely self-contained (no imports from the host
application) so that it can be dropped into any dolfinx container.

Friction Heating Model
----------------------
The volumetric heat generation rate at the contact interface is:

    Q = mu * P * v / d_interface

where:
    - mu: friction coefficient
    - P: contact pressure (Pa)
    - v = 2 * pi * f * A: peak sliding velocity
    - f: ultrasonic frequency (Hz)
    - A: vibration amplitude (m)
    - d_interface: interface layer thickness (m) for volumetric conversion

Usage
-----
::

    docker run --rm \\
        -v /tmp/exchange:/exchange \\
        dolfinx/dolfinx:stable \\
        python /exchange/<run_id>/script.py

Environment
-----------
The working directory is set by Docker; all paths are resolved relative
to the script location (which is inside the run directory).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("thermal_solver")

# The script is placed at /exchange/<run_id>/script.py by FEniCSxRunner.
SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR  # same directory
OUTPUT_DIR = RUN_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_result(data: dict) -> None:
    """Write result.json to the output directory."""
    result_path = OUTPUT_DIR / "result.json"
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, default=_json_default)
    logger.info("Wrote result to %s", result_path)


def _json_default(obj):
    """JSON serializer for numpy types."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------


def load_config() -> dict:
    """Load config.json from the run directory."""
    config_path = RUN_DIR / "config.json"
    if not config_path.exists():
        logger.error("config.json not found in %s", RUN_DIR)
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as fp:
        return json.load(fp)


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------


def load_or_create_mesh(config: dict):
    """Load the mesh from file or create a simple test geometry.

    Returns a dolfinx mesh.
    """
    import dolfinx
    import dolfinx.mesh
    from mpi4py import MPI

    mesh_file = RUN_DIR / "mesh.msh"

    if mesh_file.exists():
        logger.info("Loading mesh from %s", mesh_file)
        try:
            import dolfinx.io.gmshio as gmshio
            mesh, cell_tags, facet_tags = gmshio.read_from_msh(
                str(mesh_file), MPI.COMM_WORLD, rank=0, gdim=3
            )
            logger.info(
                "Mesh loaded: %d cells, %d vertices",
                mesh.topology.index_map(3).size_local,
                mesh.topology.index_map(0).size_local,
            )
            return mesh, cell_tags, facet_tags
        except Exception as exc:
            logger.warning("gmshio failed: %s -- creating test mesh", exc)

    # Fallback: simple box mesh
    logger.info("Creating test box mesh (no mesh file provided)")
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[-0.025, -0.015, 0.0], [0.025, 0.015, 0.020]],
        [20, 12, 16],
        dolfinx.mesh.CellType.tetrahedron,
    )
    return mesh, None, None


# ---------------------------------------------------------------------------
# Material / thermal properties
# ---------------------------------------------------------------------------

# Default thermal properties for common materials.
# Keys: k (W/m.K), Cp (J/kg.K), rho (kg/m3)
_THERMAL_DEFAULTS = {
    "Titanium Ti-6Al-4V": {"k": 6.7, "Cp": 526.3, "rho": 4430.0},
    "Tool Steel": {"k": 44.5, "Cp": 475.0, "rho": 7800.0},
    "ABS": {"k": 0.17, "Cp": 1400.0, "rho": 1050.0},
    "Aluminum 7075-T6": {"k": 130.0, "Cp": 960.0, "rho": 2810.0},
    "Steel AISI 4140": {"k": 42.7, "Cp": 473.0, "rho": 7850.0},
    "Nylon 6": {"k": 0.25, "Cp": 1700.0, "rho": 1140.0},
    "Polycarbonate": {"k": 0.20, "Cp": 1200.0, "rho": 1200.0},
    "PEEK": {"k": 0.25, "Cp": 1340.0, "rho": 1320.0},
}


def get_thermal_properties(material_name: str) -> dict:
    """Get thermal properties for a material."""
    if material_name in _THERMAL_DEFAULTS:
        return dict(_THERMAL_DEFAULTS[material_name])
    logger.warning(
        "No thermal properties for '%s'; using ABS defaults", material_name
    )
    return dict(_THERMAL_DEFAULTS["ABS"])


def compute_heat_generation(
    contact_pressure_pa: float,
    frequency_hz: float,
    amplitude_m: float,
    friction_coefficient: float,
    interface_thickness_m: float = 0.001,
) -> float:
    """Compute volumetric heat generation rate.

    Q = mu * P * v / d_interface

    where v = 2 * pi * f * A  (peak sliding velocity).

    Parameters
    ----------
    contact_pressure_pa : float
        Contact pressure in Pascals.
    frequency_hz : float
        Ultrasonic frequency in Hz.
    amplitude_m : float
        Vibration amplitude in metres.
    friction_coefficient : float
        Coulomb friction coefficient.
    interface_thickness_m : float
        Thickness of the interface layer for volumetric conversion.

    Returns
    -------
    float
        Volumetric heat generation rate in W/m^3.
    """
    v_peak = 2.0 * math.pi * frequency_hz * amplitude_m
    q_surface = friction_coefficient * contact_pressure_pa * v_peak  # W/m^2
    q_volumetric = q_surface / interface_thickness_m  # W/m^3
    return q_volumetric


# ---------------------------------------------------------------------------
# Thermal solver
# ---------------------------------------------------------------------------


def solve_thermal(config: dict):
    """Set up and solve the transient heat equation.

    The governing equation is:

        rho * Cp * dT/dt = k * nabla^2(T) + Q

    Discretised with backward Euler in time and linear FEM in space.

    Parameters
    ----------
    config : dict
        Full solver configuration from config.json.

    Returns
    -------
    dict
        Solver result data including temperature fields and history.
    """
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    thermal_cfg = config.get("thermal", {})
    excitation = config.get("excitation", {})
    contact_cfg = config.get("contact", {})
    materials = config.get("materials", {})

    # Workpiece material thermal properties
    workpiece_name = materials.get("workpiece", {}).get("name", "ABS")
    thermal_props = get_thermal_properties(workpiece_name)
    k_thermal = thermal_props["k"]       # W/m.K
    Cp = thermal_props["Cp"]             # J/kg.K
    rho = thermal_props["rho"]           # kg/m^3

    # Excitation parameters
    frequency_hz = float(excitation.get("frequency_hz", 20000))
    amplitude_um = float(excitation.get("amplitude_um", 30.0))
    amplitude_m = amplitude_um * 1e-6

    # Contact parameters
    friction_coeff = float(contact_cfg.get("friction_coefficient", 0.3))
    contact_pressure_pa = float(
        contact_cfg.get("contact_pressure_pa", 1e6)
    )  # Default 1 MPa

    # Time-stepping
    weld_time_s = float(thermal_cfg.get("weld_time_s", 0.5))
    n_steps = int(thermal_cfg.get("n_time_steps", 50))
    dt = weld_time_s / n_steps

    # Initial temperature
    initial_temp_c = float(thermal_cfg.get("initial_temp_c", 25.0))
    initial_temp_k = initial_temp_c + 273.15

    # Melt temperature (for melt zone prediction)
    melt_temp_c = float(thermal_cfg.get("melt_temp_c", 200.0))  # For ABS
    melt_temp_k = melt_temp_c + 273.15

    # Heat generation
    interface_thickness = float(
        thermal_cfg.get("interface_thickness_m", 0.001)
    )
    Q_volumetric = compute_heat_generation(
        contact_pressure_pa=contact_pressure_pa,
        frequency_hz=frequency_hz,
        amplitude_m=amplitude_m,
        friction_coefficient=friction_coeff,
        interface_thickness_m=interface_thickness,
    )
    logger.info("Heat generation rate: %.3e W/m^3", Q_volumetric)

    # ------------------------------------------------------------------
    # Mesh
    # ------------------------------------------------------------------
    mesh, cell_tags, facet_tags = load_or_create_mesh(config)
    gdim = mesh.geometry.dim
    coords = mesh.geometry.x

    # ------------------------------------------------------------------
    # Function space (scalar temperature)
    # ------------------------------------------------------------------
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    # Temperature at current and previous time steps
    T_n = dolfinx.fem.Function(V, name="T_n")  # previous step
    T_h = dolfinx.fem.Function(V, name="T_h")  # solution

    # Set initial temperature
    T_n.x.array[:] = initial_temp_k
    T_h.x.array[:] = initial_temp_k

    # Trial and test functions
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ------------------------------------------------------------------
    # Heat source: apply Q at the contact interface (bottom face)
    # ------------------------------------------------------------------
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    z_tol = (z_max - z_min) * 0.05

    # Identify bottom boundary facets for contact heat source
    mesh.topology.create_connectivity(gdim - 1, gdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    facet_midpoints = dolfinx.mesh.compute_midpoints(
        mesh, gdim - 1, boundary_facets
    )
    bottom_facets = boundary_facets[facet_midpoints[:, 2] < z_min + z_tol]

    contact_tag = 10
    marked_facets = bottom_facets
    marked_values = np.full(len(bottom_facets), contact_tag, dtype=np.int32)
    sort_idx = np.argsort(marked_facets)
    ft = dolfinx.mesh.meshtags(
        mesh, gdim - 1,
        marked_facets[sort_idx],
        marked_values[sort_idx],
    )
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

    # Surface heat flux (W/m^2) at contact interface
    q_surface = (
        friction_coeff * contact_pressure_pa
        * 2.0 * math.pi * frequency_hz * amplitude_m
    )
    q_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(q_surface))

    # Volumetric heat source (applied throughout the domain near interface)
    Q_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(Q_volumetric))

    # ------------------------------------------------------------------
    # Weak form: backward Euler
    # rho*Cp*(T - T_n)/dt * v + k*grad(T).grad(v) = Q*v + q_s*v*ds
    # ------------------------------------------------------------------
    k_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(k_thermal))
    rho_cp = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType(rho * Cp)
    )
    dt_const = dolfinx.fem.Constant(mesh, PETSc.ScalarType(dt))

    # Bilinear form (LHS)
    a_form = (
        (rho_cp / dt_const) * T * v * ufl.dx
        + k_const * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
    )

    # Linear form (RHS)
    L_form = (
        (rho_cp / dt_const) * T_n * v * ufl.dx
        + Q_const * v * ufl.dx
        + q_const * v * ds(contact_tag)
    )

    # Convective boundary condition on top face (ambient cooling)
    h_conv = dolfinx.fem.Constant(mesh, PETSc.ScalarType(10.0))  # W/m^2.K
    T_ambient = dolfinx.fem.Constant(mesh, PETSc.ScalarType(initial_temp_k))

    top_facets = boundary_facets[facet_midpoints[:, 2] > z_max - z_tol]
    top_tag = 20
    all_marked_facets = np.concatenate([marked_facets, top_facets])
    all_marked_values = np.concatenate([
        marked_values,
        np.full(len(top_facets), top_tag, dtype=np.int32),
    ])
    sort_idx_all = np.argsort(all_marked_facets)
    ft_all = dolfinx.mesh.meshtags(
        mesh, gdim - 1,
        all_marked_facets[sort_idx_all],
        all_marked_values[sort_idx_all],
    )
    ds_all = ufl.Measure("ds", domain=mesh, subdomain_data=ft_all)

    # Add convection to LHS and RHS
    a_form += h_conv * T * v * ds_all(top_tag)
    L_form_full = L_form + h_conv * T_ambient * v * ds_all(top_tag)

    # ------------------------------------------------------------------
    # Compile forms and create solver
    # ------------------------------------------------------------------
    bilinear = dolfinx.fem.form(a_form)
    linear = dolfinx.fem.form(L_form_full)

    A = dolfinx.fem.petsc.assemble_matrix(bilinear)
    A.assemble()
    b = dolfinx.fem.petsc.create_vector(linear)

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------
    thermal_history = []
    max_temps = []
    mean_temps = []
    current_time = 0.0

    logger.info(
        "Starting thermal solve: %d steps, dt=%.4f s, total=%.3f s",
        n_steps, dt, weld_time_s,
    )

    for step in range(n_steps):
        current_time += dt

        # Reassemble RHS
        with b.localForm() as loc:
            loc.set(0)
        dolfinx.fem.petsc.assemble_vector(b, linear)
        b.ghostUpdate(
            addv=PETSc.InsertMode.ADD_VALUES,
            mode=PETSc.ScatterMode.REVERSE,
        )

        # Solve
        solver.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()

        # Update previous time step
        T_n.x.array[:] = T_h.x.array[:]

        # Record temperature history
        T_array = T_h.x.array
        max_T = float(np.max(T_array))
        mean_T = float(np.mean(T_array))
        max_temps.append(round(max_T - 273.15, 2))  # Convert to Celsius
        mean_temps.append(round(mean_T - 273.15, 2))

        # Record snapshots at regular intervals
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            thermal_history.append({
                "time_s": round(current_time, 6),
                "max_temp_c": round(max_T - 273.15, 2),
                "mean_temp_c": round(mean_T - 273.15, 2),
                "min_temp_c": round(float(np.min(T_array)) - 273.15, 2),
            })

        if step % 10 == 0:
            logger.info(
                "Step %d/%d: t=%.4f s, T_max=%.1f C, T_mean=%.1f C",
                step + 1, n_steps, current_time,
                max_T - 273.15, mean_T - 273.15,
            )

    t_solve = time.monotonic() - t_start
    logger.info("Thermal solve completed in %.2f s", t_solve)

    # ------------------------------------------------------------------
    # Post-process: melt zone prediction
    # ------------------------------------------------------------------
    T_final = T_h.x.array
    T_final_c = T_final - 273.15
    melt_mask = T_final_c > melt_temp_c
    n_melt_nodes = int(np.sum(melt_mask))
    n_total_nodes = len(T_final)
    melt_fraction = n_melt_nodes / max(n_total_nodes, 1)

    # Estimate melt zone volume (rough: fraction of total domain volume)
    domain_vol = (
        (coords[:, 0].max() - coords[:, 0].min())
        * (coords[:, 1].max() - coords[:, 1].min())
        * (coords[:, 2].max() - coords[:, 2].min())
    )
    melt_volume_mm3 = float(domain_vol * melt_fraction * 1e9)  # m^3 -> mm^3

    # Temperature distribution at final time (sampled)
    temp_distribution = T_final_c.tolist()[:200]

    result = {
        "status": "success",
        "solve_time_s": round(t_solve, 3),
        "n_time_steps": n_steps,
        "weld_time_s": weld_time_s,
        "dt_s": dt,
        # Temperature results
        "max_temperature_c": round(float(np.max(T_final_c)), 2),
        "mean_temperature_c": round(float(np.mean(T_final_c)), 2),
        "min_temperature_c": round(float(np.min(T_final_c)), 2),
        "initial_temperature_c": initial_temp_c,
        # Temperature distribution
        "temperature_distribution": temp_distribution,
        # Melt zone
        "melt_zone": {
            "melt_temp_c": melt_temp_c,
            "melt_fraction": round(melt_fraction, 4),
            "melt_volume_mm3": round(melt_volume_mm3, 2),
            "n_melt_nodes": n_melt_nodes,
            "n_total_nodes": n_total_nodes,
        },
        # Thermal history
        "thermal_history": thermal_history,
        "max_temp_history": max_temps,
        "mean_temp_history": mean_temps,
        # Heat source info
        "heat_generation_rate_w_m3": Q_volumetric,
        "surface_heat_flux_w_m2": q_surface,
        # Material info
        "material": workpiece_name,
        "thermal_conductivity_w_mk": k_thermal,
        "specific_heat_j_kgk": Cp,
        "density_kg_m3": rho,
        # Mesh info
        "n_dofs": V.dofmap.index_map.size_local,
        "n_cells": mesh.topology.index_map(gdim).size_local,
    }

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Run the thermal solver."""
    logger.info("=" * 60)
    logger.info("Thermal Solver -- FEniCSx Docker Script")
    logger.info("=" * 60)
    logger.info("Run directory: %s", RUN_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")

    try:
        result = solve_thermal(config)
        logger.info("Thermal solver completed successfully")
    except Exception as exc:
        logger.exception("Thermal solver failed: %s", exc)
        result = {
            "status": "error",
            "error": str(exc),
        }

    # Write output
    _write_result(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
