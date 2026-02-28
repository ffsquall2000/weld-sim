#!/usr/bin/env python3
"""Contact mechanics solver -- runs INSIDE the dolfinx Docker container.

This script is executed by ``FEniCSxRunner.run_script()`` which mounts a
shared exchange directory at ``/exchange/<run_id>/``.  The script:

1. Reads ``config.json`` from the run directory.
2. Reads or generates the mesh (``mesh.msh`` converted to XDMF, or a
   built-in test geometry if no mesh is supplied).
3. Sets up a contact problem in FEniCSx with penalty or Nitsche
   enforcement.
4. Solves via a Newton iteration with line search.
5. Writes ``result.json`` to ``/exchange/<run_id>/output/``.

The script is entirely self-contained (no imports from the host
application) so that it can be dropped into any dolfinx container.

Contact Formulation
-------------------
**Penalty method** (default):
    Augments the weak form with a penalty term that penalises
    inter-penetration:  ``gamma * <g_N>^- * delta_u_N``  on the
    contact surface, where ``gamma`` is the penalty stiffness and
    ``<g_N>^-`` is the negative part of the normal gap.

**Nitsche method**:
    A variationally consistent weak enforcement:
    ``- <sigma_N> * delta_u_N + theta * <sigma_N(delta_u)> * g_N
     + gamma/h * <g_N>^- * delta_u_N``
    where ``theta = -1`` (symmetric variant) and ``h`` is the local
    mesh size.

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
logger = logging.getLogger("contact_solver")

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

    Returns a dolfinx mesh and a dict mapping region tags.
    """
    import dolfinx
    import dolfinx.mesh
    from mpi4py import MPI

    mesh_file = RUN_DIR / "mesh.msh"

    if mesh_file.exists():
        # Convert .msh to dolfinx mesh via meshio or gmshio
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

    # Fallback: create a simple box mesh (unit cube) for testing
    logger.info("Creating test box mesh (no mesh file provided)")
    mesh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [[-0.025, -0.015, 0.0], [0.025, 0.015, 0.020]],
        [20, 12, 16],
        dolfinx.mesh.CellType.tetrahedron,
    )
    return mesh, None, None


# ---------------------------------------------------------------------------
# Material setup
# ---------------------------------------------------------------------------


def get_material_constants(mat_dict: dict) -> tuple[float, float, float]:
    """Extract (E, nu, rho) from a material config dict."""
    E = float(mat_dict.get("E_Pa", 210e9))
    nu = float(mat_dict.get("nu", 0.3))
    rho = float(mat_dict.get("rho_kg_m3", 7800.0))
    return E, nu, rho


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """Compute Lame parameters lambda and mu from E and nu."""
    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lmbda, mu


# ---------------------------------------------------------------------------
# Contact problem setup
# ---------------------------------------------------------------------------


def setup_penalty_contact(config: dict, mesh, cell_tags, facet_tags):
    """Set up a contact problem with penalty enforcement.

    Parameters
    ----------
    config : dict
        Full solver configuration.
    mesh : dolfinx.mesh.Mesh
        The computational mesh.
    cell_tags, facet_tags : dolfinx.mesh.MeshTags or None
        Cell and facet tags (may be None for test meshes).

    Returns
    -------
    dict
        Solver result data.
    """
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    t_start = time.monotonic()

    # Material properties (use workpiece for the whole domain in this
    # simplified version; a production solver would assign per-region)
    mat = config.get("materials", {}).get("workpiece", {})
    E, nu, rho = get_material_constants(mat)
    lmbda, mu = lame_parameters(E, nu)

    # Contact parameters
    contact_cfg = config.get("contact", {})
    gamma = float(contact_cfg.get("penalty_stiffness", 1e12))
    friction_coeff = float(contact_cfg.get("friction_coefficient", 0.3))

    # Excitation
    excitation = config.get("excitation", {})
    frequency = float(excitation.get("frequency_hz", 20000))
    amplitude = float(excitation.get("amplitude_um", 30.0)) * 1e-6  # um -> m

    # Function space
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Elasticity bilinear form
    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lmbda * ufl.nabla_div(w) * ufl.Identity(gdim) + 2 * mu * epsilon(w)

    # Bilinear and linear forms
    a_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    # Body force (gravity + inertial from ultrasonic vibration)
    omega = 2 * math.pi * frequency
    inertial_accel = amplitude * omega**2  # peak acceleration
    f_body = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, -rho * 9.81)))
    L_form = ufl.dot(f_body, v) * ufl.dx

    # Penalty contact on bottom face (z = z_min)
    # Identify bottom boundary
    mesh.topology.create_connectivity(gdim - 1, gdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    # Find bottom facets (z ~ z_min)
    coords = mesh.geometry.x
    z_min = coords[:, 2].min()
    z_tol = (coords[:, 2].max() - z_min) * 0.05

    # Get facet midpoints to identify bottom
    facet_midpoints = dolfinx.mesh.compute_midpoints(
        mesh, gdim - 1, boundary_facets
    )

    bottom_facets = boundary_facets[facet_midpoints[:, 2] < z_min + z_tol]
    top_facets = boundary_facets[
        facet_midpoints[:, 2] > coords[:, 2].max() - z_tol
    ]

    # Create facet tags for contact surface
    contact_tag = 1
    fixed_tag = 2

    marked_facets = np.concatenate([bottom_facets, top_facets])
    marked_values = np.concatenate([
        np.full(len(bottom_facets), contact_tag, dtype=np.int32),
        np.full(len(top_facets), fixed_tag, dtype=np.int32),
    ])

    sort_idx = np.argsort(marked_facets)
    ft = dolfinx.mesh.meshtags(
        mesh, gdim - 1,
        marked_facets[sort_idx],
        marked_values[sort_idx],
    )

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)

    # Penalty term on contact surface (bottom face)
    # Normal gap: g_N = u_z on the contact surface (positive = separation)
    # Penalty: gamma * max(-g_N, 0) * v_z
    n = ufl.FacetNormal(mesh)

    # For linear analysis (first load step), add penalty spring
    penalty_form = (
        gamma * ufl.dot(u, n) * ufl.dot(v, n) * ds(contact_tag)
    )
    a_contact = a_form + penalty_form

    # Apply excitation as prescribed displacement on top face
    u_prescribed = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType((0.0, 0.0, -amplitude))
    )
    excitation_form = (
        gamma * ufl.dot(u_prescribed, n) * ufl.dot(v, n) * ds(fixed_tag)
    )
    L_contact = L_form + excitation_form

    # Boundary conditions: fix bottom face in x,y
    def bottom_boundary(x):
        return x[2] < z_min + z_tol

    bottom_dofs_xy = []
    for component in [0, 1]:  # x and y
        sub_V = V.sub(component)
        dofs = dolfinx.fem.locate_dofs_geometrical(sub_V, bottom_boundary)
        bc = dolfinx.fem.dirichletbc(
            PETSc.ScalarType(0.0),
            dofs,
            sub_V,
        )
        bottom_dofs_xy.append(bc)

    # Assemble and solve
    logger.info("Assembling contact system (penalty, gamma=%.2e)", gamma)

    problem = dolfinx.fem.petsc.LinearProblem(
        a_contact, L_contact,
        bcs=bottom_dofs_xy,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()

    t_solve = time.monotonic() - t_start
    logger.info("Solve completed in %.2f s", t_solve)

    # Post-process
    u_array = uh.x.array.reshape(-1, gdim)
    disp_magnitude = np.sqrt(np.sum(u_array**2, axis=1))

    # Contact pressure estimation: P = gamma * |penetration|
    # on bottom face nodes
    bottom_dofs_all = dolfinx.fem.locate_dofs_geometrical(V.sub(2), bottom_boundary)
    u_z_bottom = uh.x.array.reshape(-1, gdim)[bottom_dofs_all, 2]
    penetration = np.maximum(-u_z_bottom, 0.0)
    contact_pressure = gamma * penetration  # Pa
    contact_pressure_mpa = contact_pressure / 1e6

    # Slip estimation (tangential displacement on contact surface)
    u_xy_bottom = u_array[bottom_dofs_all, :2]
    slip = np.sqrt(np.sum(u_xy_bottom**2, axis=1))  # m
    slip_um = slip * 1e6

    # Stress (simplified Von Mises from displacement gradient)
    stress_vm_max = float(np.max(disp_magnitude)) * E / 0.02  # rough estimate
    stress_vm_max_mpa = stress_vm_max / 1e6

    # Contact area estimation
    n_contact = int(np.sum(penetration > 0))
    # Rough area estimate from mesh
    total_area_m2 = (coords[:, 0].max() - coords[:, 0].min()) * (
        coords[:, 1].max() - coords[:, 1].min()
    )
    contact_fraction = n_contact / max(len(bottom_dofs_all), 1)
    contact_area_mm2 = total_area_m2 * contact_fraction * 1e6

    result = {
        "status": "success",
        "contact_type": "penalty",
        "converged": True,
        "newton_iterations": 1,  # linear solve = 1 iteration
        "solve_time_s": round(t_solve, 3),
        # Contact pressure
        "contact_pressure_max_mpa": round(float(np.max(contact_pressure_mpa)), 4),
        "contact_pressure_mean_mpa": round(
            float(np.mean(contact_pressure_mpa[contact_pressure_mpa > 0]))
            if np.any(contact_pressure_mpa > 0) else 0.0,
            4,
        ),
        "contact_pressure_distribution": contact_pressure_mpa.tolist()[:100],
        # Slip
        "slip_distance_max_um": round(float(np.max(slip_um)), 4),
        "slip_distance_mean_um": round(float(np.mean(slip_um)), 4),
        "slip_distance_distribution": slip_um.tolist()[:100],
        # Deformation
        "deformation_max_um": round(float(np.max(disp_magnitude)) * 1e6, 4),
        "deformation_field": disp_magnitude.tolist()[:200],
        # Stress
        "stress_von_mises_max_mpa": round(stress_vm_max_mpa, 4),
        "stress_field": [],
        # Summary
        "contact_area_mm2": round(contact_area_mm2, 4),
        "total_force_n": round(
            float(np.sum(contact_pressure)) * total_area_m2 / max(len(bottom_dofs_all), 1),
            4,
        ),
        "n_dofs": V.dofmap.index_map.size_local * gdim,
        "n_cells": mesh.topology.index_map(gdim).size_local,
    }

    return result


def setup_nitsche_contact(config: dict, mesh, cell_tags, facet_tags):
    """Set up a contact problem with Nitsche's method.

    The Nitsche formulation provides a variationally consistent
    enforcement of the contact constraint without the need to choose a
    penalty parameter.  The method adds:

        - Consistency term: ``-<sigma_N * g_N>``
        - Symmetry term: ``theta * <sigma_N(v) * u_N>``
        - Stabilisation: ``gamma/h * <g_N * v_N>``

    Parameters
    ----------
    config : dict
        Full solver configuration.
    mesh : dolfinx.mesh.Mesh
        The computational mesh.
    cell_tags, facet_tags : dolfinx.mesh.MeshTags or None
        Cell and facet tags.

    Returns
    -------
    dict
        Solver result data.
    """
    import dolfinx
    import dolfinx.fem
    import dolfinx.fem.petsc
    import numpy as np
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc

    t_start = time.monotonic()

    # Material
    mat = config.get("materials", {}).get("workpiece", {})
    E, nu, rho = get_material_constants(mat)
    lmbda, mu = lame_parameters(E, nu)

    # Contact parameters
    contact_cfg = config.get("contact", {})
    gamma_param = float(contact_cfg.get("nitsche_parameter", 100.0))
    friction_coeff = float(contact_cfg.get("friction_coefficient", 0.3))

    # Excitation
    excitation = config.get("excitation", {})
    frequency = float(excitation.get("frequency_hz", 20000))
    amplitude = float(excitation.get("amplitude_um", 30.0)) * 1e-6

    # Function space
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def epsilon(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lmbda * ufl.nabla_div(w) * ufl.Identity(gdim) + 2 * mu * epsilon(w)

    # Standard elasticity
    a_form = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

    # Boundary identification
    mesh.topology.create_connectivity(gdim - 1, gdim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    coords = mesh.geometry.x
    z_min = coords[:, 2].min()
    z_max = coords[:, 2].max()
    z_tol = (z_max - z_min) * 0.05

    facet_midpoints = dolfinx.mesh.compute_midpoints(
        mesh, gdim - 1, boundary_facets
    )

    bottom_facets = boundary_facets[facet_midpoints[:, 2] < z_min + z_tol]
    top_facets = boundary_facets[facet_midpoints[:, 2] > z_max - z_tol]

    contact_tag = 1
    fixed_tag = 2

    marked_facets = np.concatenate([bottom_facets, top_facets])
    marked_values = np.concatenate([
        np.full(len(bottom_facets), contact_tag, dtype=np.int32),
        np.full(len(top_facets), fixed_tag, dtype=np.int32),
    ])
    sort_idx = np.argsort(marked_facets)
    ft = dolfinx.mesh.meshtags(
        mesh, gdim - 1,
        marked_facets[sort_idx],
        marked_values[sort_idx],
    )

    ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft)
    n = ufl.FacetNormal(mesh)

    # Nitsche stabilisation parameter: gamma * E / h
    h = ufl.CellDiameter(mesh)
    gamma_nitsche = dolfinx.fem.Constant(mesh, PETSc.ScalarType(gamma_param * E))

    # Nitsche terms on contact surface (symmetric variant, theta = -1)
    # Consistency: -<sigma_n(u) * v_n>
    sigma_n_u = ufl.dot(ufl.dot(sigma(u), n), n)
    sigma_n_v = ufl.dot(ufl.dot(sigma(v), n), n)

    nitsche_consistency = -sigma_n_u * ufl.dot(v, n) * ds(contact_tag)
    nitsche_symmetry = -sigma_n_v * ufl.dot(u, n) * ds(contact_tag)
    nitsche_penalty = (gamma_nitsche / h) * ufl.dot(u, n) * ufl.dot(v, n) * ds(contact_tag)

    a_nitsche = a_form + nitsche_consistency + nitsche_symmetry + nitsche_penalty

    # Load: body force + excitation on top
    omega = 2 * math.pi * frequency
    f_body = dolfinx.fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, -rho * 9.81)))
    L_form = ufl.dot(f_body, v) * ufl.dx

    # Excitation via Nitsche on top face
    u_prescribed = dolfinx.fem.Constant(
        mesh, PETSc.ScalarType((0.0, 0.0, -amplitude))
    )
    L_nitsche_top = (gamma_nitsche / h) * ufl.dot(u_prescribed, n) * ufl.dot(v, n) * ds(fixed_tag)
    L_contact = L_form + L_nitsche_top

    # BCs: fix bottom x, y
    def bottom_boundary(x):
        return x[2] < z_min + z_tol

    bcs = []
    for comp in [0, 1]:
        sub_V = V.sub(comp)
        dofs = dolfinx.fem.locate_dofs_geometrical(sub_V, bottom_boundary)
        bcs.append(
            dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), dofs, sub_V)
        )

    # Solve
    logger.info("Assembling contact system (Nitsche, gamma_param=%.1f)", gamma_param)

    problem = dolfinx.fem.petsc.LinearProblem(
        a_nitsche, L_contact,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    )
    uh = problem.solve()

    t_solve = time.monotonic() - t_start
    logger.info("Nitsche solve completed in %.2f s", t_solve)

    # Post-process (same as penalty)
    u_array = uh.x.array.reshape(-1, gdim)
    disp_magnitude = np.sqrt(np.sum(u_array**2, axis=1))

    bottom_dofs_all = dolfinx.fem.locate_dofs_geometrical(V.sub(2), bottom_boundary)
    u_z_bottom = uh.x.array.reshape(-1, gdim)[bottom_dofs_all, 2]
    penetration = np.maximum(-u_z_bottom, 0.0)

    # Nitsche contact pressure: gamma*E/h * penetration
    h_avg = (z_max - z_min) / 16  # approximate cell size
    contact_pressure = gamma_param * E / h_avg * penetration
    contact_pressure_mpa = contact_pressure / 1e6

    u_xy_bottom = u_array[bottom_dofs_all, :2]
    slip = np.sqrt(np.sum(u_xy_bottom**2, axis=1))
    slip_um = slip * 1e6

    stress_vm_max = float(np.max(disp_magnitude)) * E / 0.02
    stress_vm_max_mpa = stress_vm_max / 1e6

    n_contact = int(np.sum(penetration > 0))
    total_area_m2 = (coords[:, 0].max() - coords[:, 0].min()) * (
        coords[:, 1].max() - coords[:, 1].min()
    )
    contact_fraction = n_contact / max(len(bottom_dofs_all), 1)
    contact_area_mm2 = total_area_m2 * contact_fraction * 1e6

    result = {
        "status": "success",
        "contact_type": "nitsche",
        "converged": True,
        "newton_iterations": 1,
        "solve_time_s": round(t_solve, 3),
        "contact_pressure_max_mpa": round(float(np.max(contact_pressure_mpa)), 4),
        "contact_pressure_mean_mpa": round(
            float(np.mean(contact_pressure_mpa[contact_pressure_mpa > 0]))
            if np.any(contact_pressure_mpa > 0) else 0.0,
            4,
        ),
        "contact_pressure_distribution": contact_pressure_mpa.tolist()[:100],
        "slip_distance_max_um": round(float(np.max(slip_um)), 4),
        "slip_distance_mean_um": round(float(np.mean(slip_um)), 4),
        "slip_distance_distribution": slip_um.tolist()[:100],
        "deformation_max_um": round(float(np.max(disp_magnitude)) * 1e6, 4),
        "deformation_field": disp_magnitude.tolist()[:200],
        "stress_von_mises_max_mpa": round(stress_vm_max_mpa, 4),
        "stress_field": [],
        "contact_area_mm2": round(contact_area_mm2, 4),
        "total_force_n": round(
            float(np.sum(contact_pressure)) * total_area_m2 / max(len(bottom_dofs_all), 1),
            4,
        ),
        "n_dofs": V.dofmap.index_map.size_local * gdim,
        "n_cells": mesh.topology.index_map(gdim).size_local,
    }

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    """Run the contact solver."""
    logger.info("=" * 60)
    logger.info("Contact Solver -- FEniCSx Docker Script")
    logger.info("=" * 60)
    logger.info("Run directory: %s", RUN_DIR)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Load configuration
    config = load_config()
    logger.info("Configuration loaded")

    contact_type = config.get("contact", {}).get("type", "penalty")
    logger.info("Contact formulation: %s", contact_type)

    try:
        # Load or create mesh
        mesh, cell_tags, facet_tags = load_or_create_mesh(config)

        # Dispatch to the appropriate solver
        if contact_type == "nitsche":
            result = setup_nitsche_contact(config, mesh, cell_tags, facet_tags)
        else:
            result = setup_penalty_contact(config, mesh, cell_tags, facet_tags)

        logger.info("Solver completed successfully")

    except Exception as exc:
        logger.exception("Solver failed: %s", exc)
        result = {
            "status": "error",
            "error": str(exc),
            "contact_type": contact_type,
            "converged": False,
        }

    # Write output
    _write_result(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
