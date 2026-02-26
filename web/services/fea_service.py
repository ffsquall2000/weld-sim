"""FEA modal analysis service -- pure numpy/scipy, no OpenGL dependencies.

Implements a simplified finite element solver for ultrasonic horn modal analysis:
  1. Structured hex mesh generation (numpy)
  2. 8-node hexahedral element stiffness and mass matrices
  3. Sparse global assembly
  4. Shift-invert eigenvalue solve near target frequency (scipy.sparse.linalg.eigsh)
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    FEA_MATERIALS,
    get_material,
)

logger = logging.getLogger(__name__)

# Mesh density presets: (nx_per_10mm, ny_per_10mm, nz_per_10mm)
_MESH_DENSITY: dict[str, tuple[int, int, int]] = {
    "coarse": (2, 2, 2),
    "medium": (3, 3, 3),
    "fine": (5, 5, 5),
}


class FEAService:
    """Pure numpy/scipy FEA modal analysis for ultrasonic horns."""

    def run_modal_analysis(
        self,
        horn_type: str,
        width_mm: float,
        height_mm: float,
        length_mm: float,
        material: str,
        frequency_khz: float,
        mesh_density: str = "medium",
    ) -> dict:
        """Run modal analysis and return results for the web frontend."""
        mat = get_material(material)
        if mat is None:
            raise ValueError(
                f"Unknown material '{material}'. "
                f"Available: {list(FEA_MATERIALS.keys())}"
            )

        density_key = mesh_density if mesh_density in _MESH_DENSITY else "medium"
        per10 = _MESH_DENSITY[density_key]

        # Calculate mesh divisions
        nx = max(2, int(width_mm / 10 * per10[0]))
        ny = max(3, int(height_mm / 10 * per10[1]))
        nz = max(2, int(length_mm / 10 * per10[2]))

        # Cap to avoid huge meshes
        max_nodes = 8000  # keep solve fast (<10s)
        while (nx + 1) * (ny + 1) * (nz + 1) > max_nodes:
            nx = max(2, nx - 1)
            ny = max(3, ny - 1)
            nz = max(2, nz - 1)

        t0 = time.perf_counter()

        # Step 1: Generate mesh
        vertices, elements = self._generate_hex_mesh(
            width_mm, height_mm, length_mm, nx, ny, nz, horn_type
        )
        node_count = len(vertices)
        element_count = len(elements)

        # Step 2: Assemble K and M
        K, M = self._assemble_global(vertices, elements, mat)

        # Step 3: Apply boundary conditions (fix bottom face)
        fixed_dofs = self._get_fixed_dofs(vertices, height_mm)
        K, M = self._apply_bc(K, M, fixed_dofs)

        # Step 4: Eigenvalue solve
        target_hz = frequency_khz * 1000.0
        sigma = (2 * math.pi * target_hz) ** 2
        n_modes = min(10, K.shape[0] - 2)

        try:
            eigenvalues, eigenvectors = eigsh(
                K, k=n_modes, M=M, sigma=sigma, which="LM"
            )
        except Exception:
            logger.warning("Shift-invert failed, using SM fallback")
            eigenvalues, eigenvectors = eigsh(
                K, k=min(6, K.shape[0] - 2), M=M, which="SM"
            )

        solve_time = time.perf_counter() - t0

        # Step 5: Build mode shapes
        mode_shapes = []
        for i, eigval in enumerate(eigenvalues):
            freq = math.sqrt(max(abs(eigval), 0.0)) / (2 * math.pi)
            disp = eigenvectors[:, i]

            # Classify mode type from displacement pattern
            disp_3d = disp.reshape(-1, 3)[:node_count]
            dx_rms = float(np.sqrt(np.mean(disp_3d[:, 0] ** 2)))
            dy_rms = float(np.sqrt(np.mean(disp_3d[:, 1] ** 2)))
            dz_rms = float(np.sqrt(np.mean(disp_3d[:, 2] ** 2)))
            total_rms = max(dx_rms + dy_rms + dz_rms, 1e-12)

            if dy_rms / total_rms > 0.6:
                mode_type = "longitudinal"
            elif dx_rms / total_rms > 0.5 or dz_rms / total_rms > 0.5:
                mode_type = "flexural"
            else:
                mode_type = "torsional"

            mode_shapes.append(
                {
                    "frequency_hz": round(freq, 1),
                    "mode_type": mode_type,
                    "participation_factor": round(
                        float(np.max(np.abs(disp_3d))), 6
                    ),
                    "effective_mass_ratio": round(
                        float(np.sum(disp_3d**2) / max(len(disp_3d), 1)), 6
                    ),
                    "displacement_max": round(
                        float(np.max(np.abs(disp_3d))), 6
                    ),
                }
            )

        mode_shapes.sort(key=lambda m: m["frequency_hz"])

        # Find closest mode to target
        closest = min(mode_shapes, key=lambda m: abs(m["frequency_hz"] - target_hz))
        deviation = abs(closest["frequency_hz"] - target_hz) / target_hz * 100

        # Generate visualization mesh (surface triangles)
        vis_mesh = self._generate_surface_mesh(vertices, elements)

        # Estimate max stress (simplified: from displacement gradient)
        E = mat["E_pa"]
        max_disp = max(m["displacement_max"] for m in mode_shapes)
        stress_estimate = E * max_disp / (height_mm * 1e-3) / 1e6  # rough MPa

        return {
            "mode_shapes": mode_shapes,
            "closest_mode_hz": closest["frequency_hz"],
            "target_frequency_hz": target_hz,
            "frequency_deviation_percent": round(deviation, 2),
            "node_count": node_count,
            "element_count": element_count,
            "solve_time_s": round(solve_time, 3),
            "mesh": vis_mesh,
            "stress_max_mpa": round(stress_estimate, 1),
            "temperature_max_c": None,
        }

    # ------------------------------------------------------------------
    # Mesh generation
    # ------------------------------------------------------------------

    def _generate_hex_mesh(
        self,
        w: float,
        h: float,
        l_: float,
        nx: int,
        ny: int,
        nz: int,
        horn_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate structured hexahedral mesh."""
        # Node coordinates
        x = np.linspace(-w / 2, w / 2, nx + 1)
        y = np.linspace(-h / 2, h / 2, ny + 1)
        z = np.linspace(-l_ / 2, l_ / 2, nz + 1)

        # Create 3D grid
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Apply horn-type shape modification
        if horn_type == "exponential":
            # Taper from bottom to top
            for j in range(ny + 1):
                t = (Y[0, j, 0] + h / 2) / h  # 0 at bottom, 1 at top
                exp_denom = 1.0 - math.exp(-3)
                scale = 1.0 - 0.5 * (1.0 - math.exp(-3 * t)) / exp_denom
                X[:, j, :] *= scale
                Z[:, j, :] *= scale
        elif horn_type == "cylindrical":
            # Make roughly circular cross-section
            for i in range(nx + 1):
                for k in range(nz + 1):
                    rx = X[i, 0, k] / (w / 2) if w > 0 else 0.0
                    rz = Z[i, 0, k] / (l_ / 2) if l_ > 0 else 0.0
                    r = math.sqrt(rx**2 + rz**2)
                    if r > 1.0:
                        scale = 1.0 / r
                        X[i, :, k] *= scale
                        Z[i, :, k] *= scale

        vertices = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Element connectivity (8-node hex)
        elements = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    n0 = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
                    n1 = n0 + 1
                    n2 = n0 + (nz + 1)
                    n3 = n2 + 1
                    n4 = n0 + (ny + 1) * (nz + 1)
                    n5 = n4 + 1
                    n6 = n4 + (nz + 1)
                    n7 = n6 + 1
                    elements.append([n0, n1, n3, n2, n4, n5, n7, n6])

        return vertices, np.array(elements, dtype=np.int32)

    # ------------------------------------------------------------------
    # FEA assembly
    # ------------------------------------------------------------------

    def _assemble_global(
        self,
        vertices: np.ndarray,
        elements: np.ndarray,
        mat: dict,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Assemble global stiffness and mass matrices."""
        ndof = len(vertices) * 3
        E = mat["E_pa"]
        nu = mat["nu"]
        rho = mat["rho_kg_m3"]

        # Elasticity matrix (3D isotropic)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        D = np.array(
            [
                [lam + 2 * mu, lam, lam, 0, 0, 0],
                [lam, lam + 2 * mu, lam, 0, 0, 0],
                [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ]
        )

        # Use lumped mass for efficiency
        rows_k: list[int] = []
        cols_k: list[int] = []
        vals_k: list[float] = []
        mass_diag = np.zeros(ndof)

        # Gauss points for 2x2x2 integration
        gp = 1.0 / math.sqrt(3)
        gauss_pts = [
            (-gp, -gp, -gp),
            (gp, -gp, -gp),
            (gp, gp, -gp),
            (-gp, gp, -gp),
            (-gp, -gp, gp),
            (gp, -gp, gp),
            (gp, gp, gp),
            (-gp, gp, gp),
        ]

        for elem in elements:
            coords = vertices[elem]  # (8, 3)
            ke = np.zeros((24, 24))
            me = 0.0  # element mass

            for xi, eta, zeta in gauss_pts:
                dN = self._hex_dshape(xi, eta, zeta)  # (3, 8)
                J = dN @ coords  # (3, 3)
                detJ = float(np.linalg.det(J))
                if detJ <= 0:
                    detJ = abs(detJ) + 1e-12

                dNdx = np.linalg.solve(J, dN)  # (3, 8)

                # Strain-displacement matrix B (6x24)
                B = np.zeros((6, 24))
                for n in range(8):
                    c = 3 * n
                    B[0, c] = dNdx[0, n]
                    B[1, c + 1] = dNdx[1, n]
                    B[2, c + 2] = dNdx[2, n]
                    B[3, c] = dNdx[1, n]
                    B[3, c + 1] = dNdx[0, n]
                    B[4, c + 1] = dNdx[2, n]
                    B[4, c + 2] = dNdx[1, n]
                    B[5, c] = dNdx[2, n]
                    B[5, c + 2] = dNdx[0, n]

                ke += B.T @ D @ B * detJ  # weight=1 for each Gauss point
                me += rho * detJ

            # Scatter to global
            dofs: list[int] = []
            for n in elem:
                dofs.extend([n * 3, n * 3 + 1, n * 3 + 2])

            for ii in range(24):
                for jj in range(24):
                    if abs(ke[ii, jj]) > 1e-20:
                        rows_k.append(dofs[ii])
                        cols_k.append(dofs[jj])
                        vals_k.append(float(ke[ii, jj]))

            # Lumped mass: distribute element mass equally to 8 nodes
            node_mass = me / 8.0
            for n in elem:
                for d in range(3):
                    mass_diag[n * 3 + d] += node_mass

        K = sparse.coo_matrix(
            (vals_k, (rows_k, cols_k)), shape=(ndof, ndof)
        ).tocsr()

        # Symmetrize K
        K = (K + K.T) / 2

        M = sparse.diags(mass_diag)

        return K, M

    @staticmethod
    def _hex_dshape(xi: float, eta: float, zeta: float) -> np.ndarray:
        """Shape function derivatives for 8-node hex element.

        Returns dN/d(xi,eta,zeta) as (3, 8) matrix.
        """
        dN = np.zeros((3, 8))
        signs = [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ]
        for i, (si, sj, sk) in enumerate(signs):
            dN[0, i] = si * (1 + sj * eta) * (1 + sk * zeta) / 8
            dN[1, i] = sj * (1 + si * xi) * (1 + sk * zeta) / 8
            dN[2, i] = sk * (1 + si * xi) * (1 + sj * eta) / 8
        return dN

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    @staticmethod
    def _get_fixed_dofs(vertices: np.ndarray, height_mm: float) -> list[int]:
        """Get DOFs to fix (bottom face nodes)."""
        y_min = float(vertices[:, 1].min())
        tol = height_mm * 0.01  # 1% tolerance
        fixed: list[int] = []
        for i, v in enumerate(vertices):
            if v[1] <= y_min + tol:
                fixed.extend([i * 3, i * 3 + 1, i * 3 + 2])
        return fixed

    @staticmethod
    def _apply_bc(
        K: sparse.csr_matrix,
        M: sparse.csr_matrix,
        fixed_dofs: list[int],
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Apply boundary conditions by zeroing rows/cols and setting diagonal."""
        K_lil = K.tolil()
        M_lil = M.tolil()
        diag_vals = K.diagonal()
        big = float(diag_vals.max()) * 1e6

        for dof in fixed_dofs:
            K_lil[dof, :] = 0
            K_lil[:, dof] = 0
            K_lil[dof, dof] = big
            M_lil[dof, dof] = 1.0  # prevent singularity

        return K_lil.tocsr(), M_lil.tocsr()

    # ------------------------------------------------------------------
    # Visualization mesh
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_surface_mesh(
        vertices: np.ndarray, elements: np.ndarray
    ) -> dict[str, list]:
        """Extract surface faces for Three.js visualization."""
        # Hex face definitions (node indices within each element)
        hex_faces = [
            [0, 1, 5, 4],  # front  (z-)
            [2, 3, 7, 6],  # back   (z+)
            [0, 3, 2, 1],  # bottom (y-)
            [4, 5, 6, 7],  # top    (y+)
            [0, 4, 7, 3],  # left   (x-)
            [1, 2, 6, 5],  # right  (x+)
        ]

        # Count face occurrences to find surface faces
        face_count: dict[tuple[int, ...], int] = {}
        face_nodes: dict[tuple[int, ...], list[int]] = {}
        for elem in elements:
            for hf in hex_faces:
                face_key = tuple(sorted(int(elem[i]) for i in hf))
                face_count[face_key] = face_count.get(face_key, 0) + 1
                if face_key not in face_nodes:
                    face_nodes[face_key] = [int(elem[i]) for i in hf]

        # Surface faces appear only once
        surface_triangles: list[list[int]] = []
        for face_key, count in face_count.items():
            if count == 1:
                nodes = face_nodes[face_key]
                # Split quad into two triangles
                surface_triangles.append([nodes[0], nodes[1], nodes[2]])
                surface_triangles.append([nodes[0], nodes[2], nodes[3]])

        # Round vertices for JSON
        verts = [
            [round(float(v[0]), 3), round(float(v[1]), 3), round(float(v[2]), 3)]
            for v in vertices
        ]

        return {"vertices": verts, "faces": surface_triangles}
