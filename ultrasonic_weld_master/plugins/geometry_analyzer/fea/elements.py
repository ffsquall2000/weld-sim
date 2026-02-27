"""TET10 quadratic tetrahedral element formulation.

Implements the 10-node quadratic tetrahedral element with:
- Quadratic shape functions in barycentric (natural) coordinates
- 4-point Gauss quadrature for tetrahedra
- Strain-displacement (B) matrix
- Element stiffness matrix (30x30)
- Consistent element mass matrix (30x30)
- Stress computation at arbitrary points

Node numbering follows the standard Bathe convention:
- Nodes 0-3: corner nodes
- Nodes 4-9: mid-edge nodes

Natural (barycentric) coordinates: L1=xi, L2=eta, L3=zeta, L4=1-xi-eta-zeta

Node layout in natural coordinates::

    Node 0: (1, 0, 0)       corner
    Node 1: (0, 1, 0)       corner
    Node 2: (0, 0, 1)       corner
    Node 3: (0, 0, 0)       corner
    Node 4: (0.5, 0.5, 0)   mid-edge 0-1
    Node 5: (0, 0.5, 0.5)   mid-edge 1-2
    Node 6: (0.5, 0, 0.5)   mid-edge 0-2
    Node 7: (0.5, 0, 0)     mid-edge 0-3
    Node 8: (0, 0.5, 0)     mid-edge 1-3
    Node 9: (0, 0, 0.5)     mid-edge 2-3
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class TET10Element:
    """10-node quadratic tetrahedral finite element (TET10).

    Each node has 3 translational DOFs (u_x, u_y, u_z), giving 30 DOFs
    per element. The element uses 4-point Gauss quadrature which is exact
    for degree-2 polynomials over the tetrahedral domain.

    Attributes
    ----------
    NODE_NATURAL_COORDS : NDArray[np.float64]
        (10, 3) array of natural coordinates (xi, eta, zeta) for each node.
    GAUSS_POINTS : NDArray[np.float64]
        (4, 4) array where each row is [xi, eta, zeta, weight].
    """

    # -- Natural coordinates for the 10 nodes ---------------------------------
    NODE_NATURAL_COORDS: NDArray[np.float64] = np.array([
        [1.0, 0.0, 0.0],       # Node 0: corner
        [0.0, 1.0, 0.0],       # Node 1: corner
        [0.0, 0.0, 1.0],       # Node 2: corner
        [0.0, 0.0, 0.0],       # Node 3: corner
        [0.5, 0.5, 0.0],       # Node 4: mid-edge 0-1
        [0.0, 0.5, 0.5],       # Node 5: mid-edge 1-2
        [0.5, 0.0, 0.5],       # Node 6: mid-edge 0-2
        [0.5, 0.0, 0.0],       # Node 7: mid-edge 0-3
        [0.0, 0.5, 0.0],       # Node 8: mid-edge 1-3
        [0.0, 0.0, 0.5],       # Node 9: mid-edge 2-3
    ], dtype=np.float64)

    # -- 4-point Gauss quadrature for tetrahedra (degree 2 exact) --------------
    # Barycentric coordinates: (a, b, b, b) and permutations
    # a = (5 + 3*sqrt(5)) / 20 = 0.5854101966249685
    # b = (5 - sqrt(5)) / 20   = 0.1381966011250105
    # Each weight = 1/24 (sum = 4/24 = 1/6 = volume of unit tet)
    _A: float = 0.5854101966249685
    _B: float = 0.1381966011250105
    _W: float = 1.0 / 24.0

    # Columns: xi (L1), eta (L2), zeta (L3), weight
    # The 4th barycentric coordinate L4 = 1 - xi - eta - zeta is implicit.
    GAUSS_POINTS: NDArray[np.float64] = np.array([
        [_A, _B, _B, _W],  # L1=a, L2=b, L3=b, L4=b
        [_B, _A, _B, _W],  # L1=b, L2=a, L3=b, L4=b
        [_B, _B, _A, _W],  # L1=b, L2=b, L3=a, L4=b
        [_B, _B, _B, _W],  # L1=b, L2=b, L3=b, L4=a
    ], dtype=np.float64)

    # -- Higher-order Gauss quadrature for mass matrix (degree 4 exact) --------
    # The consistent mass matrix requires integrating N_i*N_j (degree 4).
    # The 4-point rule is only degree 2, so we use an 11-point rule
    # (Keast, 1986) that is exact for degree 4 polynomials on tetrahedra.
    # Reference: P. Keast, "Moderate degree tetrahedral quadrature formulas",
    #            CMAME, 55:339-348, 1986.
    #
    # Points with barycentric coords (L1, L2, L3, L4) and weights.
    # All weights sum to 1/6 (volume of unit tet).
    GAUSS_POINTS_HIGH: NDArray[np.float64] = np.array([
        # 1 point with multiplicity 1: centroid (1/4, 1/4, 1/4, 1/4)
        [0.25, 0.25, 0.25, -0.01315555555555556],
        # 4 points with multiplicity 4: (a1, b1, b1, b1) and permutations
        # a1 = 0.07142857142857143, b1 = 0.30952380952380953
        [0.07142857142857143, 0.30952380952380953, 0.30952380952380953,
         0.007622222222222222],
        [0.30952380952380953, 0.07142857142857143, 0.30952380952380953,
         0.007622222222222222],
        [0.30952380952380953, 0.30952380952380953, 0.07142857142857143,
         0.007622222222222222],
        [0.30952380952380953, 0.30952380952380953, 0.30952380952380953,
         0.007622222222222222],
        # 6 points with multiplicity 6: (a2, a2, b2, b2) and permutations
        # a2 = 0.05635083268962915, b2 = 0.44364916731037085
        [0.05635083268962915, 0.05635083268962915, 0.44364916731037085,
         0.02488888888888889],
        [0.05635083268962915, 0.44364916731037085, 0.05635083268962915,
         0.02488888888888889],
        [0.05635083268962915, 0.44364916731037085, 0.44364916731037085,
         0.02488888888888889],
        [0.44364916731037085, 0.05635083268962915, 0.05635083268962915,
         0.02488888888888889],
        [0.44364916731037085, 0.05635083268962915, 0.44364916731037085,
         0.02488888888888889],
        [0.44364916731037085, 0.44364916731037085, 0.05635083268962915,
         0.02488888888888889],
    ], dtype=np.float64)

    # Number of nodes and DOFs per element
    N_NODES: int = 10
    N_DOF: int = 30  # 10 nodes * 3 DOFs/node

    # -------------------------------------------------------------------------
    # Shape functions
    # -------------------------------------------------------------------------
    @staticmethod
    def shape_functions(xi: float, eta: float, zeta: float) -> NDArray[np.float64]:
        """Evaluate the 10 quadratic shape functions at a natural coordinate point.

        Parameters
        ----------
        xi : float
            First barycentric coordinate (L1). Range [0, 1].
        eta : float
            Second barycentric coordinate (L2). Range [0, 1].
        zeta : float
            Third barycentric coordinate (L3). Range [0, 1].

        Returns
        -------
        NDArray[np.float64]
            Shape function values, shape (10,).
        """
        L4 = 1.0 - xi - eta - zeta

        N = np.empty(10, dtype=np.float64)
        # Corner nodes: N_i = L_i * (2*L_i - 1)
        N[0] = xi * (2.0 * xi - 1.0)
        N[1] = eta * (2.0 * eta - 1.0)
        N[2] = zeta * (2.0 * zeta - 1.0)
        N[3] = L4 * (2.0 * L4 - 1.0)
        # Mid-edge nodes: N_ij = 4 * L_i * L_j
        N[4] = 4.0 * xi * eta        # mid-edge 0-1
        N[5] = 4.0 * eta * zeta      # mid-edge 1-2
        N[6] = 4.0 * xi * zeta       # mid-edge 0-2
        N[7] = 4.0 * xi * L4         # mid-edge 0-3
        N[8] = 4.0 * eta * L4        # mid-edge 1-3
        N[9] = 4.0 * zeta * L4       # mid-edge 2-3
        return N

    # -------------------------------------------------------------------------
    # Shape function derivatives
    # -------------------------------------------------------------------------
    @staticmethod
    def shape_derivatives(
        xi: float, eta: float, zeta: float
    ) -> NDArray[np.float64]:
        """Evaluate derivatives of shape functions w.r.t. natural coordinates.

        Parameters
        ----------
        xi, eta, zeta : float
            Natural (barycentric) coordinates.

        Returns
        -------
        NDArray[np.float64]
            Shape (3, 10) array. Row 0 = dN/d(xi), row 1 = dN/d(eta),
            row 2 = dN/d(zeta).
        """
        L4 = 1.0 - xi - eta - zeta

        dN = np.empty((3, 10), dtype=np.float64)

        # Derivatives of L4 w.r.t. xi, eta, zeta are all -1.

        # dN/d(xi)
        dN[0, 0] = 4.0 * xi - 1.0                    # d/dxi [xi*(2xi-1)]
        dN[0, 1] = 0.0                                 # d/dxi [eta*(2eta-1)]
        dN[0, 2] = 0.0                                 # d/dxi [zeta*(2zeta-1)]
        dN[0, 3] = -4.0 * L4 + 1.0                    # d/dxi [L4*(2L4-1)] = (2L4-1)*(-1) + L4*2*(-1)
        dN[0, 4] = 4.0 * eta                           # d/dxi [4*xi*eta]
        dN[0, 5] = 0.0                                 # d/dxi [4*eta*zeta]
        dN[0, 6] = 4.0 * zeta                          # d/dxi [4*xi*zeta]
        dN[0, 7] = 4.0 * (L4 - xi)                    # d/dxi [4*xi*L4] = 4*(L4 + xi*(-1))
        dN[0, 8] = -4.0 * eta                          # d/dxi [4*eta*L4] = 4*eta*(-1)
        dN[0, 9] = -4.0 * zeta                         # d/dxi [4*zeta*L4] = 4*zeta*(-1)

        # dN/d(eta)
        dN[1, 0] = 0.0                                 # d/deta [xi*(2xi-1)]
        dN[1, 1] = 4.0 * eta - 1.0                    # d/deta [eta*(2eta-1)]
        dN[1, 2] = 0.0                                 # d/deta [zeta*(2zeta-1)]
        dN[1, 3] = -4.0 * L4 + 1.0                    # d/deta [L4*(2L4-1)]
        dN[1, 4] = 4.0 * xi                            # d/deta [4*xi*eta]
        dN[1, 5] = 4.0 * zeta                          # d/deta [4*eta*zeta]
        dN[1, 6] = 0.0                                 # d/deta [4*xi*zeta]
        dN[1, 7] = -4.0 * xi                           # d/deta [4*xi*L4] = 4*xi*(-1)
        dN[1, 8] = 4.0 * (L4 - eta)                   # d/deta [4*eta*L4] = 4*(L4 + eta*(-1))
        dN[1, 9] = -4.0 * zeta                         # d/deta [4*zeta*L4] = 4*zeta*(-1)

        # dN/d(zeta)
        dN[2, 0] = 0.0                                 # d/dzeta [xi*(2xi-1)]
        dN[2, 1] = 0.0                                 # d/dzeta [eta*(2eta-1)]
        dN[2, 2] = 4.0 * zeta - 1.0                   # d/dzeta [zeta*(2zeta-1)]
        dN[2, 3] = -4.0 * L4 + 1.0                    # d/dzeta [L4*(2L4-1)]
        dN[2, 4] = 0.0                                 # d/dzeta [4*xi*eta]
        dN[2, 5] = 4.0 * eta                           # d/dzeta [4*eta*zeta]
        dN[2, 6] = 4.0 * xi                            # d/dzeta [4*xi*zeta]
        dN[2, 7] = -4.0 * xi                           # d/dzeta [4*xi*L4] = 4*xi*(-1)
        dN[2, 8] = -4.0 * eta                          # d/dzeta [4*eta*L4] = 4*eta*(-1)
        dN[2, 9] = 4.0 * (L4 - zeta)                  # d/dzeta [4*zeta*L4] = 4*(L4 + zeta*(-1))

        return dN

    # -------------------------------------------------------------------------
    # Jacobian and B-matrix
    # -------------------------------------------------------------------------
    def strain_displacement(
        self,
        coords: NDArray[np.float64],
        xi: float,
        eta: float,
        zeta: float,
    ) -> tuple[NDArray[np.float64], float]:
        """Compute the strain-displacement matrix B and Jacobian determinant.

        Parameters
        ----------
        coords : NDArray[np.float64]
            (10, 3) array of nodal physical coordinates.
        xi, eta, zeta : float
            Natural coordinates of the evaluation point.

        Returns
        -------
        B : NDArray[np.float64]
            (6, 30) strain-displacement matrix in Voigt notation.
            Rows correspond to [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz].
        det_J : float
            Determinant of the Jacobian matrix (must be positive for
            a valid element).
        """
        # Shape function derivatives in natural coordinates: (3, 10)
        dN_nat = self.shape_derivatives(xi, eta, zeta)

        # Jacobian: J = dN_nat @ coords  ->  (3, 10) @ (10, 3) = (3, 3)
        # J[i, j] = sum_k dN_nat[i, k] * coords[k, j]
        # Maps from natural to physical: dx/d(xi), dx/d(eta), etc.
        J = dN_nat @ coords  # (3, 3)

        det_J = np.linalg.det(J)
        if det_J <= 0.0:
            raise ValueError(
                f"Non-positive Jacobian determinant ({det_J:.6e}). "
                "Check element node ordering and coordinates."
            )

        # Inverse Jacobian
        J_inv = np.linalg.inv(J)  # (3, 3)

        # Shape function derivatives in physical coordinates: (3, 10)
        # dN_phys[i, k] = sum_j J_inv[i, j] * dN_nat[j, k]
        dN_phys = J_inv @ dN_nat  # (3, 10)

        # Assemble B matrix (6 x 30)
        # Voigt notation: [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]
        B = np.zeros((6, 30), dtype=np.float64)

        for i in range(10):
            col = 3 * i
            dNx = dN_phys[0, i]  # dN_i/dx
            dNy = dN_phys[1, i]  # dN_i/dy
            dNz = dN_phys[2, i]  # dN_i/dz

            # eps_xx = du/dx
            B[0, col] = dNx
            # eps_yy = dv/dy
            B[1, col + 1] = dNy
            # eps_zz = dw/dz
            B[2, col + 2] = dNz
            # gamma_xy = du/dy + dv/dx
            B[3, col] = dNy
            B[3, col + 1] = dNx
            # gamma_yz = dv/dz + dw/dy
            B[4, col + 1] = dNz
            B[4, col + 2] = dNy
            # gamma_xz = du/dz + dw/dx
            B[5, col] = dNz
            B[5, col + 2] = dNx

        return B, det_J

    # -------------------------------------------------------------------------
    # Element stiffness matrix
    # -------------------------------------------------------------------------
    def stiffness_matrix(
        self,
        coords: NDArray[np.float64],
        D: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the 30x30 element stiffness matrix via Gauss quadrature.

        K_e = sum_{gp} B^T * D * B * det(J) * w

        Parameters
        ----------
        coords : NDArray[np.float64]
            (10, 3) nodal physical coordinates.
        D : NDArray[np.float64]
            (6, 6) constitutive (elasticity) matrix in Voigt notation.

        Returns
        -------
        NDArray[np.float64]
            (30, 30) symmetric positive semi-definite stiffness matrix.
        """
        Ke = np.zeros((30, 30), dtype=np.float64)

        for gp in self.GAUSS_POINTS:
            xi_g, eta_g, zeta_g, w_g = gp

            B, det_J = self.strain_displacement(coords, xi_g, eta_g, zeta_g)

            # K_e += B^T * D * B * det(J) * w
            # Efficient: compute D @ B first (6x30), then B^T @ (D@B) (30x30)
            DB = D @ B  # (6, 30)
            Ke += (B.T @ DB) * det_J * w_g

        return Ke

    # -------------------------------------------------------------------------
    # Element mass matrix (consistent)
    # -------------------------------------------------------------------------
    def mass_matrix(
        self,
        coords: NDArray[np.float64],
        rho: float,
    ) -> NDArray[np.float64]:
        """Compute the 30x30 consistent element mass matrix.

        M_e = sum_{gp} rho * N_exp^T * N_exp * det(J) * w

        where N_exp is the (3, 30) expanded shape function matrix:
        [[N0, 0, 0, N1, 0, 0, ...],
         [0, N0, 0, 0, N1, 0, ...],
         [0, 0, N0, 0, 0, N1, ...]]

        Parameters
        ----------
        coords : NDArray[np.float64]
            (10, 3) nodal physical coordinates.
        rho : float
            Material density [kg/m^3].

        Returns
        -------
        NDArray[np.float64]
            (30, 30) symmetric positive definite consistent mass matrix.
        """
        Me = np.zeros((30, 30), dtype=np.float64)

        # Use higher-order quadrature for mass matrix because the integrand
        # N_i * N_j is degree 4 (product of two quadratic functions).
        # The 4-point rule (degree 2) is insufficient; the 11-point rule
        # (degree 4) provides exact integration.
        for gp in self.GAUSS_POINTS_HIGH:
            xi_g, eta_g, zeta_g, w_g = gp

            N = self.shape_functions(xi_g, eta_g, zeta_g)  # (10,)

            # Compute Jacobian determinant (we need it for the volume mapping)
            dN_nat = self.shape_derivatives(xi_g, eta_g, zeta_g)
            J = dN_nat @ coords
            det_J = np.linalg.det(J)

            if det_J <= 0.0:
                raise ValueError(
                    f"Non-positive Jacobian determinant ({det_J:.6e}) "
                    "in mass matrix computation."
                )

            # Build expanded shape function matrix N_exp (3, 30)
            # Instead of building the full matrix, we use the outer product
            # directly: N_exp^T @ N_exp has a block structure.
            # M_e[3i+a, 3j+a] += rho * N[i] * N[j] * det_J * w  for a=0,1,2
            factor = rho * det_J * w_g

            # Vectorized assembly: compute outer product N_i * N_j
            NN = np.outer(N, N) * factor  # (10, 10)
            for i in range(10):
                for j in range(10):
                    val = NN[i, j]
                    for a in range(3):
                        Me[3 * i + a, 3 * j + a] += val

        return Me

    # -------------------------------------------------------------------------
    # Stress computation
    # -------------------------------------------------------------------------
    def stress_at_point(
        self,
        coords: NDArray[np.float64],
        u_e: NDArray[np.float64],
        D: NDArray[np.float64],
        xi: float,
        eta: float,
        zeta: float,
    ) -> NDArray[np.float64]:
        """Compute the stress tensor (Voigt notation) at a natural coordinate point.

        sigma = D * B * u_e

        Parameters
        ----------
        coords : NDArray[np.float64]
            (10, 3) nodal physical coordinates.
        u_e : NDArray[np.float64]
            (30,) element displacement vector.
        D : NDArray[np.float64]
            (6, 6) constitutive matrix.
        xi, eta, zeta : float
            Natural coordinates of the evaluation point.

        Returns
        -------
        NDArray[np.float64]
            (6,) stress vector [sigma_xx, sigma_yy, sigma_zz,
            tau_xy, tau_yz, tau_xz].
        """
        B, _det_J = self.strain_displacement(coords, xi, eta, zeta)
        strain = B @ u_e  # (6,)
        stress = D @ strain  # (6,)
        return stress

    # -------------------------------------------------------------------------
    # Utility: compute element volume
    # -------------------------------------------------------------------------
    def volume(self, coords: NDArray[np.float64]) -> float:
        """Compute the element volume via Gauss quadrature.

        Parameters
        ----------
        coords : NDArray[np.float64]
            (10, 3) nodal physical coordinates.

        Returns
        -------
        float
            Element volume.
        """
        vol = 0.0
        for gp in self.GAUSS_POINTS:
            xi_g, eta_g, zeta_g, w_g = gp
            dN_nat = self.shape_derivatives(xi_g, eta_g, zeta_g)
            J = dN_nat @ coords
            det_J = np.linalg.det(J)
            vol += det_J * w_g
        return vol

    # -------------------------------------------------------------------------
    # Utility: build isotropic elasticity matrix
    # -------------------------------------------------------------------------
    @staticmethod
    def isotropic_elasticity_matrix(
        E: float, nu: float
    ) -> NDArray[np.float64]:
        """Build the 6x6 isotropic linear-elastic constitutive matrix.

        Uses Voigt notation ordering:
        [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_xz].

        Parameters
        ----------
        E : float
            Young's modulus [Pa].
        nu : float
            Poisson's ratio [-].

        Returns
        -------
        NDArray[np.float64]
            (6, 6) symmetric positive definite elasticity matrix.
        """
        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        D = np.array([
            [lam + 2 * mu, lam,          lam,          0.0, 0.0, 0.0],
            [lam,          lam + 2 * mu, lam,          0.0, 0.0, 0.0],
            [lam,          lam,          lam + 2 * mu, 0.0, 0.0, 0.0],
            [0.0,          0.0,          0.0,          mu,  0.0, 0.0],
            [0.0,          0.0,          0.0,          0.0, mu,  0.0],
            [0.0,          0.0,          0.0,          0.0, 0.0, mu ],
        ], dtype=np.float64)

        return D

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TET10Element(nodes={self.N_NODES}, dofs={self.N_DOF}, "
            f"gauss_points={len(self.GAUSS_POINTS)})"
        )
