"""Tests for TET10 quadratic tetrahedral element."""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element


class TestTET10ShapeFunctions:
    def setup_method(self):
        self.elem = TET10Element()

    def test_partition_of_unity(self):
        """Sum of shape functions = 1 at any point."""
        for _ in range(10):
            xi, eta, zeta = np.random.dirichlet([1, 1, 1, 1])[:3]
            N = self.elem.shape_functions(xi, eta, zeta)
            assert abs(np.sum(N) - 1.0) < 1e-12

    def test_kronecker_delta_at_nodes(self):
        """N_i(node_j) = delta_ij."""
        nat_coords = self.elem.NODE_NATURAL_COORDS
        for i, (xi, eta, zeta) in enumerate(nat_coords):
            N = self.elem.shape_functions(xi, eta, zeta)
            for j in range(10):
                expected = 1.0 if i == j else 0.0
                assert abs(N[j] - expected) < 1e-12

    def test_shape_derivatives_sum_to_zero(self):
        """Sum of dN/d(xi) = 0."""
        dN = self.elem.shape_derivatives(0.25, 0.25, 0.25)
        assert dN.shape == (3, 10)
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)


class TestTET10StiffnessMatrix:
    def setup_method(self):
        self.elem = TET10Element()
        # Physical coords must correspond to the natural coord ordering:
        # Node 0 (natural 1,0,0) -> physical (1,0,0)
        # Node 1 (natural 0,1,0) -> physical (0,1,0)
        # Node 2 (natural 0,0,1) -> physical (0,0,1)
        # Node 3 (natural 0,0,0) -> physical (0,0,0)
        # Mid-edge nodes at midpoints
        self.coords = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
            [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        ], dtype=float)
        E, nu = 113.8e9, 0.342
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.D = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

    def test_stiffness_symmetric(self):
        Ke = self.elem.stiffness_matrix(self.coords, self.D)
        assert Ke.shape == (30, 30)
        assert np.allclose(Ke, Ke.T, atol=1e-6)

    def test_stiffness_positive_semidefinite(self):
        Ke = self.elem.stiffness_matrix(self.coords, self.D)
        eigvals = np.linalg.eigvalsh(Ke)
        # 6 rigid body modes (zero eigenvalues), rest positive
        n_zero = np.sum(eigvals < 1e-6 * eigvals.max())
        assert n_zero == 6
        assert np.all(eigvals[6:] > 0)


class TestTET10MassMatrix:
    def setup_method(self):
        self.elem = TET10Element()
        self.coords = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
            [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        ], dtype=float)

    def test_mass_matrix_symmetric(self):
        Me = self.elem.mass_matrix(self.coords, rho=4430.0)
        assert Me.shape == (30, 30)
        assert np.allclose(Me, Me.T, atol=1e-10)

    def test_mass_matrix_positive_definite(self):
        Me = self.elem.mass_matrix(self.coords, rho=4430.0)
        eigvals = np.linalg.eigvalsh(Me)
        assert np.all(eigvals > 0)

    def test_total_mass_correct(self):
        """Rigid body translation should recover total mass = rho * V.

        For a consistent mass matrix, a unit translation in any direction
        gives u^T * M * u = rho * V (the total element mass).
        """
        Me = self.elem.mass_matrix(self.coords, rho=4430.0)
        expected_mass = 4430.0 * (1.0 / 6.0)  # Vol of unit tet = 1/6
        # Unit translation in x-direction
        u_x = np.zeros(30)
        for i in range(10):
            u_x[3 * i] = 1.0
        total_mass = u_x @ Me @ u_x
        assert abs(total_mass - expected_mass) / expected_mass < 1e-10


class TestTET10PatchTest:
    """Patch test: constant strain field must be exactly reproduced."""

    def test_uniform_tension(self):
        elem = TET10Element()
        coords = np.array([
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0],
            [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        ], dtype=float)

        E_val, nu = 113.8e9, 0.342
        lam = E_val * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E_val / (2 * (1 + nu))
        D = np.array([
            [lam + 2*mu, lam, lam, 0, 0, 0],
            [lam, lam + 2*mu, lam, 0, 0, 0],
            [lam, lam, lam + 2*mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ])

        # Uniform strain: eps_x = 0.001, rest from Poisson
        eps_x = 0.001
        u_e = np.zeros(30)
        for i in range(10):
            u_e[3*i] = eps_x * coords[i, 0]
            u_e[3*i+1] = -nu * eps_x * coords[i, 1]
            u_e[3*i+2] = -nu * eps_x * coords[i, 2]

        stress = elem.stress_at_point(coords, u_e, D, 0.25, 0.25, 0.25)
        expected = D @ np.array([eps_x, -nu*eps_x, -nu*eps_x, 0, 0, 0])
        assert np.allclose(stress, expected, rtol=1e-10)
