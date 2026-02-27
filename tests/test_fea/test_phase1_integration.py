"""Phase 1 integration: mesh a horn, verify element quality, verify material lookup."""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.elements import TET10Element
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import get_material


class TestPhase1Integration:
    def test_mesh_to_element_pipeline(self):
        """Generate mesh, then compute element stiffness for first element."""
        mesher = GmshMesher()
        mesh = mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=6.0,
            order=2,
        )
        assert mesh.element_type == "TET10"

        # Get material
        mat = get_material("Titanium Ti-6Al-4V")
        E, nu = mat["E_pa"], mat["nu"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        D = np.zeros((6, 6))
        D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
        D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = lam
        D[3, 3] = D[4, 4] = D[5, 5] = mu

        # Compute element stiffness for first element
        elem = TET10Element()
        coords = mesh.nodes[mesh.elements[0]]  # (10, 3)
        Ke = elem.stiffness_matrix(coords, D)
        assert Ke.shape == (30, 30)
        assert np.allclose(Ke, Ke.T, atol=1e-4)

        # Compute element mass
        Me = elem.mass_matrix(coords, rho=mat["rho_kg_m3"])
        assert Me.shape == (30, 30)
        assert np.all(np.linalg.eigvalsh(Me) > -1e-10)

    def test_half_wavelength_dimension(self):
        """Mesh dimensions should accommodate a half-wavelength at 20 kHz."""
        mat = get_material("Titanium Ti-6Al-4V")
        c = mat["acoustic_velocity_m_s"]
        f = 20000.0
        half_wave_mm = c / (2 * f) * 1000  # ~126.7 mm

        mesher = GmshMesher()
        mesh = mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": half_wave_mm},
            mesh_size=5.0,
            order=2,
        )

        # Verify mesh length matches
        y_extent = mesh.nodes[:, 1].max() - mesh.nodes[:, 1].min()
        assert abs(y_extent - half_wave_mm / 1000) < 0.001  # tolerance 1mm
