"""Tests for the post-processing module (AmplitudeAnalyzer).

Synthetic tests (1--9) use analytically constructed data and do NOT
require Gmsh.  Test 10 runs a real harmonic analysis through
SolverA and then post-processes the result with AmplitudeAnalyzer.

Synthetic datasets
------------------
Most tests construct a Lorentzian (single-degree-of-freedom) FRF:

    H(f) = 1 / (1 - (f/f0)^2 + j * 2*zeta*(f/f0))

so that all quantities (resonance frequency, Q-factor, phase at
resonance) have known analytical values.
"""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.post_processing import (
    AmplitudeAnalyzer,
    FRFResult,
    GainCurveResult,
    ResonanceInfo,
    UniformityResult,
)


# ===================================================================
# Helpers for building synthetic datasets
# ===================================================================

def _lorentzian_frf(
    frequencies_hz: np.ndarray,
    f0: float,
    zeta: float,
) -> np.ndarray:
    """Single-DOF Lorentzian FRF (complex).

    H(f) = 1 / (1 - r^2 + 2j*zeta*r)   where r = f / f0

    Peak amplitude ~ 1 / (2*zeta) at r = 1.
    """
    r = frequencies_hz / f0
    return 1.0 / (1.0 - r ** 2 + 2j * zeta * r)


def _build_synthetic_analyzer(
    f0: float = 20000.0,
    zeta: float = 0.005,
    freq_min: float = 18000.0,
    freq_max: float = 22000.0,
    n_freq: int = 401,
    n_excit: int = 10,
    n_resp: int = 10,
    resp_amplitude_scale: np.ndarray | None = None,
    excitation_amplitude: float = 0.0,
    longitudinal_axis: str = "y",
) -> AmplitudeAnalyzer:
    """Create an AmplitudeAnalyzer from a synthetic Lorentzian FRF.

    The displacement field is constructed so that:
    - Excitation nodes all have unit amplitude (the denominator of the
      FRF is always 1.0 for the ``excitation_amplitude=0`` / force-
      excitation path).
    - Response nodes have amplitude = |H(f)| * resp_amplitude_scale[i].

    Parameters
    ----------
    resp_amplitude_scale : (n_resp,) or None
        Per-node amplitude multiplier for the response face.  If
        ``None`` every response node gets the same amplitude (uniform).
    """
    frequencies = np.linspace(freq_min, freq_max, n_freq)
    H = _lorentzian_frf(frequencies, f0, zeta)  # (n_freq,) complex

    n_nodes = n_excit + n_resp
    n_dof = n_nodes * 3

    # Build displacement field:  (n_freq, n_dof) complex
    disp = np.zeros((n_freq, n_dof), dtype=np.complex128)

    axis_idx = {"x": 0, "y": 1, "z": 2}[longitudinal_axis]

    # Excitation nodes: indices 0..n_excit-1
    excit_ids = np.arange(n_excit)
    for nid in excit_ids:
        dof = 3 * nid + axis_idx
        disp[:, dof] = 1.0 + 0.0j  # unit amplitude at all frequencies

    # Response nodes: indices n_excit..n_excit+n_resp-1
    resp_ids = np.arange(n_excit, n_excit + n_resp)

    if resp_amplitude_scale is None:
        resp_amplitude_scale = np.ones(n_resp)

    for i, nid in enumerate(resp_ids):
        dof = 3 * nid + axis_idx
        disp[:, dof] = H * resp_amplitude_scale[i]

    # Node coordinates -- place on a flat face (Z=0 plane for Y-axis horns)
    coords = np.zeros((n_nodes, 3), dtype=np.float64)
    # Spread response nodes on a grid for quadrant tests
    if n_resp >= 4:
        side = int(np.ceil(np.sqrt(n_resp)))
        for i in range(n_resp):
            row = i // side
            col = i % side
            nid = resp_ids[i]
            # Use transverse axes (not the longitudinal one)
            if axis_idx == 1:
                coords[nid, 0] = (col - side / 2) * 0.001  # X
                coords[nid, 2] = (row - side / 2) * 0.001  # Z
                coords[nid, 1] = 0.1  # Y (top face)
            elif axis_idx == 0:
                coords[nid, 1] = (col - side / 2) * 0.001
                coords[nid, 2] = (row - side / 2) * 0.001
                coords[nid, 0] = 0.1
            else:
                coords[nid, 0] = (col - side / 2) * 0.001
                coords[nid, 1] = (row - side / 2) * 0.001
                coords[nid, 2] = 0.1

    return AmplitudeAnalyzer(
        frequencies_hz=frequencies,
        displacement_amplitudes=disp,
        node_coords=coords,
        excitation_node_ids=excit_ids,
        response_node_ids=resp_ids,
        excitation_amplitude=excitation_amplitude,
        longitudinal_axis=longitudinal_axis,
    )


# ===================================================================
# 1. test_frf_peak_at_known_frequency
# ===================================================================

class TestFRFPeakAtKnownFrequency:
    """Verify resonance detection on a synthetic FRF."""

    def test_frf_peak_at_known_frequency(self):
        """The detected resonance should be within 0.5% of the true
        resonance frequency embedded in the Lorentzian."""
        f0 = 20000.0
        analyzer = _build_synthetic_analyzer(f0=f0, zeta=0.005, n_freq=801)
        frf = analyzer.compute_frf()

        assert isinstance(frf, FRFResult)
        assert frf.peak_gain > 0.0
        rel_err = abs(frf.resonance_freq_hz - f0) / f0
        assert rel_err < 0.005, (
            f"Resonance at {frf.resonance_freq_hz:.1f} Hz, expected "
            f"{f0:.1f} Hz (error {rel_err*100:.2f}%)"
        )

    def test_find_resonance_matches_frf(self):
        """find_resonance() should agree with compute_frf()."""
        analyzer = _build_synthetic_analyzer(f0=20000.0, zeta=0.01)
        frf = analyzer.compute_frf()
        res = analyzer.find_resonance()

        assert isinstance(res, ResonanceInfo)
        assert res.frequency_hz == frf.resonance_freq_hz
        assert res.peak_gain == frf.peak_gain
        assert res.index == int(np.argmax(frf.amplitude))


# ===================================================================
# 2. test_q_factor_from_synthetic_lorentzian
# ===================================================================

class TestQFactorFromSyntheticLorentzian:
    """Q-factor should match the known analytical value for an SDOF system."""

    @pytest.mark.parametrize("zeta", [0.05, 0.01, 0.005])
    def test_q_factor_accuracy(self, zeta: float):
        """Q_analytical = 1 / (2*zeta).  We allow 5% tolerance."""
        Q_expected = 1.0 / (2.0 * zeta)
        # Wider sweep and more points for narrow peaks
        n_freq = max(801, int(20 * Q_expected))
        bw = max(4000.0, 4 * 20000.0 / Q_expected)
        analyzer = _build_synthetic_analyzer(
            f0=20000.0,
            zeta=zeta,
            freq_min=20000.0 - bw / 2,
            freq_max=20000.0 + bw / 2,
            n_freq=n_freq,
        )
        Q = analyzer.compute_q_factor()
        assert Q > 0.0, "Q-factor should be positive"
        rel_err = abs(Q - Q_expected) / Q_expected
        assert rel_err < 0.05, (
            f"Q={Q:.1f}, expected {Q_expected:.1f} (error {rel_err*100:.2f}%)"
        )


# ===================================================================
# 3. test_uniformity_perfect_uniform
# ===================================================================

class TestUniformityPerfectUniform:
    """When all response nodes have the same amplitude, U=1, U'=1,
    asymmetry=0."""

    def test_uniformity_perfect(self):
        analyzer = _build_synthetic_analyzer(
            n_resp=16, resp_amplitude_scale=np.ones(16)
        )
        result = analyzer.compute_uniformity()

        assert isinstance(result, UniformityResult)
        assert result.uniformity_U == pytest.approx(1.0, abs=1e-12)
        assert result.uniformity_U_prime == pytest.approx(1.0, abs=1e-12)
        assert result.asymmetry_pct == pytest.approx(0.0, abs=1e-6)


# ===================================================================
# 4. test_uniformity_non_uniform
# ===================================================================

class TestUniformityNonUniform:
    """Non-uniform amplitudes should give U < 1."""

    def test_uniformity_non_uniform(self):
        """One node has half the amplitude -> U < 1."""
        scale = np.ones(16)
        scale[0] = 0.5  # one node at half amplitude
        analyzer = _build_synthetic_analyzer(n_resp=16, resp_amplitude_scale=scale)
        result = analyzer.compute_uniformity()

        assert result.uniformity_U < 1.0
        assert result.uniformity_U_prime < 1.0
        # min/mean should reflect the 0.5 node
        expected_U = 0.5 / np.mean(scale)
        assert result.uniformity_U == pytest.approx(expected_U, rel=1e-6)


# ===================================================================
# 5. test_asymmetry_symmetric
# ===================================================================

class TestAsymmetrySymmetric:
    """Symmetric node amplitudes should give near-zero asymmetry."""

    def test_asymmetry_symmetric_nodes(self):
        """All response nodes same amplitude -> asymmetry ~ 0%."""
        analyzer = _build_synthetic_analyzer(
            n_resp=16, resp_amplitude_scale=np.ones(16)
        )
        result = analyzer.compute_uniformity()
        assert result.asymmetry_pct < 1.0  # essentially 0


# ===================================================================
# 6. test_asymmetry_asymmetric
# ===================================================================

class TestAsymmetryAsymmetric:
    """Deliberately asymmetric amplitudes should give asymmetry > 0."""

    def test_asymmetry_one_quadrant_larger(self):
        """Place 16 response nodes on a 4x4 grid.  Make one quadrant
        have 50% more amplitude -> measurable asymmetry."""
        # Build 16 response nodes
        n_resp = 16
        scale = np.ones(n_resp)

        # With the grid layout in _build_synthetic_analyzer, the first
        # few nodes will be in one spatial quadrant.  We make the first
        # quadrant 1.5x larger.
        scale[:4] = 1.5

        analyzer = _build_synthetic_analyzer(
            n_resp=n_resp, resp_amplitude_scale=scale
        )
        result = analyzer.compute_uniformity()
        assert result.asymmetry_pct > 0.0, (
            f"Expected asymmetry > 0, got {result.asymmetry_pct:.4f}%"
        )


# ===================================================================
# 7. test_gain_curve_shape
# ===================================================================

class TestGainCurveShape:
    """Gain curve should have a peak with falloff on both sides."""

    def test_gain_curve_peak_with_falloff(self):
        """The gain at the resonance should be higher than at the
        edges of the sweep."""
        analyzer = _build_synthetic_analyzer(
            f0=20000.0, zeta=0.01, n_freq=401
        )
        gc = analyzer.compute_gain_curve()

        assert isinstance(gc, GainCurveResult)
        assert gc.peak_gain > 0.0
        assert gc.q_factor > 0.0

        # Gain at edges should be significantly smaller than peak
        edge_gain = max(gc.gain_amplitude[0], gc.gain_amplitude[-1])
        assert gc.peak_gain > 2.0 * edge_gain, (
            f"Peak gain {gc.peak_gain:.2f} is not sufficiently larger "
            f"than edge gain {edge_gain:.2f}"
        )

    def test_gain_curve_has_phase(self):
        """Phase array should be populated and same length as frequencies."""
        analyzer = _build_synthetic_analyzer(f0=20000.0, zeta=0.01)
        gc = analyzer.compute_gain_curve()

        assert len(gc.gain_phase_deg) == len(gc.frequencies_hz)
        assert not np.all(gc.gain_phase_deg == 0.0), (
            "Phase should not be all zeros for a resonant system"
        )


# ===================================================================
# 8. test_nodal_amplitude_extraction
# ===================================================================

class TestNodalAmplitudeExtraction:
    """Extract amplitudes at a specific frequency index."""

    def test_nodal_amplitude_at_frequency(self):
        """nodal_amplitude_at_frequency should return the correct shape
        and non-negative values."""
        n_excit, n_resp = 10, 12
        analyzer = _build_synthetic_analyzer(
            n_excit=n_excit, n_resp=n_resp
        )
        n_nodes = n_excit + n_resp

        # Pick the midpoint frequency
        mid_idx = len(analyzer.frequencies_hz) // 2
        amps = analyzer.nodal_amplitude_at_frequency(mid_idx)

        assert amps.shape == (n_nodes,)
        assert np.all(amps >= 0.0)

    def test_excitation_nodes_have_unit_amplitude(self):
        """In the synthetic setup, excitation nodes should have |U_y|=1."""
        n_excit = 8
        analyzer = _build_synthetic_analyzer(n_excit=n_excit, n_resp=10)
        amps = analyzer.nodal_amplitude_at_frequency(0)

        # Excitation nodes (0..n_excit-1) should have amplitude ~1.0
        for i in range(n_excit):
            assert amps[i] == pytest.approx(1.0, abs=1e-10)


# ===================================================================
# 9. test_phase_at_resonance
# ===================================================================

class TestPhaseAtResonance:
    """For an SDOF system, the phase should be ~ -90 degrees at resonance."""

    def test_phase_near_minus_90_at_resonance(self):
        """The Lorentzian H(f) = 1/(1-r^2 + 2j*zeta*r) has
        angle(H) = -90 degrees at r=1 (resonance)."""
        analyzer = _build_synthetic_analyzer(
            f0=20000.0, zeta=0.01, n_freq=801
        )
        frf = analyzer.compute_frf()
        res = analyzer.find_resonance()

        phase_at_res = frf.phase_deg[res.index]
        # Phase should be near -90 degrees (within 5 degrees tolerance
        # since the discrete frequency grid may not land exactly at f0)
        assert abs(phase_at_res - (-90.0)) < 5.0, (
            f"Phase at resonance = {phase_at_res:.1f} deg, "
            f"expected ~ -90 deg"
        )


# ===================================================================
# 10. test_post_processing_from_harmonic_result (real FEA)
# ===================================================================

class TestPostProcessingFromHarmonicResult:
    """Run a real harmonic analysis and post-process with AmplitudeAnalyzer.

    Requires Gmsh.
    """

    @pytest.fixture(scope="class")
    def harmonic_result(self):
        """Module-scoped fixture: run a real harmonic analysis."""
        gmsh = pytest.importorskip("gmsh")  # noqa: F841 - needed for import check

        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
            HarmonicConfig,
        )
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
            get_material,
        )
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import (
            GmshMesher,
        )
        from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import (
            SolverA,
        )

        material = "Titanium Ti-6Al-4V"
        mat = get_material(material)
        c = mat["acoustic_velocity_m_s"]
        length_mm = c / (2.0 * 20000.0) * 1000.0

        mesher = GmshMesher()
        mesh = mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": length_mm},
            mesh_size=8.0,
            order=2,
        )

        config = HarmonicConfig(
            mesh=mesh,
            material_name=material,
            freq_min_hz=19000.0,
            freq_max_hz=21000.0,
            n_freq_points=51,
            damping_model="hysteretic",
            damping_ratio=0.005,
            excitation_node_set="bottom_face",
            response_node_set="top_face",
        )

        solver = SolverA()
        return solver.harmonic_analysis(config), mesh

    def test_real_fea_gain_positive(self, harmonic_result):
        """Gain from the real FEA should be positive."""
        result, mesh = harmonic_result

        excit_ids = np.asarray(mesh.node_sets["bottom_face"], dtype=np.intp)
        resp_ids = np.asarray(mesh.node_sets["top_face"], dtype=np.intp)

        analyzer = AmplitudeAnalyzer(
            frequencies_hz=result.frequencies_hz,
            displacement_amplitudes=result.displacement_amplitudes,
            node_coords=mesh.nodes,
            excitation_node_ids=excit_ids,
            response_node_ids=resp_ids,
            excitation_amplitude=0.0,  # force excitation
            longitudinal_axis="y",
        )

        frf = analyzer.compute_frf()
        assert frf.peak_gain > 0.0, (
            f"Expected positive gain, got {frf.peak_gain}"
        )

    def test_real_fea_uniformity_positive(self, harmonic_result):
        """Uniformity from real FEA should be in [0, 1]."""
        result, mesh = harmonic_result

        excit_ids = np.asarray(mesh.node_sets["bottom_face"], dtype=np.intp)
        resp_ids = np.asarray(mesh.node_sets["top_face"], dtype=np.intp)

        analyzer = AmplitudeAnalyzer(
            frequencies_hz=result.frequencies_hz,
            displacement_amplitudes=result.displacement_amplitudes,
            node_coords=mesh.nodes,
            excitation_node_ids=excit_ids,
            response_node_ids=resp_ids,
            excitation_amplitude=0.0,
            longitudinal_axis="y",
        )

        unif = analyzer.compute_uniformity()
        assert 0.0 <= unif.uniformity_U <= 1.0 + 1e-9
        assert 0.0 <= unif.uniformity_U_prime <= 1.0 + 1e-9

    def test_real_fea_q_factor_positive(self, harmonic_result):
        """Q-factor from real FEA should be non-negative."""
        result, mesh = harmonic_result

        excit_ids = np.asarray(mesh.node_sets["bottom_face"], dtype=np.intp)
        resp_ids = np.asarray(mesh.node_sets["top_face"], dtype=np.intp)

        analyzer = AmplitudeAnalyzer(
            frequencies_hz=result.frequencies_hz,
            displacement_amplitudes=result.displacement_amplitudes,
            node_coords=mesh.nodes,
            excitation_node_ids=excit_ids,
            response_node_ids=resp_ids,
            excitation_amplitude=0.0,
            longitudinal_axis="y",
        )

        q = analyzer.compute_q_factor()
        assert q >= 0.0

    def test_real_fea_gain_curve(self, harmonic_result):
        """Gain curve from real FEA should be populated."""
        result, mesh = harmonic_result

        excit_ids = np.asarray(mesh.node_sets["bottom_face"], dtype=np.intp)
        resp_ids = np.asarray(mesh.node_sets["top_face"], dtype=np.intp)

        analyzer = AmplitudeAnalyzer(
            frequencies_hz=result.frequencies_hz,
            displacement_amplitudes=result.displacement_amplitudes,
            node_coords=mesh.nodes,
            excitation_node_ids=excit_ids,
            response_node_ids=resp_ids,
            excitation_amplitude=0.0,
            longitudinal_axis="y",
        )

        gc = analyzer.compute_gain_curve()
        assert isinstance(gc, GainCurveResult)
        assert gc.peak_gain > 0.0
        assert len(gc.frequencies_hz) == len(result.frequencies_hz)


# ===================================================================
# Extra edge-case tests
# ===================================================================

class TestEdgeCases:
    """Edge-case and integration tests."""

    def test_displacement_excitation_mode(self):
        """When excitation_amplitude > 0, gain = |resp| / prescribed."""
        f0 = 20000.0
        zeta = 0.01
        prescribed = 1e-6
        analyzer = _build_synthetic_analyzer(
            f0=f0,
            zeta=zeta,
            excitation_amplitude=prescribed,
        )
        frf = analyzer.compute_frf()
        # Peak gain should be |H_peak| / prescribed * mean(resp_scale)
        # since resp nodes get H * 1.0 and we divide by prescribed
        # In our synthetic setup, excit nodes have amp=1, resp have |H|
        # So gain = mean(|H|) / prescribed
        # H_peak ~ 1/(2*zeta) = 50
        # gain_peak ~ 50 / 1e-6 = 5e7
        assert frf.peak_gain > 1.0  # definitely large

    def test_q_factor_peak_at_edge_returns_zero(self):
        """If the peak is at the first or last frequency, Q may be 0."""
        # Make peak at the very start of the sweep
        analyzer = _build_synthetic_analyzer(
            f0=18000.0,  # below freq_min
            zeta=0.01,
            freq_min=18000.0,
            freq_max=22000.0,
        )
        q = analyzer.compute_q_factor()
        # Peak will be at or near the left edge, so Q might be 0
        # (no lower crossing). This is a valid edge case.
        assert q >= 0.0

    def test_invalid_axis_raises(self):
        """Invalid longitudinal_axis should raise ValueError."""
        with pytest.raises(ValueError, match="longitudinal_axis"):
            AmplitudeAnalyzer(
                frequencies_hz=np.array([1.0]),
                displacement_amplitudes=np.zeros((1, 3), dtype=np.complex128),
                node_coords=np.zeros((1, 3)),
                excitation_node_ids=np.array([0]),
                response_node_ids=np.array([0]),
                longitudinal_axis="w",
            )

    def test_few_response_nodes_asymmetry_zero(self):
        """Fewer than 4 response nodes should give asymmetry=0."""
        analyzer = _build_synthetic_analyzer(n_resp=3)
        result = analyzer.compute_uniformity()
        assert result.asymmetry_pct == 0.0

    def test_x_axis_longitudinal(self):
        """Analyzer should work with X as the longitudinal axis."""
        analyzer = _build_synthetic_analyzer(
            longitudinal_axis="x", n_resp=10
        )
        frf = analyzer.compute_frf()
        assert frf.peak_gain > 0.0

    def test_z_axis_longitudinal(self):
        """Analyzer should work with Z as the longitudinal axis."""
        analyzer = _build_synthetic_analyzer(
            longitudinal_axis="z", n_resp=10
        )
        frf = analyzer.compute_frf()
        assert frf.peak_gain > 0.0
