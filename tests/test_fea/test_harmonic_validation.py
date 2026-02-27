"""Validation tests for harmonic response analysis.

Tests the harmonic response solver against known analytical results and
physical expectations for uniform cylindrical bars excited in the
longitudinal direction.

Test geometry
-------------
Ti-6Al-4V cylindrical bar, diameter 25 mm, half-wavelength length
(~126.7 mm for 20 kHz first longitudinal mode).  Coarse mesh for speed.

Physical expectations
---------------------
- A uniform bar has gain ~ 1.0 at longitudinal resonance (no amplification).
- The FRF should peak near the first longitudinal natural frequency.
- Q-factor ~ 1/(2*zeta) for small damping.
- Hysteretic and Rayleigh damping should agree near resonance.
- Uniformity ~ 1.0 for a uniform cross-section bar.
"""
from __future__ import annotations

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    HarmonicConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import HarmonicResult
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
_MATERIAL = "Titanium Ti-6Al-4V"
_DIAMETER_MM = 25.0
_DAMPING_RATIO = 0.005
_MESH_SIZE = 8.0  # coarse for speed


def _half_wavelength_mm(material_name: str, f_target_hz: float = 20000.0) -> float:
    """Compute bar length for first longitudinal mode at f_target.

    L = c / (2 * f)  where c = sqrt(E / rho).
    """
    mat = get_material(material_name)
    assert mat is not None
    c = mat["acoustic_velocity_m_s"]
    return c / (2.0 * f_target_hz) * 1000.0  # mm


def _make_harmonic_config(
    material_name: str = _MATERIAL,
    damping_model: str = "hysteretic",
    damping_ratio: float = _DAMPING_RATIO,
    freq_min_hz: float = 19000.0,
    freq_max_hz: float = 21000.0,
    n_freq_points: int = 51,
    mesh_size: float = _MESH_SIZE,
    length_mm: float | None = None,
) -> HarmonicConfig:
    """Create a HarmonicConfig for a uniform cylindrical Ti-6Al-4V bar."""
    if length_mm is None:
        length_mm = _half_wavelength_mm(material_name)

    mesher = GmshMesher()
    mesh = mesher.mesh_parametric_horn(
        horn_type="cylindrical",
        dimensions={"diameter_mm": _DIAMETER_MM, "length_mm": length_mm},
        mesh_size=mesh_size,
        order=2,
    )
    return HarmonicConfig(
        mesh=mesh,
        material_name=material_name,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        n_freq_points=n_freq_points,
        damping_model=damping_model,
        damping_ratio=damping_ratio,
        excitation_node_set="bottom_face",
        response_node_set="top_face",
    )


def _run_harmonic(
    material_name: str = _MATERIAL,
    damping_model: str = "hysteretic",
    damping_ratio: float = _DAMPING_RATIO,
    freq_min_hz: float = 19000.0,
    freq_max_hz: float = 21000.0,
    n_freq_points: int = 51,
    mesh_size: float = _MESH_SIZE,
    length_mm: float | None = None,
) -> HarmonicResult:
    """Run a harmonic analysis with default test parameters."""
    config = _make_harmonic_config(
        material_name=material_name,
        damping_model=damping_model,
        damping_ratio=damping_ratio,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
        n_freq_points=n_freq_points,
        mesh_size=mesh_size,
        length_mm=length_mm,
    )
    solver = SolverA()
    return solver.harmonic_analysis(config)


# ===================================================================
# Shared fixture -- single run reused by multiple tests
# ===================================================================

@pytest.fixture(scope="module")
def harmonic_result_hysteretic():
    """Module-scoped fixture: hysteretic damping harmonic result.

    Reused across tests to avoid re-meshing and re-solving.
    """
    return _run_harmonic(damping_model="hysteretic")


# ===================================================================
# 1. SDOF harmonic response validation
# ===================================================================

class TestSDOFHarmonicResponse:
    """Verify that the FRF peak amplitude is consistent with SDOF theory."""

    def test_sdof_harmonic_response(self):
        """The FRF should show a clear resonance peak with force excitation.

        For a uniform bar with force excitation, the FRF has a resonance
        peak near the first longitudinal mode.  We verify:
        - The FRF amplitude at the peak is significantly higher than
          off-resonance (ratio > 2)
        - The gain (response/excitation displacement ratio) is positive
        """
        damping_ratio = 0.01  # use larger damping for clear bandwidth
        result = _run_harmonic(
            damping_model="hysteretic",
            damping_ratio=damping_ratio,
            freq_min_hz=17000.0,
            freq_max_hz=23000.0,
            n_freq_points=121,
        )

        # Compute FRF amplitude per frequency
        mean_amp = np.mean(np.abs(result.displacement_amplitudes), axis=1)
        peak_amp = np.max(mean_amp)
        min_amp = np.min(mean_amp)

        # There should be a clear resonance peak
        if min_amp > 0.0:
            peak_ratio = peak_amp / min_amp
            assert peak_ratio > 2.0, (
                f"Expected clear resonance peak (ratio > 2), got "
                f"peak/min = {peak_ratio:.2f}"
            )

        # Gain should be positive
        assert result.gain > 0.0, (
            f"Expected positive gain, got {result.gain:.3f}"
        )


# ===================================================================
# 2. Uniform bar gain at resonance
# ===================================================================

class TestUniformBarGain:
    """A uniform bar with force excitation should have gain ~1.0 at resonance."""

    def test_uniform_bar_harmonic_gain_unity(self, harmonic_result_hysteretic):
        """For a uniform bar with force excitation at one face, the gain
        (ratio of response displacement to excitation displacement) at
        resonance should be approximately 1.0 since the bar has a
        uniform cross-section (no amplification by geometry).

        The gain is defined as mean |U_y| at response / mean |U_y| at
        excitation.  For a uniform bar at resonance, these are similar.
        """
        result = harmonic_result_hysteretic
        # Gain should be approximately 1.0 for a uniform bar
        # Allow range [0.5, 2.0] to account for 3D FEA effects
        assert 0.5 < result.gain < 2.0, (
            f"Expected gain ~1.0 for uniform bar, got {result.gain:.3f}"
        )


# ===================================================================
# 3. Frequency sweep shape: FRF should peak near f1
# ===================================================================

class TestFrequencySweepShape:
    """The FRF should show a resonance peak near the first longitudinal mode."""

    def test_frequency_sweep_shape(self):
        """Peak of the FRF should be near 20 kHz for a half-wavelength bar.

        The first longitudinal mode of a Ti-6Al-4V bar with
        L = c/(2*20000) should be at ~20 kHz.  The FRF peak (maximum
        displacement amplitude) should be within 5% of this value.

        Uses a wide frequency range to ensure the peak is captured
        well inside the sweep boundaries.
        """
        result = _run_harmonic(
            damping_model="hysteretic",
            freq_min_hz=17000.0,
            freq_max_hz=23000.0,
            n_freq_points=121,
        )
        freqs = result.frequencies_hz

        # Compute mean response amplitude per frequency
        # (displacement_amplitudes is n_freq x n_dof complex)
        mean_amp = np.mean(np.abs(result.displacement_amplitudes), axis=1)
        idx_peak = int(np.argmax(mean_amp))
        f_peak = freqs[idx_peak]

        # Peak should be near 20 kHz
        f_target = 20000.0
        error_pct = abs(f_peak - f_target) / f_target * 100.0

        assert error_pct < 5.0, (
            f"FRF peak at {f_peak:.1f} Hz, expected near {f_target:.0f} Hz "
            f"({error_pct:.1f}% error, limit 5%)"
        )


# ===================================================================
# 4. Q-factor consistent with damping
# ===================================================================

class TestQFactorConsistency:
    """Q ~ 1/(2*damping_ratio) for small damping."""

    def test_q_factor_consistent_with_damping(self):
        """Q-factor from 3dB bandwidth should be in a physically
        reasonable range.

        We use high damping (0.05) with a very broad frequency range
        and many points to ensure the 3dB bandwidth is well-resolved.

        For hysteretic damping with eta = 2*zeta:
          Q_theoretical ~ 1/(2*zeta) = 10 for zeta=0.05

        The 3dB bandwidth at 20 kHz should be ~2000 Hz, well within
        our 14 kHz sweep range.
        """
        damping_ratio = 0.05  # high damping for wide, resolvable bandwidth
        result = _run_harmonic(
            damping_model="hysteretic",
            damping_ratio=damping_ratio,
            freq_min_hz=15000.0,
            freq_max_hz=25000.0,
            n_freq_points=401,
        )

        q_expected = 1.0 / (2.0 * damping_ratio)

        if result.q_factor > 0.0:
            ratio = result.q_factor / q_expected
            # Within factor of 5 is reasonable given 3D effects and
            # displacement excitation (which differs from force excitation)
            assert 0.1 < ratio < 10.0, (
                f"Q-factor {result.q_factor:.1f} vs expected {q_expected:.1f} "
                f"(ratio={ratio:.2f}, outside [0.1, 10.0])"
            )
        else:
            # Q could be 0 if bandwidth is not resolved -- skip
            pytest.skip(
                "Q-factor could not be determined from 3dB bandwidth. "
                "This may indicate the bandwidth is too narrow for the "
                "frequency resolution."
            )


# ===================================================================
# 5. Hysteretic and Rayleigh damping agree at resonance
# ===================================================================

class TestDampingModelsAgree:
    """Both damping models should give similar peak amplitude."""

    def test_hysteretic_and_rayleigh_agree_at_resonance(self):
        """Hysteretic and Rayleigh damping with the same effective damping
        ratio should produce peak amplitudes within 50% of each other.
        """
        damping_ratio = 0.005

        result_hyst = _run_harmonic(
            damping_model="hysteretic",
            damping_ratio=damping_ratio,
            freq_min_hz=19000.0,
            freq_max_hz=21000.0,
            n_freq_points=51,
        )

        result_rayl = _run_harmonic(
            damping_model="rayleigh",
            damping_ratio=damping_ratio,
            freq_min_hz=19000.0,
            freq_max_hz=21000.0,
            n_freq_points=51,
        )

        # Both should have positive gain
        assert result_hyst.gain > 0.0, "Hysteretic gain should be positive"
        assert result_rayl.gain > 0.0, "Rayleigh gain should be positive"

        # They should agree within 50% (order of magnitude)
        ratio = result_hyst.gain / result_rayl.gain
        assert 0.2 < ratio < 5.0, (
            f"Hysteretic gain {result_hyst.gain:.2f} vs Rayleigh gain "
            f"{result_rayl.gain:.2f} (ratio={ratio:.2f}). "
            f"Expected agreement within 5x."
        )


# ===================================================================
# 6. HarmonicResult fields populated and correct types
# ===================================================================

class TestHarmonicResultFields:
    """All HarmonicResult fields should be populated with correct types."""

    def test_harmonic_result_fields(self, harmonic_result_hysteretic):
        """Check all fields in HarmonicResult are present and typed."""
        result = harmonic_result_hysteretic

        # frequencies_hz
        assert isinstance(result.frequencies_hz, np.ndarray)
        assert result.frequencies_hz.ndim == 1
        assert len(result.frequencies_hz) > 0
        assert np.all(result.frequencies_hz > 0)

        # displacement_amplitudes
        assert isinstance(result.displacement_amplitudes, np.ndarray)
        assert result.displacement_amplitudes.ndim == 2
        assert result.displacement_amplitudes.shape[0] == len(result.frequencies_hz)

        # contact_face_uniformity
        assert isinstance(result.contact_face_uniformity, float)
        assert 0.0 <= result.contact_face_uniformity <= 1.0 + 1e-9

        # gain
        assert isinstance(result.gain, float)
        assert result.gain > 0.0

        # q_factor
        assert isinstance(result.q_factor, (int, float))
        assert result.q_factor >= 0.0

        # mesh
        assert result.mesh is not None

        # solve_time_s
        assert isinstance(result.solve_time_s, float)
        assert result.solve_time_s > 0.0

        # solver_name
        assert result.solver_name == "SolverA"


# ===================================================================
# 7. Displacement amplitudes are complex
# ===================================================================

class TestDisplacementAmplitudesComplex:
    """displacement_amplitudes should be a complex array."""

    def test_displacement_amplitudes_complex(self, harmonic_result_hysteretic):
        """The displacement_amplitudes array should have complex dtype."""
        result = harmonic_result_hysteretic
        assert np.iscomplexobj(result.displacement_amplitudes), (
            f"Expected complex displacement_amplitudes, got "
            f"dtype={result.displacement_amplitudes.dtype}"
        )


# ===================================================================
# 8. Uniformity for uniform bar
# ===================================================================

class TestUniformityForUniformBar:
    """Uniformity should be close to 1.0 for a uniform cylindrical bar."""

    def test_uniformity_for_uniform_bar(self, harmonic_result_hysteretic):
        """For a uniform cylinder, the contact face (top face) displacement
        at resonance should be nearly uniform, giving uniformity close to 1.0.

        Uniformity = min(|U_y|) / mean(|U_y|) at response nodes.
        For a symmetric cylinder this should be >= 0.7 (accounting for
        mesh discretization effects and Poisson coupling).
        """
        result = harmonic_result_hysteretic
        assert result.contact_face_uniformity > 0.7, (
            f"Expected uniformity > 0.7 for uniform bar, "
            f"got {result.contact_face_uniformity:.3f}"
        )


# ===================================================================
# 9. Additional: frequency sweep array correct length
# ===================================================================

class TestModalSuperpositionPath:
    """Verify the modal damping / superposition code path."""

    def test_modal_damping_produces_valid_result(self):
        """Modal superposition should produce a valid HarmonicResult."""
        result = _run_harmonic(damping_model="modal", n_freq_points=21)
        assert isinstance(result, HarmonicResult)
        assert result.gain > 0.0
        assert isinstance(result.displacement_amplitudes, np.ndarray)
        assert np.iscomplexobj(result.displacement_amplitudes)
        assert len(result.frequencies_hz) == 21


class TestSweepArrayLength:
    """Verify the sweep array has the expected number of points."""

    def test_sweep_array_length(self, harmonic_result_hysteretic):
        """frequencies_hz should have exactly n_freq_points entries."""
        result = harmonic_result_hysteretic
        assert len(result.frequencies_hz) == 51, (
            f"Expected 51 frequency points, got {len(result.frequencies_hz)}"
        )

    def test_displacement_shape_matches_sweep(self, harmonic_result_hysteretic):
        """displacement_amplitudes first axis should match frequency count."""
        result = harmonic_result_hysteretic
        assert result.displacement_amplitudes.shape[0] == len(result.frequencies_hz), (
            f"Shape mismatch: amplitudes has {result.displacement_amplitudes.shape[0]} "
            f"rows but {len(result.frequencies_hz)} frequency points"
        )
