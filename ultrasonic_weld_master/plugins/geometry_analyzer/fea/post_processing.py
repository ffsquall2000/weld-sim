"""Post-processing for harmonic response analysis results.

Provides the ``AmplitudeAnalyzer`` class that computes:

* **FRF** (Frequency Response Function) -- amplitude gain and phase vs frequency.
* **Resonance detection** -- peak amplitude in the FRF curve.
* **Q-factor** -- quality factor from 3 dB bandwidth.
* **Uniformity** -- displacement uniformity across the response (contact) face.
* **Asymmetry** -- quadrant-based asymmetry of the response face amplitudes.

The analyzer is designed to work with the complex displacement fields produced
by ``SolverA.harmonic_analysis`` but is independent of the solver; only
``numpy`` arrays are required as input.

Coordinate convention
---------------------
The longitudinal axis defaults to ``"y"`` (index 1 in each 3-DOF node),
matching the convention used by ``GmshMesher``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

_AXIS_MAP = {"x": 0, "y": 1, "z": 2}


@dataclass
class FRFResult:
    """Frequency Response Function result.

    Attributes
    ----------
    frequencies_hz : (n_freq,)
        Sweep frequencies in Hz.
    amplitude : (n_freq,)
        |H(omega)| -- amplitude of the FRF at each frequency.
    phase_deg : (n_freq,)
        Phase angle of H(omega) in degrees.
    resonance_freq_hz : float
        Frequency of peak amplitude.
    peak_gain : float
        Maximum FRF amplitude.
    """

    frequencies_hz: np.ndarray
    amplitude: np.ndarray
    phase_deg: np.ndarray
    resonance_freq_hz: float
    peak_gain: float


@dataclass
class ResonanceInfo:
    """Resonance detection result.

    Attributes
    ----------
    frequency_hz : float
        Resonance frequency in Hz.
    peak_gain : float
        Amplitude gain at resonance.
    index : int
        Index into the frequencies array.
    """

    frequency_hz: float
    peak_gain: float
    index: int


@dataclass
class UniformityResult:
    """Uniformity assessment of the response face.

    Attributes
    ----------
    uniformity_U : float
        min(|U_long|) / mean(|U_long|) at the response face.
        Target >= 0.85 for good weld quality.
    uniformity_U_prime : float
        mean(|U_long|) / max(|U_long|) -- normalised uniformity.
    asymmetry_pct : float
        Quadrant-based asymmetry percentage.  Target < 5 %.
    node_amplitudes : (n_resp,)
        |U_long| at each response node.
    """

    uniformity_U: float
    uniformity_U_prime: float
    asymmetry_pct: float
    node_amplitudes: np.ndarray


@dataclass
class GainCurveResult:
    """Gain curve (amplitude and phase vs frequency).

    Attributes
    ----------
    frequencies_hz : (n_freq,)
        Sweep frequencies.
    gain_amplitude : (n_freq,)
        Amplitude gain at each frequency.
    gain_phase_deg : (n_freq,)
        Phase in degrees at each frequency.
    resonance_freq_hz : float
        Resonance frequency.
    peak_gain : float
        Maximum gain.
    q_factor : float
        Quality factor from 3 dB bandwidth.
    """

    frequencies_hz: np.ndarray
    gain_amplitude: np.ndarray
    gain_phase_deg: np.ndarray
    resonance_freq_hz: float
    peak_gain: float
    q_factor: float


# ---------------------------------------------------------------------------
# AmplitudeAnalyzer
# ---------------------------------------------------------------------------


class AmplitudeAnalyzer:
    """Post-processing for harmonic response analysis results.

    Computes amplitude gain, uniformity, FRF curves, Q-factor,
    and related metrics from harmonic displacement fields.

    Parameters
    ----------
    frequencies_hz : (n_freq,)
        Sweep frequencies in Hz.
    displacement_amplitudes : (n_freq, n_dof) complex
        Complex displacement at each frequency and DOF.
    node_coords : (n_nodes, 3)
        Node positions in metres.
    excitation_node_ids : (n_excit,)
        0-based node indices for the excitation face.
    response_node_ids : (n_resp,)
        0-based node indices for the response face (horn tip).
    excitation_amplitude : float
        Prescribed displacement amplitude [m].  Used when computing
        gain as response / prescribed for displacement excitation.
        Set to ``0.0`` to use force-excitation gain formula instead
        (response / excitation face displacement).
    longitudinal_axis : str
        ``'x'``, ``'y'``, or ``'z'``.  Default ``'y'``.
    """

    def __init__(
        self,
        frequencies_hz: np.ndarray,
        displacement_amplitudes: np.ndarray,
        node_coords: np.ndarray,
        excitation_node_ids: np.ndarray,
        response_node_ids: np.ndarray,
        excitation_amplitude: float = 1e-6,
        longitudinal_axis: str = "y",
    ) -> None:
        self.frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
        self.displacement_amplitudes = np.asarray(
            displacement_amplitudes, dtype=np.complex128
        )
        self.node_coords = np.asarray(node_coords, dtype=np.float64)
        self.excitation_node_ids = np.asarray(excitation_node_ids, dtype=np.intp)
        self.response_node_ids = np.asarray(response_node_ids, dtype=np.intp)
        self.excitation_amplitude = float(excitation_amplitude)
        self.longitudinal_axis = longitudinal_axis.lower().strip()

        if self.longitudinal_axis not in _AXIS_MAP:
            raise ValueError(
                f"longitudinal_axis must be 'x', 'y', or 'z', "
                f"got {self.longitudinal_axis!r}"
            )

        self._axis_idx = _AXIS_MAP[self.longitudinal_axis]

        # Pre-compute longitudinal DOF indices for excitation and response faces
        self._excit_long_dofs = 3 * self.excitation_node_ids + self._axis_idx
        self._resp_long_dofs = 3 * self.response_node_ids + self._axis_idx

        # Cache for computed results
        self._frf_result: FRFResult | None = None
        self._resonance_info: ResonanceInfo | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _response_amplitudes_at(
        self, freq_index: int
    ) -> NDArray[np.float64]:
        """Return |U_long| at each response node for a given frequency index."""
        return np.abs(
            self.displacement_amplitudes[freq_index, self._resp_long_dofs]
        )

    def _excitation_amplitudes_at(
        self, freq_index: int
    ) -> NDArray[np.float64]:
        """Return |U_long| at each excitation node for a given frequency index."""
        return np.abs(
            self.displacement_amplitudes[freq_index, self._excit_long_dofs]
        )

    def _mean_response_per_freq(self) -> NDArray[np.float64]:
        """Mean |U_long| at response nodes for each frequency."""
        return np.mean(
            np.abs(self.displacement_amplitudes[:, self._resp_long_dofs]),
            axis=1,
        )

    def _mean_excitation_per_freq(self) -> NDArray[np.float64]:
        """Mean |U_long| at excitation nodes for each frequency."""
        return np.mean(
            np.abs(self.displacement_amplitudes[:, self._excit_long_dofs]),
            axis=1,
        )

    def _complex_response_mean(self) -> NDArray[np.complex128]:
        """Mean complex U_long at response nodes for each frequency."""
        return np.mean(
            self.displacement_amplitudes[:, self._resp_long_dofs],
            axis=1,
        )

    def _complex_excitation_mean(self) -> NDArray[np.complex128]:
        """Mean complex U_long at excitation nodes for each frequency."""
        return np.mean(
            self.displacement_amplitudes[:, self._excit_long_dofs],
            axis=1,
        )

    # ------------------------------------------------------------------
    # 1. compute_frf
    # ------------------------------------------------------------------

    def compute_frf(self) -> FRFResult:
        """Compute the Frequency Response Function.

        For displacement excitation (``excitation_amplitude > 0``):
            H(omega) = mean(U_resp_long) / excitation_amplitude

        For force excitation (``excitation_amplitude == 0``):
            H(omega) = mean(U_resp_long) / mean(U_excit_long)

        Returns
        -------
        FRFResult
            FRF amplitude, phase, and resonance information.
        """
        resp_complex = self._complex_response_mean()  # (n_freq,) complex

        if self.excitation_amplitude > 0.0:
            # Displacement excitation: gain = |U_resp| / prescribed amplitude
            H = resp_complex / self.excitation_amplitude
        else:
            # Force excitation: gain = U_resp / U_excit (complex ratio)
            excit_complex = self._complex_excitation_mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                H = np.where(
                    np.abs(excit_complex) > 0.0,
                    resp_complex / excit_complex,
                    0.0 + 0.0j,
                )

        amplitude = np.abs(H)
        phase_deg = np.angle(H, deg=True)

        idx_peak = int(np.argmax(amplitude))
        resonance_freq_hz = float(self.frequencies_hz[idx_peak])
        peak_gain = float(amplitude[idx_peak])

        self._frf_result = FRFResult(
            frequencies_hz=self.frequencies_hz.copy(),
            amplitude=amplitude,
            phase_deg=phase_deg,
            resonance_freq_hz=resonance_freq_hz,
            peak_gain=peak_gain,
        )
        return self._frf_result

    # ------------------------------------------------------------------
    # 2. find_resonance
    # ------------------------------------------------------------------

    def find_resonance(self) -> ResonanceInfo:
        """Find the frequency of peak amplitude in the FRF.

        Returns
        -------
        ResonanceInfo
            Resonance frequency, peak gain, and index.
        """
        if self._frf_result is None:
            self.compute_frf()

        frf = self._frf_result
        assert frf is not None  # guaranteed by compute_frf above

        idx = int(np.argmax(frf.amplitude))
        self._resonance_info = ResonanceInfo(
            frequency_hz=float(self.frequencies_hz[idx]),
            peak_gain=float(frf.amplitude[idx]),
            index=idx,
        )
        return self._resonance_info

    # ------------------------------------------------------------------
    # 3. compute_q_factor
    # ------------------------------------------------------------------

    def compute_q_factor(self) -> float:
        """Compute Q-factor from the 3 dB bandwidth.

        Q = f_res / (f_hi - f_lo)

        where f_lo and f_hi are the frequencies at which |H| drops to
        peak / sqrt(2).  Linear interpolation is used for accurate
        crossing detection between discrete frequency points.

        Returns
        -------
        float
            Q-factor.  Returns ``0.0`` if bandwidth cannot be
            determined (e.g. peak at sweep edge or very narrow peak).
        """
        if self._frf_result is None:
            self.compute_frf()

        frf = self._frf_result
        assert frf is not None

        amplitude = frf.amplitude
        frequencies = self.frequencies_hz

        idx_peak = int(np.argmax(amplitude))
        peak_val = amplitude[idx_peak]
        f_res = frequencies[idx_peak]

        if peak_val <= 0.0:
            return 0.0

        threshold = peak_val / np.sqrt(2.0)

        # --- Find f_lo: search left from peak ---------------------------------
        f_lo: float | None = None
        for i in range(idx_peak - 1, -1, -1):
            if amplitude[i] <= threshold:
                g_low, g_high = amplitude[i], amplitude[i + 1]
                f_low, f_high = frequencies[i], frequencies[i + 1]
                dg = g_high - g_low
                if dg != 0.0:
                    frac = (threshold - g_low) / dg
                    f_lo = f_low + frac * (f_high - f_low)
                else:
                    f_lo = f_low
                break

        # --- Find f_hi: search right from peak --------------------------------
        f_hi: float | None = None
        for i in range(idx_peak + 1, len(amplitude)):
            if amplitude[i] <= threshold:
                g_high_prev, g_low_curr = amplitude[i - 1], amplitude[i]
                f_prev, f_curr = frequencies[i - 1], frequencies[i]
                dg = g_high_prev - g_low_curr
                if dg != 0.0:
                    frac = (g_high_prev - threshold) / dg
                    f_hi = f_prev + frac * (f_curr - f_prev)
                else:
                    f_hi = f_curr
                break

        if f_lo is not None and f_hi is not None:
            bandwidth = f_hi - f_lo
            if bandwidth > 0.0:
                return f_res / bandwidth

        return 0.0

    # ------------------------------------------------------------------
    # 4. compute_uniformity
    # ------------------------------------------------------------------

    def compute_uniformity(
        self, freq_index: int | None = None
    ) -> UniformityResult:
        """Compute displacement uniformity at the response face.

        Parameters
        ----------
        freq_index : int or None
            Frequency index to evaluate.  If ``None``, the resonance
            index (peak FRF amplitude) is used.

        Returns
        -------
        UniformityResult
            Uniformity metrics and per-node amplitudes.
        """
        if freq_index is None:
            res = self.find_resonance()
            freq_index = res.index

        node_amps = self._response_amplitudes_at(freq_index)

        mean_amp = float(np.mean(node_amps))
        max_amp = float(np.max(node_amps))
        min_amp = float(np.min(node_amps))

        # Uniformity U: min / mean  (target >= 0.85)
        if mean_amp > 0.0:
            uniformity_U = min_amp / mean_amp
        else:
            uniformity_U = 0.0

        # Normalised uniformity U': mean / max
        if max_amp > 0.0:
            uniformity_U_prime = mean_amp / max_amp
        else:
            uniformity_U_prime = 0.0

        # Asymmetry via quadrant decomposition
        asymmetry_pct = self._compute_quadrant_asymmetry(
            node_amps, self.response_node_ids
        )

        return UniformityResult(
            uniformity_U=uniformity_U,
            uniformity_U_prime=uniformity_U_prime,
            asymmetry_pct=asymmetry_pct,
            node_amplitudes=node_amps,
        )

    def _compute_quadrant_asymmetry(
        self,
        node_amps: NDArray[np.float64],
        node_ids: NDArray[np.intp],
    ) -> float:
        """Compute asymmetry by dividing response face into quadrants.

        The response face nodes are split into four quadrants based on
        their coordinates relative to the centroid of the face, using
        the two transverse axes (i.e. the axes *not* the longitudinal
        axis).

        asymmetry_pct = (max(avg_q) - min(avg_q)) / mean(avg_q) * 100

        Returns 0.0 if there are fewer than 4 response nodes (not
        enough to form meaningful quadrants).
        """
        n_nodes = len(node_ids)
        if n_nodes < 4:
            return 0.0

        coords = self.node_coords[node_ids]  # (n_resp, 3)

        # Determine transverse axes
        axes = [0, 1, 2]
        axes.remove(self._axis_idx)
        ax0, ax1 = axes[0], axes[1]

        centroid_0 = np.mean(coords[:, ax0])
        centroid_1 = np.mean(coords[:, ax1])

        # Assign each node to a quadrant
        q0 = coords[:, ax0] >= centroid_0
        q1 = coords[:, ax1] >= centroid_1

        quadrant_masks = [
            q0 & q1,        # quadrant 0: +ax0, +ax1
            q0 & ~q1,       # quadrant 1: +ax0, -ax1
            ~q0 & q1,       # quadrant 2: -ax0, +ax1
            ~q0 & ~q1,      # quadrant 3: -ax0, -ax1
        ]

        quadrant_avgs = []
        for mask in quadrant_masks:
            if np.any(mask):
                quadrant_avgs.append(float(np.mean(node_amps[mask])))

        if len(quadrant_avgs) < 2:
            return 0.0

        mean_of_avgs = np.mean(quadrant_avgs)
        if mean_of_avgs <= 0.0:
            return 0.0

        asymmetry = (max(quadrant_avgs) - min(quadrant_avgs)) / mean_of_avgs * 100.0
        return float(asymmetry)

    # ------------------------------------------------------------------
    # 5. compute_gain_curve
    # ------------------------------------------------------------------

    def compute_gain_curve(self) -> GainCurveResult:
        """Compute the gain curve (amplitude and phase vs frequency).

        Returns
        -------
        GainCurveResult
            Gain amplitude, phase, resonance frequency, peak gain,
            and Q-factor.
        """
        frf = self.compute_frf()
        q = self.compute_q_factor()

        return GainCurveResult(
            frequencies_hz=frf.frequencies_hz.copy(),
            gain_amplitude=frf.amplitude.copy(),
            gain_phase_deg=frf.phase_deg.copy(),
            resonance_freq_hz=frf.resonance_freq_hz,
            peak_gain=frf.peak_gain,
            q_factor=q,
        )

    # ------------------------------------------------------------------
    # 6. nodal_amplitude_at_frequency
    # ------------------------------------------------------------------

    def nodal_amplitude_at_frequency(
        self, freq_index: int
    ) -> NDArray[np.float64]:
        """Extract |U_long| at *all* nodes for a specific frequency.

        Parameters
        ----------
        freq_index : int
            Index into the frequency array.

        Returns
        -------
        (n_nodes,) float
            Longitudinal displacement amplitude at each node.
        """
        n_nodes = self.node_coords.shape[0]
        long_dofs = 3 * np.arange(n_nodes) + self._axis_idx
        return np.abs(self.displacement_amplitudes[freq_index, long_dofs])


# ---------------------------------------------------------------------------
# Gain chain result
# ---------------------------------------------------------------------------


@dataclass
class GainChainResult:
    """Result of gain chain computation through an ultrasonic stack.

    Attributes
    ----------
    components : list[dict]
        Per-component gain data.  Each dict has keys:
        ``name``, ``input_amp``, ``output_amp``, ``gain``.
    total_gain : float
        Product of all component gains.
    """

    components: list[dict]
    total_gain: float


# ---------------------------------------------------------------------------
# Standalone gain chain function
# ---------------------------------------------------------------------------


def compute_gain_chain(
    displacement: np.ndarray,
    component_interfaces: list[dict],
) -> GainChainResult:
    """Compute amplitude gain through each component in an ultrasonic stack.

    For each component, the gain is the ratio of the mean longitudinal
    displacement amplitude at the output face to the mean amplitude at
    the input face.  The total stack gain is the product of all
    component gains.

    Parameters
    ----------
    displacement : (n_dof,) complex or float
        Displacement vector at a single frequency (e.g. resonance).
        Can be complex (from harmonic analysis) or real.
    component_interfaces : list[dict]
        One entry per component, ordered from input to output.
        Each dict must have keys:

        - ``name`` : str -- component identifier.
        - ``input_dofs`` : array-like of int -- DOF indices at the
          input (bottom) face of this component.
        - ``output_dofs`` : array-like of int -- DOF indices at the
          output (top) face of this component.

    Returns
    -------
    GainChainResult
        Per-component gain data and total stack gain.

    Raises
    ------
    ValueError
        If ``component_interfaces`` is empty, or a component has
        empty input or output DOF lists.

    Notes
    -----
    The displacement array is typically the complex harmonic response
    at the resonance frequency.  The absolute value ``|U|`` is taken
    before averaging.
    """
    if len(component_interfaces) == 0:
        raise ValueError(
            "component_interfaces must contain at least one component."
        )

    displacement = np.asarray(displacement)

    components_out: list[dict] = []
    total_gain = 1.0

    for comp in component_interfaces:
        name = comp["name"]
        input_dofs = np.asarray(comp["input_dofs"], dtype=np.intp)
        output_dofs = np.asarray(comp["output_dofs"], dtype=np.intp)

        if len(input_dofs) == 0:
            raise ValueError(
                f"Component {name!r} has empty input_dofs."
            )
        if len(output_dofs) == 0:
            raise ValueError(
                f"Component {name!r} has empty output_dofs."
            )

        input_amp = float(np.mean(np.abs(displacement[input_dofs])))
        output_amp = float(np.mean(np.abs(displacement[output_dofs])))

        if input_amp > 0.0:
            gain = output_amp / input_amp
        else:
            gain = 0.0

        components_out.append({
            "name": name,
            "input_amp": input_amp,
            "output_amp": output_amp,
            "gain": gain,
        })
        total_gain *= gain

    logger.info(
        "Gain chain: %d components, total_gain=%.4f",
        len(components_out),
        total_gain,
    )

    return GainChainResult(
        components=components_out,
        total_gain=total_gain,
    )
