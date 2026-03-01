"""Mode shape classifier for FEA modal analysis results.

Classifies each mode by type (longitudinal, flexural, torsional, compound),
detects parasitic modes near a target frequency, identifies nodal planes
for longitudinal modes, and computes effective modal mass.

This enhances the basic classification in ``SolverA._classify_mode`` with:
- Torsional mode detection via angular momentum about the principal axis.
- Parasitic mode alerts (CRITICAL / WARNING / OK).
- Nodal plane identification via zero-crossing interpolation.
- Full effective modal mass computation.

Algorithm details
-----------------
1. **Displacement Ratio Classification**: Extract per-axis displacement
   energy ratios R_x, R_y, R_z from eigenvectors.  If R_longitudinal > 0.70
   the mode is longitudinal; if max(R_lateral1, R_lateral2) > 0.60 it is
   flexural; otherwise check for torsional, then compound.

2. **Torsional Detection**: Compute the normalised angular momentum about
   the longitudinal axis.  If the normalised value exceeds 0.60, the mode
   is torsional.

3. **Parasitic Mode Detection**: For each non-longitudinal mode, compute
   frequency separation from the target longitudinal mode.  Flag as
   CRITICAL (<3%), WARNING (<5%), or OK.

4. **Nodal Plane Identification**: For longitudinal modes, find where
   the axial displacement crosses zero along the longitudinal axis via
   linear interpolation.

5. **Effective Modal Mass**: Mass-weighted participation factors in
   each axis direction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Axis index mapping
# ---------------------------------------------------------------------------
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
_LONGITUDINAL_RATIO_THRESHOLD = 0.70
_FLEXURAL_RATIO_THRESHOLD = 0.60
_TORSIONAL_ANGULAR_MOMENTUM_THRESHOLD = 0.60

# Parasitic mode separation thresholds (percentage)
_PARASITIC_CRITICAL_PCT = 3.0
_PARASITIC_WARNING_PCT = 5.0


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ClassifiedMode:
    """A single classified mode.

    Attributes
    ----------
    mode_number : int
        1-based mode number in the analysis order.
    frequency_hz : float
        Natural frequency in Hz.
    mode_type : str
        One of ``"longitudinal"``, ``"flexural"``, ``"torsional"``,
        ``"compound"``.
    displacement_ratios : np.ndarray
        Array ``[R_x, R_y, R_z]`` of displacement energy ratios that
        sum to 1.0.
    effective_mass : np.ndarray
        Array ``[M_eff_x, M_eff_y, M_eff_z]`` of effective modal masses
        in each axis direction (kg).
    nodal_plane_y : float or None
        Y-coordinate (in meters) of the nodal plane for longitudinal
        modes.  ``None`` for non-longitudinal modes or if no crossing
        is found.
    parasitic_flag : str
        ``"OK"``, ``"WARNING"``, or ``"CRITICAL"`` indicating proximity
        to the target longitudinal mode.
    separation_pct : float or None
        Frequency separation from the target longitudinal mode as a
        percentage.  ``None`` for the target mode itself and for
        longitudinal modes.
    """
    mode_number: int
    frequency_hz: float
    mode_type: str
    displacement_ratios: np.ndarray
    effective_mass: np.ndarray
    nodal_plane_y: Optional[float]
    parasitic_flag: str
    separation_pct: Optional[float]


@dataclass
class ClassificationResult:
    """Full classification result for all analysed modes.

    Attributes
    ----------
    modes : list[ClassifiedMode]
        All classified modes, in order of ascending frequency.
    target_mode_index : int
        Index into *modes* of the primary longitudinal mode nearest
        ``f_target``.
    parasitic_modes : list[ClassifiedMode]
        Non-longitudinal modes within 5% of the target frequency.
    total_mass_kg : float
        Total structural mass computed from the mass matrix via
        ``e^T M e`` for a unit translation direction.
    """
    modes: list[ClassifiedMode]
    target_mode_index: int
    parasitic_modes: list[ClassifiedMode]
    total_mass_kg: float


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------
class ModeClassifier:
    """Classify mode shapes from modal analysis.

    Takes node coordinates and the global mass matrix and classifies
    eigenvectors by mode type, detects parasitic modes near a target
    frequency, identifies nodal planes, and computes effective modal
    mass.

    Parameters
    ----------
    node_coords : np.ndarray, shape (n_nodes, 3)
        Node coordinates in meters.
    M : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
        Global consistent mass matrix in CSR format.
    longitudinal_axis : str
        Which axis is the longitudinal direction: ``"x"``, ``"y"``,
        or ``"z"``.  Defaults to ``"y"`` (GmshMesher convention).

    Examples
    --------
    >>> classifier = ModeClassifier(mesh.nodes, M, longitudinal_axis="y")
    >>> result = classifier.classify(freqs, mode_shapes, target_frequency_hz=20000.0)
    >>> print(result.target_mode_index)
    """

    def __init__(
        self,
        node_coords: NDArray[np.float64],
        M: sp.csr_matrix,
        longitudinal_axis: str = "y",
    ) -> None:
        if longitudinal_axis not in _AXIS_INDEX:
            raise ValueError(
                f"longitudinal_axis must be 'x', 'y', or 'z', "
                f"got {longitudinal_axis!r}"
            )

        self._coords = np.asarray(node_coords, dtype=np.float64)
        self._M = M
        self._long_axis = longitudinal_axis
        self._long_idx = _AXIS_INDEX[longitudinal_axis]

        # Pre-determine the two lateral axis indices
        all_axes = [0, 1, 2]
        all_axes.remove(self._long_idx)
        self._lat_idx_1 = all_axes[0]
        self._lat_idx_2 = all_axes[1]

        self._n_nodes = self._coords.shape[0]
        self._n_dof = 3 * self._n_nodes

        # Pre-compute unit direction vectors for effective mass
        self._e = np.zeros((3, self._n_dof), dtype=np.float64)
        self._e[0, 0::3] = 1.0  # x
        self._e[1, 1::3] = 1.0  # y
        self._e[2, 2::3] = 1.0  # z

        # Pre-compute M @ e for each direction
        self._Me = np.zeros((3, self._n_dof), dtype=np.float64)
        for d in range(3):
            self._Me[d] = M @ self._e[d]

        # Total mass (u^T M u for unit translation)
        self._total_mass = float(self._e[0] @ self._Me[0])

        logger.info(
            "ModeClassifier: %d nodes, %d DOFs, longitudinal_axis=%r, "
            "total_mass=%.6f kg",
            self._n_nodes,
            self._n_dof,
            self._long_axis,
            self._total_mass,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        frequencies_hz: NDArray[np.float64],
        mode_shapes: NDArray[np.float64],
        target_frequency_hz: float = 20000.0,
    ) -> ClassificationResult:
        """Classify all modes.

        Parameters
        ----------
        frequencies_hz : np.ndarray, shape (n_modes,)
            Natural frequencies in Hz.
        mode_shapes : np.ndarray, shape (n_modes, n_dof)
            Mass-normalised mode shapes (row per mode).
        target_frequency_hz : float
            Target operating frequency in Hz (default 20 kHz).

        Returns
        -------
        ClassificationResult
            Full classification including mode types, parasitic alerts,
            nodal planes, and effective mass.
        """
        frequencies_hz = np.asarray(frequencies_hz, dtype=np.float64)
        mode_shapes = np.asarray(mode_shapes, dtype=np.float64)
        n_modes = len(frequencies_hz)

        if mode_shapes.shape != (n_modes, self._n_dof):
            raise ValueError(
                f"mode_shapes shape {mode_shapes.shape} does not match "
                f"expected ({n_modes}, {self._n_dof})"
            )

        # Step 1: Classify each mode and compute properties
        classified: list[ClassifiedMode] = []

        for i in range(n_modes):
            phi = mode_shapes[i]
            freq = float(frequencies_hz[i])

            # Displacement ratios
            disp_ratios = self._displacement_ratios(phi)

            # Mode type classification
            mode_type = self._classify_type(phi, disp_ratios)

            # Effective modal mass
            eff_mass = self._effective_mass(phi)

            # Nodal plane (longitudinal modes only)
            nodal_plane = None
            if mode_type == "longitudinal":
                nodal_plane = self._find_nodal_plane(phi)

            classified.append(
                ClassifiedMode(
                    mode_number=i + 1,
                    frequency_hz=freq,
                    mode_type=mode_type,
                    displacement_ratios=disp_ratios,
                    effective_mass=eff_mass,
                    nodal_plane_y=nodal_plane,
                    parasitic_flag="OK",  # placeholder; updated below
                    separation_pct=None,
                )
            )

        # Step 2: Find target longitudinal mode
        target_idx = self._find_target_mode(classified, target_frequency_hz)

        # Step 3: Parasitic mode detection
        if target_idx >= 0:
            f_target = classified[target_idx].frequency_hz
            parasitic_list: list[ClassifiedMode] = []

            for cm in classified:
                if cm.mode_type == "longitudinal":
                    # Longitudinal modes are not parasitic
                    cm.parasitic_flag = "OK"
                    cm.separation_pct = None
                else:
                    sep_pct = abs(cm.frequency_hz - f_target) / f_target * 100.0
                    cm.separation_pct = sep_pct
                    if sep_pct < _PARASITIC_CRITICAL_PCT:
                        cm.parasitic_flag = "CRITICAL"
                        parasitic_list.append(cm)
                    elif sep_pct < _PARASITIC_WARNING_PCT:
                        cm.parasitic_flag = "WARNING"
                        parasitic_list.append(cm)
                    else:
                        cm.parasitic_flag = "OK"
        else:
            # No longitudinal mode found; all modes are OK
            parasitic_list = []
            logger.warning(
                "No longitudinal mode found near target %.1f Hz. "
                "Parasitic detection skipped.",
                target_frequency_hz,
            )

        result = ClassificationResult(
            modes=classified,
            target_mode_index=target_idx,
            parasitic_modes=parasitic_list,
            total_mass_kg=self._total_mass,
        )

        logger.info(
            "Classification complete: %d modes, target_index=%d, "
            "%d parasitic modes",
            n_modes,
            target_idx,
            len(parasitic_list),
        )

        return result

    # ------------------------------------------------------------------
    # Displacement ratios
    # ------------------------------------------------------------------

    @staticmethod
    def _displacement_ratios(phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute displacement energy ratios [R_x, R_y, R_z].

        Parameters
        ----------
        phi : (n_dof,) array
            Mode shape vector.

        Returns
        -------
        np.ndarray, shape (3,)
            ``[R_x, R_y, R_z]`` where each is the fraction of
            displacement energy in that axis.  Sums to 1.0.
        """
        u_x = phi[0::3]
        u_y = phi[1::3]
        u_z = phi[2::3]

        sum_x2 = float(np.sum(u_x ** 2))
        sum_y2 = float(np.sum(u_y ** 2))
        sum_z2 = float(np.sum(u_z ** 2))
        total = sum_x2 + sum_y2 + sum_z2

        if total == 0.0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)

        return np.array(
            [sum_x2 / total, sum_y2 / total, sum_z2 / total],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Mode type classification
    # ------------------------------------------------------------------

    def _classify_type(
        self,
        phi: NDArray[np.float64],
        disp_ratios: NDArray[np.float64],
    ) -> str:
        """Classify a single mode by type.

        Uses displacement ratios first, then checks torsional via
        angular momentum if neither longitudinal nor flexural.

        Classification rules:

        1. If the longitudinal ratio R_long > 0.70 -> ``"longitudinal"``.
        2. If any single lateral ratio > 0.60 -> ``"flexural"``.
        3. If the *combined* lateral ratio R_lat1 + R_lat2 > 0.70
           (dominant lateral motion) -> ``"flexural"``.  This catches
           bending modes in rotationally symmetric geometries where
           the displacement couples equally into both lateral axes.
        4. Check torsional via angular momentum criterion.
        5. Otherwise -> ``"compound"``.

        Parameters
        ----------
        phi : (n_dof,) array
            Mode shape vector.
        disp_ratios : (3,) array
            ``[R_x, R_y, R_z]`` displacement ratios.

        Returns
        -------
        str
            ``"longitudinal"``, ``"flexural"``, ``"torsional"``, or
            ``"compound"``.
        """
        R_long = disp_ratios[self._long_idx]
        R_lat1 = disp_ratios[self._lat_idx_1]
        R_lat2 = disp_ratios[self._lat_idx_2]

        if R_long > _LONGITUDINAL_RATIO_THRESHOLD:
            return "longitudinal"

        if max(R_lat1, R_lat2) > _FLEXURAL_RATIO_THRESHOLD:
            return "flexural"

        # For rotationally symmetric bodies (cylinders), bending modes
        # couple into both lateral axes equally.  Check combined lateral
        # energy ratio.
        R_lateral_combined = R_lat1 + R_lat2
        if R_lateral_combined > _LONGITUDINAL_RATIO_THRESHOLD:
            # Only classify as flexural if the mode is NOT torsional.
            # Torsional modes also have high lateral displacement, but
            # with a distinct angular momentum signature.
            if not self._is_torsional(phi):
                return "flexural"

        # Check torsional
        if self._is_torsional(phi):
            return "torsional"

        return "compound"

    # ------------------------------------------------------------------
    # Torsional detection
    # ------------------------------------------------------------------

    def _is_torsional(self, phi: NDArray[np.float64]) -> bool:
        """Check if a mode is torsional via angular momentum about the
        longitudinal axis.

        Computes the normalised angular momentum:

        .. math::

            L_y = \\sum_i (z_i u_{x,i} - x_i u_{z,i})

        normalised by:

        .. math::

            L_{y,\\text{norm}} = \\frac{|L_y|}{
                \\sqrt{\\sum_i (x_i^2 + z_i^2) \\cdot
                       \\sum_i (u_{x,i}^2 + u_{z,i}^2)}}

        For the general case where the longitudinal axis is not ``"y"``,
        the formula generalises: the two lateral axes play the roles of
        ``x`` and ``z`` in the cross product.

        Parameters
        ----------
        phi : (n_dof,) array
            Mode shape vector.

        Returns
        -------
        bool
            ``True`` if the mode is torsional.
        """
        # Extract coordinates in the lateral plane
        # lat1 plays the role of "x" and lat2 plays the role of "z"
        c_lat1 = self._coords[:, self._lat_idx_1]
        c_lat2 = self._coords[:, self._lat_idx_2]

        # Extract displacements in the lateral plane
        u_lat1 = phi[self._lat_idx_1::3]
        u_lat2 = phi[self._lat_idx_2::3]

        # Angular momentum about the longitudinal axis
        # L = sum(c_lat2 * u_lat1 - c_lat1 * u_lat2)
        L = float(np.sum(c_lat2 * u_lat1 - c_lat1 * u_lat2))

        # Normalisation denominator
        coord_sq_sum = float(np.sum(c_lat1 ** 2 + c_lat2 ** 2))
        disp_sq_sum = float(np.sum(u_lat1 ** 2 + u_lat2 ** 2))

        denom = np.sqrt(coord_sq_sum * disp_sq_sum)

        if denom < 1e-30:
            return False

        L_norm = abs(L) / denom

        return L_norm > _TORSIONAL_ANGULAR_MOMENTUM_THRESHOLD

    # ------------------------------------------------------------------
    # Effective modal mass
    # ------------------------------------------------------------------

    def _effective_mass(self, phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute effective modal mass in each axis direction.

        For mass-normalised eigenvector phi:

        .. math::

            M_{\\text{eff},d} = (\\phi^T M e_d)^2

        Parameters
        ----------
        phi : (n_dof,) array
            Mass-normalised mode shape vector.

        Returns
        -------
        np.ndarray, shape (3,)
            ``[M_eff_x, M_eff_y, M_eff_z]`` in kg.
        """
        result = np.zeros(3, dtype=np.float64)
        for d in range(3):
            gamma = float(phi @ self._Me[d])
            result[d] = gamma ** 2
        return result

    # ------------------------------------------------------------------
    # Nodal plane identification
    # ------------------------------------------------------------------

    def _find_nodal_plane(
        self, phi: NDArray[np.float64]
    ) -> Optional[float]:
        """Find the nodal plane (zero-crossing) of longitudinal
        displacement along the longitudinal axis.

        Collects (y_coord, u_long) for all nodes, sorts by y_coord,
        finds sign changes in u_long, and interpolates to find the
        zero-crossing location.

        For modes with multiple nodal planes, returns the one closest
        to the centre of the geometry along the longitudinal axis.

        Parameters
        ----------
        phi : (n_dof,) array
            Mode shape vector.

        Returns
        -------
        float or None
            Y-coordinate (meters) of the nodal plane, or ``None`` if
            no zero crossing is found.
        """
        # Extract longitudinal displacement
        u_long = phi[self._long_idx::3]
        y_coords = self._coords[:, self._long_idx]

        # Sort by longitudinal coordinate
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        u_sorted = u_long[sort_idx]

        # Find sign changes
        crossings: list[float] = []
        for j in range(len(u_sorted) - 1):
            u_a = u_sorted[j]
            u_b = u_sorted[j + 1]

            if u_a * u_b < 0:
                # Linear interpolation
                y_a = y_sorted[j]
                y_b = y_sorted[j + 1]
                abs_a = abs(u_a)
                abs_b = abs(u_b)
                y_nodal = y_a + abs_a / (abs_a + abs_b) * (y_b - y_a)
                crossings.append(float(y_nodal))

        if not crossings:
            return None

        # Return the crossing closest to the centre
        y_min = float(y_coords.min())
        y_max = float(y_coords.max())
        y_centre = (y_min + y_max) / 2.0

        best = min(crossings, key=lambda yc: abs(yc - y_centre))
        return best

    # ------------------------------------------------------------------
    # Target mode identification
    # ------------------------------------------------------------------

    @staticmethod
    def _find_target_mode(
        modes: list[ClassifiedMode],
        target_freq: float,
    ) -> int:
        """Find the longitudinal mode closest to the target frequency.

        Parameters
        ----------
        modes : list[ClassifiedMode]
            Classified modes.
        target_freq : float
            Target frequency in Hz.

        Returns
        -------
        int
            Index into *modes* of the best longitudinal match.
            Returns -1 if no longitudinal modes exist.
        """
        longitudinal_indices = [
            i for i, m in enumerate(modes) if m.mode_type == "longitudinal"
        ]

        if not longitudinal_indices:
            return -1

        best_idx = min(
            longitudinal_indices,
            key=lambda i: abs(modes[i].frequency_hz - target_freq),
        )
        return best_idx

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_mass_kg(self) -> float:
        """Total structural mass in kg."""
        return self._total_mass

    @property
    def longitudinal_axis(self) -> str:
        """The longitudinal axis label."""
        return self._long_axis

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return self._n_nodes

    @property
    def n_dof(self) -> int:
        """Number of degrees of freedom."""
        return self._n_dof

    def __repr__(self) -> str:
        return (
            f"ModeClassifier(n_nodes={self._n_nodes}, "
            f"longitudinal_axis={self._long_axis!r}, "
            f"total_mass={self._total_mass:.6f} kg)"
        )
