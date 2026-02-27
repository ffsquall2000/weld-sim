# FEA Core Solver A -- Detailed Design Document

> **Version**: v1.0
> **Date**: 2026-02-27
> **Status**: Draft
> **Module path**: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/`
> **Dependencies**: numpy >= 1.24, scipy >= 1.11, optional: scikit-sparse (CHOLMOD)
> **Target accuracy**: frequency error < 1%, stress error < 5%
> **Scale**: up to 100K+ nodes (300K+ DOF), 15--40 kHz ultrasonic range

---

## 0. Module Layout

```
ultrasonic_weld_master/plugins/geometry_analyzer/fea/
    __init__.py
    material_properties.py      # existing -- FEA_MATERIALS database
    mesh_io.py                  # Gmsh .msh reader, node/element extraction     (~400 LOC)
    elements.py                 # TET10, HEX20 shape functions + integration    (~900 LOC)
    assembly.py                 # Global K, M assembly with DOF mapping          (~500 LOC)
    boundary_conditions.py      # BC application (free-free, clamped, MPC)       (~350 LOC)
    modal_solver.py             # Modal analysis (eigenvalue)                    (~600 LOC)
    harmonic_solver.py          # Harmonic response analysis                     (~550 LOC)
    static_solver.py            # Linear static stress analysis                  (~400 LOC)
    fatigue.py                  # Fatigue/life assessment                        (~450 LOC)
    assembly_coupling.py        # Multi-body horn+booster+transducer coupling    (~500 LOC)
    postprocess.py              # Stress recovery, mode classification, output   (~500 LOC)
    solver_config.py            # Dataclasses for solver configuration           (~200 LOC)
    utils.py                    # Coordinate transforms, sparse helpers          (~200 LOC)
                                                                   Total: ~5,550 LOC
```

---

## 1. Modal Analysis (Eigenvalue Problem)

### 1.1 Governing Equation

The undamped free-vibration generalized eigenvalue problem:

```
K * phi_i = omega_i^2 * M * phi_i
```

where:
- **K** is the global stiffness matrix (n x n, symmetric positive semi-definite for free-free)
- **M** is the global consistent mass matrix (n x n, symmetric positive definite)
- **omega_i** = 2 * pi * f_i is the i-th circular natural frequency
- **phi_i** is the i-th mode shape vector

### 1.2 Boundary Condition Variants

| Variant | Description | Implementation |
|---------|-------------|----------------|
| Free-free | Standard horn tuning; first 6 modes are rigid-body (f ~ 0) | Shift-invert with sigma near target frequency; discard modes where f < 10 Hz |
| Clamped at nodal plane | Fix all DOFs on nodal plane surface nodes | Zero rows/columns in K, M for constrained DOFs (penalty or elimination) |
| Pre-stressed modal | Bolt preload -> geometric stiffness K_sigma -> (K + K_sigma) * phi = omega^2 * M * phi | Two-step: static solve for preload stress, then build K_sigma, then eigensolve |

### 1.3 Shift-Invert Strategy

For ultrasonic applications, we seek modes near a target frequency f_target (e.g., 20 kHz). The shift-invert transformation uses:

```
sigma = (2 * pi * f_target)^2

Solve: (K - sigma * M)^{-1} * M * phi = mu * phi

where mu = 1 / (omega^2 - sigma)
```

Modes closest to f_target have the largest |mu|, so they converge first in ARPACK's Lanczos iteration.

**Implementation**: `scipy.sparse.linalg.eigsh(K, k=n_modes, M=M, sigma=sigma, which='LM')`

- k = 20--50 (request 20+ modes; more for parasitic mode detection)
- `which='LM'` finds largest magnitude eigenvalues of the shifted operator
- OPmode uses the factorization of (K - sigma * M); for 300K DOF this is the bottleneck

### 1.4 Mode Classification

Each mode shape phi_i (dimension n_dof = 3 * n_nodes) is decomposed into x, y, z displacement components:

```python
u_x = phi_i[0::3]   # x-displacements at all nodes
u_y = phi_i[1::3]   # y-displacements at all nodes
u_z = phi_i[2::3]   # z-displacements at all nodes
```

**Participation factors** (mass-weighted):

```
Gamma_x = (phi_i^T * M * e_x) / (phi_i^T * M * phi_i)
Gamma_y = (phi_i^T * M * e_y) / (phi_i^T * M * phi_i)
Gamma_z = (phi_i^T * M * e_z) / (phi_i^T * M * phi_i)
```

where e_x, e_y, e_z are unit direction vectors expanded to full DOF size (e_x = [1,0,0, 1,0,0, ...]).

**Displacement ratios** for classification:

```
R_long = sum(u_z^2) / (sum(u_x^2) + sum(u_y^2) + sum(u_z^2))
R_flex = max(sum(u_x^2), sum(u_y^2)) / (sum(u_x^2) + sum(u_y^2) + sum(u_z^2))
```

**Torsional detection**: compute angular momentum about the principal axis:

```
L_z = sum_i m_i * (x_i * u_y_i - y_i * u_x_i)
L_z_normalized = |L_z| / sqrt(sum_i m_i * (x_i^2 + y_i^2) * (u_x_i^2 + u_y_i^2))
```

Classification rules (z = horn longitudinal axis):

| Condition | Mode Type |
|-----------|-----------|
| R_long > 0.70 | Longitudinal |
| R_flex > 0.60 and R_long < 0.30 | Flexural |
| L_z_normalized > 0.60 | Torsional |
| else | Compound / coupled |

### 1.5 Parasitic Mode Detection

For a target longitudinal mode at f_target:

```
For each non-longitudinal mode i with frequency f_i:
    delta_f = |f_i - f_target|
    separation_ratio = delta_f / f_target * 100  (percent)

    if separation_ratio < 3.0:
        flag = CRITICAL   # parasitic mode too close
    elif separation_ratio < 5.0:
        flag = WARNING    # needs monitoring
    else:
        flag = OK
```

### 1.6 Nodal Plane Location

The nodal plane is where the longitudinal displacement u_z crosses zero. For the target longitudinal mode:

```
For each node i, store (z_i, u_z_i).
Sort by z coordinate.
Find zero-crossings by detecting sign changes in u_z.
Interpolate: z_nodal = z_a + |u_z_a| / (|u_z_a| + |u_z_b|) * (z_b - z_a)
```

### 1.7 Effective Modal Mass

```
M_eff_x_i = (phi_i^T * M * e_x)^2 / (phi_i^T * M * phi_i)
M_eff_y_i = (phi_i^T * M * e_y)^2 / (phi_i^T * M * phi_i)
M_eff_z_i = (phi_i^T * M * e_z)^2 / (phi_i^T * M * phi_i)
```

The total effective modal mass ratio should sum to approximately the total mass:

```
sum_i M_eff_z_i / M_total ~ 1.0   (completeness check)
```

### 1.8 Data Structures

```python
@dataclass
class ModalAnalysisConfig:
    n_modes: int = 30                    # number of modes to extract
    f_target_hz: float = 20000.0         # target frequency for shift-invert
    f_min_hz: float = 100.0              # discard rigid-body modes below this
    bc_type: str = "free_free"           # free_free | clamped_nodal_plane | pre_stressed
    clamped_node_set: Optional[np.ndarray] = None   # node IDs for clamped BC
    preload_force_n: float = 0.0         # bolt preload for pre-stressed modal
    preload_face_node_set: Optional[np.ndarray] = None

@dataclass
class ModeResult:
    mode_number: int
    frequency_hz: float
    omega_rad_s: float
    mode_shape: np.ndarray               # (n_dof,) eigenvector
    mode_type: str                       # longitudinal | flexural | torsional | compound
    participation_factors: np.ndarray    # (3,) -- x, y, z
    effective_mass: np.ndarray           # (3,) -- x, y, z [kg]
    displacement_ratios: np.ndarray      # (3,) -- R_x, R_y, R_z
    nodal_plane_z: Optional[float]       # z-coordinate of nodal plane (longitudinal modes)
    parasitic_flag: str                  # OK | WARNING | CRITICAL

@dataclass
class ModalAnalysisResult:
    modes: list[ModeResult]
    target_mode_index: int               # index of mode closest to f_target that is longitudinal
    total_mass_kg: float
    effective_mass_ratio: float          # completeness check
    parasitic_modes: list[dict]          # [{mode_idx, freq, separation_pct, type}, ...]
    solve_time_s: float
```

### 1.9 Python Interface

```python
class ModalSolver:
    """Eigenvalue solver for modal analysis of ultrasonic components."""

    def __init__(self, K: sp.csr_matrix, M: sp.csr_matrix,
                 node_coords: np.ndarray, config: ModalAnalysisConfig):
        """
        Parameters
        ----------
        K : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
            Global stiffness matrix.
        M : scipy.sparse.csr_matrix, shape (n_dof, n_dof)
            Global consistent mass matrix.
        node_coords : np.ndarray, shape (n_nodes, 3)
            Node coordinates [x, y, z] in meters.
        config : ModalAnalysisConfig
        """
        ...

    def solve(self) -> ModalAnalysisResult:
        """Execute eigenvalue extraction via ARPACK shift-invert."""
        ...

    def _apply_boundary_conditions(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Apply BCs by DOF elimination or penalty method. Returns modified (K, M)."""
        ...

    def _compute_prestress_stiffness(self) -> sp.csr_matrix:
        """Build geometric stiffness K_sigma from preload stress field."""
        ...

    def _classify_mode(self, phi: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
        """Classify mode type. Returns (type_str, participation_factors, disp_ratios)."""
        ...

    def _find_nodal_planes(self, phi: np.ndarray) -> Optional[float]:
        """Find z-coordinate where longitudinal displacement crosses zero."""
        ...

    def _effective_modal_mass(self, phi: np.ndarray) -> np.ndarray:
        """Compute effective modal mass in x, y, z directions."""
        ...

    def _detect_parasitic_modes(self, modes: list[ModeResult]) -> list[dict]:
        """Flag non-longitudinal modes within 5% of target frequency."""
        ...
```

**Estimated LOC**: ~600 lines.

---

## 2. Harmonic Response Analysis

### 2.1 Governing Equation

The steady-state response to harmonic excitation at angular frequency omega:

```
[-omega^2 * M + j * omega * C + K] * U(omega) = F(omega)
```

or equivalently with the dynamic stiffness matrix:

```
D(omega) * U(omega) = F(omega)

where D(omega) = K - omega^2 * M + j * omega * C
```

All quantities are complex. U(omega) is the complex displacement amplitude at each DOF.

### 2.2 Damping Models

**Model A: Structural (hysteretic) damping via loss factor eta**

```
C_structural is replaced by: K_complex = K * (1 + j * eta)

where eta = 1 / Q   (Q is the quality factor)

D(omega) = K * (1 + j * eta) - omega^2 * M
```

Typical Q-factors for ultrasonic horn materials:

| Material | Q-factor | eta = 1/Q |
|----------|----------|-----------|
| Ti-6Al-4V | 8000--15000 | 6.7e-5 -- 1.25e-4 |
| Al 7075-T6 | 5000--10000 | 1.0e-4 -- 2.0e-4 |
| Steel D2 | 10000--20000 | 5.0e-5 -- 1.0e-4 |
| CPM steels | 8000--15000 | 6.7e-5 -- 1.25e-4 |

**Model B: Rayleigh (proportional) damping**

```
C = alpha * M + beta * K
```

Frequency-fitted coefficients from two target frequencies (f1, f2) and damping ratios (zeta1, zeta2):

```
omega_1 = 2 * pi * f_1
omega_2 = 2 * pi * f_2

| omega_1^{-1}   omega_1 |   | alpha |     | zeta_1 |
|                         | * |       | = 2 |        |
| omega_2^{-1}   omega_2 |   | beta  |     | zeta_2 |
```

Solving:

```
alpha = 2 * (zeta_1 * omega_2 - zeta_2 * omega_1) / (omega_2^2 - omega_1^2) * omega_1 * omega_2
beta  = 2 * (zeta_2 * omega_2 - zeta_1 * omega_1) / (omega_2^2 - omega_1^2)
```

For ultrasonic applications, typically set:
- f_1 = 0.8 * f_target, f_2 = 1.2 * f_target
- zeta_1 = zeta_2 = 1 / (2 * Q)

### 2.3 Frequency Sweep

```
f_center = f_resonance (from modal analysis)
f_min = f_center * (1 - sweep_pct / 100)
f_max = f_center * (1 + sweep_pct / 100)
sweep_pct = 2.0 (default)
n_points = 50--200

freq_array = np.linspace(f_min, f_max, n_points)

For each f in freq_array:
    omega = 2 * pi * f
    D = K * (1 + j * eta) - omega^2 * M       # structural damping
    # OR: D = K + j * omega * C - omega^2 * M  # Rayleigh damping
    U = spsolve(D, F)
    store U(omega)
```

### 2.4 Excitation Definition

**Displacement excitation** (prescribed at input face):

```
Given u_prescribed at input face node set S_in:
Partition DOFs into free (f) and prescribed (p):

D_ff * U_f + D_fp * U_p = F_f = 0
D_pf * U_f + D_pp * U_p = F_p (reaction)

U_f = -D_ff^{-1} * D_fp * U_p
```

**Force excitation** (applied at input face):

```
F = 0 everywhere except F_z at input face nodes = F_total / n_input_nodes
```

### 2.5 Output Quantities

**Amplitude gain** (output-to-input ratio):

```
For displacement excitation:
    u_out = mean(|U_z|) at output face (horn tip)
    u_in  = |u_prescribed|
    gain  = u_out / u_in

FRF (Frequency Response Function):
    H(omega) = U_output(omega) / U_input(omega)     # complex
    |H(omega)| = amplitude gain
    angle(H(omega)) = phase response
```

**Amplitude uniformity at contact face**:

```
u_z_nodes = |U_z| at all contact face nodes
u_avg = mean(u_z_nodes)
u_max = max(u_z_nodes)
u_min = min(u_z_nodes)

U  = u_min / u_avg           # uniformity ratio (U >= 0.85 target)
U' = u_avg / u_max           # normalized uniformity
```

**Amplitude asymmetry**:

```
Divide contact face into quadrants Q1..Q4 (by centroid-relative coordinates).
avg_i = mean(|U_z|) in quadrant i
asymmetry = (max(avg_i) - min(avg_i)) / mean(avg_i) * 100  [percent]
Target: asymmetry < 5%
```

### 2.6 Data Structures

```python
@dataclass
class HarmonicAnalysisConfig:
    excitation_type: str = "displacement"   # displacement | force
    excitation_amplitude: float = 1.0e-6    # [m] for displacement, [N] for force
    excitation_direction: str = "z"         # x | y | z
    input_face_nodes: Optional[np.ndarray] = None    # node IDs
    output_face_nodes: Optional[np.ndarray] = None   # node IDs (horn tip)
    contact_face_nodes: Optional[np.ndarray] = None  # node IDs (workpiece contact)
    damping_model: str = "structural"       # structural | rayleigh
    Q_factor: float = 10000.0
    rayleigh_alpha: float = 0.0             # only if damping_model == rayleigh
    rayleigh_beta: float = 0.0
    f_center_hz: float = 20000.0
    sweep_percent: float = 2.0
    n_sweep_points: int = 100

@dataclass
class HarmonicResponseResult:
    frequencies_hz: np.ndarray              # (n_points,)
    gain_amplitude: np.ndarray              # (n_points,) |H(f)|
    gain_phase_deg: np.ndarray              # (n_points,) angle(H(f)) in degrees
    resonance_freq_hz: float                # frequency of peak gain
    peak_gain: float                        # maximum |H|
    uniformity_U: float                     # at resonance
    uniformity_U_prime: float               # at resonance
    asymmetry_percent: float                # at resonance
    displacement_field_at_resonance: np.ndarray  # (n_dof,) complex
    stress_field_at_resonance: np.ndarray        # (n_elements, n_gauss, 6) complex
    solve_time_s: float
```

### 2.7 Python Interface

```python
class HarmonicSolver:
    """Frequency-domain harmonic response solver."""

    def __init__(self, K: sp.csr_matrix, M: sp.csr_matrix,
                 node_coords: np.ndarray, elements: list[Element],
                 config: HarmonicAnalysisConfig):
        ...

    def solve(self) -> HarmonicResponseResult:
        """Sweep through frequency range, solving complex system at each point."""
        ...

    def _build_damping_matrix(self) -> Optional[sp.csr_matrix]:
        """Build C matrix for Rayleigh damping; None for structural damping."""
        ...

    def _build_dynamic_stiffness(self, omega: float) -> sp.csc_matrix:
        """Assemble D(omega) = K*(1+j*eta) - omega^2*M  or  K + j*omega*C - omega^2*M."""
        ...

    def _apply_displacement_bc(self, D: sp.csc_matrix, u_prescribed: np.ndarray
                               ) -> tuple[sp.csc_matrix, np.ndarray]:
        """Partition and reduce for prescribed displacement excitation."""
        ...

    def _compute_gain(self, U: np.ndarray) -> tuple[float, float]:
        """Compute amplitude gain and phase from complex displacement field."""
        ...

    def _compute_uniformity(self, U: np.ndarray) -> tuple[float, float, float]:
        """Compute U, U', and asymmetry at contact face."""
        ...

    def _recover_stress(self, U: np.ndarray) -> np.ndarray:
        """Recover stress field from displacement (element-by-element B*D*u)."""
        ...
```

**Estimated LOC**: ~550 lines.

---

## 3. Static Stress Analysis

### 3.1 Governing Equation

Linear static equilibrium:

```
K * u = f
```

where:
- **K** is the global stiffness matrix (n_dof x n_dof)
- **u** is the displacement vector (n_dof x 1)
- **f** is the external load vector (n_dof x 1)

### 3.2 Load Cases

**Case 1: Bolt preload at threaded connection**

```
Given: Preload force F_bolt [N], bolt axis direction d_bolt, bolt face node set S_bolt.

Distribute force equally:
    f_i = F_bolt * d_bolt / |S_bolt|   for each node i in S_bolt

Constrain: opposite face nodes fixed in bolt direction.
```

**Case 2: Clamping force at nodal plane**

```
Given: Clamping force F_clamp [N], nodal plane node set S_np.

Apply distributed pressure: p = F_clamp / A_np
f_i = p * A_i * n_i   (using tributary area A_i and surface normal n_i)
```

**Case 3: Contact pressure at horn-workpiece interface**

```
Given: Applied pressure p_contact [Pa], contact face node set S_contact.

f_i = p_contact * A_i * n_i   for each node i in S_contact
```

### 3.3 Stress Recovery at Integration Points

For each element e, at each Gauss point g:

```
sigma_g = D_e * B_g * u_e
```

where:
- D_e is the 6x6 elasticity matrix (Voigt notation) for element e
- B_g is the 6x(3*n_elem_nodes) strain-displacement matrix at Gauss point g
- u_e is the element displacement vector extracted from global u

The stress vector in Voigt notation:

```
sigma = [sigma_xx, sigma_yy, sigma_zz, tau_yz, tau_xz, tau_xy]^T
```

**Von Mises equivalent stress**:

```
sigma_vm = sqrt(0.5 * ((sigma_xx - sigma_yy)^2 + (sigma_yy - sigma_zz)^2 +
                         (sigma_zz - sigma_xx)^2 + 6 * (tau_xy^2 + tau_yz^2 + tau_xz^2)))
```

**Principal stresses** from the 3x3 stress tensor eigenvalue decomposition:

```
sigma_tensor = [[sigma_xx, tau_xy, tau_xz],
                [tau_xy, sigma_yy, tau_yz],
                [tau_xz, tau_yz, sigma_zz]]

eigenvalues -> sigma_1 >= sigma_2 >= sigma_3  (principal stresses)
eigenvectors -> n_1, n_2, n_3                 (principal directions)
```

**Reaction forces** at constrained DOFs:

```
R = K * u - f    (at constrained DOFs, R gives reaction; at free DOFs, R ~ 0)
```

### 3.4 Data Structures

```python
@dataclass
class StaticAnalysisConfig:
    load_type: str = "bolt_preload"         # bolt_preload | clamp_force | contact_pressure
    load_magnitude: float = 10000.0         # [N] for forces, [Pa] for pressure
    load_direction: np.ndarray = field(default_factory=lambda: np.array([0, 0, -1]))
    load_face_nodes: Optional[np.ndarray] = None
    constraint_type: str = "fixed"          # fixed | symmetry | prescribed
    constraint_face_nodes: Optional[np.ndarray] = None
    constraint_dofs: Optional[list[int]] = None  # which DOFs to fix: 0=x, 1=y, 2=z

@dataclass
class StaticAnalysisResult:
    displacement: np.ndarray                # (n_dof,)
    stress_at_gauss: np.ndarray             # (n_elements, n_gauss, 6) Voigt
    von_mises_at_gauss: np.ndarray          # (n_elements, n_gauss)
    principal_stresses: np.ndarray          # (n_elements, n_gauss, 3) sorted desc
    principal_directions: np.ndarray        # (n_elements, n_gauss, 3, 3) eigenvectors
    reaction_forces: np.ndarray             # (n_constrained_dofs,)
    max_von_mises_mpa: float
    max_displacement_mm: float
    solve_time_s: float
```

### 3.5 Python Interface

```python
class StaticSolver:
    """Linear static stress solver."""

    def __init__(self, K: sp.csr_matrix, node_coords: np.ndarray,
                 elements: list[Element], config: StaticAnalysisConfig):
        ...

    def solve(self) -> StaticAnalysisResult:
        """Solve K*u = f and recover stresses."""
        ...

    def _build_load_vector(self) -> np.ndarray:
        """Construct global force vector from load case definition."""
        ...

    def _apply_constraints(self, K: sp.csr_matrix, f: np.ndarray
                           ) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Apply Dirichlet BCs via penalty or elimination. Returns (K_mod, f_mod, bc_mask)."""
        ...

    def _recover_element_stress(self, u: np.ndarray, element: Element
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute stress at all Gauss points of one element.
        Returns (stress_voigt, von_mises, principal_stresses)."""
        ...

    def _compute_reaction_forces(self, K: sp.csr_matrix, u: np.ndarray,
                                  f: np.ndarray) -> np.ndarray:
        """R = K*u - f at constrained DOFs."""
        ...
```

**Estimated LOC**: ~400 lines.

---

## 4. Fatigue / Life Assessment

### 4.1 Approach Overview

Ultrasonic horns experience high-cycle fatigue at 10^9+ cycles (20 kHz * 50,000 seconds = 10^9). The assessment uses:

1. Von Mises equivalent alternating stress from harmonic response
2. Material S-N curve data
3. Goodman mean-stress correction
4. Stress concentration factors

### 4.2 Von Mises Alternating Stress

From the harmonic response at resonance, the complex stress field sigma(t) = Re(sigma_complex * e^{j*omega*t}):

```
sigma_a = |sigma_complex|    (amplitude of each component)

sigma_vm_alt = sqrt(0.5 * ((sigma_a_xx^2 + sigma_a_yy^2 + sigma_a_zz^2)
                            - sigma_a_xx*sigma_a_yy - sigma_a_yy*sigma_a_zz - sigma_a_zz*sigma_a_xx
                            + 3*(tau_a_xy^2 + tau_a_yz^2 + tau_a_xz^2)))
```

Note: For purely harmonic loading with zero mean (typical for ultrasonic vibration), this is the maximum Von Mises stress amplitude over one cycle.

### 4.3 S-N Curve Database

Power-law form for the S-N curve in the high-cycle regime:

```
sigma_a = sigma_f' * (2 * N)^b

where:
    sigma_f' = fatigue strength coefficient [MPa]
    b = fatigue strength exponent (negative)
    N = number of cycles to failure
```

At the endurance limit (N = 10^9 for ultrasonic):

```
sigma_e = sigma_f' * (2e9)^b
```

Material database entries:

| Material | sigma_f' [MPa] | b | sigma_e at 10^9 [MPa] | sigma_UTS [MPa] | Q-factor |
|----------|----------------|------|------------------------|------------------|----------|
| Ti-6Al-4V | 1700 | -0.095 | 330 | 950 | 8000--15000 |
| Al 7075-T6 | 1050 | -0.110 | 105 | 572 | 5000--10000 |
| Steel D2 (58 HRC) | 2200 | -0.080 | 550 | 1620 | 10000--20000 |
| CPM 10V (62 HRC) | 2400 | -0.085 | 510 | 2100 | 8000--15000 |
| M2 HSS (64 HRC) | 2500 | -0.082 | 560 | 2200 | 10000--18000 |

### 4.4 Stress Concentration Factor Kt

Geometric stress concentration at features (fillets, slots, threaded holes):

```
sigma_local = Kt * sigma_nominal

Kt values (from Peterson's):
    - Fillet radius R at step: Kt = 1 + 2 * sqrt(t/R)  (approximate for r/d << 1)
    - Threaded connection: Kt = 3.0 -- 5.0
    - Slot/groove: Kt = 1.5 -- 3.0
    - Smooth surface: Kt = 1.0
```

In the FEA context, stress concentration is inherently captured if the mesh is sufficiently refined at the feature. The Kt factor is used only when:
- The mesh is coarse near a feature (submodeling not used)
- Analytical correction is needed for threaded connections not explicitly modeled

### 4.5 Goodman Diagram (Mean + Alternating Stress)

For cases with nonzero mean stress (e.g., bolt preload + vibration):

```
sigma_mean = stress from static analysis (bolt preload)
sigma_alt  = stress from harmonic analysis (vibration amplitude)

Goodman criterion:
    sigma_alt / sigma_e + sigma_mean / sigma_UTS = 1 / SF

Safety factor:
    SF = 1 / (sigma_alt / sigma_e + sigma_mean / sigma_UTS)

Target: SF >= 2.0
```

For pure alternating stress (no mean, typical free-free horn vibration):

```
SF = sigma_e / (Kt * sigma_vm_alt_max)
```

### 4.6 Critical Location Identification

```
For each Gauss point (element e, point g):
    sigma_alt_eg = Kt * sigma_vm_alt[e, g]
    SF_eg = sigma_e / sigma_alt_eg                    # simplified (no mean)
    # OR: SF_eg = 1 / (sigma_alt_eg/sigma_e + sigma_mean_eg/sigma_UTS)  # Goodman

Sort all (e, g) by SF ascending.
Top 10 lowest SF locations are "critical failure locations."
Map back to physical coordinates via element shape function interpolation.
```

### 4.7 Data Structures

```python
@dataclass
class FatigueMaterialData:
    name: str
    sigma_f_prime_mpa: float       # fatigue strength coefficient
    b_exponent: float              # fatigue strength exponent
    sigma_e_mpa: float             # endurance limit at 10^9 cycles
    sigma_uts_mpa: float           # ultimate tensile strength
    sigma_yield_mpa: float         # yield strength

@dataclass
class FatigueAnalysisConfig:
    material: str = "Ti-6Al-4V"
    Kt_global: float = 1.0         # global Kt (applied everywhere)
    Kt_regions: dict = field(default_factory=dict)  # {element_set_name: Kt}
    target_SF: float = 2.0
    n_critical_locations: int = 10
    include_mean_stress: bool = False  # set True if bolt preload exists

@dataclass
class FatigueResult:
    min_safety_factor: float
    safety_factor_field: np.ndarray      # (n_elements, n_gauss)
    critical_locations: list[dict]       # [{elem_id, gauss_pt, x, y, z, sigma_alt, SF}, ...]
    passes_target: bool                  # min_safety_factor >= target_SF
    goodman_data: Optional[dict]         # {sigma_alt_range, sigma_mean_range, SF_contour}
```

### 4.8 Python Interface

```python
class FatigueAssessor:
    """High-cycle fatigue assessment for ultrasonic components."""

    # Class-level S-N database
    SN_DATABASE: dict[str, FatigueMaterialData] = {
        "Ti-6Al-4V": FatigueMaterialData("Ti-6Al-4V", 1700, -0.095, 330, 950, 880),
        "Al 7075-T6": FatigueMaterialData("Al 7075-T6", 1050, -0.110, 105, 572, 503),
        "Steel D2": FatigueMaterialData("Steel D2", 2200, -0.080, 550, 1620, 1620),
        "CPM 10V": FatigueMaterialData("CPM 10V", 2400, -0.085, 510, 2100, 2100),
        "M2 HSS": FatigueMaterialData("M2 HSS", 2500, -0.082, 560, 2200, 2200),
    }

    def __init__(self, harmonic_result: HarmonicResponseResult,
                 static_result: Optional[StaticAnalysisResult],
                 elements: list[Element], node_coords: np.ndarray,
                 config: FatigueAnalysisConfig):
        ...

    def assess(self) -> FatigueResult:
        """Run fatigue assessment on harmonic stress field."""
        ...

    def _compute_alternating_von_mises(self) -> np.ndarray:
        """Extract VM alternating stress from complex harmonic stress field."""
        ...

    def _apply_stress_concentration(self, sigma: np.ndarray) -> np.ndarray:
        """Apply Kt factors globally and per-region."""
        ...

    def _compute_safety_factors(self, sigma_alt: np.ndarray,
                                 sigma_mean: Optional[np.ndarray]) -> np.ndarray:
        """Compute SF at every Gauss point using Goodman or simple ratio."""
        ...

    def _find_critical_locations(self, SF: np.ndarray) -> list[dict]:
        """Identify n worst locations and map to physical coordinates."""
        ...

    def _build_goodman_diagram_data(self, sigma_alt: np.ndarray,
                                     sigma_mean: np.ndarray) -> dict:
        """Construct data for Goodman diagram visualization."""
        ...
```

**Estimated LOC**: ~450 lines.

---

## 5. Assembly Coupling (Multi-Body Analysis)

### 5.1 Overview

A full ultrasonic stack consists of:

```
[Transducer] --threaded joint-- [Booster] --threaded joint-- [Horn]
```

Each component is meshed independently. They are coupled at interfaces.

### 5.2 Interface Coupling Methods

**Method A: Bonded (shared DOF merge)**

For matching meshes at the interface (ideal case):

```
Given interface node pairs (i_A, i_B) where node i_A on body A coincides with node i_B on body B.
Merge DOFs: replace all occurrences of DOF(i_B) with DOF(i_A) in body B's matrices.
The merged system has n_dof_total = n_dof_A + n_dof_B - 3 * n_interface_pairs.
```

**Method B: Tied contact (Multi-Point Constraints)**

For non-matching meshes (general case), use MPC to enforce displacement continuity:

```
For each slave node s on surface B, find the master element face on surface A
that contains the projection of s.

Constraint: u_s = sum_m N_m(xi_s, eta_s) * u_m

where N_m are shape functions of the master face evaluated at the projected
natural coordinates (xi_s, eta_s) of the slave node.
```

This is enforced via Lagrange multipliers or the penalty method.

**Penalty method formulation**:

```
K_contact = alpha_p * integral_{Gamma_c} (N_s - N_m)^T * (N_s - N_m) dGamma

Add K_contact to the global stiffness matrix.
alpha_p = 10^3 * max(diagonal(K))  (penalty stiffness)
```

**Lagrange multiplier formulation**:

```
Augmented system:
[K    G^T] [u]     [f]
[G    0  ] [lambda] [0]

where G is the constraint matrix, lambda are Lagrange multipliers (interface forces).
```

### 5.3 Joint Stiffness Modeling

Threaded connections are not modeled with explicit thread geometry. Instead:

```
K_joint = distributed spring stiffness at the thread interface

For a standard M10 threaded stud at ultrasonic frequency:
    k_axial = E * A_thread / L_engage   [N/m]
    k_bending ~ 0.5 * k_axial           (reduced due to thread compliance)

A_thread = pi/4 * d_minor^2
L_engage = engagement length (typically 1.0--1.5 * d_nominal)

The joint is modeled as:
    - Bonded (glued) interface for axial/shear DOFs
    - Reduced stiffness springs for the engaged thread region
```

### 5.4 Impedance Matching Computation

The acoustic impedance at each component:

```
Z = rho * c * A

where:
    rho = material density [kg/m^3]
    c = sqrt(E / rho) = longitudinal wave speed [m/s]
    A = cross-section area at the interface [m^2]
```

Transmission coefficient between adjacent components:

```
T_{AB} = 4 * Z_A * Z_B / (Z_A + Z_B)^2
```

Overall stack transmission efficiency:

```
eta_stack = product(T_{i,i+1}) for all interfaces
```

### 5.5 Gain Through Full Stack

From the harmonic response of the coupled assembly:

```
u_transducer_output = prescribed displacement at transducer-booster interface
u_booster_output    = |U_z| at booster-horn interface (from FEA)
u_horn_tip          = |U_z| at horn tip (from FEA)

Booster gain (FEA) = u_booster_output / u_transducer_output
Horn gain (FEA)    = u_horn_tip / u_booster_output
Stack gain (FEA)   = u_horn_tip / u_transducer_output
```

Compare FEA-computed gains with catalog values for validation.

### 5.6 Data Structures

```python
@dataclass
class ComponentMesh:
    name: str                              # "horn", "booster", "transducer"
    node_coords: np.ndarray                # (n_nodes, 3) in component local frame
    elements: list[Element]
    material_name: str
    interface_nodes: dict[str, np.ndarray]  # {"top": [...], "bottom": [...]}
    K_local: Optional[sp.csr_matrix] = None
    M_local: Optional[sp.csr_matrix] = None

@dataclass
class AssemblyCouplingConfig:
    coupling_method: str = "tied_contact"   # bonded | tied_contact
    penalty_factor: float = 1e3             # multiplier on max(diag(K))
    joint_stiffness_reduction: float = 0.8  # fraction of full bonded stiffness
    component_order: list[str] = field(
        default_factory=lambda: ["transducer", "booster", "horn"]
    )

@dataclass
class AssemblyResult:
    K_global: sp.csr_matrix
    M_global: sp.csr_matrix
    dof_map: dict[str, np.ndarray]         # component_name -> global DOF indices
    interface_node_pairs: list[tuple]       # [(node_A, node_B), ...]
    impedance: dict[str, float]            # component_name -> Z
    transmission_coefficients: dict[str, float]  # interface_name -> T
    stack_efficiency: float
    n_total_dof: int
```

### 5.7 Python Interface

```python
class AssemblyCoupler:
    """Couples multiple ultrasonic component meshes into a single system."""

    def __init__(self, components: list[ComponentMesh], config: AssemblyCouplingConfig):
        ...

    def couple(self) -> AssemblyResult:
        """Build coupled global K, M from individual component matrices."""
        ...

    def _merge_dofs_bonded(self, K_A: sp.csr_matrix, K_B: sp.csr_matrix,
                            pairs: list[tuple]) -> sp.csr_matrix:
        """Merge matching interface nodes by DOF identification."""
        ...

    def _build_mpc_constraints(self, surface_A_nodes: np.ndarray,
                                surface_B_nodes: np.ndarray,
                                coords_A: np.ndarray, coords_B: np.ndarray
                                ) -> sp.csr_matrix:
        """Build G constraint matrix for tied contact via closest-point projection."""
        ...

    def _compute_impedance(self, component: ComponentMesh) -> float:
        """Z = rho * c * A for the component interface."""
        ...

    def _transmission_coefficient(self, Z_A: float, Z_B: float) -> float:
        """T = 4*Z_A*Z_B / (Z_A + Z_B)^2."""
        ...
```

**Estimated LOC**: ~500 lines.

---

## 6. Element Technology

### 6.1 TET10 -- 10-Node Quadratic Tetrahedron

**Nodes**: 4 corner nodes + 6 mid-edge nodes = 10 nodes, 30 DOFs per element.

**Natural coordinates**: (xi, eta, zeta) with L1 = 1 - xi - eta - zeta, L2 = xi, L3 = eta, L4 = zeta.

**Shape functions**:

```
N1  = L1 * (2*L1 - 1)          # corner 1
N2  = L2 * (2*L2 - 1)          # corner 2
N3  = L3 * (2*L3 - 1)          # corner 3
N4  = L4 * (2*L4 - 1)          # corner 4
N5  = 4 * L1 * L2              # mid-edge 1-2
N6  = 4 * L2 * L3              # mid-edge 2-3
N7  = 4 * L1 * L3              # mid-edge 1-3
N8  = 4 * L1 * L4              # mid-edge 1-4
N9  = 4 * L2 * L4              # mid-edge 2-4
N10 = 4 * L3 * L4              # mid-edge 3-4
```

**Integration**: 4-point Gauss quadrature on tetrahedron.

Gauss points in barycentric coordinates (L1, L2, L3, L4):

```
Point 1: (a, b, b, b)   weight = 1/24
Point 2: (b, a, b, b)   weight = 1/24
Point 3: (b, b, a, b)   weight = 1/24
Point 4: (b, b, b, a)   weight = 1/24

where a = (5 + 3*sqrt(5)) / 20 = 0.5854101966...
      b = (5 - sqrt(5)) / 20   = 0.1381966011...

(Note: weights sum to 1/6, the volume of the reference tetrahedron)
```

### 6.2 HEX20 -- 20-Node Quadratic Hexahedron (Serendipity)

**Nodes**: 8 corner nodes + 12 mid-edge nodes = 20 nodes, 60 DOFs per element.

**Natural coordinates**: (xi, eta, zeta) each in [-1, +1].

**Shape functions**:

Corner nodes (i = 1..8 at positions (xi_i, eta_i, zeta_i) = combinations of +/-1):

```
N_i = (1/8) * (1 + xi_i*xi) * (1 + eta_i*eta) * (1 + zeta_i*zeta)
      * (xi_i*xi + eta_i*eta + zeta_i*zeta - 2)
```

Mid-edge nodes (3 families):

```
For mid-edge node at xi_i = 0:
    N_i = (1/4) * (1 - xi^2) * (1 + eta_i*eta) * (1 + zeta_i*zeta)

For mid-edge node at eta_i = 0:
    N_i = (1/4) * (1 + xi_i*xi) * (1 - eta^2) * (1 + zeta_i*zeta)

For mid-edge node at zeta_i = 0:
    N_i = (1/4) * (1 + xi_i*xi) * (1 + eta_i*eta) * (1 - zeta^2)
```

**Integration**: 3x3x3 = 27-point Gauss quadrature.

```
1D Gauss points for 3-point rule:
    xi_1 = -sqrt(3/5), w_1 = 5/9
    xi_2 = 0,          w_2 = 8/9
    xi_3 = +sqrt(3/5), w_3 = 5/9

3D points: all combinations (xi_i, eta_j, zeta_k), weight = w_i * w_j * w_k
Total: 27 points, weights sum to 8.0 (volume of reference cube [-1,1]^3)
```

### 6.3 Jacobian and B-Matrix

For either element type, at each Gauss point:

**Jacobian**:

```
J = [dN/d_natural]^T * X_elem

where:
    dN/d_natural is (3, n_nodes) -- derivatives of shape functions w.r.t. natural coords
    X_elem is (n_nodes, 3) -- element node physical coordinates
    J is (3, 3) -- Jacobian matrix

det(J) > 0 required (positive element volume)
```

**Physical shape function derivatives**:

```
dN/d_physical = J^{-1} * dN/d_natural
```

**Strain-displacement matrix B** (6 x 3*n_nodes):

```
For node i (columns 3*i, 3*i+1, 3*i+2):

B[:, 3*i:3*i+3] = [[dNi/dx,    0,       0    ],
                    [0,        dNi/dy,   0    ],
                    [0,        0,       dNi/dz],
                    [0,        dNi/dz,  dNi/dy],
                    [dNi/dz,   0,       dNi/dx],
                    [dNi/dy,   dNi/dx,  0     ]]
```

### 6.4 Elasticity Matrix D

For isotropic linear elastic material (Voigt notation, 6x6):

```
lambda = E * nu / ((1 + nu) * (1 - 2*nu))
mu     = E / (2 * (1 + nu))

D = [[lambda + 2*mu, lambda,       lambda,       0,  0,  0 ],
     [lambda,       lambda + 2*mu, lambda,       0,  0,  0 ],
     [lambda,       lambda,       lambda + 2*mu, 0,  0,  0 ],
     [0,            0,            0,            mu, 0,  0 ],
     [0,            0,            0,            0,  mu, 0 ],
     [0,            0,            0,            0,  0,  mu]]
```

### 6.5 Element Stiffness and Mass Matrices

**Stiffness**:

```
K_e = sum_{g=1}^{n_gauss} B_g^T * D * B_g * det(J_g) * w_g
```

- TET10: 30x30, summed over 4 Gauss points
- HEX20: 60x60, summed over 27 Gauss points

**Consistent mass** (full integration, required for modal accuracy):

```
M_e = rho * sum_{g=1}^{n_gauss} N_g^T * N_g * det(J_g) * w_g
```

where N_g is the (3, 3*n_nodes) shape function matrix at Gauss point g:

```
N_g = [[N1, 0,  0,  N2, 0,  0, ...],
       [0,  N1, 0,  0,  N2, 0, ...],
       [0,  0,  N1, 0,  0,  N2,...]]
```

This gives a full (non-diagonal) mass matrix which is critical for < 1% frequency accuracy.

### 6.6 Temperature-Dependent Properties

Material properties E(T), nu(T), rho(T) are interpolated from tabulated data:

```python
# Temperature table for Ti-6Al-4V
temp_points = [20, 100, 200, 300, 400, 500]   # degrees C
E_values    = [113.8e9, 110.0e9, 105.0e9, 100.0e9, 94.0e9, 86.0e9]  # Pa
nu_values   = [0.342, 0.345, 0.350, 0.355, 0.360, 0.370]

E_at_T = np.interp(T, temp_points, E_values)
```

At each element, the temperature field (from a prior thermal analysis or prescribed) determines the local D matrix.

### 6.7 Data Structures

```python
@dataclass
class GaussPoint:
    """Quadrature point in natural coordinates with weight."""
    natural_coords: np.ndarray   # (3,) or (4,) for barycentric
    weight: float

class ElementType(enum.Enum):
    TET10 = "tet10"
    HEX20 = "hex20"

@dataclass
class Element:
    elem_id: int
    elem_type: ElementType
    node_ids: np.ndarray         # (10,) or (20,) global node IDs
    material_name: str

class ElementLibrary:
    """Shape functions, derivatives, and Gauss integration for TET10 and HEX20."""

    @staticmethod
    def shape_functions_tet10(L: np.ndarray) -> np.ndarray:
        """Evaluate TET10 shape functions at barycentric coords L = (L1, L2, L3, L4).
        Returns: (10,) array of shape function values."""
        ...

    @staticmethod
    def shape_derivatives_tet10(L: np.ndarray) -> np.ndarray:
        """Derivatives dN/d(L1, L2, L3) for TET10.
        Returns: (3, 10) array."""
        ...

    @staticmethod
    def gauss_points_tet10() -> list[GaussPoint]:
        """4-point quadrature rule for tetrahedron.
        Returns: list of 4 GaussPoint objects."""
        ...

    @staticmethod
    def shape_functions_hex20(xi: np.ndarray) -> np.ndarray:
        """Evaluate HEX20 shape functions at (xi, eta, zeta) in [-1,1]^3.
        Returns: (20,) array."""
        ...

    @staticmethod
    def shape_derivatives_hex20(xi: np.ndarray) -> np.ndarray:
        """Derivatives dN/d(xi, eta, zeta) for HEX20.
        Returns: (3, 20) array."""
        ...

    @staticmethod
    def gauss_points_hex20() -> list[GaussPoint]:
        """3x3x3 = 27-point Gauss rule for hexahedron.
        Returns: list of 27 GaussPoint objects."""
        ...

    @staticmethod
    def compute_jacobian(dN_dnat: np.ndarray, node_coords: np.ndarray
                         ) -> tuple[np.ndarray, float]:
        """Compute Jacobian J and det(J).
        Parameters:
            dN_dnat: (3, n_nodes) shape function derivatives in natural coords
            node_coords: (n_nodes, 3) physical coordinates
        Returns: (J: (3,3), det_J: float)"""
        ...

    @staticmethod
    def compute_B_matrix(dN_dphys: np.ndarray, n_nodes: int) -> np.ndarray:
        """Build 6 x (3*n_nodes) strain-displacement matrix.
        Parameters:
            dN_dphys: (3, n_nodes) shape function derivatives in physical coords
        Returns: (6, 3*n_nodes) B matrix"""
        ...

    @staticmethod
    def elasticity_matrix(E: float, nu: float) -> np.ndarray:
        """6x6 isotropic elasticity matrix in Voigt notation."""
        ...

    def element_stiffness(self, element: Element, node_coords: np.ndarray,
                          E: float, nu: float) -> np.ndarray:
        """Compute element stiffness matrix K_e.
        Returns: (n_elem_dof, n_elem_dof) dense array."""
        ...

    def element_mass(self, element: Element, node_coords: np.ndarray,
                     rho: float) -> np.ndarray:
        """Compute element consistent mass matrix M_e.
        Returns: (n_elem_dof, n_elem_dof) dense array."""
        ...
```

**Estimated LOC**: ~900 lines.

---

## 7. Solver Infrastructure

### 7.1 Mesh I/O (Gmsh Reader)

Reads Gmsh `.msh` format (v4.1 ASCII and binary):

```python
@dataclass
class FEAMesh:
    nodes: np.ndarray              # (n_nodes, 3) coordinates in meters
    node_ids: np.ndarray           # (n_nodes,) global Gmsh node IDs
    elements: list[Element]        # list of TET10 / HEX20 elements
    physical_groups: dict[str, list[int]]  # group_name -> [elem_ids]
    node_sets: dict[str, np.ndarray]       # set_name -> node_id array
    n_nodes: int
    n_elements: int

class GmshReader:
    """Read Gmsh .msh files and extract TET10/HEX20 mesh data."""

    def read(self, filepath: str) -> FEAMesh:
        """Parse .msh file. Supports v4.1 ASCII format.
        Extracts:
            - Nodes section ($Nodes)
            - Elements section ($Elements), filtering for elem types 11 (TET10) and 17 (HEX20)
            - Physical groups ($PhysicalNames)
            - Node sets from physical group membership
        """
        ...

    def _parse_nodes(self, lines: list[str], offset: int) -> tuple[np.ndarray, np.ndarray]:
        ...

    def _parse_elements(self, lines: list[str], offset: int) -> list[Element]:
        ...

    def _extract_surface_nodes(self, mesh: FEAMesh, group_name: str) -> np.ndarray:
        """Get node IDs belonging to a named physical group (surface)."""
        ...
```

**Estimated LOC**: ~400 lines.

### 7.2 Global Assembly

Element-by-element assembly into global sparse matrices using COO format then converting to CSR:

```python
class GlobalAssembler:
    """Assemble global stiffness and mass matrices from element contributions."""

    def __init__(self, mesh: FEAMesh, element_lib: ElementLibrary,
                 material_props: dict[str, dict]):
        """
        Parameters
        ----------
        mesh : FEAMesh
        element_lib : ElementLibrary
        material_props : dict mapping material_name -> {E_pa, nu, rho_kg_m3, ...}
        """
        self.mesh = mesh
        self.element_lib = element_lib
        self.material_props = material_props
        self.n_dof = 3 * mesh.n_nodes
        self._node_id_to_index: dict[int, int] = {}  # Gmsh ID -> sequential index

    def assemble(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Assemble global K and M in CSR format.

        Algorithm:
        1. Pre-allocate COO arrays: rows, cols, vals_K, vals_M
           Estimate nnz: n_elements * (n_elem_dof^2)
           (TET10: 900 per elem, HEX20: 3600 per elem)
        2. For each element:
           a. Get node coordinates for element
           b. Look up material properties (with temperature if applicable)
           c. Compute K_e, M_e via element_lib
           d. Compute DOF mapping: dof_map = [3*idx, 3*idx+1, 3*idx+2 for each node]
           e. Scatter into COO arrays
        3. Create sp.coo_matrix, convert to csr_matrix
        4. Symmetrize: K = (K + K.T) / 2 (numerical cleanup)

        Returns (K, M) as scipy.sparse.csr_matrix.
        """
        ...

    def _element_dof_map(self, element: Element) -> np.ndarray:
        """Map element node IDs to global DOF indices.
        Returns: (n_elem_dof,) array of global DOF indices."""
        ...

    def _scatter_to_coo(self, K_e: np.ndarray, dof_map: np.ndarray,
                         rows: list, cols: list, vals: list) -> None:
        """Scatter dense element matrix into COO triplets."""
        ...
```

**Memory estimate for 300K DOF system**:

```
K and M storage:
    TET10 mesh, ~100K nodes, ~500K elements:
        nnz per element: ~900 (30x30, upper triangle ~465)
        Total nnz ~ 500K * 900 = 450M (before assembly overlap)
        After assembly (shared nodes): ~50M--100M nonzeros
        Memory: ~100M * 8 bytes = ~800 MB (double precision)

    With symmetric storage (upper triangle only): ~400 MB
    CSR overhead: row_ptr (300K * 4 bytes) + col_ind (50M * 4 bytes) ~ 200 MB
    Total K + M: ~1.2 GB for full storage, ~600 MB symmetric
```

**Vectorized assembly optimization**:

```python
def assemble_vectorized(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Vectorized assembly using numpy broadcasting.

    Instead of Python loop over elements, batch elements of the same type:
    1. Group all TET10 elements, extract all node coords as (n_tet10, 10, 3) array
    2. Compute all K_e in batch using numpy einsum:
       K_batch = np.einsum('eig,ij,ejh,e->egh', B_batch, D, B_batch, det_J * w)
    3. Build COO indices for all elements at once
    4. Create sparse matrix in one shot

    This achieves ~10-20x speedup over pure Python loops.
    """
    ...
```

**Estimated LOC**: ~500 lines.

### 7.3 Linear Solvers

```python
class LinearSolverBackend:
    """Abstraction over sparse linear solver backends."""

    def __init__(self, backend: str = "auto"):
        """
        backend: "auto" | "scipy_spsolve" | "cholmod" | "scipy_cg"

        auto selection logic:
            if scikit-sparse available and matrix is SPD:
                use CHOLMOD (fastest for K*u=f)
            elif n_dof < 500_000:
                use scipy.sparse.linalg.spsolve (SuperLU)
            else:
                use scipy.sparse.linalg.cg with ILU preconditioner
        """
        ...

    def solve_real(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solve A*x = b for real-valued systems."""
        ...

    def solve_complex(self, A: sp.csc_matrix, b: np.ndarray) -> np.ndarray:
        """Solve A*x = b for complex-valued systems (harmonic analysis)."""
        ...

    def factorize(self, A: sp.csc_matrix) -> object:
        """Pre-factorize A for repeated solves with different RHS.
        Returns opaque factor object."""
        ...

    def solve_factored(self, factor: object, b: np.ndarray) -> np.ndarray:
        """Solve using pre-computed factorization."""
        ...
```

**Solver performance characteristics**:

| Method | Use Case | Memory | Time (300K DOF) | Notes |
|--------|----------|--------|-----------------|-------|
| `scipy.sparse.linalg.spsolve` | Static, harmonic | 2-4x nnz | 5-30 s | SuperLU direct |
| CHOLMOD (`scikit-sparse`) | Static (SPD only) | 1.5-3x nnz | 2-10 s | Cholesky factorization |
| `scipy.sparse.linalg.eigsh` | Modal | Lanczos vectors | 30-120 s | ARPACK shift-invert |
| `scipy.sparse.linalg.cg` | Large static | ~1x nnz | 10-60 s | Iterative, needs precond |

### 7.4 Eigenvalue Solver Wrapper

```python
class EigenSolverWrapper:
    """Wrapper around scipy.sparse.linalg.eigsh with shift-invert configuration."""

    def solve(self, K: sp.csr_matrix, M: sp.csr_matrix,
              n_modes: int, sigma: float,
              tol: float = 1e-8, maxiter: int = 1000
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve generalized eigenvalue problem K*phi = omega^2 * M * phi
        using shift-invert around sigma.

        Parameters
        ----------
        K : stiffness matrix (n, n) CSR
        M : mass matrix (n, n) CSR
        n_modes : number of eigenvalues to extract
        sigma : shift value (= (2*pi*f_target)^2)
        tol : convergence tolerance for ARPACK
        maxiter : maximum Lanczos iterations

        Returns
        -------
        eigenvalues : (n_modes,) omega^2 values, sorted ascending
        eigenvectors : (n, n_modes) mode shapes, columns are mass-normalized

        Implementation:
        1. Compute OPinv = factorize(K - sigma * M) using CHOLMOD or SuperLU
        2. Call eigsh(K, k=n_modes, M=M, sigma=sigma, which='LM',
                      tol=tol, maxiter=maxiter, OPinv=OPinv)
        3. Sort by ascending eigenvalue
        4. Mass-normalize: phi_i = phi_i / sqrt(phi_i^T * M * phi_i)
        """
        ...
```

### 7.5 Boundary Condition Application

```python
class BoundaryConditionApplicator:
    """Apply various boundary conditions to global matrices."""

    @staticmethod
    def apply_fixed_dofs(K: sp.csr_matrix, M: sp.csr_matrix, f: np.ndarray,
                         fixed_dofs: np.ndarray, method: str = "elimination"
                         ) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
        """
        Fix specified DOFs to zero displacement.

        method = "elimination":
            Remove rows/columns for fixed DOFs.
            K_red, M_red have size (n_free, n_free).
            Must track free-DOF mapping for solution expansion.

        method = "penalty":
            K[i,i] += penalty_value for each fixed DOF i
            M[i,i] = 0 (or very small) for each fixed DOF i
            penalty_value = 1e20 * max(abs(K.diagonal()))
            Preserves matrix size but may degrade conditioning.
        """
        ...

    @staticmethod
    def apply_prescribed_displacement(K: sp.csr_matrix, f: np.ndarray,
                                       prescribed_dofs: np.ndarray,
                                       prescribed_values: np.ndarray
                                       ) -> tuple[sp.csr_matrix, np.ndarray]:
        """Apply nonzero prescribed displacements (e.g., excitation in harmonic).
        Modifies RHS: f_i -= K[i, j] * u_prescribed_j for all free DOFs i."""
        ...

    @staticmethod
    def apply_mpc(K: sp.csr_matrix, M: sp.csr_matrix,
                  master_dofs: np.ndarray, slave_dofs: np.ndarray,
                  constraint_matrix: np.ndarray
                  ) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """Apply multi-point constraints via transformation.
        u_slave = T * u_master
        K_new = T^T * K * T
        M_new = T^T * M * T
        """
        ...
```

**Estimated LOC for all infrastructure**: ~700 lines (mesh_io + assembly + solvers + BC).

---

## 8. Postprocessing Module

### 8.1 Stress Recovery and Extrapolation

Integration-point stresses are recovered and optionally extrapolated to nodes:

```python
class PostProcessor:
    """Stress recovery, field extrapolation, and result output."""

    def recover_gauss_point_stresses(self, U: np.ndarray, elements: list[Element],
                                      node_coords: np.ndarray, materials: dict
                                      ) -> np.ndarray:
        """Compute stress at all Gauss points for all elements.
        Returns: (n_elements, max_n_gauss, 6) Voigt stress components.

        For TET10: shape (n_elem, 4, 6)
        For HEX20: shape (n_elem, 27, 6)
        """
        ...

    def von_mises_stress(self, stress_voigt: np.ndarray) -> np.ndarray:
        """Compute Von Mises from Voigt stress tensor.
        Input:  (..., 6) with [sxx, syy, szz, tyz, txz, txy]
        Output: (...) scalar Von Mises stress.

        sigma_vm = sqrt(0.5*((s1-s2)^2 + (s2-s3)^2 + (s3-s1)^2 + 6*(t12^2+t23^2+t13^2)))
        """
        ...

    def principal_stresses(self, stress_voigt: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Compute principal stresses and directions.
        Returns: (principal_values: (..., 3), principal_dirs: (..., 3, 3))
        Eigendecomposition of symmetric 3x3 stress tensor at each point.
        """
        ...

    def extrapolate_to_nodes(self, gauss_values: np.ndarray, elements: list[Element]
                              ) -> np.ndarray:
        """Extrapolate Gauss point values to element nodes, then average at shared nodes.
        Uses inverse-distance or shape-function-based extrapolation.
        Returns: (n_nodes,) or (n_nodes, n_components) nodal values.
        """
        ...

    def compute_amplitude_uniformity(self, U_complex: np.ndarray,
                                      face_nodes: np.ndarray,
                                      node_coords: np.ndarray,
                                      direction: int = 2
                                      ) -> dict:
        """Compute uniformity metrics at a face.
        Returns: {U, U_prime, asymmetry_pct, max_amp, min_amp, avg_amp}
        """
        ...
```

**Estimated LOC**: ~500 lines.

---

## 9. Solver Configuration (Top-Level Orchestration)

### 9.1 Master Config

```python
@dataclass
class FEASolverConfig:
    """Top-level configuration for the FEA solver pipeline."""
    mesh_file: str                              # path to .msh file
    material_name: str = "Ti-6Al-4V"            # primary material
    temperature_c: float = 25.0                 # operating temperature
    analysis_types: list[str] = field(
        default_factory=lambda: ["modal"]        # modal | harmonic | static | fatigue
    )
    modal: ModalAnalysisConfig = field(default_factory=ModalAnalysisConfig)
    harmonic: HarmonicAnalysisConfig = field(default_factory=HarmonicAnalysisConfig)
    static: StaticAnalysisConfig = field(default_factory=StaticAnalysisConfig)
    fatigue: FatigueAnalysisConfig = field(default_factory=FatigueAnalysisConfig)
    assembly: Optional[AssemblyCouplingConfig] = None
    solver_backend: str = "auto"                # auto | scipy | cholmod
    n_threads: int = 1                          # for future parallel support
    output_dir: str = "./fea_results"
```

### 9.2 Solver Pipeline

```python
class FEASolverPipeline:
    """Orchestrates the full FEA analysis workflow."""

    def __init__(self, config: FEASolverConfig):
        self.config = config
        self.mesh: Optional[FEAMesh] = None
        self.K: Optional[sp.csr_matrix] = None
        self.M: Optional[sp.csr_matrix] = None
        self.results: dict[str, Any] = {}

    def run(self) -> dict:
        """Execute the full analysis pipeline.

        Steps:
        1. Read mesh from Gmsh file
        2. Look up material properties (with temperature interpolation)
        3. Assemble global K, M matrices
        4. If assembly coupling: couple multiple component meshes
        5. Run requested analyses in order:
           a. Modal analysis (required input for harmonic/fatigue)
           b. Static analysis (if bolt preload or clamping)
           c. Harmonic response (uses modal results for frequency targeting)
           d. Fatigue assessment (uses harmonic + static stress fields)
        6. Post-process and write results

        Returns: dict of {analysis_type: result_dataclass}
        """
        ...

    def _read_mesh(self) -> FEAMesh: ...
    def _get_material(self) -> dict: ...
    def _assemble_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]: ...
    def _run_modal(self) -> ModalAnalysisResult: ...
    def _run_harmonic(self, modal: ModalAnalysisResult) -> HarmonicResponseResult: ...
    def _run_static(self) -> StaticAnalysisResult: ...
    def _run_fatigue(self, harmonic: HarmonicResponseResult,
                     static: Optional[StaticAnalysisResult]) -> FatigueResult: ...
    def _export_results(self) -> None: ...
```

---

## 10. Integration with Existing Codebase

### 10.1 Plugin Registration

The FEA solver integrates as a sub-module of the existing `geometry_analyzer` plugin:

```python
# ultrasonic_weld_master/plugins/geometry_analyzer/fea/__init__.py

from .solver_config import FEASolverConfig
from .modal_solver import ModalSolver, ModalAnalysisConfig, ModalAnalysisResult
from .harmonic_solver import HarmonicSolver, HarmonicAnalysisConfig, HarmonicResponseResult
from .static_solver import StaticSolver, StaticAnalysisConfig, StaticAnalysisResult
from .fatigue import FatigueAssessor, FatigueAnalysisConfig, FatigueResult
from .assembly_coupling import AssemblyCoupler, AssemblyCouplingConfig, AssemblyResult
from .elements import ElementLibrary
from .mesh_io import GmshReader, FEAMesh

__all__ = [
    "FEASolverConfig", "FEASolverPipeline",
    "ModalSolver", "ModalAnalysisConfig", "ModalAnalysisResult",
    "HarmonicSolver", "HarmonicAnalysisConfig", "HarmonicResponseResult",
    "StaticSolver", "StaticAnalysisConfig", "StaticAnalysisResult",
    "FatigueAssessor", "FatigueAnalysisConfig", "FatigueResult",
    "AssemblyCoupler", "AssemblyCouplingConfig", "AssemblyResult",
    "ElementLibrary", "GmshReader", "FEAMesh",
]
```

### 10.2 Material Database Bridge

The FEA solver reads from the existing `FEA_MATERIALS` database in `material_properties.py`:

```python
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material, FEA_MATERIALS,
)

# Usage in solver:
mat = get_material("Ti-6Al-4V")
E = mat["E_pa"]           # 113.8e9
nu = mat["nu"]             # 0.342
rho = mat["rho_kg_m3"]    # 4430.0
```

Extensions needed in `material_properties.py`:

```python
# Add to FEA_MATERIALS entries:
"fatigue_sigma_f_prime_mpa": 1700.0,    # fatigue strength coefficient
"fatigue_b_exponent": -0.095,           # fatigue strength exponent
"fatigue_endurance_mpa": 330.0,         # endurance limit at 10^9 cycles
"sigma_uts_mpa": 950.0,                 # ultimate tensile strength
"Q_factor_range": [8000, 15000],        # quality factor range
"E_temperature_table": {                # temperature-dependent Young's modulus
    "temp_c": [20, 100, 200, 300, 400, 500],
    "E_pa": [113.8e9, 110.0e9, 105.0e9, 100.0e9, 94.0e9, 86.0e9],
}
```

### 10.3 Event Bus Integration

```python
# Emit events during FEA solve for progress tracking:
event_bus.emit("fea.analysis.started", {"type": "modal", "n_dof": 300000})
event_bus.emit("fea.assembly.progress", {"percent": 45})
event_bus.emit("fea.solve.progress", {"iteration": 50, "residual": 1e-6})
event_bus.emit("fea.analysis.completed", {"type": "modal", "time_s": 45.2})
```

---

## 11. Validation Targets and Acceptance Criteria

| Metric | Target | Validation Method |
|--------|--------|-------------------|
| Natural frequency error | < 1% | Compare with ANSYS/Abaqus on benchmark horn geometry |
| Von Mises stress error | < 5% | Compare with commercial FEA at same mesh density |
| Mode shape MAC value | > 0.99 | Modal Assurance Criterion vs. commercial FEA |
| Amplitude uniformity accuracy | +/- 2% | Compare U, U' with experimental laser vibrometry |
| Solve time (modal, 100K nodes) | < 120 s | Benchmark on Apple M1/M2 MacBook |
| Solve time (harmonic, 50 points) | < 300 s | Benchmark with factorization reuse |
| Memory usage (100K nodes) | < 4 GB | Monitor via tracemalloc |
| Mass conservation | sum(M_eff)/M_total > 0.95 | Effective mass completeness check |

---

## 12. Estimated Lines of Code Summary

| Module | File | LOC |
|--------|------|-----|
| Mesh I/O | `mesh_io.py` | 400 |
| Element library | `elements.py` | 900 |
| Global assembly | `assembly.py` | 500 |
| Boundary conditions | `boundary_conditions.py` | 350 |
| Modal solver | `modal_solver.py` | 600 |
| Harmonic solver | `harmonic_solver.py` | 550 |
| Static solver | `static_solver.py` | 400 |
| Fatigue assessment | `fatigue.py` | 450 |
| Assembly coupling | `assembly_coupling.py` | 500 |
| Post-processing | `postprocess.py` | 500 |
| Solver config | `solver_config.py` | 200 |
| Utilities | `utils.py` | 200 |
| **Total** | | **~5,550** |

---

## 13. Dependencies

```
# Required (already in project)
numpy>=1.24
scipy>=1.11

# Optional (performance)
scikit-sparse>=0.4.8       # CHOLMOD for 3-5x faster direct solves on SPD systems

# Optional (meshing, already partially present)
gmsh>=4.11                 # for mesh generation (Gmsh Python API)
```

No new mandatory dependencies beyond numpy and scipy, which are already in the project's `requirements.txt`.
