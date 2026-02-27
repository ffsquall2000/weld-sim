## 8. Phased Implementation Roadmap

> **Timeline**: 14 weeks (7 two-week phases)
> **Team assumption**: 1-2 developers, full-time
> **Baseline**: Existing `web/services/fea_service.py` (838 LOC, HEX8 structured mesh, simplified modal + harmonic)
> **Target**: Production-grade FEA system with Gmsh meshing, TET10 elements, validated modal/harmonic/stress solvers, piezoelectric coupling, and full assembly analysis

### Dependency Graph

```
Phase 1 ─────► Phase 2 ─────► Phase 3 ─────► Phase 4
(Foundation)   (Modal)        (Harmonic)      (Stress/Fatigue)
                                                    │
Phase 1 ─────► Phase 5 ◄──────────────────────────┘
               (Assembly)
                    │
                    ▼
               Phase 6 ─────► Phase 7
               (FEniCSx)      (Advanced)
```

---

### Phase 1: Foundation (Weeks 1-2)

**Goal**: Replace the structured hex mesh generator and HEX8 element with Gmsh-based unstructured meshing and TET10 quadratic tetrahedral elements. Establish the solver interface abstraction and material database that all subsequent phases depend on.

**Dependencies**: None (starting point)

#### 1.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 1a | Gmsh mesh generation wrapper | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py` | **Create** |
| 1b | STEP file import via Gmsh | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py` | (same file, `import_step()` method) |
| 1c | TET10 element implementation | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/elements.py` | **Create** |
| 1d | Solver interface abstraction | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_interface.py` | **Create** |
| 1e | Result container objects | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/results.py` | **Create** |
| 1f | Enhanced material database | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/material_properties.py` | **Modify** |
| 1g | FEA configuration dataclasses | `ultrasonic_weld_master/plugins/geometry_analyzer/fea/config.py` | **Create** |
| 1h | Unit tests | `tests/test_fea/test_mesher.py`, `tests/test_fea/test_elements.py`, `tests/test_fea/test_materials.py` | **Create** |

#### 1.2 Detailed Specifications

**1a/1b -- Gmsh Mesh Generation (`mesher.py`, ~450 LOC)**

```python
class GmshMesher:
    """Gmsh-based mesh generation for ultrasonic horn/booster/assembly."""

    def mesh_parametric_horn(
        self,
        horn_type: str,          # flat/cylindrical/exponential/stepped/blade
        dimensions: dict,        # width_mm, height_mm, length_mm, ...
        mesh_size: float = 2.0,  # characteristic element size in mm
        order: int = 2,          # 1=TET4, 2=TET10
        refinement_zones: list[dict] | None = None,  # [{center, radius, size}]
    ) -> "FEAMesh"

    def mesh_from_step(
        self,
        step_path: str,
        mesh_size: float = 2.0,
        order: int = 2,
    ) -> "FEAMesh"

    def mesh_from_cadquery(
        self,
        cq_shape,               # CadQuery Workplane object
        mesh_size: float = 2.0,
        order: int = 2,
    ) -> "FEAMesh"

@dataclass
class FEAMesh:
    nodes: np.ndarray           # (N, 3) node coordinates in meters
    elements: np.ndarray        # (E, 10) node connectivity for TET10
    element_type: str           # "TET10" or "TET4"
    node_sets: dict[str, np.ndarray]  # named node sets: "top_face", "bottom_face", etc.
    element_sets: dict[str, np.ndarray]
    surface_tris: np.ndarray    # (F, 3) surface triangulation for visualization
    mesh_stats: dict            # {min_quality, mean_quality, num_nodes, num_elements}
```

The mesher uses Gmsh's Python API (`gmsh.model.occ` for geometry, `gmsh.model.mesh` for meshing). Parametric horns are built using Gmsh's OpenCASCADE kernel -- cylinders via `addCylinder`, boxes via `addBox`, exponential profiles via `addThruSections` with spline lofts. Node sets are identified geometrically: top face = nodes where `y == y_max` within tolerance, bottom face = `y == y_min`, etc. The CadQuery bridge exports to a temporary STEP buffer and feeds it to Gmsh, avoiding any direct OCC dependency in the mesher.

**1c -- TET10 Element (`elements.py`, ~500 LOC)**

```python
class TET10Element:
    """10-node quadratic tetrahedral element (Bathe convention)."""

    # 4-point Gauss quadrature for TET10 (degree 2, exact for quadratic)
    GAUSS_POINTS: np.ndarray   # (4, 4) -- [xi, eta, zeta, weight]

    def shape_functions(self, xi, eta, zeta) -> np.ndarray:
        """10 shape functions N_i at natural coordinates. Returns (10,)."""

    def shape_derivatives(self, xi, eta, zeta) -> np.ndarray:
        """dN/d(xi,eta,zeta) at natural coordinates. Returns (3, 10)."""

    def stiffness_matrix(self, coords: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Element stiffness matrix K_e (30x30)."""

    def mass_matrix(self, coords: np.ndarray, rho: float) -> np.ndarray:
        """Consistent mass matrix M_e (30x30)."""

    def strain_displacement(self, coords: np.ndarray, xi, eta, zeta) -> tuple[np.ndarray, float]:
        """B-matrix (6x30) and det(J) at a natural coordinate point."""

    def stress_at_point(self, coords, u_e, D, xi, eta, zeta) -> np.ndarray:
        """Stress tensor (6,) in Voigt notation at a point."""
```

TET10 uses the standard quadratic tetrahedral shape functions with 10 nodes (4 vertices + 6 mid-edge). The Gauss quadrature uses 4 integration points (Keast rule, degree 2) which is exact for quadratic displacement interpolation. The consistent mass matrix is computed by numerical integration of `rho * N^T N * det(J)` over the element, not lumped -- this is critical for accurate eigenfrequencies in ultrasonic range.

Verification: A single TET10 element under uniform tension must reproduce the exact stress with zero discretization error. The patch test (constant strain field over an irregular mesh of TET10 elements) must pass to machine precision.

**1d -- Solver Interface (`solver_interface.py`, ~200 LOC)**

```python
class SolverInterface(ABC):
    """Abstract base for FEA solvers (Solver A = numpy/scipy, Solver B = FEniCSx)."""

    @abstractmethod
    def modal_analysis(self, config: ModalConfig) -> ModalResult: ...

    @abstractmethod
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult: ...

    @abstractmethod
    def static_analysis(self, config: StaticConfig) -> StaticResult: ...

class SolverA(SolverInterface):
    """NumPy/SciPy-based solver (default, no external dependencies)."""

class SolverB(SolverInterface):
    """FEniCSx-based solver (plugin, requires Docker or native install)."""
```

**1e -- Result Containers (`results.py`, ~250 LOC)**

```python
@dataclass
class ModalResult:
    frequencies_hz: np.ndarray          # Natural frequencies
    mode_shapes: np.ndarray             # (n_modes, n_dof) eigenvectors
    mode_types: list[str]               # "longitudinal", "flexural", "torsional"
    effective_mass_ratios: np.ndarray    # Modal participation factors
    mesh: FEAMesh                       # Reference to the mesh
    solve_time_s: float
    solver_name: str                    # "SolverA" or "SolverB"

@dataclass
class HarmonicResult:
    frequencies_hz: np.ndarray
    displacement_amplitudes: np.ndarray # (n_freq, n_dof) complex amplitudes
    contact_face_uniformity: float      # 0-1
    gain: float                         # amplitude ratio output/input
    q_factor: float
    mesh: FEAMesh
    solve_time_s: float

@dataclass
class StaticResult:
    displacement: np.ndarray            # (n_dof,)
    stress_vm: np.ndarray               # (n_elements,) Von Mises at centroids
    stress_tensor: np.ndarray           # (n_elements, 6) full stress tensor
    max_stress_mpa: float
    mesh: FEAMesh
    solve_time_s: float

@dataclass
class FatigueResult:
    safety_factors: np.ndarray          # per-element Goodman safety factor
    min_safety_factor: float
    critical_location: np.ndarray       # [x, y, z] of worst element
    estimated_life_cycles: float
    sn_curve_name: str
```

**1f -- Material Database Enhancement (`material_properties.py`, modify existing ~200 LOC)**

Add to each material entry:
- `damping_ratio` (loss factor eta, for harmonic analysis, typically 0.001-0.01 for metals)
- `fatigue_sn_coefficients` (S-N curve parameters: `sigma_f`, `b`, `sigma_e` for endurance limit)
- `acoustic_velocity_m_s` (longitudinal wave speed, for validation: `c = sqrt(E/rho)`)
- `temperature_dependent` (boolean, for future thermal coupling)

Add new materials needed for transducer modeling:
- PZT-4 (piezoelectric ceramic): `E_pa`, `d33`, `d31`, `eps_33`, `k_t` (coupling coefficient)
- PZT-8 (hard PZT for high-power): same structure
- Brass C360 (electrode/backing material)
- Stainless Steel 316L (booster/housing material)

**1g -- FEA Configuration (`config.py`, ~150 LOC)**

```python
@dataclass
class ModalConfig:
    mesh: FEAMesh
    material: dict
    n_modes: int = 20
    target_frequency_hz: float = 20000.0
    boundary_conditions: str = "free-free"  # "free-free" | "fixed-free" | "custom"
    fixed_node_sets: list[str] = field(default_factory=list)

@dataclass
class HarmonicConfig:
    mesh: FEAMesh
    material: dict
    freq_range: tuple[float, float] = (16000.0, 24000.0)
    n_freq_points: int = 201
    damping_model: str = "hysteretic"  # "hysteretic" | "rayleigh" | "modal"
    damping_ratio: float = 0.005
    excitation_node_set: str = "bottom_face"
    response_node_set: str = "top_face"

@dataclass
class StaticConfig:
    mesh: FEAMesh
    material: dict
    loads: list[dict]       # [{type: "pressure", node_set: "...", value: ...}]
    boundary_conditions: list[dict]
```

#### 1.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| TET10 patch test (constant strain) | Exact stress recovery | < 1e-10 relative error |
| TET10 single element uniform tension | Analytical stress | < 1e-10 relative error |
| Gmsh mesh of 25mm diameter x 80mm cylinder | Generates valid TET10 mesh | No inverted elements (min Jacobian > 0) |
| Gmsh STEP import of sample horn file | Mesh with correct bounding box | Bounding box within 0.1mm of input |
| Material lookup for all existing aliases | Returns correct properties | Exact match |
| New PZT-4 material has all required fields | All piezo fields present | Existence check |
| Mesh quality metric (scaled Jacobian) | > 0.3 average | Statistical check on 100 meshes |

#### 1.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/mesher.py` | ~450 | -- |
| `fea/elements.py` | ~500 | -- |
| `fea/solver_interface.py` | ~200 | -- |
| `fea/results.py` | ~250 | -- |
| `fea/config.py` | ~150 | -- |
| `fea/material_properties.py` | -- | ~100 |
| `tests/test_fea/test_mesher.py` | ~200 | -- |
| `tests/test_fea/test_elements.py` | ~300 | -- |
| `tests/test_fea/test_materials.py` | ~100 | -- |
| **Total** | **~2,150** | **~100** |

#### 1.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gmsh Python API version incompatibility | Medium | High | Pin `gmsh>=4.12` in requirements; test in CI with specific version |
| CadQuery-to-Gmsh bridge fails for complex horn shapes | Medium | Medium | Fallback: export CadQuery to STEP temp file, import via Gmsh STEP reader |
| TET10 consistent mass matrix computation is slow for large meshes | Low | Medium | Pre-compute element matrices in batched vectorized form; provide lumped mass option |
| Gmsh requires OpenGL for visualization (server environment) | High | Low | Use `gmsh.option.setNumber("General.Terminal", 1)` and run headless; never call `gmsh.fltk` |

#### 1.6 Demoable at End of Phase

- **CLI demo**: Generate a TET10 mesh for a 20 kHz cylindrical horn (D=25mm, L=80mm, Ti-6Al-4V) via a script. Print mesh statistics (node count, element count, quality metrics). Export mesh to VTK for ParaView visualization.
- **STEP import demo**: Load a sample `.step` file, mesh it, print bounding box and mesh stats.
- **Unit test suite**: `pytest tests/test_fea/ -v` passes with 100% coverage of elements, mesher, and materials.

---

### Phase 2: Core Modal Analysis (Weeks 3-4)

**Goal**: Implement the free-free modal analysis solver -- the single most important capability. Validate against analytical solutions for simple geometries.

**Dependencies**: Phase 1 (mesh, elements, material database, solver interface)

#### 2.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 2a | Sparse global assembly (K, M) | `fea/assembler.py` | **Create** |
| 2b | Free-free modal solver (SolverA) | `fea/solver_a.py` | **Create** |
| 2c | Mode classification engine | `fea/mode_classifier.py` | **Create** |
| 2d | Parasitic mode detection | `fea/mode_classifier.py` | (same file) |
| 2e | Nodal plane identification | `fea/mode_classifier.py` | (same file) |
| 2f | Web API integration | `web/services/fea_service.py` | **Modify** (add new codepath, preserve old) |
| 2g | Validation benchmark suite | `tests/test_fea/test_modal_validation.py` | **Create** |
| 2h | Unit tests | `tests/test_fea/test_assembler.py`, `tests/test_fea/test_solver_a.py` | **Create** |

#### 2.2 Detailed Specifications

**2a -- Global Assembly (`assembler.py`, ~350 LOC)**

```python
class GlobalAssembler:
    """Assemble global stiffness and mass matrices from element contributions."""

    def assemble(
        self,
        mesh: FEAMesh,
        element: TET10Element,
        material: dict,
    ) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
        """Returns (K_global, M_global) as sparse CSR matrices."""
```

Assembly uses the COO (coordinate) format for accumulation, then converts to CSR for solving. The DOF numbering is `[u_x0, u_y0, u_z0, u_x1, u_y1, u_z1, ...]`, i.e., 3 DOFs per node, interleaved. For a mesh with 10,000 nodes, this gives 30,000 DOFs and K/M matrices of size 30,000 x 30,000 (but sparse, typically ~1% fill).

Performance target: Assembly of a 10,000-node mesh in under 5 seconds. Strategy: vectorize the element loop using pre-allocated arrays and `numpy` broadcasting where possible. If the pure Python loop is too slow, provide a Cython/Numba fallback for the inner B-matrix computation.

**2b -- Modal Solver (`solver_a.py`, ~600 LOC)**

```python
class SolverA(SolverInterface):
    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Free-free or fixed-free eigenvalue analysis."""
```

Implementation details:
1. Assemble K and M using `GlobalAssembler`
2. For free-free: use `scipy.sparse.linalg.eigsh` with shift-invert (`sigma = (2*pi*f_target)^2`). Request `n_modes + 6` modes to account for 6 rigid-body modes (which will be at ~0 Hz)
3. Filter rigid-body modes: discard modes where `freq < 100 Hz`
4. Sort remaining modes by frequency
5. Classify each mode using `ModeClassifier`
6. Return `ModalResult` with all mode data

Key improvement over the existing `fea_service.py`: The current code applies fixed boundary conditions at the bottom face. In reality, ultrasonic horns and boosters vibrate in **free-free** conditions (held at the nodal plane, which is a zero-displacement plane for the operating mode). Free-free analysis is the standard for sonotrode tuning.

**2c/2d/2e -- Mode Classification (`mode_classifier.py`, ~400 LOC)**

```python
class ModeClassifier:
    """Classify mode shapes as longitudinal, flexural, or torsional."""

    def classify(
        self,
        eigenvector: np.ndarray,
        mesh: FEAMesh,
    ) -> ModeInfo:
        """Returns mode type, nodal plane locations, participation factors."""

    def detect_parasitic_modes(
        self,
        modal_result: ModalResult,
        target_freq_hz: float,
        min_separation_hz: float = 500.0,
    ) -> list[ParasiticModeWarning]:
        """Identify dangerous parasitic modes near the operating frequency."""

    def find_nodal_planes(
        self,
        eigenvector: np.ndarray,
        mesh: FEAMesh,
        axis: str = "y",  # longitudinal axis
    ) -> list[float]:
        """Find axial positions where displacement is zero (nodal planes)."""
```

Classification algorithm:
1. Reshape eigenvector to (N, 3) displacement field
2. Compute RMS displacement in each direction: `d_x`, `d_y`, `d_z`
3. Compute angular momentum about the longitudinal axis (y-axis) for torsional detection: `L = sum(r x v)` where `r` is the radial position and `v` is the displacement
4. Classification rules:
   - **Longitudinal**: `d_y / (d_x + d_y + d_z) > 0.7` AND angular momentum ratio < 0.1
   - **Flexural**: `max(d_x, d_z) / (d_x + d_y + d_z) > 0.5` AND angular momentum ratio < 0.2
   - **Torsional**: angular momentum ratio > 0.3
   - **Coupled**: does not meet any single-mode criterion

Parasitic mode detection:
- For each non-longitudinal mode within `min_separation_hz` of the target frequency, generate a warning with the mode shape, frequency, and separation distance
- Industry rule of thumb: separation should be > 1000 Hz for 20 kHz horns (5% of operating frequency)

Nodal plane identification:
- Along the longitudinal axis, sample the axial (y) displacement at cross-sections
- Find zero-crossings using linear interpolation between mesh slices
- Report nodal plane positions in mm from the input face

**2f -- Web API Integration (`fea_service.py`, modify ~100 LOC)**

Add a `use_gmsh: bool = False` flag to `run_modal_analysis`. When `True`, route through the new Gmsh + TET10 + SolverA pipeline. When `False`, use the existing HEX8 code path. This enables gradual migration -- the frontend can offer a toggle, and validation tests can compare both paths.

```python
def run_modal_analysis(self, ..., use_gmsh: bool = False) -> dict:
    if use_gmsh:
        return self._run_modal_gmsh(...)  # New pipeline
    else:
        return self._run_modal_legacy(...)  # Existing HEX8 pipeline
```

#### 2.3 Validation Criteria

| Test | Geometry | Expected Result | Tolerance |
|------|----------|-----------------|-----------|
| Uniform bar, free-free, 1st longitudinal | Ti-6Al-4V, L=130.8mm, D=25mm | f1 = c/(2L) = 19,440 Hz (c = sqrt(E/rho) = 5,087 m/s) | < 1% error |
| Uniform bar, free-free, 1st longitudinal | Al 7075-T6, L=126.6mm, D=25mm | f1 = c/(2L) = 19,986 Hz (c = 5,057 m/s) | < 1% error |
| Uniform bar, free-free, 1st longitudinal | Steel D2, L=131.5mm, D=25mm | f1 = c/(2L) = 19,885 Hz (c = 5,229 m/s) | < 1% error |
| Stepped horn, known Branson benchmark | Ti-6Al-4V, D1=50mm, D2=25mm, L=130mm | f1 within 2% of published Branson data | < 2% error |
| Mode classification: 1st mode of uniform cylinder | -- | Classified as "longitudinal" | Exact |
| Mode classification: 2nd mode of uniform cylinder | -- | Classified as "flexural" | Exact |
| Parasitic mode detection: cylinder with nearby flexural | D=25mm, L=130mm | Warning generated with correct flexural freq | Within 2% |
| Nodal plane: half-wave horn | -- | Single nodal plane at L/2 | < 1mm error |
| Mesh convergence: 3 refinement levels | Coarse/medium/fine | Frequencies converge monotonically | Confirmed |

**Key analytical validation formula**:

For a uniform bar in free-free vibration, the n-th longitudinal natural frequency is:
```
f_n = n * c / (2 * L)    where c = sqrt(E / rho)
```

For n=1, this gives the fundamental half-wave resonance. This is the primary validation benchmark because it has an exact analytical solution and is the direct analog of a simple cylindrical horn operating at its tuned frequency.

#### 2.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/assembler.py` | ~350 | -- |
| `fea/solver_a.py` | ~600 | -- |
| `fea/mode_classifier.py` | ~400 | -- |
| `web/services/fea_service.py` | -- | ~100 |
| `tests/test_fea/test_assembler.py` | ~200 | -- |
| `tests/test_fea/test_solver_a.py` | ~250 | -- |
| `tests/test_fea/test_modal_validation.py` | ~350 | -- |
| **Total** | **~2,150** | **~100** |

#### 2.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `eigsh` convergence failure for ill-conditioned K/M | Medium | High | Implement fallback: (1) try different `sigma` shifts, (2) use `ARPACK` with mode="cayley", (3) convert to dense and use `scipy.linalg.eigh` for small problems |
| Assembly too slow for large meshes (>50k nodes) | Medium | Medium | Profile and vectorize; cap mesh size for web requests with configurable limit |
| Free-free rigid body modes contaminate results | Low | High | Robust rigid-body filter: discard modes where `freq < max(100, f_target * 0.01)` Hz |
| Mode misclassification for complex horn shapes | Medium | Medium | Validate against known horn designs; allow user override of classification |

#### 2.6 Demoable at End of Phase

- **Web demo**: Hit `POST /api/v1/geometry/fea/run` with `use_gmsh=true` for a 20 kHz titanium cylindrical horn. Response includes mode shapes with correct classifications, frequencies within 1% of analytical, and 3D visualization mesh.
- **Validation report**: Auto-generated table comparing computed vs. analytical frequencies for 3 materials, printed to console and saved as JSON.
- **Parasitic mode demo**: Analyze a horn geometry known to have a nearby flexural mode; demonstrate the warning is generated.

---

### Phase 3: Harmonic Response + Amplitude (Weeks 5-6)

**Goal**: Compute frequency response functions, amplitude gain, and contact-face amplitude uniformity. This enables the core design feedback loop: "does this horn deliver uniform amplitude at the weld surface?"

**Dependencies**: Phase 2 (modal solver, assembly)

#### 3.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 3a | Harmonic response solver | `fea/solver_a.py` | **Modify** (add `harmonic_analysis`) |
| 3b | Q-factor / hysteretic damping model | `fea/damping.py` | **Create** |
| 3c | Amplitude gain calculator | `fea/post_processing.py` | **Create** |
| 3d | Amplitude uniformity analyzer | `fea/post_processing.py` | (same file) |
| 3e | Frequency response curve generator | `fea/post_processing.py` | (same file) |
| 3f | Web API: harmonic endpoint enhancement | `web/routers/acoustic.py`, `web/services/fea_service.py` | **Modify** |
| 3g | Validation tests | `tests/test_fea/test_harmonic_validation.py` | **Create** |

#### 3.2 Detailed Specifications

**3a -- Harmonic Response Solver (add to `solver_a.py`, ~300 LOC)**

Two approaches, configurable:

**Direct method** (default for narrow-band sweeps, < 50 frequency points):
```
Z(omega) = K - omega^2 * M + i * eta * K    (hysteretic damping)
           or K - omega^2 * M + i*omega*C    (viscous/Rayleigh damping)
u(omega) = Z^{-1} * F
```
Solved using `scipy.sparse.linalg.spsolve` at each frequency point.

**Modal superposition** (for wide-band sweeps, > 50 frequency points):
```
u(omega) = sum_r [ phi_r * phi_r^T * F ] / [ omega_r^2 - omega^2 + i * eta * omega_r^2 ]
```
Uses the eigenvalues/eigenvectors from Phase 2 modal analysis. Much faster for many frequency points because the modal basis is pre-computed.

**3b -- Damping Models (`damping.py`, ~200 LOC)**

```python
class DampingModel(ABC):
    @abstractmethod
    def get_damping_matrix(self, K, M, omega) -> sparse.csr_matrix | complex: ...

class HystereticDamping(DampingModel):
    """Constant loss factor: C_eff = i * eta * K"""
    def __init__(self, eta: float = 0.005): ...

class RayleighDamping(DampingModel):
    """C = alpha*M + beta*K, fitted to two target frequencies."""
    def __init__(self, zeta: float = 0.01, f1: float = 18000, f2: float = 22000): ...

class ModalDamping(DampingModel):
    """Per-mode damping ratios."""
    def __init__(self, zeta_per_mode: np.ndarray): ...
```

For ultrasonic horns, hysteretic (constant loss factor) damping is the most physically appropriate because the loss factor of metals is approximately frequency-independent in the 15-40 kHz range. Typical values: Ti-6Al-4V eta = 0.002-0.005, steel eta = 0.003-0.008, aluminum eta = 0.001-0.003.

**3c/3d/3e -- Post-Processing (`post_processing.py`, ~400 LOC)**

```python
class AmplitudeAnalyzer:
    def compute_gain(
        self,
        harmonic_result: HarmonicResult,
        input_node_set: str = "bottom_face",
        output_node_set: str = "top_face",
    ) -> float:
        """Amplitude gain = mean(|u_output|) / mean(|u_input|) at resonance."""

    def compute_uniformity(
        self,
        harmonic_result: HarmonicResult,
        face_node_set: str = "top_face",
    ) -> tuple[float, np.ndarray]:
        """Returns (uniformity_ratio, per_node_amplitudes).
        uniformity = 1 - std(amplitudes)/mean(amplitudes)
        Target: > 0.95 for good welding."""

    def frequency_response_curve(
        self,
        harmonic_result: HarmonicResult,
        response_dof: int | str = "top_face_y_mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (frequencies, amplitudes) for plotting FRF."""

    def find_resonance_peak(
        self,
        frf_freqs: np.ndarray,
        frf_amps: np.ndarray,
    ) -> tuple[float, float, float]:
        """Returns (peak_freq, peak_amplitude, Q_factor).
        Q = f_peak / bandwidth_3dB"""
```

#### 3.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| Uniform cylinder at resonance: gain = 1.0 | Gain ~ 1.0 (no taper = no amplification) | < 5% |
| Exponential horn (area ratio 4:1): gain ~ 2.0 | Gain ~ 2.0 | < 10% |
| Stepped horn (area ratio 4:1): gain ~ 2.0 | Gain ~ 2.0 | < 10% |
| Q-factor of Ti-6Al-4V horn (eta=0.003) | Q ~ 1/(2*eta) ~ 167 | Within 20% |
| FRF peak frequency matches modal frequency | f_peak == f_modal | < 0.1% |
| Uniformity of cylindrical horn: > 0.99 | Near-perfect uniformity (axisymmetric) | > 0.98 |
| Uniformity of blade horn: < 0.95 | Poor uniformity expected | Qualitative check |

#### 3.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/solver_a.py` | ~300 | -- |
| `fea/damping.py` | ~200 | -- |
| `fea/post_processing.py` | ~400 | -- |
| `web/routers/acoustic.py` | -- | ~50 |
| `web/services/fea_service.py` | -- | ~100 |
| `tests/test_fea/test_harmonic_validation.py` | ~300 | -- |
| **Total** | **~1,200** | **~150** |

#### 3.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `spsolve` with complex matrix is slow | Medium | Medium | Use modal superposition for > 50 freq points; use `scipy.sparse.linalg.splu` factorization for repeated solves near resonance |
| Damping value uncertainty (eta not known precisely) | High | Medium | Default to literature values; expose as user-configurable parameter; sensitivity analysis in output |
| Amplitude uniformity metric is sensitive to mesh resolution at contact face | Medium | Medium | Require minimum 4 elements across contact face width; interpolate to regular grid before computing statistics |

#### 3.6 Demoable at End of Phase

- **FRF plot**: Generate a frequency response curve (16-24 kHz) for a 20 kHz titanium horn, showing a clear resonance peak. Export as JSON data suitable for Chart.js or Plotly frontend rendering.
- **Gain + Uniformity report**: For an exponential horn, report computed gain and uniformity ratio. Compare gain against the analytical area-ratio formula.
- **Damping comparison**: Show FRF for the same horn with three different damping values (eta = 0.001, 0.005, 0.01) to demonstrate the effect on peak sharpness and Q-factor.

---

### Phase 4: Static Stress + Fatigue (Weeks 7-8)

**Goal**: Compute static stress distribution, Von Mises stress at integration points, fatigue life assessment with S-N curves, and pre-stressed modal analysis for bolt preload effects.

**Dependencies**: Phase 2 (assembly, solver), Phase 3 (harmonic results for cyclic stress extraction)

#### 4.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 4a | Static stress solver | `fea/solver_a.py` | **Modify** (add `static_analysis`) |
| 4b | Von Mises at integration points | `fea/stress_recovery.py` | **Create** |
| 4c | Fatigue assessment module | `fea/fatigue.py` | **Create** |
| 4d | Stress concentration factor calculator | `fea/stress_recovery.py` | (same file) |
| 4e | Pre-stressed modal analysis | `fea/solver_a.py` | **Modify** |
| 4f | Web API: stress + fatigue endpoints | `web/routers/acoustic.py`, `web/services/fea_service.py` | **Modify** |
| 4g | Validation and tests | `tests/test_fea/test_stress_validation.py`, `tests/test_fea/test_fatigue.py` | **Create** |

#### 4.2 Detailed Specifications

**4a -- Static Stress Solver (add to `solver_a.py`, ~200 LOC)**

```python
def static_analysis(self, config: StaticConfig) -> StaticResult:
    """Solve K * u = F with boundary conditions."""
```

Supports load types:
- `"pressure"`: Uniform pressure on a node set (converted to consistent nodal forces via surface integration)
- `"force"`: Concentrated force at specific nodes
- `"bolt_preload"`: Axial force on a cylindrical surface (for bolt holes)
- `"gravity"`: Body force from self-weight

Boundary conditions:
- `"fixed"`: All 3 DOFs = 0 at specified nodes
- `"symmetry"`: Normal DOF = 0 at symmetry plane
- `"nodal_plane_support"`: Fix axial DOF at the nodal plane location (the standard mounting condition for sonotrodes)

**4b -- Stress Recovery (`stress_recovery.py`, ~400 LOC)**

```python
class StressRecovery:
    def compute_element_stress(
        self,
        mesh: FEAMesh,
        displacement: np.ndarray,
        material: dict,
        at_gauss_points: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (von_mises_stress, full_stress_tensor).
        von_mises: (n_elements * n_gauss_points,) if at_gauss_points
                   (n_elements,) if averaged at centroids.
        stress_tensor: (n_points, 6) in Voigt notation [sxx, syy, szz, sxy, syz, sxz]."""

    def extrapolate_to_nodes(
        self,
        element_stress: np.ndarray,
        mesh: FEAMesh,
    ) -> np.ndarray:
        """Extrapolate Gauss-point stresses to nodes using superconvergent patch recovery (SPR)."""

    def find_stress_hotspots(
        self,
        von_mises: np.ndarray,
        mesh: FEAMesh,
        n_hotspots: int = 10,
    ) -> list[dict]:
        """Return top stress hotspots with location, stress value, and surrounding context."""

    def stress_concentration_factor(
        self,
        von_mises: np.ndarray,
        mesh: FEAMesh,
        nominal_stress: float,
    ) -> float:
        """Kt = sigma_max / sigma_nominal"""
```

The improvement over the existing `_compute_stress_hotspots` in `fea_service.py` is significant: the current code evaluates strain/stress only at the element centroid (1 point per element). The new code evaluates at all 4 Gauss points of TET10 elements, then optionally performs superconvergent patch recovery (SPR) to extrapolate to nodes for smooth stress contour plots.

**4c -- Fatigue Assessment (`fatigue.py`, ~350 LOC)**

```python
class FatigueAnalyzer:
    # Built-in S-N curve data
    SN_CURVES: dict[str, dict] = {
        "Ti-6Al-4V": {
            "sigma_f": 1200e6,    # fatigue strength coefficient [Pa]
            "b": -0.095,          # Basquin exponent
            "sigma_e": 510e6,     # endurance limit [Pa] at 1e7 cycles
            "R": -1,              # stress ratio for the S-N data
        },
        "Steel D2": {
            "sigma_f": 2100e6,
            "b": -0.08,
            "sigma_e": 750e6,
            "R": -1,
        },
        "Aluminum 7075-T6": {
            "sigma_f": 700e6,
            "b": -0.12,
            "sigma_e": 159e6,
            "R": -1,
        },
    }

    def goodman_safety_factor(
        self,
        sigma_a: float,       # alternating stress amplitude [Pa]
        sigma_m: float,       # mean stress [Pa]
        sigma_e: float,       # endurance limit [Pa]
        sigma_uts: float,     # ultimate tensile strength [Pa]
    ) -> float:
        """Goodman diagram safety factor: SF = 1 / (sigma_a/sigma_e + sigma_m/sigma_uts)"""

    def assess_fatigue(
        self,
        harmonic_result: HarmonicResult,
        static_result: StaticResult | None,  # for mean stress from preload
        material: dict,
        operating_frequency_hz: float,
        target_life_hours: float = 1000.0,  # design life
    ) -> FatigueResult:
        """Full fatigue assessment combining dynamic and static stresses."""

    def sn_life(
        self,
        sigma_a: float,
        material_name: str,
    ) -> float:
        """Predicted cycles to failure from S-N curve."""
```

Fatigue assessment workflow:
1. Extract cyclic stress amplitude from harmonic analysis: `sigma_a = |sigma_harmonic|` at each element
2. Extract mean stress from static analysis (bolt preload, gravity): `sigma_m = sigma_static`
3. Compute Goodman-corrected equivalent stress: `sigma_eq = sigma_a / (1 - sigma_m/sigma_uts)`
4. Compare against endurance limit with safety factor
5. For elements exceeding endurance limit, compute predicted life from S-N curve
6. Report minimum safety factor, critical location, and estimated life

**4e -- Pre-Stressed Modal Analysis (add to `solver_a.py`, ~150 LOC)**

```python
def prestressed_modal_analysis(
    self,
    config: ModalConfig,
    preload_result: StaticResult,
) -> ModalResult:
    """Modal analysis with geometric stiffness from preload.

    K_eff = K_linear + K_geometric(sigma_preload)
    (K_eff - omega^2 * M) * phi = 0
    """
```

This is important for analyzing the effect of bolt preload on the natural frequencies of the horn. Bolt preload introduces a static stress field that modifies the effective stiffness (stress stiffening). For ultrasonic horns, bolt preload typically raises the resonant frequency by 50-200 Hz.

The geometric stiffness matrix `K_geometric` is computed from the pre-stress field using the standard Green-Lagrange formulation for TET10 elements.

#### 4.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| Cylinder under uniform tension: Von Mises = applied stress | Exact | < 0.1% |
| Cylinder with step change (Kt known): SCF matches Peterson's | Published Kt | < 5% |
| Ti-6Al-4V horn at 20 kHz, 50 um amplitude: fatigue SF > 2 | Typical for well-designed horn | Qualitative sanity check |
| Bolt preload 30 kN on M12: frequency shift ~ +100-200 Hz | Published range for similar horns | Qualitative |
| S-N curve for Ti-6Al-4V: N=1e7 at sigma_e=510 MPa | Exact from curve data | Exact |

#### 4.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/stress_recovery.py` | ~400 | -- |
| `fea/fatigue.py` | ~350 | -- |
| `fea/solver_a.py` | ~350 | -- |
| `web/routers/acoustic.py` | -- | ~80 |
| `web/services/fea_service.py` | -- | ~120 |
| `tests/test_fea/test_stress_validation.py` | ~250 | -- |
| `tests/test_fea/test_fatigue.py` | ~200 | -- |
| **Total** | **~1,550** | **~200** |

#### 4.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Geometric stiffness matrix implementation is complex for TET10 | Medium | High | Use published formulation (Bathe 2014, Ch.6); validate with known benchmark (Euler column buckling) |
| S-N data uncertainty for specific alloy heat treatments | High | Medium | Use conservative data from MMPDS/Metallic Materials Properties; allow user to input custom S-N parameters |
| SPR stress extrapolation at boundaries/singularities | Medium | Medium | Use simple averaging at boundary nodes; flag singularity locations (sharp corners) as unreliable |

#### 4.6 Demoable at End of Phase

- **Stress contour map**: For a stepped horn under harmonic excitation, generate a per-element Von Mises stress map. Export as JSON color-mapped data for 3D rendering.
- **Fatigue report**: For a 20 kHz titanium horn at 50 um amplitude, output a fatigue assessment: minimum safety factor, critical location, estimated life in hours.
- **Pre-stressed modal**: Show the frequency shift caused by bolt preload (10, 20, 30 kN) in a table.

---

### Phase 5: Booster + Assembly (Weeks 9-10)

**Goal**: Extend the system to model boosters (half-wave transformers) and full transducer+booster+horn assemblies. Enable gain chain verification and full-stack modal analysis.

**Dependencies**: Phase 1 (mesher), Phase 2 (modal), Phase 3 (harmonic), Phase 4 (stress)

#### 5.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 5a | Booster parametric geometry | `fea/booster_generator.py` | **Create** |
| 5b | Multi-body assembly builder | `fea/assembly_builder.py` | **Create** |
| 5c | Interface coupling (bonded/tied) | `fea/assembly_builder.py` | (same file) |
| 5d | Full-stack analysis workflow | `fea/workflow.py` | **Create** |
| 5e | Gain chain verification | `fea/post_processing.py` | **Modify** |
| 5f | Web API: assembly endpoints | `web/routers/assembly.py` | **Create** |
| 5g | Tests | `tests/test_fea/test_booster.py`, `tests/test_fea/test_assembly.py` | **Create** |

#### 5.2 Detailed Specifications

**5a -- Booster Generator (`booster_generator.py`, ~300 LOC)**

```python
class BoosterGenerator:
    """Generate parametric booster (half-wave transformer) geometries."""

    def generate_stepped_booster(
        self,
        d_input: float,         # Input diameter (mm)
        d_output: float,        # Output diameter (mm)
        total_length: float,    # Total length (mm), must be ~lambda/2
        material: str = "Titanium Ti-6Al-4V",
        frequency_hz: float = 20000,
    ) -> "FEAMesh":
        """Stepped booster: two cylinders of different diameter.
        Gain = (D_input / D_output)^2 (area ratio)."""

    def generate_exponential_booster(
        self,
        d_input: float,
        d_output: float,
        total_length: float,
        material: str = "Titanium Ti-6Al-4V",
        frequency_hz: float = 20000,
    ) -> "FEAMesh":
        """Exponential booster: smooth taper following exp(-alpha*x).
        Gain = D_input / D_output."""

    def generate_catenoidal_booster(
        self,
        d_input: float,
        d_output: float,
        total_length: float,
        material: str = "Titanium Ti-6Al-4V",
        frequency_hz: float = 20000,
    ) -> "FEAMesh":
        """Catenoidal booster: cosh profile, lowest stress concentration."""

    def auto_length(
        self,
        material: str,
        frequency_hz: float,
    ) -> float:
        """Compute half-wavelength length: L = c / (2*f)."""
```

Standard booster types in the ultrasonic welding industry:
- **Stepped (1:1.5, 1:2, 1:2.5)**: Most common, sharp diameter change at the nodal plane
- **Exponential**: Smooth taper, lower stress concentration than stepped
- **Catenoidal**: Cosh profile, theoretically optimal stress distribution

**5b/5c -- Assembly Builder (`assembly_builder.py`, ~500 LOC)**

```python
class AssemblyBuilder:
    """Build multi-component assemblies with interface coupling."""

    def add_component(
        self,
        name: str,
        mesh: FEAMesh,
        material: dict,
        position: np.ndarray = np.zeros(3),  # [x, y, z] offset
    ) -> None:

    def add_interface(
        self,
        component_a: str,
        node_set_a: str,
        component_b: str,
        node_set_b: str,
        coupling_type: str = "bonded",  # "bonded" | "tied" | "contact"
    ) -> None:

    def build(self) -> tuple[FEAMesh, dict]:
        """Merge component meshes and apply interface constraints.
        Returns (merged_mesh, material_map) where material_map maps
        element ranges to material properties."""
```

Interface coupling implementation:
- **Bonded** (default): Merge coincident nodes at the interface. For non-matching meshes, use multi-point constraints (MPC) via Lagrange multipliers or penalty method.
- **Tied**: Same as bonded but implemented via penalty springs to allow slight relative motion. Penalty stiffness = 100 * max(K_diagonal).
- **Contact** (Phase 7): Full contact with friction, only in FEniCSx solver.

For the typical transducer+booster+horn stack:
1. Transducer output face is bonded to booster input face
2. Booster output face is bonded to horn input face (stud connection)
3. The assembly is analyzed as a single coupled system

**5d -- Full-Stack Workflow (`workflow.py`, ~300 LOC)**

```python
class FullStackWorkflow:
    """End-to-end analysis workflow for transducer+booster+horn."""

    def run_full_analysis(
        self,
        horn_mesh: FEAMesh,
        horn_material: dict,
        booster_mesh: FEAMesh | None = None,
        booster_material: dict | None = None,
        transducer_mesh: FEAMesh | None = None,
        transducer_material: dict | None = None,
        frequency_hz: float = 20000,
        amplitude_um: float = 50,
    ) -> dict:
        """Run the complete analysis chain:
        1. Assemble components
        2. Modal analysis (free-free)
        3. Harmonic response at operating frequency
        4. Stress analysis
        5. Fatigue assessment
        6. Gain chain verification

        Returns a comprehensive report dict."""

    def verify_gain_chain(
        self,
        harmonic_result: HarmonicResult,
        assembly_info: dict,
    ) -> dict:
        """Verify the amplitude gain at each interface:
        - Transducer output amplitude
        - Booster input/output ratio (should match design gain)
        - Horn input/output ratio (should match design gain)
        - Total system gain = booster_gain * horn_gain
        """
```

#### 5.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| Stepped booster (D50:D25, Ti): gain = 4.0 | Area ratio gain | < 10% |
| Catenoidal booster: lower max stress than stepped | Qualitative comparison | Stress ratio < 0.7 |
| Full assembly (booster 1:2 + horn 1:1): total gain ~ 2.0 | Sum of component gains | < 15% |
| Assembly modal: first longitudinal mode near target freq | 20,000 Hz for properly tuned stack | < 1% |
| Gain chain: booster gain * horn gain = total measured gain | Consistency check | < 5% |

#### 5.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/booster_generator.py` | ~300 | -- |
| `fea/assembly_builder.py` | ~500 | -- |
| `fea/workflow.py` | ~300 | -- |
| `fea/post_processing.py` | -- | ~150 |
| `web/routers/assembly.py` | ~200 | -- |
| `web/services/fea_service.py` | -- | ~150 |
| `tests/test_fea/test_booster.py` | ~200 | -- |
| `tests/test_fea/test_assembly.py` | ~300 | -- |
| **Total** | **~1,800** | **~300** |

#### 5.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Non-matching meshes at interfaces are hard to couple | High | High | Phase 1: require matching meshes at interfaces (Gmsh can generate matching boundaries). Phase 2: add MPC for non-matching later. |
| Assembly mesh too large for single scipy solve | Medium | High | Target < 100k DOFs for assembly; use mesh coarsening far from interfaces; provide solver time estimate before running |
| Booster length auto-tuning iterates too many times | Low | Medium | Use analytical formula for initial guess; limit Newton iterations to 10 |

#### 5.6 Demoable at End of Phase

- **Full stack demo**: Analyze a complete 20 kHz stack (transducer equivalent mass + booster 1:2 + cylindrical horn). Show mode shapes, frequency, gain at each interface, and stress distribution.
- **Booster design tool**: Input desired gain ratio and frequency; output optimal booster dimensions for stepped/exponential/catenoidal profiles.
- **Gain chain report**: Table showing amplitude at each interface point, with pass/fail against design targets.

---

### Phase 6: FEniCSx Plugin (Weeks 11-12)

**Goal**: Implement Solver B using FEniCSx for piezoelectric coupled analysis, impedance spectrum computation, and cross-validation against Solver A.

**Dependencies**: Phase 2 (modal, for cross-validation), Phase 5 (assembly, for transducer modeling)

#### 6.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 6a | Docker deployment configuration | `docker/Dockerfile.fenics`, `docker/docker-compose.yml` | **Create** |
| 6b | FEniCSx solver implementation | `fea/solver_b.py` | **Create** |
| 6c | Piezoelectric element formulation | `fea/solver_b.py` | (same file) |
| 6d | Impedance spectrum calculator | `fea/solver_b.py` | (same file) |
| 6e | Cross-validation harness (A vs B) | `fea/cross_validation.py` | **Create** |
| 6f | Web API: solver selection + impedance endpoint | `web/routers/acoustic.py`, `web/services/fea_service.py` | **Modify** |
| 6g | Tests | `tests/test_fea/test_solver_b.py`, `tests/test_fea/test_cross_validation.py` | **Create** |

#### 6.2 Detailed Specifications

**6a -- Docker Setup (`docker/`, ~100 LOC config)**

```dockerfile
# docker/Dockerfile.fenics
FROM dolfinx/dolfinx:v0.8.0
RUN pip install gmsh meshio numpy scipy
COPY ultrasonic_weld_master/plugins/geometry_analyzer/fea/ /app/fea/
COPY web/ /app/web/
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8001"]
```

```yaml
# docker/docker-compose.yml
services:
  solver-b:
    build:
      context: .
      dockerfile: docker/Dockerfile.fenics
    ports:
      - "8001:8001"
    volumes:
      - ./data:/app/data
    environment:
      - SOLVER_MODE=fenics
```

FEniCSx runs in its own Docker container because it has complex native dependencies (PETSc, MPI, HDF5) that are difficult to install natively on macOS. The main application communicates with the FEniCSx solver via HTTP API or direct Python import when running inside the same container.

**6b/6c -- FEniCSx Solver (`solver_b.py`, ~700 LOC)**

```python
class SolverB(SolverInterface):
    """FEniCSx-based solver with piezoelectric coupling."""

    def __init__(self, mode: str = "local"):
        """mode: "local" (direct import) or "remote" (HTTP to Docker container)"""

    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Modal analysis using SLEPc eigensolver via FEniCSx."""

    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Harmonic response using PETSc complex solver."""

    def piezoelectric_modal(
        self,
        config: ModalConfig,
        piezo_material: dict,   # PZT-4 or PZT-8 properties
        electrode_nodes: dict,  # {"top": node_set, "bottom": node_set}
    ) -> ModalResult:
        """Coupled electromechanical modal analysis.
        Solves: [K_uu  K_uv] [u]     [M 0] [u]
                [K_vu  K_vv] [V] = w^2[0 0] [V]
        where u = displacement, V = electric potential."""

    def impedance_spectrum(
        self,
        config: HarmonicConfig,
        piezo_material: dict,
        electrode_nodes: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute electrical impedance Z(f) = V(f) / I(f) over frequency range.
        Returns (frequencies, impedances) for impedance analyzer comparison."""
```

The piezoelectric formulation couples the mechanical displacement field `u` with the electric potential field `V` through the constitutive equations:
```
T = c^E * S - e^T * E     (stress = stiffness*strain - piezo_coupling*electric_field)
D = e * S + eps^S * E      (electric_displacement = piezo_coupling*strain + permittivity*electric_field)
```
where `c^E` is the elastic stiffness at constant electric field, `e` is the piezoelectric coupling matrix, and `eps^S` is the permittivity at constant strain.

This requires a mixed FE formulation with 4 DOFs per node in the piezoelectric region (3 displacement + 1 electric potential) and 3 DOFs per node elsewhere. FEniCSx handles this naturally through its `MixedElement` functionality.

**6d -- Impedance Spectrum**

The impedance spectrum is the primary experimental measurement used to tune transducers. It shows:
- **Resonance frequency (f_r)**: Minimum impedance (series resonance)
- **Anti-resonance frequency (f_a)**: Maximum impedance (parallel resonance)
- **Effective coupling coefficient**: `k_eff^2 = 1 - (f_r/f_a)^2`
- **Mechanical Q-factor**: From the sharpness of the impedance minimum

Computing this from FEA allows direct comparison with impedance analyzer measurements, which is the gold standard for transducer validation.

**6e -- Cross-Validation (`cross_validation.py`, ~250 LOC)**

```python
class CrossValidator:
    """Compare Solver A and Solver B results for the same problem."""

    def compare_modal(
        self,
        result_a: ModalResult,
        result_b: ModalResult,
        tolerance_freq_percent: float = 1.0,
    ) -> dict:
        """Compare eigenfrequencies and mode shapes.
        Returns {freq_errors, mac_matrix, passed}."""

    def modal_assurance_criterion(
        self,
        modes_a: np.ndarray,
        modes_b: np.ndarray,
    ) -> np.ndarray:
        """MAC matrix: MAC(i,j) = |phi_a_i . phi_b_j|^2 / (|phi_a_i|^2 * |phi_b_j|^2)
        Diagonal should be > 0.9 for matching modes."""
```

#### 6.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| SolverA vs SolverB modal frequencies (same mesh) | Same eigenfrequencies | < 0.5% difference |
| MAC matrix diagonal (SolverA vs SolverB) | > 0.95 for first 10 modes | All > 0.90 |
| PZT-4 transducer impedance: f_r near design frequency | Known from datasheet | < 2% |
| k_eff for PZT-4 disk | ~0.33 (published for PZT-4) | < 10% |
| Docker container starts and responds to health check | HTTP 200 on /health | Pass/fail |

#### 6.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `docker/Dockerfile.fenics` | ~30 | -- |
| `docker/docker-compose.yml` | ~30 | -- |
| `fea/solver_b.py` | ~700 | -- |
| `fea/cross_validation.py` | ~250 | -- |
| `web/routers/acoustic.py` | -- | ~80 |
| `web/services/fea_service.py` | -- | ~100 |
| `tests/test_fea/test_solver_b.py` | ~300 | -- |
| `tests/test_fea/test_cross_validation.py` | ~200 | -- |
| **Total** | **~1,510** | **~180** |

#### 6.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| FEniCSx API changes between versions | Medium | High | Pin to `dolfinx==0.8.0` in Docker; abstract all FEniCSx calls behind `SolverB` class |
| PETSc complex number build required for harmonic | Medium | Medium | Use `dolfinx/dolfinx:v0.8.0` image which includes complex PETSc build |
| Docker not available on all target machines | High | Medium | SolverB is optional; SolverA handles all non-piezoelectric analysis; detect Docker availability at startup |
| Piezoelectric formulation debugging is time-intensive | High | High | Start with known benchmark (PZT-4 disk, published in IEEE UFFC); implement and validate the simplest case first |

#### 6.6 Demoable at End of Phase

- **Docker one-liner**: `docker-compose up solver-b` starts the FEniCSx solver container.
- **Cross-validation report**: Analyze a 20 kHz titanium horn with both solvers, produce a comparison table showing frequency agreement and MAC values.
- **Impedance spectrum plot**: For a PZT-4 transducer disk, generate the impedance vs. frequency curve showing clear resonance and anti-resonance peaks.

---

### Phase 7: Advanced Features + Polish (Weeks 13-14)

**Goal**: Add thermal coupling, contact analysis, STEP import for arbitrary geometry, mesh convergence automation, and the frontend 3D viewer and interactive charts.

**Dependencies**: All previous phases

#### 7.1 Deliverables

| # | Deliverable | File(s) | Action |
|---|-------------|---------|--------|
| 7a | Thermal coupling (SolverB) | `fea/solver_b.py` | **Modify** |
| 7b | Contact analysis (SolverB) | `fea/solver_b.py` | **Modify** |
| 7c | STEP import for arbitrary geometry | `fea/mesher.py` | **Modify** (enhance `mesh_from_step`) |
| 7d | Mesh convergence automation | `fea/mesh_convergence.py` | **Create** |
| 7e | Frontend: 3D viewer component | `frontend/src/components/FEAViewer3D.vue` | **Create** |
| 7f | Frontend: FRF/impedance charts | `frontend/src/components/FrequencyResponseChart.vue` | **Create** |
| 7g | Frontend: assembly builder UI | `frontend/src/views/AssemblyView.vue` | **Create** |
| 7h | Legacy migration completion | `web/services/fea_service.py` | **Modify** (remove old HEX8 code) |
| 7i | Integration tests | `tests/test_fea/test_integration.py` | **Create** |

#### 7.2 Detailed Specifications

**7a -- Thermal Coupling (add to `solver_b.py`, ~250 LOC)**

```python
def thermomechanical_analysis(
    self,
    config: ThermalConfig,
    heat_generation: np.ndarray,  # per-element heat generation rate [W/m^3]
) -> ThermalResult:
    """Coupled thermal-mechanical analysis.
    1. Solve heat equation: div(k * grad(T)) + Q = 0
    2. Apply thermal expansion: epsilon_thermal = alpha * (T - T_ref)
    3. Solve mechanical with thermal strain as initial strain.
    """
```

Heat generation in ultrasonic welding comes from:
- Hysteretic damping: `Q = pi * f * eta * sigma_max^2 / E` [W/m^3]
- Interface friction: modeled as a surface heat source at the weld interface

This is solved as a one-way coupled problem: mechanical vibration generates heat, heat changes material properties and introduces thermal stress, but the thermal field does not significantly change the vibration pattern at steady-state.

**7b -- Contact Analysis (add to `solver_b.py`, ~300 LOC)**

```python
def contact_analysis(
    self,
    config: ContactConfig,
    friction_coefficient: float = 0.3,
) -> ContactResult:
    """Nonlinear contact analysis at the weld interface.
    Uses penalty method with augmented Lagrangian.
    Computes contact pressure distribution and slip/stick zones."""
```

Contact analysis at the horn-workpiece interface determines:
- Contact pressure distribution (affects weld quality)
- Slip/stick transitions (affects energy dissipation)
- Effective contact area under dynamic loading

This is a nonlinear problem requiring iterative solution (Newton-Raphson) and is computationally expensive. It is implemented exclusively in FEniCSx (SolverB) because FEniCSx provides built-in support for nonlinear variational problems and contact mechanics.

**7c -- STEP Import Enhancement (modify `mesher.py`, ~100 LOC)**

Enhance the existing `mesh_from_step` to handle:
- Multi-body STEP files (automatically identify separate solids)
- Face identification for boundary condition application (using face normals and positions)
- Automatic mesh refinement at small features (fillets, slots, holes)
- Defeaturing: optional removal of small features below a threshold size

**7d -- Mesh Convergence (`mesh_convergence.py`, ~300 LOC)**

```python
class MeshConvergenceStudy:
    """Automated mesh convergence analysis."""

    def run(
        self,
        geometry_params: dict,
        analysis_type: str = "modal",  # "modal" | "harmonic" | "stress"
        mesh_sizes: list[float] | None = None,  # [4.0, 2.0, 1.0, 0.5] mm
        target_quantity: str = "frequency_hz",
        convergence_threshold: float = 0.005,  # 0.5% relative change
    ) -> MeshConvergenceResult:
        """Run analysis at multiple mesh refinement levels.
        Returns convergence plot data and recommended mesh size."""

@dataclass
class MeshConvergenceResult:
    mesh_sizes: list[float]
    node_counts: list[int]
    target_values: list[float]
    relative_changes: list[float]
    converged: bool
    recommended_mesh_size: float
    richardson_extrapolation: float  # extrapolated "exact" value
```

The convergence study runs the analysis at 3-5 mesh sizes and checks if the result has converged (relative change between last two refinements < threshold). Richardson extrapolation is used to estimate the mesh-independent value.

**7e/7f/7g -- Frontend Components**

**3D Viewer (`FEAViewer3D.vue`, ~500 LOC)**: Three.js-based viewer for displaying mesh, mode shapes (animated), stress contours, and amplitude distribution. Uses the mesh data already returned by the API (`{vertices, faces}` format). Features:
- Orbit/pan/zoom camera controls
- Color mapping for Von Mises stress or displacement magnitude
- Mode shape animation (sinusoidal scaling of displacement field)
- Cross-section cutting plane
- Node/element picking for inspection

**FRF Chart (`FrequencyResponseChart.vue`, ~200 LOC)**: Chart.js or Plotly-based frequency response plot. Displays:
- Amplitude vs. frequency (log scale)
- Phase vs. frequency
- Impedance vs. frequency (from Phase 6)
- Interactive cursor showing frequency, amplitude, and phase at hover position
- Peak markers with Q-factor annotation

**Assembly Builder (`AssemblyView.vue`, ~400 LOC)**: Visual tool for configuring transducer+booster+horn assemblies:
- Drag-and-drop component selection
- Parameter input for each component (dimensions, material)
- Preview of assembled geometry
- "Analyze" button launching the full-stack workflow
- Gain chain visualization showing amplitude at each interface

**7h -- Legacy Migration**

The final step: remove the old HEX8 codepath from `web/services/fea_service.py`. The `use_gmsh` flag added in Phase 2 is now always `True`. All API contracts remain the same -- only the internal implementation changes. The old `_generate_hex_mesh`, `_assemble_global`, `_hex_dshape`, `_apply_bc`, and related methods (lines 566-838 of the current file) are deleted.

#### 7.3 Validation Criteria

| Test | Expected | Tolerance |
|------|----------|-----------|
| Thermal: horn temperature rise after 10s continuous operation | Literature range (10-50 C for Ti) | Qualitative |
| Contact: Hertz contact pressure on cylinder-flat | Analytical Hertz formula | < 5% |
| STEP import: sample horn STEP file meshes correctly | Valid mesh, no inverted elements | Pass/fail |
| Mesh convergence: frequency converges within 4 refinements | Converged flag = true | Pass/fail |
| 3D viewer: renders 50k-node mesh at 30 FPS | Performance benchmark | Pass/fail |
| Full integration: API returns correct schema | Pydantic validation passes | Pass/fail |

#### 7.4 Estimated Lines of Code

| File | New LOC | Modified LOC |
|------|---------|-------------|
| `fea/solver_b.py` | ~550 | -- |
| `fea/mesher.py` | -- | ~100 |
| `fea/mesh_convergence.py` | ~300 | -- |
| `frontend/src/components/FEAViewer3D.vue` | ~500 | -- |
| `frontend/src/components/FrequencyResponseChart.vue` | ~200 | -- |
| `frontend/src/views/AssemblyView.vue` | ~400 | -- |
| `web/services/fea_service.py` | -- | ~200 (remove ~300 old LOC) |
| `tests/test_fea/test_integration.py` | ~400 | -- |
| **Total** | **~2,350** | **~300** |

#### 7.5 Risk Factors and Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Three.js performance with large meshes (>100k triangles) | Medium | Medium | Use instanced rendering; decimate mesh for display (keep full mesh for analysis); use WebGL2 |
| Contact analysis convergence issues | High | Medium | Provide good initial guess from linear analysis; limit Newton iterations; expose convergence params to user |
| Legacy migration breaks existing tests | Low | High | Run all existing tests after migration; keep old code path available via environment variable for 1 release |
| STEP import fails for some CAD systems' output | Medium | Medium | Test with STEP files from SolidWorks, Fusion360, NX, and FreeCAD; handle parser errors gracefully |

#### 7.6 Demoable at End of Phase

- **Full product demo**: End-to-end workflow in the web UI: select components, configure assembly, run analysis, view 3D results with animated mode shapes, inspect FRF, review stress and fatigue report.
- **Mesh convergence report**: Automated study showing convergence of the 1st longitudinal frequency over 4 mesh levels, with recommended mesh size.
- **STEP import**: Upload a real horn STEP file from a customer, mesh it, run modal analysis, show results in 3D viewer.

---

### Cross-Cutting Concerns

#### Testing Strategy

**Unit Tests** (per-phase, in `tests/test_fea/`):

| Phase | Test File | Coverage Target |
|-------|-----------|----------------|
| 1 | `test_elements.py`, `test_mesher.py`, `test_materials.py` | Element formulation, mesh generation, material lookup |
| 2 | `test_assembler.py`, `test_solver_a.py`, `test_modal_validation.py` | Assembly correctness, eigenvalue accuracy, analytical benchmarks |
| 3 | `test_harmonic_validation.py` | FRF shape, gain accuracy, uniformity calculation |
| 4 | `test_stress_validation.py`, `test_fatigue.py` | Stress recovery accuracy, S-N curve correctness, Goodman diagram |
| 5 | `test_booster.py`, `test_assembly.py` | Booster geometry, interface coupling, gain chain |
| 6 | `test_solver_b.py`, `test_cross_validation.py` | FEniCSx results, A-vs-B agreement |
| 7 | `test_integration.py` | End-to-end API tests, STEP import, mesh convergence |

**Integration Tests** (per-phase, cumulative):

```python
# tests/test_fea/test_integration.py
class TestEndToEnd:
    def test_cylindrical_horn_modal(self):
        """Full pipeline: geometry -> mesh -> modal -> results for a standard horn."""

    def test_full_stack_analysis(self):
        """Booster + horn assembly: mesh -> modal -> harmonic -> stress -> fatigue."""

    def test_step_import_to_results(self):
        """STEP file -> mesh -> modal -> validate frequency."""

    def test_api_contract(self):
        """FastAPI endpoint returns correct Pydantic schema."""
```

**Validation Benchmarks** (golden reference data):

Store benchmark results in `tests/test_fea/benchmarks/`:
```
benchmarks/
  uniform_bar_ti64_20khz.json      # Analytical: f = c/(2L)
  uniform_bar_al7075_20khz.json    # Analytical: f = c/(2L)
  uniform_bar_d2_20khz.json        # Analytical: f = c/(2L)
  stepped_horn_branson_20khz.json  # Published manufacturer data
  pzt4_disk_impedance.json         # Published IEEE UFFC data
```

Each benchmark file contains:
```json
{
  "geometry": { ... },
  "material": "...",
  "expected_results": {
    "frequency_hz": 20000,
    "tolerance_percent": 1.0,
    "source": "Analytical: f = sqrt(E/rho) / (2*L)"
  }
}
```

**Test Execution Plan**:
```bash
# Unit tests (fast, no external dependencies)
pytest tests/test_fea/ -v --ignore=tests/test_fea/test_solver_b.py

# Full suite including FEniCSx (requires Docker)
pytest tests/test_fea/ -v

# Validation benchmarks only
pytest tests/test_fea/ -v -m benchmark

# Performance regression test
pytest tests/test_fea/ -v -m performance --benchmark-json=benchmark.json
```

#### Deployment Plan

**Phase 1-5 (SolverA only, no Docker required)**:

```bash
# Server installation
pip install numpy scipy gmsh cadquery meshio
pip install -e .   # Install the package

# Verify Gmsh works headless
python -c "import gmsh; gmsh.initialize(); gmsh.finalize(); print('OK')"

# Start web server
python run_web.py
```

Requirements additions to `pyproject.toml`:
```toml
[project.optional-dependencies]
fea = [
    "gmsh>=4.12",
    "meshio>=5.3",
]
fea-cad = [
    "cadquery>=2.4",
]
```

**Phase 6+ (SolverB with Docker)**:

```bash
# Build and start FEniCSx container
docker-compose -f docker/docker-compose.yml up -d solver-b

# Verify SolverB is accessible
curl http://localhost:8001/api/v1/health

# Main app automatically detects SolverB availability
python run_web.py  # Connects to SolverB at localhost:8001
```

**Production deployment (full stack)**:

```yaml
# docker/docker-compose.prod.yml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - SOLVER_B_URL=http://solver-b:8001
    depends_on:
      - solver-b

  solver-b:
    build:
      context: .
      dockerfile: docker/Dockerfile.fenics
    expose:
      - "8001"
```

#### Migration Plan: Replacing Existing `fea_service.py`

The migration from the current simplified FEA to the production-grade system is **gradual and reversible** at every step.

**Phase 2 (Week 3)**: Add `use_gmsh` flag to API
```python
# web/services/fea_service.py
def run_modal_analysis(self, ..., use_gmsh: bool = False):
    if use_gmsh:
        return self._run_modal_gmsh(...)     # New TET10 pipeline
    else:
        return self._run_modal_legacy(...)   # Existing HEX8 pipeline (unchanged)
```
- Frontend adds a toggle switch: "Use advanced solver"
- Both paths return the same JSON schema
- Old tests continue to pass unchanged

**Phase 3 (Week 5)**: Extend `use_gmsh` to harmonic analysis
```python
def run_acoustic_analysis(self, ..., use_gmsh: bool = False):
    if use_gmsh:
        return self._run_acoustic_gmsh(...)   # New pipeline
    else:
        return self._run_acoustic_legacy(...)  # Existing pipeline
```

**Phase 5 (Week 9)**: New assembly endpoints are added alongside existing ones
- `POST /api/v1/assembly/analyze` (new, assembly-only)
- Existing `POST /api/v1/geometry/fea/run` and `POST /api/v1/acoustic/analyze` unchanged

**Phase 7 (Week 13)**: Default switches to new pipeline
```python
def run_modal_analysis(self, ..., use_gmsh: bool = True):  # Default changed
    ...
```
- Old HEX8 code path kept behind `use_gmsh=False` for one release
- Mark as deprecated in API docs

**Phase 7 (Week 14)**: Old code removed
- Delete methods: `_generate_hex_mesh`, `_assemble_global`, `_hex_dshape`, `_get_fixed_dofs`, `_apply_bc`, `_generate_surface_mesh` (approximately 300 LOC)
- Old `_prepare_model`, `_eigen_solve`, `_classify_modes` refactored to delegate to new classes
- Net effect: `fea_service.py` shrinks from ~838 LOC to ~200 LOC (thin adapter calling into `fea/` module)

**Rollback procedure** at any phase: Set `use_gmsh=False` in the API call or environment variable `FEA_USE_LEGACY=1`. The old code path is preserved and tested until the final removal in Week 14.

#### Total Lines of Code Summary

| Phase | New LOC | Modified LOC | Cumulative New |
|-------|---------|-------------|----------------|
| 1: Foundation | 2,150 | 100 | 2,150 |
| 2: Modal | 2,150 | 100 | 4,300 |
| 3: Harmonic | 1,200 | 150 | 5,500 |
| 4: Stress/Fatigue | 1,550 | 200 | 7,050 |
| 5: Assembly | 1,800 | 300 | 8,850 |
| 6: FEniCSx | 1,510 | 180 | 10,360 |
| 7: Advanced | 2,350 | 300 | 12,710 |
| **Total** | **~12,710** | **~1,330** | -- |

Final `fea/` module structure after all phases:

```
ultrasonic_weld_master/plugins/geometry_analyzer/fea/
  __init__.py
  config.py              # ~150 LOC  -- Analysis configuration dataclasses
  mesher.py              # ~550 LOC  -- Gmsh mesh generation + STEP import
  elements.py            # ~500 LOC  -- TET10 element formulation
  assembler.py           # ~350 LOC  -- Global matrix assembly
  material_properties.py # ~300 LOC  -- Material database (enhanced)
  solver_interface.py    # ~200 LOC  -- Abstract solver interface
  solver_a.py            # ~1,150 LOC -- NumPy/SciPy solver (modal, harmonic, static, prestressed)
  solver_b.py            # ~1,250 LOC -- FEniCSx solver (piezo, thermal, contact)
  results.py             # ~250 LOC  -- Result container dataclasses
  mode_classifier.py     # ~400 LOC  -- Mode shape classification + parasitic detection
  damping.py             # ~200 LOC  -- Damping models
  post_processing.py     # ~550 LOC  -- Gain, uniformity, FRF analysis
  stress_recovery.py     # ~400 LOC  -- Von Mises, SPR, hotspots
  fatigue.py             # ~350 LOC  -- S-N curves, Goodman, safety factors
  booster_generator.py   # ~300 LOC  -- Parametric booster geometries
  assembly_builder.py    # ~500 LOC  -- Multi-body assembly
  workflow.py            # ~300 LOC  -- Full-stack analysis orchestration
  mesh_convergence.py    # ~300 LOC  -- Automated convergence study
  cross_validation.py    # ~250 LOC  -- Solver A vs B comparison
```
