# FEA System API Endpoints & Analysis Scenarios -- Design Document

**Date**: 2026-02-27
**Status**: Draft
**Scope**: REST API design, Pydantic schemas, async pipeline, analysis workflows, frontend integration
**Parent document**: Web Frontend + API Service Design (2026-02-26)

---

## Table of Contents

1. [API Endpoint Design](#1-api-endpoint-design)
2. [Request/Response Schemas (Pydantic Models)](#2-requestresponse-schemas)
3. [Async Analysis Pipeline](#3-async-analysis-pipeline)
4. [Analysis Scenarios (Detailed Workflows)](#4-analysis-scenarios)
5. [Frontend Integration Points](#5-frontend-integration-points)

---

## 1. API Endpoint Design

All endpoints live under the `/api/v1` prefix, consistent with the existing application
(`web/app.py` registers routers with `prefix="/api/v1"`). New routers are added for
components, assemblies, advanced analysis, and the async job system.

### 1.1 Component Management

These endpoints manage persisted component definitions (horn, booster, transducer) that
can be referenced by ID in analysis and assembly operations.

#### Horn CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/components/horns` | Create a horn definition (parametric or from STEP import) |
| `GET` | `/components/horns` | List all saved horn definitions |
| `GET` | `/components/horns/{horn_id}` | Get a single horn definition by ID |
| `PUT` | `/components/horns/{horn_id}` | Update a horn definition |
| `DELETE` | `/components/horns/{horn_id}` | Delete a horn definition |
| `POST` | `/components/horns/generate` | Generate parametric horn geometry (preview, no persist) |
| `POST` | `/components/horns/import-step` | Upload STEP file, extract geometry, create horn definition |
| `GET` | `/components/horns/{horn_id}/mesh` | Get tessellated mesh for 3D viewer |
| `GET` | `/components/horns/{horn_id}/download` | Download STEP or STL file (`?format=step\|stl`) |

#### Booster CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/components/boosters` | Create a booster definition |
| `GET` | `/components/boosters` | List all saved booster definitions |
| `GET` | `/components/boosters/{booster_id}` | Get a single booster definition |
| `PUT` | `/components/boosters/{booster_id}` | Update a booster definition |
| `DELETE` | `/components/boosters/{booster_id}` | Delete a booster definition |
| `POST` | `/components/boosters/generate` | Generate parametric booster geometry (preview) |
| `POST` | `/components/boosters/import-step` | Upload STEP, create booster definition |

#### Transducer CRUD

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/components/transducers` | Create a transducer definition |
| `GET` | `/components/transducers` | List all saved transducer definitions |
| `GET` | `/components/transducers/{transducer_id}` | Get a single transducer definition |
| `PUT` | `/components/transducers/{transducer_id}` | Update a transducer definition |
| `DELETE` | `/components/transducers/{transducer_id}` | Delete a transducer definition |

#### Materials

Extends the existing `/materials` router:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/fea/materials` | List all FEA materials with full mechanical/thermal properties |
| `GET` | `/fea/materials/{name}` | Get single material by name (alias-aware) |

### 1.2 Analysis Endpoints

All analysis endpoints follow a two-phase pattern:

- **Synchronous (small models)**: direct `POST` returning results immediately for mesh sizes under the `SYNC_THRESHOLD` (default: 5000 nodes).
- **Asynchronous (large models)**: `POST` returns a `job_id`; client polls `/jobs/{job_id}` for status and results.

The `solver_backend` field in every request controls which solver runs:
`"solver_a"` (numpy/scipy, default), `"solver_b"` (FEniCSx), or `"cross_validate"` (both).

#### 1.2.1 Modal Analysis

```
POST /api/v1/analysis/modal
```

Run eigenvalue analysis to find natural frequencies and mode shapes.

**Boundary condition variants** (set via `bc_type` field):
- `free_free` -- unconstrained (both ends free)
- `clamped` -- bottom face fixed (existing behavior in `FEAService`)
- `pre_stressed` -- modal with pre-stress stiffening from static load

#### 1.2.2 Harmonic Response Analysis

```
POST /api/v1/analysis/harmonic
```

Frequency-domain forced response. Sweeps a frequency range around the target,
computes FRF (frequency response function), amplitudes, phase, and gain.

#### 1.2.3 Static Stress Analysis

```
POST /api/v1/analysis/stress
```

Static structural analysis under specified loads (clamping force, operational pressure).
Returns Von Mises stress contour, principal stresses, and safety factors.

#### 1.2.4 Fatigue Assessment

```
POST /api/v1/analysis/fatigue
```

Uses stress results (from a prior analysis or computed inline) combined with S-N curve
data, load spectrum, and operating cycles to predict fatigue life.

#### 1.2.5 Acoustic Analysis (Combined Pipeline)

```
POST /api/v1/analysis/acoustic
```

Runs the full acoustic pipeline in sequence: modal -> harmonic -> stress -> amplitude
distribution. This is the existing `AcousticAnalysisResponse` path, now formalized
with the unified schema and async support.

#### 1.2.6 Piezoelectric Analysis (Solver B only)

```
POST /api/v1/analysis/piezoelectric
```

Coupled electro-mechanical analysis of the transducer element. Requires FEniCSx backend.
Returns electrical impedance, mechanical displacement at operating frequency, and
electro-mechanical coupling coefficient.

If `solver_backend` is `"solver_a"`, the server returns `HTTP 422` with message
`"Piezoelectric analysis requires Solver B (FEniCSx)"`.

#### 1.2.7 Thermal Analysis (Solver B only)

```
POST /api/v1/analysis/thermal
```

Transient or steady-state thermal analysis accounting for frictional heat generation
at the contact face and conduction through the horn body. Requires FEniCSx.

Same 422 guard as piezoelectric.

#### 1.2.8 Cross-Validation

```
POST /api/v1/analysis/cross-validate
```

Runs the same analysis configuration on both Solver A and Solver B, returning side-by-side
results with discrepancy metrics (frequency deviation, stress delta, amplitude delta).

### 1.3 Assembly Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/assemblies` | Create assembly from component IDs |
| `GET` | `/assemblies` | List all saved assemblies |
| `GET` | `/assemblies/{assembly_id}` | Get assembly definition with component details |
| `PUT` | `/assemblies/{assembly_id}` | Update assembly (reorder, replace component) |
| `DELETE` | `/assemblies/{assembly_id}` | Delete assembly |
| `POST` | `/assemblies/{assembly_id}/analyze` | Run full-stack analysis on assembly |
| `POST` | `/assemblies/{assembly_id}/impedance` | Compute impedance spectrum |
| `GET` | `/assemblies/{assembly_id}/mesh` | Get combined assembly mesh for 3D viewer |

### 1.4 Job Management (Async Pipeline)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/jobs/{job_id}` | Get job status, progress, and results (when complete) |
| `GET` | `/jobs` | List recent jobs (`?status=running\|complete\|failed&limit=50`) |
| `POST` | `/jobs/{job_id}/cancel` | Cancel a running analysis job |
| `DELETE` | `/jobs/{job_id}` | Delete job and its cached results |
| `WS` | `/jobs/{job_id}/progress` | WebSocket for real-time progress streaming |

### 1.5 Results & Reporting

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/results/{job_id}` | Get full analysis results (JSON) |
| `GET` | `/results/{job_id}/export` | Export results (`?format=vtk\|json\|pdf`) |
| `GET` | `/results/{job_id}/mesh` | Get deformed mesh / mode shape mesh for 3D viewer |
| `GET` | `/results/{job_id}/contour` | Get stress/displacement contour data for overlay |
| `POST` | `/results/compare` | Compare two or more analysis results side-by-side |
| `POST` | `/analysis/convergence` | Run mesh convergence study (multiple mesh densities) |

### 1.6 Complete Endpoint Map

```
/api/v1/
  components/
    horns/                          POST GET
    horns/{horn_id}                 GET PUT DELETE
    horns/generate                  POST
    horns/import-step               POST
    horns/{horn_id}/mesh            GET
    horns/{horn_id}/download        GET
    boosters/                       POST GET
    boosters/{booster_id}           GET PUT DELETE
    boosters/generate               POST
    boosters/import-step            POST
    transducers/                    POST GET
    transducers/{transducer_id}     GET PUT DELETE
  analysis/
    modal                           POST
    harmonic                        POST
    stress                          POST
    fatigue                         POST
    acoustic                        POST
    piezoelectric                   POST  (Solver B only)
    thermal                         POST  (Solver B only)
    cross-validate                  POST
    convergence                     POST
  assemblies/
    /                               POST GET
    {assembly_id}                   GET PUT DELETE
    {assembly_id}/analyze           POST
    {assembly_id}/impedance         POST
    {assembly_id}/mesh              GET
  jobs/
    /                               GET
    {job_id}                        GET DELETE
    {job_id}/cancel                 POST
    {job_id}/progress               WS
  results/
    {job_id}                        GET
    {job_id}/export                 GET
    {job_id}/mesh                   GET
    {job_id}/contour                GET
    compare                         POST
  fea/
    materials                       GET
    materials/{name}                GET
```

---

## 2. Request/Response Schemas

All models use Pydantic `BaseModel`. File location: `web/schemas/fea.py` (new file).

### 2.1 Enumerations

```python
from enum import Enum

class ComponentType(str, Enum):
    HORN = "horn"
    BOOSTER = "booster"
    TRANSDUCER = "transducer"

class HornProfileType(str, Enum):
    FLAT = "flat"
    CYLINDRICAL = "cylindrical"
    EXPONENTIAL = "exponential"
    CATENOIDAL = "catenoidal"
    STEPPED = "stepped"
    BLADE = "blade"
    CONICAL = "conical"
    BARBELL = "barbell"

class BoosterProfileType(str, Enum):
    CYLINDRICAL = "cylindrical"
    STEPPED = "stepped"
    EXPONENTIAL = "exponential"
    CATENOIDAL = "catenoidal"
    CONICAL = "conical"

class BoundaryCondition(str, Enum):
    FREE_FREE = "free_free"
    CLAMPED = "clamped"
    PRE_STRESSED = "pre_stressed"

class SolverBackend(str, Enum):
    SOLVER_A = "solver_a"       # numpy/scipy
    SOLVER_B = "solver_b"       # FEniCSx
    CROSS_VALIDATE = "cross_validate"

class MeshDensity(str, Enum):
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    CUSTOM = "custom"

class AnalysisType(str, Enum):
    MODAL = "modal"
    HARMONIC = "harmonic"
    STRESS = "stress"
    FATIGUE = "fatigue"
    ACOUSTIC = "acoustic"
    PIEZOELECTRIC = "piezoelectric"
    THERMAL = "thermal"

class JobStatus(str, Enum):
    QUEUED = "queued"
    MESHING = "meshing"
    SOLVING = "solving"
    POST_PROCESSING = "post_processing"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ModeType(str, Enum):
    LONGITUDINAL = "longitudinal"
    FLEXURAL = "flexural"
    TORSIONAL = "torsional"
    RADIAL = "radial"
    COUPLED = "coupled"

class EdgeTreatment(str, Enum):
    NONE = "none"
    CHAMFER = "chamfer"
    FILLET = "fillet"
    COMPOUND = "compound"

class KnurlType(str, Enum):
    NONE = "none"
    LINEAR = "linear"
    CROSS_HATCH = "cross_hatch"
    DIAMOND = "diamond"
    CONICAL = "conical"
    SPHERICAL = "spherical"
```

### 2.2 Component Geometry Inputs

#### Horn Definition

```python
class HornGeometryInput(BaseModel):
    """Parametric horn geometry definition."""
    profile_type: HornProfileType = HornProfileType.CYLINDRICAL
    # Primary dimensions
    input_diameter_mm: float = Field(gt=0, description="Input end diameter")
    output_diameter_mm: float = Field(gt=0, description="Output end diameter")
    length_mm: float = Field(gt=0, description="Total horn length (axial)")
    # Stepped horn only
    step_position_ratio: Optional[float] = Field(
        default=None, ge=0.2, le=0.8,
        description="Axial position of step as ratio of total length"
    )
    step_diameter_mm: Optional[float] = Field(default=None, gt=0)
    # Slot details (for slotted horns)
    slot_count: int = Field(default=0, ge=0, le=12)
    slot_width_mm: Optional[float] = Field(default=None, gt=0)
    slot_depth_mm: Optional[float] = Field(default=None, gt=0)
    # Contact face
    contact_face_width_mm: Optional[float] = Field(default=None, gt=0)
    contact_face_length_mm: Optional[float] = Field(default=None, gt=0)
    # Knurl
    knurl_type: KnurlType = KnurlType.NONE
    knurl_pitch_mm: float = Field(default=1.0, gt=0)
    knurl_tooth_width_mm: float = Field(default=0.5, gt=0)
    knurl_depth_mm: float = Field(default=0.3, ge=0)
    # Edge treatment
    edge_treatment: EdgeTreatment = EdgeTreatment.NONE
    chamfer_radius_mm: float = Field(default=0.0, ge=0)
    chamfer_angle_deg: float = Field(default=45.0, gt=0, le=90)

class HornFromSTEPInput(BaseModel):
    """Horn created by importing a STEP file.
    The STEP file is uploaded via multipart/form-data alongside this JSON body."""
    name: str = Field(min_length=1, max_length=200)
    material: str = "Titanium Ti-6Al-4V"
    operating_frequency_khz: float = Field(default=20.0, gt=0)
    contact_face_direction: str = Field(
        default="Y+",
        pattern=r"^[XYZ][+-]$",
        description="Direction normal to contact face for amplitude extraction"
    )

class HornDefinition(BaseModel):
    """Persisted horn definition (response from GET/POST)."""
    horn_id: str
    name: str
    source: str  # "parametric" | "step_import"
    material: str
    operating_frequency_khz: float
    geometry: Optional[HornGeometryInput] = None   # set if parametric
    step_file_hash: Optional[str] = None           # set if from STEP
    volume_mm3: float
    surface_area_mm2: float
    bounding_box: list[float]  # [x_min, y_min, z_min, x_max, y_max, z_max]
    has_cad_export: bool
    created_at: str
    updated_at: str
```

#### Booster Definition

```python
class BoosterGeometryInput(BaseModel):
    """Parametric booster geometry definition."""
    profile_type: BoosterProfileType = BoosterProfileType.STEPPED
    input_diameter_mm: float = Field(gt=0, description="Input end diameter")
    output_diameter_mm: float = Field(gt=0, description="Output end diameter")
    length_mm: float = Field(gt=0, description="Total booster length")
    flange_position_ratio: float = Field(
        default=0.5, ge=0.2, le=0.8,
        description="Mounting flange position as ratio of total length"
    )
    flange_width_mm: float = Field(default=10.0, gt=0)
    gain_ratio: float = Field(
        gt=0,
        description="Target mechanical gain ratio (output_amp / input_amp)"
    )

class BoosterDefinition(BaseModel):
    """Persisted booster definition."""
    booster_id: str
    name: str
    source: str
    material: str
    geometry: Optional[BoosterGeometryInput] = None
    gain_ratio: float
    node_location_ratio: float  # position of zero-displacement node (ratio of length)
    volume_mm3: float
    created_at: str
    updated_at: str
```

#### Transducer Definition

```python
class TransducerDefinition(BaseModel):
    """Transducer (converter) definition for assembly modeling."""
    transducer_id: str
    name: str
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    nominal_frequency_khz: float = Field(gt=0)
    impedance_ohms: float = Field(default=10.0, gt=0)
    max_amplitude_um: float = Field(gt=0)
    # Piezo stack
    piezo_material: str = Field(default="PZT-4")
    piezo_d33_pm_v: float = Field(
        default=289.0,
        description="Piezoelectric charge constant d33 [pC/N]"
    )
    # Mechanical dimensions
    front_mass_diameter_mm: float = Field(gt=0)
    front_mass_length_mm: float = Field(gt=0)
    back_mass_diameter_mm: float = Field(gt=0)
    back_mass_length_mm: float = Field(gt=0)
    front_mass_material: str = "Titanium Ti-6Al-4V"
    back_mass_material: str = "Steel D2"
    created_at: str
    updated_at: str
```

### 2.3 Material Assignment

```python
class MaterialAssignment(BaseModel):
    """Material properties for FEA -- references the FEA_MATERIALS database."""
    name: str = Field(
        description="Material name or alias (case-insensitive). "
                    "e.g. 'Titanium Ti-6Al-4V', 'ti64', 'D2'"
    )
    # Override fields (optional -- defaults come from FEA_MATERIALS lookup)
    E_gpa: Optional[float] = Field(default=None, gt=0)
    density_kg_m3: Optional[float] = Field(default=None, gt=0)
    poisson_ratio: Optional[float] = Field(default=None, gt=0, lt=0.5)
    yield_mpa: Optional[float] = Field(default=None, gt=0)
    thermal_conductivity_w_mk: Optional[float] = Field(default=None, gt=0)
    specific_heat_j_kgk: Optional[float] = Field(default=None, gt=0)
    cte_1_k: Optional[float] = Field(default=None, gt=0)
    damping_ratio: float = Field(
        default=0.01, ge=0.0001, le=0.2,
        description="Modal damping ratio (zeta). Default 1% for titanium horns."
    )
```

### 2.4 Analysis Configuration Inputs

#### Modal Analysis Request

```python
class ModalAnalysisRequest(BaseModel):
    """Configuration for modal (eigenvalue) analysis."""
    # Source: either a persisted component ID or inline geometry
    component_id: Optional[str] = Field(
        default=None,
        description="Reference to a saved horn/booster definition"
    )
    component_type: ComponentType = ComponentType.HORN
    # Inline geometry (used when component_id is None)
    geometry: Optional[HornGeometryInput | BoosterGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    # Analysis parameters
    bc_type: BoundaryCondition = BoundaryCondition.FREE_FREE
    target_frequency_khz: float = Field(default=20.0, gt=0)
    num_modes: int = Field(default=10, ge=1, le=50)
    frequency_range_hz: Optional[list[float]] = Field(
        default=None,
        min_length=2, max_length=2,
        description="[f_min, f_max] filter. Modes outside this range are excluded."
    )
    # Pre-stress (only when bc_type == PRE_STRESSED)
    pre_stress_force_n: Optional[float] = Field(default=None, gt=0)
    pre_stress_direction: str = Field(default="Y")
    # Solver
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
    custom_element_size_mm: Optional[float] = Field(
        default=None, gt=0,
        description="Element size when mesh_density is CUSTOM"
    )
    # Output control
    include_mode_shapes: bool = Field(
        default=True,
        description="Include full nodal displacement vectors (can be large)"
    )
    include_mesh: bool = Field(
        default=True,
        description="Include surface mesh for 3D viewer"
    )
```

#### Harmonic Response Request

```python
class HarmonicAnalysisRequest(BaseModel):
    """Configuration for harmonic (frequency response) analysis."""
    component_id: Optional[str] = None
    component_type: ComponentType = ComponentType.HORN
    geometry: Optional[HornGeometryInput | BoosterGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    # Excitation
    excitation_type: str = Field(
        default="unit_force",
        description="'unit_force' (1N on input face) | 'displacement' (prescribed input amplitude)"
    )
    excitation_amplitude_um: Optional[float] = Field(
        default=None, gt=0,
        description="Input displacement amplitude (when excitation_type == 'displacement')"
    )
    excitation_face: str = Field(
        default="input",
        description="'input' (back face) or 'output' (front/contact face)"
    )
    # Frequency sweep
    center_frequency_khz: float = Field(default=20.0, gt=0)
    sweep_range_percent: float = Field(
        default=20.0, gt=0, le=100,
        description="Sweep +/- this percent around center frequency"
    )
    sweep_points: int = Field(default=41, ge=11, le=201)
    # Damping
    damping_model: str = Field(
        default="rayleigh",
        description="'rayleigh' | 'modal' | 'hysteretic'"
    )
    damping_ratio: float = Field(default=0.01, ge=0.0001, le=0.2)
    # Solver
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
    bc_type: BoundaryCondition = BoundaryCondition.CLAMPED
    include_mesh: bool = True
```

#### Static Stress Request

```python
class StaticStressRequest(BaseModel):
    """Configuration for static structural analysis."""
    component_id: Optional[str] = None
    component_type: ComponentType = ComponentType.HORN
    geometry: Optional[HornGeometryInput | BoosterGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    # Loads
    clamping_force_n: float = Field(
        default=1000.0, ge=0,
        description="Axial clamping/pre-load force"
    )
    contact_pressure_mpa: Optional[float] = Field(
        default=None, ge=0,
        description="Uniform pressure on contact face"
    )
    body_force_gravity: bool = Field(
        default=False,
        description="Include gravity body force"
    )
    # Thermal pre-load (optional)
    temperature_delta_c: Optional[float] = Field(
        default=None,
        description="Uniform temperature change from reference (for thermal stress)"
    )
    # Solver
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
    include_mesh: bool = True
```

#### Fatigue Assessment Request

```python
class FatigueRequest(BaseModel):
    """Configuration for fatigue life prediction."""
    component_id: Optional[str] = None
    component_type: ComponentType = ComponentType.HORN
    geometry: Optional[HornGeometryInput | BoosterGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    # Operating conditions
    operating_frequency_khz: float = Field(default=20.0, gt=0)
    operating_amplitude_um: float = Field(default=30.0, gt=0)
    duty_cycle_percent: float = Field(default=50.0, ge=1, le=100)
    cycles_per_day: int = Field(default=10000, ge=1)
    # S-N curve parameters (override material defaults)
    sn_exponent: Optional[float] = Field(default=None, description="Basquin exponent b")
    sn_coefficient_mpa: Optional[float] = Field(
        default=None, description="Fatigue strength coefficient sigma_f'"
    )
    endurance_limit_mpa: Optional[float] = Field(default=None, gt=0)
    # Stress source
    stress_from_job_id: Optional[str] = Field(
        default=None,
        description="Reuse stress results from a completed analysis job"
    )
    # Solver
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
    safety_factor_target: float = Field(default=2.0, ge=1.0)
```

#### Acoustic (Combined) Analysis Request

```python
class AcousticAnalysisRequest(BaseModel):
    """Full acoustic pipeline: modal + harmonic + stress + amplitude."""
    component_id: Optional[str] = None
    component_type: ComponentType = ComponentType.HORN
    geometry: Optional[HornGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    target_frequency_khz: float = Field(default=20.0, gt=0)
    # Harmonic sweep settings
    sweep_range_percent: float = Field(default=20.0, gt=0, le=100)
    sweep_points: int = Field(default=21, ge=11, le=201)
    damping_ratio: float = Field(default=0.01, ge=0.0001, le=0.2)
    # Amplitude extraction
    amplitude_target_um: Optional[float] = Field(
        default=None, gt=0,
        description="If set, results are scaled so mean contact amplitude matches this"
    )
    # Solver
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
    bc_type: BoundaryCondition = BoundaryCondition.CLAMPED
    include_mesh: bool = True
```

#### Piezoelectric Analysis Request (Solver B only)

```python
class PiezoelectricAnalysisRequest(BaseModel):
    """Coupled electro-mechanical analysis of transducer element."""
    transducer_id: Optional[str] = None
    transducer: Optional[TransducerDefinition] = None  # inline
    # Excitation
    driving_voltage_v: float = Field(default=100.0, gt=0)
    frequency_range_khz: list[float] = Field(
        default=[18.0, 22.0], min_length=2, max_length=2
    )
    sweep_points: int = Field(default=41, ge=11, le=201)
    # Solver (must be solver_b)
    mesh_density: MeshDensity = MeshDensity.MEDIUM
```

#### Thermal Analysis Request (Solver B only)

```python
class ThermalAnalysisRequest(BaseModel):
    """Thermal analysis for horn under operating conditions."""
    component_id: Optional[str] = None
    component_type: ComponentType = ComponentType.HORN
    geometry: Optional[HornGeometryInput] = None
    material: MaterialAssignment = MaterialAssignment(name="Titanium Ti-6Al-4V")
    # Heat generation
    operating_frequency_khz: float = Field(default=20.0, gt=0)
    operating_amplitude_um: float = Field(default=30.0, gt=0)
    contact_pressure_mpa: float = Field(default=2.0, gt=0)
    friction_coefficient: float = Field(default=0.3, gt=0)
    weld_time_s: float = Field(default=0.5, gt=0)
    # Boundary conditions
    ambient_temperature_c: float = Field(default=25.0)
    convection_coefficient_w_m2k: float = Field(default=10.0, ge=0)
    # Analysis type
    transient: bool = Field(
        default=True,
        description="True for transient, False for steady-state"
    )
    time_steps: int = Field(default=100, ge=10, le=1000)
    # Solver (must be solver_b)
    mesh_density: MeshDensity = MeshDensity.MEDIUM
```

#### Cross-Validation Request

```python
class CrossValidationRequest(BaseModel):
    """Run analysis on both solvers and compare."""
    analysis_type: AnalysisType
    # The actual analysis configuration -- one of the request types above
    # Serialized as the appropriate sub-type based on analysis_type
    config: dict = Field(
        description="Analysis configuration matching the schema for 'analysis_type'. "
                    "The solver_backend field is ignored (both are run)."
    )
    tolerance_frequency_percent: float = Field(
        default=2.0, ge=0,
        description="Acceptable frequency deviation between solvers"
    )
    tolerance_stress_percent: float = Field(
        default=5.0, ge=0,
        description="Acceptable stress deviation between solvers"
    )
```

#### Assembly Input

```python
class AssemblyCreateRequest(BaseModel):
    """Create an assembly from component IDs, ordered from transducer to horn."""
    name: str = Field(min_length=1, max_length=200)
    # Components in stack order: transducer (bottom) -> booster -> horn (top)
    transducer_id: str
    booster_id: str
    horn_id: str
    # Interface properties
    bolt_torque_nm: float = Field(default=30.0, ge=0)
    contact_stiffness_n_mm: float = Field(
        default=1e6, gt=0,
        description="Interface contact stiffness for assembly modeling"
    )

class AssemblyAnalysisRequest(BaseModel):
    """Run full-stack analysis on an assembly."""
    target_frequency_khz: float = Field(default=20.0, gt=0)
    driving_voltage_v: float = Field(default=100.0, gt=0)
    sweep_range_percent: float = Field(default=20.0, gt=0, le=100)
    sweep_points: int = Field(default=41, ge=11, le=201)
    damping_ratio: float = Field(default=0.01, ge=0.0001, le=0.2)
    include_stress: bool = True
    include_fatigue: bool = False
    solver_backend: SolverBackend = SolverBackend.SOLVER_A
    mesh_density: MeshDensity = MeshDensity.MEDIUM
```

### 2.5 Analysis Result Schemas

#### Modal Results

```python
class ModeShapeResult(BaseModel):
    """Single mode shape from modal analysis."""
    mode_number: int
    frequency_hz: float
    mode_type: ModeType
    participation_factor: float
    effective_mass_ratio: float
    displacement_max: float
    # Nodal plane locations (Z-coordinates where axial displacement crosses zero)
    nodal_planes_z_mm: list[float] = []
    # Full displacement field (optional, can be large)
    displacement_field: Optional[list[list[float]]] = Field(
        default=None,
        description="Per-node displacement vectors [[dx,dy,dz], ...]. "
                    "Omitted when include_mode_shapes=False."
    )

class ModalAnalysisResult(BaseModel):
    """Complete modal analysis result."""
    job_id: str
    analysis_type: str = "modal"
    solver_backend: str
    # Mode results
    modes: list[ModeShapeResult]
    target_frequency_hz: float
    closest_mode_hz: float
    frequency_deviation_percent: float
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    # Visualization
    mesh: Optional[dict] = Field(
        default=None,
        description='{"vertices": [[x,y,z],...], "faces": [[a,b,c],...]}'
    )
    # Metadata
    bc_type: str
    material_name: str
    created_at: str
```

#### Harmonic Response Results

```python
class FRFPoint(BaseModel):
    """Single point on the frequency response function."""
    frequency_hz: float
    amplitude: float          # magnitude of displacement
    phase_deg: float          # phase angle relative to excitation
    real: float               # real part
    imaginary: float          # imaginary part

class HarmonicAnalysisResult(BaseModel):
    """Complete harmonic response analysis result."""
    job_id: str
    analysis_type: str = "harmonic"
    solver_backend: str
    # FRF data
    frf: list[FRFPoint]
    resonance_frequency_hz: float   # frequency of peak amplitude
    anti_resonance_frequency_hz: Optional[float] = None
    peak_amplitude: float
    # Gain
    gain_at_target: float = Field(
        description="Output-to-input amplitude ratio at target frequency"
    )
    # Amplitude distribution at target frequency
    amplitude_distribution: dict = Field(
        description='{"node_positions": [[x,y,z],...], "amplitudes": [...], "phases_deg": [...]}'
    )
    amplitude_uniformity: float = Field(
        ge=0, le=1,
        description="1.0 = perfectly uniform, 0.0 = highly non-uniform"
    )
    amplitude_max_um: float
    amplitude_min_um: float
    amplitude_mean_um: float
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    mesh: Optional[dict] = None
    created_at: str
```

#### Stress Results

```python
class StressHotspot(BaseModel):
    """Single stress hotspot location."""
    location_mm: list[float]  # [x, y, z]
    von_mises_mpa: float
    principal_stresses_mpa: list[float]  # [sigma_1, sigma_2, sigma_3]
    element_index: int
    safety_factor: float = Field(
        description="yield_strength / von_mises_stress"
    )

class StressAnalysisResult(BaseModel):
    """Complete stress analysis result."""
    job_id: str
    analysis_type: str = "stress"
    solver_backend: str
    # Global stress metrics
    von_mises_max_mpa: float
    von_mises_mean_mpa: float
    safety_factor_min: float
    safety_factor_mean: float
    yield_strength_mpa: float
    # Hotspots
    hotspots: list[StressHotspot]
    # Contour data for visualization
    stress_contour: Optional[dict] = Field(
        default=None,
        description='{"element_values_mpa": [...], "node_values_mpa": [...]}'
    )
    # Displacement
    max_displacement_mm: float
    displacement_field: Optional[list[list[float]]] = None
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    mesh: Optional[dict] = None
    created_at: str
```

#### Fatigue Results

```python
class FatigueCriticalLocation(BaseModel):
    """Critical location in fatigue assessment."""
    location_mm: list[float]
    alternating_stress_mpa: float
    mean_stress_mpa: float
    predicted_life_cycles: float
    safety_factor: float
    element_index: int

class FatigueAnalysisResult(BaseModel):
    """Complete fatigue assessment result."""
    job_id: str
    analysis_type: str = "fatigue"
    solver_backend: str
    # Global fatigue metrics
    min_life_cycles: float
    min_life_hours: float = Field(
        description="Minimum life in operating hours (accounting for duty cycle)"
    )
    min_safety_factor: float
    mean_safety_factor: float
    # Critical locations
    critical_locations: list[FatigueCriticalLocation]
    # Safety factor map for visualization
    safety_factor_map: Optional[dict] = Field(
        default=None,
        description='{"element_values": [...]}'
    )
    # Input echo
    operating_frequency_khz: float
    operating_amplitude_um: float
    duty_cycle_percent: float
    # Mesh info
    node_count: int
    element_count: int
    solve_time_s: float
    created_at: str
```

#### Assembly Results

```python
class AssemblyAnalysisResult(BaseModel):
    """Full-stack assembly analysis result."""
    job_id: str
    analysis_type: str = "assembly"
    solver_backend: str
    assembly_id: str
    # Stack frequency
    stack_resonance_hz: float
    target_frequency_hz: float
    frequency_deviation_percent: float
    modes: list[ModeShapeResult]
    # Gain chain
    gain_chain: dict = Field(
        description='{"transducer_output_um": x, "booster_gain": y, '
                    '"booster_output_um": z, "horn_gain": w, '
                    '"horn_output_um": v, "total_gain": g}'
    )
    # Impedance spectrum
    impedance_spectrum: Optional[dict] = Field(
        default=None,
        description='{"frequencies_hz": [...], '
                    '"impedance_magnitude_ohm": [...], '
                    '"impedance_phase_deg": [...]}'
    )
    # Stress (optional)
    stress_results: Optional[StressAnalysisResult] = None
    fatigue_results: Optional[FatigueAnalysisResult] = None
    # Mesh
    assembly_mesh: Optional[dict] = None
    node_count: int
    element_count: int
    solve_time_s: float
    created_at: str
```

#### Cross-Validation Result

```python
class CrossValidationResult(BaseModel):
    """Side-by-side comparison of Solver A and Solver B results."""
    job_id: str
    analysis_type: str
    # Individual results
    solver_a_result: dict  # full result from Solver A
    solver_b_result: dict  # full result from Solver B
    # Discrepancy metrics
    frequency_deviation_percent: float = Field(
        description="Relative difference in primary frequency between solvers"
    )
    stress_deviation_percent: Optional[float] = Field(
        default=None,
        description="Relative difference in max stress between solvers"
    )
    amplitude_deviation_percent: Optional[float] = Field(
        default=None,
        description="Relative difference in max amplitude between solvers"
    )
    # Validation
    within_tolerance: bool
    tolerance_violations: list[str] = []
    solve_time_a_s: float
    solve_time_b_s: float
    created_at: str
```

### 2.6 Job Status Response

```python
class JobProgress(BaseModel):
    """Real-time job progress information."""
    job_id: str
    status: JobStatus
    analysis_type: AnalysisType
    progress_percent: float = Field(ge=0, le=100)
    current_stage: str = Field(
        description="Human-readable stage: 'Generating mesh', "
                    "'Assembling matrices', 'Solving eigenvalues', "
                    "'Computing stress', 'Post-processing', etc."
    )
    stage_detail: Optional[str] = Field(
        default=None,
        description="Stage sub-detail: 'Element 1500/3000', 'Mode 5/10', etc."
    )
    # Timing
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    elapsed_s: Optional[float] = None
    estimated_remaining_s: Optional[float] = None
    # Results (populated when status == COMPLETE)
    result: Optional[dict] = None
    # Error (populated when status == FAILED)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
```

### 2.7 Convergence Study

```python
class ConvergenceStudyRequest(BaseModel):
    """Run the same analysis at multiple mesh densities to check convergence."""
    analysis_type: AnalysisType = AnalysisType.MODAL
    config: dict = Field(description="Base analysis configuration")
    mesh_sizes_mm: list[float] = Field(
        default=[8.0, 5.0, 3.0, 2.0, 1.5],
        description="Element sizes to test, from coarse to fine"
    )

class ConvergenceStudyResult(BaseModel):
    """Mesh convergence study result."""
    job_id: str
    analysis_type: str
    points: list[dict] = Field(
        description='[{"element_size_mm": x, "node_count": n, '
                    '"primary_result": val, "solve_time_s": t}, ...]'
    )
    converged: bool
    converged_at_size_mm: Optional[float] = None
    richardson_extrapolation: Optional[float] = Field(
        default=None,
        description="Richardson extrapolation estimate of the exact solution"
    )
```

### 2.8 Result Export

```python
class ExportRequest(BaseModel):
    """Parameters for result export."""
    format: str = Field(
        description="'vtk' (ParaView), 'json' (raw data), 'pdf' (report)"
    )
    # PDF-specific
    include_3d_screenshots: bool = True
    include_charts: bool = True
    language: str = Field(default="en", description="'en' | 'zh-CN'")
    company_name: Optional[str] = None
    project_name: Optional[str] = None
```

---

## 3. Async Analysis Pipeline

### 3.1 Architecture Overview

```
Client                          FastAPI                         Worker Pool
  |                                |                                |
  |-- POST /analysis/modal ------->|                                |
  |                                |-- validate request              |
  |                                |-- check model size              |
  |                                |                                |
  |                                |-- IF small (< SYNC_THRESHOLD)   |
  |                                |   solve synchronously           |
  |<------ 200 {result} ----------|                                |
  |                                |                                |
  |                                |-- IF large (>= SYNC_THRESHOLD)  |
  |                                |   create Job record             |
  |                                |   submit to worker pool ------->|
  |<------ 202 {job_id} ----------|                                |
  |                                |                                |
  |-- GET /jobs/{job_id} --------->|-- query job store               |
  |<------ 200 {status, progress} |                                |
  |                                |                          [Worker]
  |                                |                           mesh  |
  |                                |                           solve |
  |                                |                           post  |
  |                                |<--- update progress ------------|
  |                                |<--- store result --------------|
  |                                |                                |
  |-- GET /jobs/{job_id} --------->|                                |
  |<- 200 {status:complete, result}|                                |
```

### 3.2 Sync vs Async Decision

The threshold is configurable via environment variable `FEA_SYNC_NODE_THRESHOLD` (default: 5000).

```python
# In the analysis router handler:
estimated_nodes = estimate_node_count(request.mesh_density, geometry_dimensions)
if estimated_nodes < settings.FEA_SYNC_NODE_THRESHOLD:
    result = solver.run(request)
    return JSONResponse(status_code=200, content=result.dict())
else:
    job = job_store.create(analysis_type=request_type, config=request.dict())
    worker_pool.submit(job.job_id, solver.run, request)
    return JSONResponse(status_code=202, content={"job_id": job.job_id})
```

### 3.3 Job Lifecycle

```
                                +---> CANCELLED
                                |
  QUEUED --> MESHING --> SOLVING --> POST_PROCESSING --> COMPLETE
    |          |          |              |
    +----------+----------+--------------+---> FAILED
```

**State definitions**:

| State | Description | Progress % |
|-------|-------------|------------|
| `QUEUED` | Job accepted, waiting for worker | 0% |
| `MESHING` | Generating finite element mesh | 5-20% |
| `SOLVING` | Running eigenvalue / harmonic / stress solve | 20-80% |
| `POST_PROCESSING` | Computing derived quantities, contours, hotspots | 80-95% |
| `COMPLETE` | Results available | 100% |
| `FAILED` | Error occurred; `error_message` populated | N/A |
| `CANCELLED` | User requested cancellation | N/A |

### 3.4 Progress Reporting

Progress is reported at granular sub-stages. The solver callbacks update a shared
`JobProgress` record in the job store.

**Sub-stage progress map for acoustic analysis (example)**:

| Stage | Sub-stage | Progress % |
|-------|-----------|------------|
| MESHING | Generating nodes | 5% |
| MESHING | Generating elements | 10% |
| MESHING | Computing connectivity | 15% |
| SOLVING | Assembling stiffness matrix | 20% |
| SOLVING | Assembling mass matrix | 25% |
| SOLVING | Applying boundary conditions | 30% |
| SOLVING | Eigenvalue solve | 35% |
| SOLVING | Classifying modes | 45% |
| SOLVING | Harmonic sweep (i/N) | 50-75% (linear) |
| POST_PROCESSING | Computing stress field | 80% |
| POST_PROCESSING | Finding hotspots | 85% |
| POST_PROCESSING | Extracting amplitudes | 90% |
| POST_PROCESSING | Generating visualization mesh | 95% |
| COMPLETE | Done | 100% |

### 3.5 WebSocket Progress Stream

```
WS /api/v1/jobs/{job_id}/progress
```

The server sends JSON frames on each progress update:

```json
{
  "job_id": "abc123",
  "status": "solving",
  "progress_percent": 62.5,
  "current_stage": "Harmonic frequency sweep",
  "stage_detail": "Frequency 15/41: 19200.0 Hz",
  "elapsed_s": 12.4,
  "estimated_remaining_s": 7.6
}
```

The client can send:
```json
{"action": "cancel"}
```
to request cancellation.

### 3.6 Worker Pool Implementation

```python
# web/services/job_manager.py

import asyncio
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Callable

class JobManager:
    """Manages async FEA analysis jobs."""

    def __init__(self, max_workers: int = 2):
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, JobRecord] = {}
        self._futures: dict[str, Future] = {}
        self._result_cache: dict[str, CachedResult] = {}

    def submit(self, job_id: str, fn: Callable, *args) -> None:
        """Submit an analysis function to the worker pool."""
        record = self._jobs[job_id]
        record.status = JobStatus.QUEUED
        future = self._executor.submit(self._run_with_progress, job_id, fn, *args)
        self._futures[job_id] = future
        future.add_done_callback(lambda f: self._on_complete(job_id, f))

    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a running job."""
        future = self._futures.get(job_id)
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                self._jobs[job_id].status = JobStatus.CANCELLED
            return cancelled
        return False

    def get_status(self, job_id: str) -> JobProgress:
        """Get current job progress."""
        ...

    def get_result(self, job_id: str) -> Optional[dict]:
        """Get cached result for a completed job."""
        cached = self._result_cache.get(job_id)
        if cached and not cached.is_expired():
            return cached.result
        return None
```

### 3.7 Result Caching

Results are cached in-memory with a configurable TTL (default: 1 hour, env var
`FEA_RESULT_CACHE_TTL_S`). The cache key is the `job_id`.

```python
@dataclass
class CachedResult:
    result: dict
    created_at: float  # time.time()
    ttl_s: float = 3600.0

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_s
```

Cache eviction:
- TTL-based: results older than `ttl_s` are evicted on next access.
- Size-based: when cache exceeds `FEA_RESULT_CACHE_MAX_MB` (default: 512), oldest
  entries are evicted.
- Manual: `DELETE /api/v1/jobs/{job_id}` removes the job and its cached result.

---

## 4. Analysis Scenarios (Detailed Workflows)

### 4.1 Scenario 1: Standalone Horn Design

This is the primary use case for horn design engineers. The workflow takes a horn
geometry (parametric or imported STEP) through a complete analysis pipeline.

```
                     +--------------+
                     | Horn Input   |
                     | (parametric  |
                     |  or STEP)    |
                     +------+-------+
                            |
                   +--------v--------+
                   | POST /components|
                   | /horns          |
                   | (persist)       |
                   +--------+--------+
                            |
                   +--------v--------+
                   | POST /analysis/ |
                   | modal           |
                   | bc=free_free    |
                   +--------+--------+
                            |
                   +--------v---------+
                   | Review modes:    |
                   | - Identify       |
                   |   longitudinal   |
                   |   mode nearest   |
                   |   20 kHz         |
                   | - Check nodal    |
                   |   plane location |
                   +--------+---------+
                            |
                   +--------v----------+
                   | POST /analysis/   |
                   | harmonic           |
                   | excitation=input   |
                   | face, target freq  |
                   +--------+----------+
                            |
               +------------v-----------+
               | Results:               |
               | - FRF curve            |
               | - Gain at target freq  |
               | - Amplitude uniformity |
               |   at contact face      |
               +------------+-----------+
                            |
                   +--------v----------+
                   | POST /analysis/   |
                   | stress             |
                   | (with harmonic     |
                   |  displacement)     |
                   +--------+----------+
                            |
                   +--------v--------+
                   | POST /analysis/ |
                   | fatigue          |
                   | stress_from_     |
                   | job_id=<stress>  |
                   +--------+--------+
                            |
                   +--------v--------+
                   | GET /results/   |
                   | {job_id}/export |
                   | ?format=pdf     |
                   +--------+--------+
                            |
                      +-----v------+
                      |  PDF Report |
                      +------------+
```

**Step-by-step API calls**:

1. **Create horn component**:
   ```
   POST /api/v1/components/horns
   {
     "name": "20kHz Exponential Horn v3",
     "geometry": {
       "profile_type": "exponential",
       "input_diameter_mm": 50.0,
       "output_diameter_mm": 30.0,
       "length_mm": 127.5,
       "knurl_type": "cross_hatch",
       "knurl_pitch_mm": 1.2,
       "knurl_depth_mm": 0.3,
       "edge_treatment": "fillet",
       "chamfer_radius_mm": 0.5
     },
     "material": "Titanium Ti-6Al-4V",
     "operating_frequency_khz": 20.0
   }
   --> 201 {"horn_id": "h_abc123", ...}
   ```

2. **Modal analysis (free-free)**:
   ```
   POST /api/v1/analysis/modal
   {
     "component_id": "h_abc123",
     "component_type": "horn",
     "bc_type": "free_free",
     "target_frequency_khz": 20.0,
     "num_modes": 15,
     "solver_backend": "solver_a",
     "mesh_density": "medium"
   }
   --> 200 (sync) or 202 {"job_id": "j_modal_001"}
   ```

3. **Review modal results**: Frontend displays mode table, user selects
   the longitudinal mode closest to 20 kHz. Suppose it is at 19,850 Hz (mode 7).

4. **Harmonic response at target mode**:
   ```
   POST /api/v1/analysis/harmonic
   {
     "component_id": "h_abc123",
     "center_frequency_khz": 19.85,
     "sweep_range_percent": 10,
     "sweep_points": 41,
     "excitation_type": "displacement",
     "excitation_amplitude_um": 20.0,
     "bc_type": "clamped"
   }
   --> 202 {"job_id": "j_harm_001"}
   ```

5. **Check amplitude uniformity**: Result includes `amplitude_uniformity: 0.87`
   and heatmap data for the contact face.

6. **Stress analysis**:
   ```
   POST /api/v1/analysis/stress
   {
     "component_id": "h_abc123",
     "clamping_force_n": 2000.0,
     "mesh_density": "fine"
   }
   --> 202 {"job_id": "j_stress_001"}
   ```

7. **Fatigue assessment**:
   ```
   POST /api/v1/analysis/fatigue
   {
     "component_id": "h_abc123",
     "stress_from_job_id": "j_stress_001",
     "operating_frequency_khz": 20.0,
     "operating_amplitude_um": 30.0,
     "duty_cycle_percent": 60,
     "cycles_per_day": 20000,
     "safety_factor_target": 2.0
   }
   --> 200 {"min_life_hours": 8500.0, "min_safety_factor": 2.3, ...}
   ```

8. **Export report**:
   ```
   GET /api/v1/results/j_harm_001/export?format=pdf
   --> 200 (PDF file download)
   ```

### 4.2 Scenario 2: Standalone Booster Design

Booster design focuses on verifying gain ratio, node location (for mounting flange),
and stress at the step transition.

```
  Booster Input (parametric)
         |
  POST /components/boosters  (persist)
         |
  POST /analysis/modal  bc=free_free
         |
  Review modes:
    - Verify first longitudinal mode at target freq
    - Extract gain ratio: max_displacement_output / max_displacement_input
    - Locate nodal plane (flange position)
         |
  POST /analysis/stress  (clamping + operational)
         |
  Report
```

**Key verification checks performed by the frontend after modal analysis**:

```
Gain Ratio Check:
  measured_gain = output_face_displacement / input_face_displacement
  expected_gain = (input_diameter / output_diameter)^2  (for stepped)
  gain_error = abs(measured_gain - expected_gain) / expected_gain * 100

Node Location Check:
  node_z = Z-coordinate where axial displacement crosses zero
  flange_z = booster_length * flange_position_ratio
  node_offset = abs(node_z - flange_z)
  WARNING if node_offset > 2 mm  (flange not at nodal plane)
```

**Step-by-step API calls**:

1. **Create booster**:
   ```
   POST /api/v1/components/boosters
   {
     "name": "1:2 Stepped Booster",
     "geometry": {
       "profile_type": "stepped",
       "input_diameter_mm": 50.0,
       "output_diameter_mm": 35.0,
       "length_mm": 126.0,
       "flange_position_ratio": 0.5,
       "flange_width_mm": 12.0,
       "gain_ratio": 2.0
     },
     "material": "Titanium Ti-6Al-4V"
   }
   --> 201 {"booster_id": "b_def456", ...}
   ```

2. **Modal analysis**:
   ```
   POST /api/v1/analysis/modal
   {
     "component_id": "b_def456",
     "component_type": "booster",
     "bc_type": "free_free",
     "target_frequency_khz": 20.0,
     "num_modes": 10,
     "include_mode_shapes": true
   }
   ```

3. **Stress analysis** (focusing on step transition):
   ```
   POST /api/v1/analysis/stress
   {
     "component_id": "b_def456",
     "component_type": "booster",
     "clamping_force_n": 3000.0,
     "mesh_density": "fine"
   }
   ```

4. **Export**:
   ```
   GET /api/v1/results/{job_id}/export?format=pdf
   ```

### 4.3 Scenario 3: Full Assembly (Horn + Booster + Transducer)

The full-stack assembly analysis models the complete ultrasonic system driven by the
piezoelectric transducer.

```
  +---[Transducer]---+---[Booster]---+---[Horn]---+
  |   PZT stack      |  Gain stage   | Contact    |
  |   Electrical     |  Nodal plane  | face       |
  |   excitation     |  (flange)     | (workpiece)|
  +------------------+---------------+------------+
```

**Workflow**:

```
  Select Components (by ID)
         |
  POST /assemblies  (create assembly)
         |
  POST /assemblies/{id}/analyze
    target_frequency_khz=20.0
    driving_voltage_v=100
    include_stress=true
    include_fatigue=true
         |
  Internal pipeline:
    1. Build combined mesh (with contact interfaces)
    2. Modal analysis of full stack
    3. Harmonic response (transducer-driven)
    4. Compute gain chain: transducer -> booster -> horn
    5. Extract impedance spectrum
    6. Stress analysis at hotspots
    7. Fatigue assessment
         |
  Return AssemblyAnalysisResult
         |
  POST /assemblies/{id}/impedance  (separate impedance sweep if needed)
         |
  Export report
```

**Step-by-step API calls**:

1. **Create assembly**:
   ```
   POST /api/v1/assemblies
   {
     "name": "20kHz Production Stack A",
     "transducer_id": "t_001",
     "booster_id": "b_def456",
     "horn_id": "h_abc123",
     "bolt_torque_nm": 35.0,
     "contact_stiffness_n_mm": 1.5e6
   }
   --> 201 {"assembly_id": "asm_789", ...}
   ```

2. **Run full analysis**:
   ```
   POST /api/v1/assemblies/asm_789/analyze
   {
     "target_frequency_khz": 20.0,
     "driving_voltage_v": 120.0,
     "sweep_range_percent": 15,
     "sweep_points": 41,
     "include_stress": true,
     "include_fatigue": true,
     "solver_backend": "solver_a",
     "mesh_density": "medium"
   }
   --> 202 {"job_id": "j_asm_001"}
   ```

3. **Poll for progress** (or connect WebSocket):
   ```
   GET /api/v1/jobs/j_asm_001
   --> 200 {
     "status": "solving",
     "progress_percent": 45,
     "current_stage": "Harmonic sweep (assembly)",
     "stage_detail": "Frequency 12/41: 19500.0 Hz"
   }
   ```

4. **Get completed results**:
   ```
   GET /api/v1/results/j_asm_001
   --> 200 {
     "stack_resonance_hz": 19920.0,
     "frequency_deviation_percent": 0.4,
     "gain_chain": {
       "transducer_output_um": 10.0,
       "booster_gain": 2.0,
       "booster_output_um": 20.0,
       "horn_gain": 1.5,
       "horn_output_um": 30.0,
       "total_gain": 3.0
     },
     "impedance_spectrum": {
       "frequencies_hz": [18000, 18100, ...],
       "impedance_magnitude_ohm": [150, 145, ...],
       "impedance_phase_deg": [-45, -42, ...]
     },
     "stress_results": { ... },
     "fatigue_results": { ... }
   }
   ```

5. **Separate impedance analysis** (optional, higher resolution):
   ```
   POST /api/v1/assemblies/asm_789/impedance
   {
     "frequency_range_khz": [18.0, 22.0],
     "sweep_points": 201
   }
   ```

6. **Export comprehensive report**:
   ```
   GET /api/v1/results/j_asm_001/export?format=pdf
   ```

### 4.4 Scenario Comparison Matrix

| Feature | Standalone Horn | Standalone Booster | Full Assembly |
|---------|----------------|-------------------|---------------|
| Modal (free-free) | Required | Required | N/A |
| Modal (assembly) | N/A | N/A | Required |
| Harmonic response | Required | Optional | Required (driven) |
| Amplitude uniformity | Required | N/A | Required |
| Gain verification | N/A | Required | Required (chain) |
| Node location | N/A | Required | Inherited |
| Impedance spectrum | N/A | N/A | Required |
| Stress analysis | Required | Required | Required |
| Fatigue assessment | Required | Optional | Required |
| Piezoelectric coupling | N/A | N/A | Optional (Solver B) |
| Thermal analysis | Optional (Solver B) | N/A | Optional (Solver B) |

---

## 5. Frontend Integration Points

### 5.1 3D Viewer (Three.js)

The existing `ThreeViewer.vue` component and `GeometryView.vue` are extended to support
FEA result visualization.

**Data sources**:

| Visualization | API Endpoint | Data Format |
|--------------|-------------|-------------|
| Raw geometry | `GET /components/horns/{id}/mesh` | `{vertices, faces}` |
| Deformed mode shape | `GET /results/{job_id}/mesh?mode=7` | `{vertices, faces, displacement_field}` |
| Stress contour | `GET /results/{job_id}/contour` | `{element_values_mpa, colormap_range}` |
| Assembly view | `GET /assemblies/{id}/mesh` | `{components: [{name, vertices, faces, transform}]}` |

**Viewer modes**:

1. **Geometry mode**: wireframe + solid, orbit/pan/zoom. Existing.
2. **Mode shape mode**: animated deformed shape. Color by displacement magnitude.
   Slider for mode selection. Animation speed control.
3. **Stress contour mode**: Von Mises color overlay on deformed mesh.
   Colorbar with range control. Click-to-inspect individual elements.
4. **Assembly mode**: multiple component meshes with distinct colors.
   Interface planes highlighted. Exploded-view toggle.

**New Vue component**: `FEAResultViewer.vue`

```
<FEAResultViewer
  :mesh="mesh"
  :mode-shapes="modeShapes"
  :stress-contour="stressContour"
  :active-mode="selectedMode"
  :view-mode="'stress'"    <!-- 'geometry' | 'mode_shape' | 'stress' | 'assembly' -->
  :animate="true"
  @mode-selected="onModeSelected"
  @node-clicked="onNodeClicked"
/>
```

### 5.2 Interactive Frequency Response Charts

Built with ECharts, extending the existing chart infrastructure.

**Chart types**:

1. **FRF Magnitude Plot**:
   - X: Frequency (Hz), Y: Amplitude (normalized or absolute)
   - Vertical dashed line at target frequency
   - Resonance peak annotation with Q-factor
   - Interactive zoom on frequency axis

2. **FRF Phase Plot**:
   - X: Frequency (Hz), Y: Phase (degrees)
   - Synchronized X-axis with magnitude plot

3. **Impedance Plot** (assembly):
   - Dual Y-axis: magnitude (ohm) + phase (degrees) vs frequency
   - Resonance/anti-resonance markers

4. **Nyquist Plot**:
   - X: Real impedance, Y: Imaginary impedance
   - Frequency annotations along the curve

**New Vue component**: `FrequencyResponseChart.vue`

```
<FrequencyResponseChart
  :frf-data="harmonicResult.frf"
  :target-frequency-hz="20000"
  :show-phase="true"
  :show-gain-annotation="true"
/>
```

### 5.3 Amplitude Distribution Heatmap

Displays the displacement amplitude distribution across the horn contact face.

**Data flow**:
```
harmonic result -> amplitude_distribution -> contact face nodes + amplitudes
  -> project to 2D plane (X-Z for Y-normal face)
  -> interpolate to regular grid
  -> render as 2D heatmap with color scale
```

**New Vue component**: `AmplitudeHeatmap.vue`

```
<AmplitudeHeatmap
  :node-positions="amplitudeDistribution.node_positions"
  :amplitudes="amplitudeDistribution.amplitudes"
  :uniformity="amplitudeUniformity"
  :target-amplitude-um="30.0"
  color-scheme="viridis"
/>
```

**Features**:
- Color scale (viridis / jet / coolwarm)
- Uniformity index displayed as percentage badge
- Min/max/mean amplitude annotations
- Click-to-inspect individual nodes
- Toggle between absolute (um) and normalized (0-1) amplitude

### 5.4 Assembly Builder UI

A drag-and-drop interface for constructing ultrasonic stack assemblies.

**Layout**:

```
+-----------------------------+---------------------------------+
| Component Library           | Assembly Canvas                 |
|                             |                                 |
| [Horns]                     |   +--[Transducer]--+            |
|   - 20kHz Exp Horn          |   |                |            |
|   - 35kHz Blade Horn        |   +--[Booster]-----+            |
|                             |   |                |            |
| [Boosters]                  |   +--[Horn]--------+            |
|   - 1:2 Stepped             |                                 |
|   - 1:1.5 Conical           | Interface Properties:           |
|                             |   Bolt torque: [35] Nm          |
| [Transducers]               |   Contact stiffness: [1.5e6]   |
|   - Branson 20kHz           |                                 |
|   - Custom 35kHz            | [Analyze Assembly] button       |
+-----------------------------+---------------------------------+
```

**New Vue component**: `AssemblyBuilder.vue`

```
<AssemblyBuilder
  :available-horns="horns"
  :available-boosters="boosters"
  :available-transducers="transducers"
  @assembly-created="onAssemblyCreated"
  @analysis-requested="onAnalysisRequested"
/>
```

**Interactions**:
- Drag component from library to canvas stack position
- Visual snap-to-connect with interface highlight
- Real-time 3D preview of combined assembly (calls `GET /assemblies/{id}/mesh`)
- Component details panel on click
- Right-click to replace or remove component

### 5.5 Analysis Progress Indicator

Displays real-time progress for long-running FEA jobs.

**New Vue component**: `AnalysisProgress.vue`

```
<AnalysisProgress
  :job-id="currentJobId"
  :show-detail="true"
  @job-complete="onJobComplete"
  @job-failed="onJobFailed"
/>
```

**Features**:
- Multi-segment progress bar (meshing | solving | post-processing)
- Current stage text + sub-detail
- Elapsed time / estimated remaining
- Cancel button
- Auto-transitions to result view on completion
- Error display with retry option on failure

**WebSocket integration** (via `composables/useJobProgress.ts`):

```typescript
// composables/useJobProgress.ts
export function useJobProgress(jobId: Ref<string | null>) {
  const status = ref<JobStatus>('queued')
  const progress = ref(0)
  const currentStage = ref('')
  const stageDetail = ref('')
  const estimatedRemaining = ref<number | null>(null)
  const result = ref<any>(null)
  const error = ref<string | null>(null)

  let ws: WebSocket | null = null

  watch(jobId, (id) => {
    if (!id) return
    ws = new WebSocket(`/api/v1/jobs/${id}/progress`)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      status.value = data.status
      progress.value = data.progress_percent
      currentStage.value = data.current_stage
      stageDetail.value = data.stage_detail || ''
      estimatedRemaining.value = data.estimated_remaining_s
      if (data.status === 'complete') {
        result.value = data.result
      }
      if (data.status === 'failed') {
        error.value = data.error_message
      }
    }
  })

  function cancel() {
    ws?.send(JSON.stringify({ action: 'cancel' }))
  }

  onUnmounted(() => ws?.close())

  return { status, progress, currentStage, stageDetail,
           estimatedRemaining, result, error, cancel }
}
```

### 5.6 Result Comparison View (Solver A vs Solver B)

Side-by-side comparison of results from cross-validation runs.

**New Vue component**: `CrossValidationView.vue`

**Layout**:

```
+-------------------------------+-------------------------------+
|        Solver A (numpy/scipy) |        Solver B (FEniCSx)     |
+-------------------------------+-------------------------------+
| Solve time: 2.3s              | Solve time: 18.7s             |
| Resonance: 19,850 Hz          | Resonance: 19,872 Hz          |
| Max stress: 145.3 MPa         | Max stress: 148.1 MPa         |
+-------------------------------+-------------------------------+
| [3D stress contour A]         | [3D stress contour B]         |
+-------------------------------+-------------------------------+
| [FRF chart A]                 | [FRF chart B]                 |
+-------------------------------+-------------------------------+

Discrepancy Report:
  Frequency deviation:  0.11%  [PASS < 2%]
  Stress deviation:     1.93%  [PASS < 5%]
  Amplitude deviation:  2.15%  [PASS < 5%]
```

**Features**:
- Synchronized 3D view (rotating one side rotates the other)
- Overlaid FRF curves on shared chart for direct comparison
- Discrepancy metrics with pass/fail badges against configurable tolerances
- Exportable comparison report (PDF)

### 5.7 New Frontend Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/fea/horn/:id?` | `HornFEAView.vue` | Horn design scenario (Scenario 1) |
| `/fea/booster/:id?` | `BoosterFEAView.vue` | Booster design scenario (Scenario 2) |
| `/fea/assembly/:id?` | `AssemblyFEAView.vue` | Full assembly scenario (Scenario 3) |
| `/fea/assembly/builder` | `AssemblyBuilderView.vue` | Drag-and-drop assembly construction |
| `/fea/results/:jobId` | `FEAResultView.vue` | Unified result viewer |
| `/fea/compare/:jobIdA/:jobIdB` | `CrossValidationView.vue` | Side-by-side comparison |
| `/fea/convergence/:jobId` | `ConvergenceView.vue` | Mesh convergence study results |

### 5.8 New Pinia Stores

```typescript
// stores/feaComponents.ts
// Manages horn, booster, transducer CRUD and local cache

// stores/feaAnalysis.ts
// Manages analysis job submission, polling, and result retrieval

// stores/feaAssembly.ts
// Manages assembly creation and assembly-level analysis
```

### 5.9 New API Client Modules

```typescript
// api/feaComponents.ts
export const feaComponentsApi = {
  // Horns
  createHorn: (data: HornCreateRequest) => apiClient.post('/components/horns', data),
  listHorns: () => apiClient.get('/components/horns'),
  getHorn: (id: string) => apiClient.get(`/components/horns/${id}`),
  updateHorn: (id: string, data: Partial<HornCreateRequest>) =>
    apiClient.put(`/components/horns/${id}`, data),
  deleteHorn: (id: string) => apiClient.delete(`/components/horns/${id}`),
  generateHorn: (data: HornGeometryInput) =>
    apiClient.post('/components/horns/generate', data),
  importHornSTEP: (file: File, metadata: HornFromSTEPInput) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('metadata', JSON.stringify(metadata))
    return apiClient.post('/components/horns/import-step', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    })
  },
  getHornMesh: (id: string) => apiClient.get(`/components/horns/${id}/mesh`),
  downloadHorn: (id: string, format: 'step' | 'stl') =>
    apiClient.get(`/components/horns/${id}/download`, {
      params: { format },
      responseType: 'blob',
    }),
  // Boosters (analogous)
  createBooster: (data: BoosterCreateRequest) => apiClient.post('/components/boosters', data),
  listBoosters: () => apiClient.get('/components/boosters'),
  // ... etc
  // Transducers (analogous)
  createTransducer: (data: TransducerCreateRequest) =>
    apiClient.post('/components/transducers', data),
  listTransducers: () => apiClient.get('/components/transducers'),
  // ... etc
}

// api/feaAnalysis.ts
export const feaAnalysisApi = {
  runModal: (req: ModalAnalysisRequest) => apiClient.post('/analysis/modal', req),
  runHarmonic: (req: HarmonicAnalysisRequest) => apiClient.post('/analysis/harmonic', req),
  runStress: (req: StaticStressRequest) => apiClient.post('/analysis/stress', req),
  runFatigue: (req: FatigueRequest) => apiClient.post('/analysis/fatigue', req),
  runAcoustic: (req: AcousticAnalysisRequest) => apiClient.post('/analysis/acoustic', req),
  runPiezoelectric: (req: PiezoelectricAnalysisRequest) =>
    apiClient.post('/analysis/piezoelectric', req),
  runThermal: (req: ThermalAnalysisRequest) => apiClient.post('/analysis/thermal', req),
  runCrossValidation: (req: CrossValidationRequest) =>
    apiClient.post('/analysis/cross-validate', req),
  runConvergence: (req: ConvergenceStudyRequest) =>
    apiClient.post('/analysis/convergence', req),
  // Jobs
  getJob: (jobId: string) => apiClient.get(`/jobs/${jobId}`),
  listJobs: (params?: { status?: string; limit?: number }) =>
    apiClient.get('/jobs', { params }),
  cancelJob: (jobId: string) => apiClient.post(`/jobs/${jobId}/cancel`),
  deleteJob: (jobId: string) => apiClient.delete(`/jobs/${jobId}`),
  // Results
  getResult: (jobId: string) => apiClient.get(`/results/${jobId}`),
  exportResult: (jobId: string, format: string) =>
    apiClient.get(`/results/${jobId}/export`, {
      params: { format },
      responseType: format === 'json' ? 'json' : 'blob',
    }),
  getResultMesh: (jobId: string, params?: { mode?: number }) =>
    apiClient.get(`/results/${jobId}/mesh`, { params }),
  getResultContour: (jobId: string) => apiClient.get(`/results/${jobId}/contour`),
  compareResults: (jobIds: string[]) =>
    apiClient.post('/results/compare', { job_ids: jobIds }),
}

// api/feaAssembly.ts
export const feaAssemblyApi = {
  createAssembly: (data: AssemblyCreateRequest) => apiClient.post('/assemblies', data),
  listAssemblies: () => apiClient.get('/assemblies'),
  getAssembly: (id: string) => apiClient.get(`/assemblies/${id}`),
  updateAssembly: (id: string, data: Partial<AssemblyCreateRequest>) =>
    apiClient.put(`/assemblies/${id}`, data),
  deleteAssembly: (id: string) => apiClient.delete(`/assemblies/${id}`),
  analyzeAssembly: (id: string, req: AssemblyAnalysisRequest) =>
    apiClient.post(`/assemblies/${id}/analyze`, req),
  getImpedance: (id: string, req?: { frequency_range_khz: number[]; sweep_points: number }) =>
    apiClient.post(`/assemblies/${id}/impedance`, req),
  getAssemblyMesh: (id: string) => apiClient.get(`/assemblies/${id}/mesh`),
}
```

---

## Appendix A: Backend File Structure (New Files)

```
web/
  routers/
    fea_components.py      # Horn/Booster/Transducer CRUD
    fea_analysis.py        # All analysis endpoints
    fea_assemblies.py      # Assembly endpoints
    fea_jobs.py            # Job management endpoints
    fea_results.py         # Result retrieval and export
  schemas/
    fea.py                 # All Pydantic models from Section 2
  services/
    fea_service.py         # (existing, extended with new analysis types)
    solver_a.py            # numpy/scipy solver wrapper
    solver_b.py            # FEniCSx solver wrapper (optional dep)
    job_manager.py         # Async job queue and worker pool
    component_store.py     # Persisted component storage (SQLite)
    result_cache.py        # TTL-based result cache
    report_generator.py    # PDF/VTK export for FEA results
```

## Appendix B: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FEA_SYNC_NODE_THRESHOLD` | `5000` | Node count below which analysis runs synchronously |
| `FEA_MAX_WORKERS` | `2` | Max concurrent analysis worker processes |
| `FEA_RESULT_CACHE_TTL_S` | `3600` | Result cache TTL in seconds |
| `FEA_RESULT_CACHE_MAX_MB` | `512` | Max result cache size in MB |
| `FEA_MAX_NODES` | `50000` | Hard limit on mesh node count |
| `FEA_SOLVER_B_AVAILABLE` | `false` | Whether FEniCSx is installed |
| `FEA_DEFAULT_DAMPING` | `0.01` | Default damping ratio |

## Appendix C: Migration from Existing Endpoints

The existing endpoints continue to work unchanged. The new FEA system endpoints live
in a separate URL namespace and do not conflict:

| Existing Endpoint | Status | Notes |
|-------------------|--------|-------|
| `POST /acoustic/analyze` | Retained | Simplified single-call acoustic analysis |
| `POST /geometry/fea/run` | Retained | Simplified single-call modal analysis |
| `POST /horn/generate` | Retained | Quick parametric preview (no persistence) |
| `GET /geometry/fea/materials` | Retained | Redirects to `GET /fea/materials` |

New endpoints provide richer functionality (persistence, async, cross-validation,
assembly modeling) while the existing endpoints remain as lightweight alternatives
for simple ad-hoc analyses.
