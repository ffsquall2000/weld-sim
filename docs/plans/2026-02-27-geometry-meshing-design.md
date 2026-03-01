# Section 4: Geometry & Meshing Layer

> **Subsystem:** `ultrasonic_weld_master.plugins.geometry_analyzer`
> **Dependencies:** CadQuery (>=2.4), Gmsh (>=4.12 Python API), numpy, scipy
> **Target accuracy:** Frequency error <1%, stress error <5%
> **Target mesh scale:** 100K--500K nodes for production, 5K--20K for interactive preview

---

## 4.1 Parametric Geometry Generator

### 4.1.1 Overview

The parametric geometry generator creates fully-defined 3D solid models of ultrasonic
acoustic stack components using CadQuery. Each component is a half-wavelength resonator
(or integer multiple) whose length is auto-computed from material wave speed and operating
frequency. The generator produces watertight B-Rep solids suitable for downstream Gmsh
meshing and FEA analysis.

**File:** `ultrasonic_weld_master/plugins/geometry_analyzer/parametric_geometry.py`
**Estimated size:** ~1,400 lines

### 4.1.2 Material Wave Speed Calculation

All half-wavelength dimensioning derives from the longitudinal bar wave speed:

```
c_bar = sqrt(E / rho)
```

where `E` is Young's modulus (Pa) and `rho` is density (kg/m^3). The half-wavelength is:

```
L_half = c_bar / (2 * f)
```

where `f` is frequency in Hz. For a 20 kHz titanium Ti-6Al-4V horn:

```
c_bar = sqrt(113.8e9 / 4430) = 5068 m/s
L_half = 5068 / (2 * 20000) = 0.12670 m = 126.7 mm
```

This is stored per-component and used as the default axial length when the user does not
override it.

### 4.1.3 Data Structures

```python
from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np


class HornType(enum.Enum):
    FLAT = "flat"
    CYLINDRICAL = "cylindrical"
    EXPONENTIAL = "exponential"
    CATENOIDAL = "catenoidal"
    BLADE = "blade"
    STEPPED = "stepped"
    BLOCK = "block"


class BoosterType(enum.Enum):
    STEPPED = "stepped"
    EXPONENTIAL = "exponential"
    CATENOIDAL = "catenoidal"
    UNIFORM = "uniform"


class ComponentType(enum.Enum):
    HORN = "horn"
    BOOSTER = "booster"
    TRANSDUCER = "transducer"


@dataclass
class SlotFeature:
    """Slot cut into a horn for amplitude uniformity improvement."""
    slot_type: str = "through"          # "through" | "blind" | "tapered"
    width_mm: float = 2.0               # slot width
    depth_mm: float = 0.0               # 0 = full through-cut
    length_mm: float = 0.0              # 0 = full length of horn face
    count: int = 1                       # number of parallel slots
    spacing_mm: float = 10.0            # center-to-center spacing
    orientation_deg: float = 0.0        # 0 = along horn length axis
    fillet_radius_mm: float = 0.5       # fillet at slot ends (stress relief)
    position_along_axis: float = 0.5    # 0.0=input face, 1.0=output face


@dataclass
class ThreadedStud:
    """Threaded stud connection between stack components."""
    thread_spec: str = "M10x1.5"        # ISO metric thread designation
    major_diameter_mm: float = 10.0
    pitch_mm: float = 1.5
    engagement_length_mm: float = 15.0  # thread engagement per side
    stud_length_mm: float = 35.0        # total stud length
    counterbore_diameter_mm: float = 16.0
    counterbore_depth_mm: float = 5.0
    preload_n: float = 15000.0          # bolt preload force


@dataclass
class HornGeometryParams:
    """Complete parameter set for horn geometry generation."""
    horn_type: HornType = HornType.CYLINDRICAL
    material: str = "Titanium Ti-6Al-4V"
    frequency_hz: float = 20000.0

    # Cross-section dimensions (input end)
    input_diameter_mm: float = 0.0      # >0 for round cross-sections
    input_width_mm: float = 50.0        # rectangular width (ignored if diameter>0)
    input_depth_mm: float = 50.0        # rectangular depth (ignored if diameter>0)

    # Cross-section dimensions (output end) -- 0 means auto-compute from gain
    output_diameter_mm: float = 0.0
    output_width_mm: float = 0.0
    output_depth_mm: float = 0.0

    # Axial dimension -- 0 means auto half-wavelength
    length_mm: float = 0.0

    # Gain (area ratio for stepped, profile parameter for others)
    gain: float = 1.5

    # Step parameters (only for STEPPED type)
    step_position_ratio: float = 0.5    # 0-1 along length
    step_fillet_mm: float = 2.0         # fillet radius at step transition

    # Slots
    slots: list[SlotFeature] = field(default_factory=list)

    # Threaded connection at input face
    input_thread: Optional[ThreadedStud] = None

    # Edge treatments
    chamfer_radius_mm: float = 0.0
    edge_treatment: str = "none"        # none | chamfer | fillet

    # Mounting ring (for clamped horns)
    mounting_ring_diameter_mm: float = 0.0    # 0 = no ring
    mounting_ring_width_mm: float = 5.0
    mounting_ring_position_ratio: float = 0.5  # along length, at nodal plane


@dataclass
class BoosterGeometryParams:
    """Parameter set for booster (amplitude transformer) geometry."""
    booster_type: BoosterType = BoosterType.STEPPED
    material: str = "Titanium Ti-6Al-4V"
    frequency_hz: float = 20000.0

    input_diameter_mm: float = 50.0     # input end (transducer side)
    output_diameter_mm: float = 0.0     # 0 = auto from gain
    length_mm: float = 0.0             # 0 = auto half-wavelength
    gain: float = 1.5

    step_position_ratio: float = 0.5
    step_fillet_mm: float = 2.0

    # Threads on both ends
    input_thread: Optional[ThreadedStud] = None
    output_thread: Optional[ThreadedStud] = None

    # Mounting flange at nodal plane
    flange_diameter_mm: float = 0.0     # 0 = no flange
    flange_width_mm: float = 8.0
    flange_position_ratio: float = 0.5


@dataclass
class TransducerGeometryParams:
    """Simplified transducer model: back mass + PZT stack + front mass."""
    material_back_mass: str = "Steel D2"
    material_front_mass: str = "Steel D2"
    frequency_hz: float = 20000.0

    back_mass_diameter_mm: float = 50.0
    back_mass_length_mm: float = 0.0    # 0 = auto quarter-wavelength

    pzt_outer_diameter_mm: float = 50.0
    pzt_inner_diameter_mm: float = 20.0  # center hole
    pzt_thickness_mm: float = 5.0
    pzt_count: int = 4                   # number of PZT rings

    front_mass_diameter_mm: float = 50.0
    front_mass_length_mm: float = 0.0    # 0 = auto

    # Output thread (connects to booster)
    output_thread: Optional[ThreadedStud] = None
    # Prestress bolt through center
    prestress_bolt_diameter_mm: float = 10.0


@dataclass
class StackAssemblyParams:
    """Full acoustic stack: transducer + booster + horn."""
    transducer: Optional[TransducerGeometryParams] = None
    booster: Optional[BoosterGeometryParams] = None
    horn: HornGeometryParams = field(default_factory=HornGeometryParams)
    frequency_hz: float = 20000.0
    # Assembly alignment axis
    alignment_axis: str = "Z"           # "X", "Y", or "Z"


@dataclass
class GeometryResult:
    """Output from geometry generation."""
    solid: object                        # CadQuery Workplane or Shape
    component_type: ComponentType
    half_wavelength_mm: float
    actual_length_mm: float
    input_area_mm2: float
    output_area_mm2: float
    gain_geometric: float               # area ratio input/output
    volume_mm3: float
    center_of_mass_mm: tuple[float, float, float]
    # Identified boundary surfaces (face indices or selectors)
    boundary_faces: dict[str, object]    # key: role, value: CQ selector
    # Metadata
    params_used: dict                    # snapshot of params after auto-fill
```

### 4.1.4 Horn Profile Functions

Each horn type defines a cross-sectional area function `A(x)` along the axial coordinate
`x in [0, L]`, where `x=0` is the input (large) end and `x=L` is the output end.

| Horn Type     | Area Profile `A(x)`                                          | Gain Formula              |
|---------------|---------------------------------------------------------------|---------------------------|
| Flat/Block    | `A(x) = A_in` (constant)                                     | 1.0                       |
| Cylindrical   | `A(x) = A_in` (round cross-section, constant)                | 1.0                       |
| Exponential   | `A(x) = A_in * exp(-2*beta*x)` where `beta = ln(G)/L`       | `sqrt(A_in / A_out)`      |
| Catenoidal    | `A(x) = A_in * (cosh(beta*(L-x)) / cosh(beta*L))^2`         | `cosh(beta*L)`            |
| Stepped       | `A(x) = A_in for x<x_step; A_out for x>=x_step`             | `A_in / A_out`            |
| Blade         | Rectangular: `w(x)` constant, `d(x)` tapers                  | `d_in / d_out`            |

### 4.1.5 Class Interface

```python
class ParametricGeometryGenerator:
    """Generate CadQuery solids for acoustic stack components."""

    def __init__(self, material_db: dict[str, dict] = None):
        """
        Args:
            material_db: Material property database. Defaults to
                         FEA_MATERIALS from material_properties.py.
        """

    # --- Public API ---

    def generate_horn(self, params: HornGeometryParams) -> GeometryResult:
        """Generate a horn solid with auto-dimensioning."""

    def generate_booster(self, params: BoosterGeometryParams) -> GeometryResult:
        """Generate a booster solid with auto-dimensioning."""

    def generate_transducer(
        self, params: TransducerGeometryParams
    ) -> GeometryResult:
        """Generate simplified transducer solid (back mass + PZT + front mass)."""

    def generate_stack(self, params: StackAssemblyParams) -> list[GeometryResult]:
        """Generate and assemble full acoustic stack.

        Returns list of GeometryResult in order:
        [transducer, booster, horn] (any may be absent).
        Components are translated along the alignment axis so that
        mating faces are coincident.
        """

    def export_step(self, results: list[GeometryResult], path: str) -> None:
        """Export one or more solids to a STEP file."""

    def export_stl(self, results: list[GeometryResult], path: str,
                   tolerance: float = 0.01) -> None:
        """Export tessellated mesh to STL."""

    # --- Internal: profile generators ---

    def _compute_half_wavelength(self, material: str, freq_hz: float) -> float:
        """Compute L_half = c_bar / (2*f)."""

    def _auto_fill_dimensions(self, params) -> None:
        """Fill in zero-valued dimensions from half-wavelength and gain."""

    def _build_horn_solid(self, params: HornGeometryParams) -> object:
        """Dispatch to type-specific builder, return CQ Workplane."""

    def _build_exponential_horn(self, params: HornGeometryParams) -> object:
        """Loft through cross-sections sampled from A(x) = A_in*exp(-2*beta*x).

        Uses 20 cross-section slices for smooth profile.
        """

    def _build_catenoidal_horn(self, params: HornGeometryParams) -> object:
        """Loft through catenoidal profile cross-sections."""

    def _build_stepped_horn(self, params: HornGeometryParams) -> object:
        """Two-section extrude with fillet at transition."""

    def _build_blade_horn(self, params: HornGeometryParams) -> object:
        """Rectangular cross-section with width >> depth, optional taper."""

    def _build_block_horn(self, params: HornGeometryParams) -> object:
        """Large rectangular block horn, typically for wide welds."""

    def _apply_slots(self, solid: object, slots: list[SlotFeature],
                     params: HornGeometryParams) -> object:
        """Cut slot features into the horn solid using boolean subtraction."""

    def _apply_thread_bore(self, solid: object, thread: ThreadedStud,
                           face_selector: str) -> object:
        """Add threaded bore (simplified as smooth counterbored hole)."""

    def _apply_mounting_ring(self, solid: object,
                             params: HornGeometryParams) -> object:
        """Add mounting ring/flange at the nodal plane."""

    def _apply_edge_treatment(self, solid: object,
                              params: HornGeometryParams) -> object:
        """Apply chamfer or fillet to selected edges."""

    def _identify_boundary_faces(self, solid: object,
                                 params) -> dict[str, object]:
        """Tag faces by role: input, output, mounting, thread, symmetry."""
```

### 4.1.6 Assembly Builder

The assembly builder stacks components along a common axis. The algorithm:

1. Generate each component individually.
2. Query the output face centroid of component N and the input face centroid of
   component N+1.
3. Translate component N+1 so its input face is coincident with the output face of N.
4. Optionally model the threaded stud as a separate cylinder body inserted into the
   aligned bores.
5. Return the list of `GeometryResult` objects with updated positions.

Thread connections are modeled as simplified smooth bores (not helical geometry) because
helical thread geometry creates extreme mesh density requirements with minimal impact on
modal frequencies (<0.1% effect per validation studies). The thread engagement zone is
tagged as a boundary surface for contact modeling in the FEA solver.

---

## 4.2 CAD Import Pipeline

### 4.2.1 Overview

The CAD import pipeline handles externally-created STEP and IGES files, validates geometry
integrity, and identifies functional surfaces for boundary condition assignment.

**File:** `ultrasonic_weld_master/plugins/geometry_analyzer/cad_import.py`
**Estimated size:** ~650 lines

### 4.2.2 Data Structures

```python
@dataclass
class ImportedFace:
    """A face identified during import with its classified role."""
    face_index: int
    face_type: str                       # planar | cylindrical | conical | toroidal | bspline
    area_mm2: float
    centroid_mm: tuple[float, float, float]
    normal_avg: tuple[float, float, float]  # average outward normal
    role: str = "unknown"                # input | output | mounting | thread | lateral | unknown
    confidence: float = 0.0


@dataclass
class ImportedBody:
    """One solid body from an imported file."""
    body_index: int
    volume_mm3: float
    surface_area_mm2: float
    bounding_box: tuple[float, float, float, float, float, float]  # xmin..zmax
    center_of_mass: tuple[float, float, float]
    principal_axis: tuple[float, float, float]  # longest inertia axis
    faces: list[ImportedFace]
    classified_type: str = "unknown"     # horn | booster | transducer | fixture | unknown
    classification_confidence: float = 0.0


@dataclass
class CADImportResult:
    """Complete result from CAD file import."""
    bodies: list[ImportedBody]
    units: str = "mm"                    # detected or assumed units
    file_format: str = "STEP"
    repair_log: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_assembly: bool = False
```

### 4.2.3 Class Interface

```python
class CADImportPipeline:
    """Import, validate, repair, and classify STEP/IGES geometry."""

    def __init__(self):
        """Initialize. Requires CadQuery/OCP."""

    # --- Public API ---

    def import_file(self, file_path: str) -> CADImportResult:
        """Import STEP or IGES file. Auto-detects format by extension.

        Performs:
          1. File parsing via CadQuery importers
          2. Unit detection (mm vs inch vs m)
          3. Geometry validation and repair
          4. Multi-body separation
          5. Body classification (horn/booster/transducer)
          6. Face role assignment
        """

    def validate_and_repair(self, shape) -> tuple[object, list[str]]:
        """Check geometry validity and attempt repair.

        Checks:
          - Watertight (closed shell)
          - No self-intersections
          - Face normal consistency
          - Minimum edge length > 0.001 mm
          - Degenerate face removal

        Returns (repaired_shape, repair_log).
        """

    def classify_body(self, body: ImportedBody) -> tuple[str, float]:
        """Classify an imported body as horn/booster/transducer/fixture.

        Heuristics:
          - Aspect ratio (length vs cross-section diameter)
          - Cross-section area variation along principal axis
          - Presence of internal bore (transducer bolt hole)
          - Presence of annular ring features (PZT stack)
          - Presence of flange features (booster mounting)
        """

    def classify_faces(self, body: ImportedBody, shape) -> list[ImportedFace]:
        """Assign roles to each face of a classified body.

        Rules:
          - Faces at axial extremes with normal aligned to principal axis
            -> input (larger area) or output (smaller area)
          - Annular flat face near nodal plane -> mounting
          - Cylindrical bore faces near axis -> thread
          - Faces perpendicular to principal axis at midpoint -> symmetry candidates
        """

    def detect_units(self, shape) -> str:
        """Detect units from bounding box scale heuristic.

        If max dimension > 1000 -> assume meters, scale to mm.
        If max dimension < 1 -> assume meters, scale to mm.
        Otherwise assume mm.
        """

    # --- Internal ---

    def _parse_step(self, file_path: str) -> object:
        """Parse STEP file via cq.importers.importStep."""

    def _parse_iges(self, file_path: str) -> object:
        """Parse IGES file via OCP IGES reader."""

    def _separate_bodies(self, shape) -> list[object]:
        """Extract individual solids from a compound shape."""

    def _compute_body_properties(self, solid) -> dict:
        """Compute volume, surface area, bounding box, CoM, principal axis."""

    def _analyze_cross_section_variation(self, solid,
                                          axis: tuple) -> np.ndarray:
        """Slice solid perpendicular to axis, return area vs position array.

        Used for horn type classification:
          - Constant area -> cylindrical/flat
          - Monotonically decreasing -> exponential or catenoidal
          - Step change -> stepped
          - Large width/depth ratio -> blade
        """
```

### 4.2.4 Feature Recognition Algorithm

Body classification uses a decision tree on geometric features:

```
1. Compute principal inertia axis -> defines "axial" direction.
2. Slice the body into 20 cross-sections along the axis.
3. Compute area array A[0..19].
4. Compute area_ratio = max(A) / min(A).

Decision:
  if body has internal cylindrical bore along axis:
      if bore has annular rings (PZT slots) -> TRANSDUCER
      else -> candidate for threaded component
  if has annular flange near midpoint:
      -> BOOSTER (confidence 0.7)
  if area_ratio < 1.05:
      -> HORN type=cylindrical or flat (from cross-section shape)
  elif area_ratio > 1.05 and area_ratio < 1.5 and smooth variation:
      -> HORN type=exponential or catenoidal
  elif area_ratio > 1.5 and step transition exists:
      -> HORN type=stepped
  elif width/depth ratio > 3 for cross-sections:
      -> HORN type=blade
  else:
      -> HORN type=block
```

Horn type sub-classification (exponential vs catenoidal) fits `A(x)` to both profiles
using least-squares and selects the one with lower residual.

---

## 4.3 Gmsh Meshing Strategy

### 4.3.1 Overview

All meshing is performed through the Gmsh Python API (`gmsh` package). The mesher
accepts CadQuery solids (exported to temporary STEP files or passed via OCC integration),
applies physics-based mesh sizing, and outputs node/element arrays ready for the FEA
solver.

**File:** `ultrasonic_weld_master/plugins/geometry_analyzer/gmsh_mesher.py`
**Estimated size:** ~1,100 lines

### 4.3.2 Element Type Selection

| Geometry Source      | Default Element | Justification                          |
|----------------------|-----------------|----------------------------------------|
| Parametric, round    | HEX20           | Structured grid possible, better accuracy per DOF |
| Parametric, box      | HEX20           | Structured grid natural for rectangular shapes |
| Parametric with slots| TET10           | Slots break structured grid topology   |
| Imported geometry    | TET10           | Arbitrary geometry requires unstructured mesh |
| Assembly interfaces  | TET10           | Conforming interface mesh requires flexibility |

HEX20 (20-node serendipity hexahedra) provides approximately 2x accuracy per DOF compared
to TET10 (10-node quadratic tetrahedra) for the same mesh density. We use HEX20 whenever
the geometry permits structured meshing, and fall back to TET10 otherwise.

### 4.3.3 Mesh Density Requirements

For acoustic/modal analysis, the critical requirement is resolving the stress wavelength.
The minimum mesh density is derived from the shortest wavelength present:

```
lambda_min = c_bar / f_max

where:
  c_bar = sqrt(E / rho)      -- longitudinal bar wave speed
  f_max = 2 * f_operating    -- capture up to 2nd harmonic

elements_per_wavelength >= 6  (for second-order elements)
element_size_max = lambda_min / 6
```

For titanium at 20 kHz:
```
c_bar = 5068 m/s
lambda_min = 5068 / 40000 = 0.1267 m = 126.7 mm
element_size_max = 126.7 / 6 = 21.1 mm (global)
```

However, local refinement zones require much finer elements (see Section 4.3.5).

### 4.3.4 Data Structures

```python
@dataclass
class MeshSizeField:
    """Defines a spatial region with a target element size."""
    field_type: str          # "box" | "ball" | "distance" | "threshold"
    target_size_mm: float
    # Box parameters
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    # Ball parameters
    center: tuple[float, float, float] = (0, 0, 0)
    radius_mm: float = 0.0
    # Distance-from-entity parameters (used for edge/face refinement)
    entity_tags: list[int] = field(default_factory=list)
    entity_dim: int = 1      # 0=point, 1=edge, 2=face


@dataclass
class MeshConfig:
    """Configuration for mesh generation."""
    # Element type preference
    element_type: str = "auto"           # "auto" | "tet10" | "hex20"
    # Global sizing
    min_element_size_mm: float = 0.5
    max_element_size_mm: float = 20.0
    elements_per_wavelength: int = 6
    # Growth rate
    growth_rate: float = 1.3             # element size growth ratio
    # Refinement
    refine_fillets: bool = True
    refine_slot_ends: bool = True
    refine_thread_roots: bool = True
    refine_contact_surfaces: bool = True
    refine_geometric_transitions: bool = True
    refinement_factor: float = 0.25      # local size = global * factor
    # Quality
    min_aspect_ratio: float = 0.2        # reject elements below this
    min_jacobian: float = 0.3            # scaled Jacobian quality threshold
    # Mesh order
    element_order: int = 2               # 1=linear, 2=quadratic
    # Assembly
    conforming_interfaces: bool = True   # conforming mesh at component interfaces
    # Performance
    num_threads: int = 4                 # Gmsh OpenMP threads
    optimize_mesh: bool = True           # run Gmsh optimization passes
    optimization_passes: int = 3


@dataclass
class MeshQualityReport:
    """Statistics on mesh quality."""
    node_count: int
    element_count: int
    element_type: str                    # "TET10" or "HEX20"
    min_aspect_ratio: float
    max_aspect_ratio: float
    mean_aspect_ratio: float
    min_jacobian: float
    mean_jacobian: float
    num_poor_elements: int               # below quality threshold
    poor_element_percent: float
    bounding_box: tuple[float, ...]
    volume_mesh_mm3: float
    passed: bool                         # True if all metrics acceptable


@dataclass
class MeshResult:
    """Complete output from meshing."""
    nodes: np.ndarray                    # (N, 3) coordinates in mm
    elements: np.ndarray                 # (E, nodes_per_elem) connectivity
    element_type: str                    # "TET10" or "HEX20"
    node_count: int
    element_count: int
    # Physical group assignments
    surface_groups: dict[str, np.ndarray]  # name -> array of face element indices
    volume_groups: dict[str, np.ndarray]   # name -> array of volume element indices
    # Quality
    quality_report: MeshQualityReport
    # Tagged boundary faces (for BC application)
    boundary_faces: dict[str, np.ndarray]  # role -> face node arrays
    # Assembly interface data (if multi-body)
    interface_node_pairs: list[tuple[int, int]]  # tied contact node pairs
```

### 4.3.5 Adaptive Refinement Zones

The mesher automatically identifies and refines the following regions:

| Zone                       | Detection Method                              | Target Size                    |
|----------------------------|-----------------------------------------------|--------------------------------|
| Geometric transitions      | Cross-section area change > 10% over 5 mm     | `global_size * 0.25`           |
| Fillets and rounds         | Edges with curvature radius < 5 mm            | `min(fillet_radius/2, 1.0) mm` |
| Slot ends                  | Concave edges at slot terminations             | `slot_fillet_radius / 2 mm`    |
| Thread root regions        | Cylindrical bore entrance edges                | `thread_pitch / 3 mm`          |
| Contact/output surfaces    | Faces tagged as "output" role                  | `global_size * 0.5`            |
| Mounting ring transitions  | Edge ring at flange-to-body junction           | `global_size * 0.3`            |
| Assembly interfaces        | Face pairs tagged for tied contact              | `global_size * 0.5` (matched)  |

### 4.3.6 Class Interface

```python
class GmshMesher:
    """Gmsh-based mesher for acoustic stack components."""

    def __init__(self, config: MeshConfig = None):
        """
        Args:
            config: Mesh configuration. Uses sensible defaults if None.
        """

    # --- Public API ---

    def mesh_solid(
        self,
        geometry_result: GeometryResult,
        config: MeshConfig = None,
    ) -> MeshResult:
        """Mesh a single solid body.

        Steps:
          1. Export CQ solid to temporary STEP file
          2. Import into Gmsh
          3. Apply physical groups for boundary faces
          4. Compute wavelength-based global element size
          5. Apply adaptive refinement fields
          6. Generate mesh (structured or unstructured based on config)
          7. Extract nodes, elements, quality metrics
          8. Validate quality; re-mesh with tighter params if needed
        """

    def mesh_assembly(
        self,
        geometry_results: list[GeometryResult],
        config: MeshConfig = None,
    ) -> MeshResult:
        """Mesh a multi-body assembly with conforming interfaces.

        Steps:
          1. Import all bodies into single Gmsh model
          2. Fragment (boolean intersect) interface surfaces for conformity
          3. Apply matched sizing at interfaces
          4. Mesh each volume
          5. Extract interface node pairs for tied contact
        """

    def run_convergence_study(
        self,
        geometry_result: GeometryResult,
        target_frequency_hz: float,
        material: dict,
        refinement_levels: int = 4,
    ) -> list[dict]:
        """Run automated mesh convergence study.

        Meshes at progressively finer densities (elements_per_wavelength =
        4, 6, 8, 12), runs eigenvalue solve at each level, and reports
        frequency convergence.

        Returns list of dicts:
          [{
              "level": 1,
              "elements_per_wavelength": 4,
              "node_count": ...,
              "element_count": ...,
              "frequency_hz": ...,
              "frequency_change_percent": ...,  # vs previous level
              "converged": bool,                # change < 0.1%
          }, ...]
        """

    # --- Internal ---

    def _compute_global_size(self, material: dict,
                              frequency_hz: float) -> float:
        """Compute max element size from wavelength criterion."""

    def _setup_refinement_fields(
        self,
        geometry_result: GeometryResult,
        global_size: float,
    ) -> None:
        """Create Gmsh size fields for adaptive refinement zones."""

    def _apply_structured_mesh(self, volume_tag: int) -> bool:
        """Attempt transfinite (structured) meshing for HEX20.

        Returns True if successful, False if geometry is too complex.
        Falls back to TET10 on failure.
        """

    def _extract_mesh_data(self) -> tuple[np.ndarray, np.ndarray, str]:
        """Extract node coordinates and element connectivity from Gmsh."""

    def _compute_quality_metrics(
        self,
        nodes: np.ndarray,
        elements: np.ndarray,
        element_type: str,
    ) -> MeshQualityReport:
        """Compute aspect ratio and Jacobian quality for all elements."""

    def _create_physical_groups(
        self,
        geometry_result: GeometryResult,
    ) -> None:
        """Map boundary_faces from GeometryResult to Gmsh physical groups."""

    def _compute_interface_pairs(
        self,
        interface_face_tags: list[tuple[int, int]],
        tolerance_mm: float = 0.01,
    ) -> list[tuple[int, int]]:
        """Find matching node pairs across interface surfaces.

        Uses KD-tree for O(N log N) nearest-neighbor matching.
        """

    def _optimize_mesh(self) -> None:
        """Run Gmsh optimization: Laplacian smoothing + optimization passes."""
```

### 4.3.7 Mesh Convergence Study Protocol

The automated convergence study follows this protocol:

1. **Level 1** (coarse): `elements_per_wavelength = 4`. Solve for first 10 eigenvalues
   near target frequency.
2. **Level 2** (medium): `elements_per_wavelength = 6`. Compare fundamental longitudinal
   mode frequency to Level 1.
3. **Level 3** (fine): `elements_per_wavelength = 8`. Compare to Level 2.
4. **Level 4** (very fine): `elements_per_wavelength = 12`. Compare to Level 3.

Convergence criterion: frequency change < 0.1% between consecutive levels. The study
reports which level first achieves convergence and recommends that density for production
analysis.

Typical mesh sizes for a 126 mm long, 50 mm diameter cylindrical horn at 20 kHz:

| Level | Elem/wavelength | Nodes   | Elements | Solve Time (est.) |
|-------|-----------------|---------|----------|--------------------|
| 1     | 4               | ~3,000  | ~2,000   | <1 s               |
| 2     | 6               | ~12,000 | ~8,000   | ~3 s               |
| 3     | 8               | ~30,000 | ~20,000  | ~15 s              |
| 4     | 12              | ~100,000| ~65,000  | ~90 s              |

### 4.3.8 Assembly Meshing Strategy

For multi-body assemblies, two interface treatment options are supported:

**Option A: Conforming mesh (default)**
- Use Gmsh's `fragment()` (boolean fragmentation) to split interface surfaces into
  matching topologies.
- Apply identical mesh sizing on both sides of the interface.
- Result: shared nodes at the interface. No contact modeling needed.
- Advantage: simplest for eigenvalue analysis, no contact nonlinearity.
- Limitation: assumes perfect bonding (appropriate for torqued threaded joints).

**Option B: Tied contact (for interface studies)**
- Mesh each body independently.
- Find nearest-neighbor node pairs across the interface gap (<0.01 mm tolerance).
- Store pairs as multi-point constraints (MPCs) for the FEA solver.
- Advantage: allows modeling of interface compliance, friction effects.
- Use case: studying the effect of bolt preload on modal frequency.

---

## 4.4 Boundary Surface Identification

### 4.4.1 Overview

Automatic boundary surface identification tags geometric faces with their functional roles,
enabling automated boundary condition (BC) assignment in the FEA solver. This subsystem
operates on both parametrically-generated and imported geometry.

**File:** `ultrasonic_weld_master/plugins/geometry_analyzer/boundary_identifier.py`
**Estimated size:** ~500 lines

### 4.4.2 Boundary Roles

| Role              | Description                                     | Typical BC               |
|-------------------|-------------------------------------------------|--------------------------|
| `input`           | Face receiving vibration from upstream component | Prescribed displacement or free |
| `output`          | Contact face / weld tip                         | Free (modal) or force (harmonic) |
| `mounting_ring`   | Flange/ring at nodal plane for clamping         | Fixed in radial direction |
| `thread_bore`     | Internal cylindrical surface for stud           | Tied contact or coupled   |
| `lateral`         | Side surfaces (free in modal analysis)          | Free                     |
| `symmetry_xz`    | Symmetry plane normal to Y                      | UY=0 (symmetric BC)      |
| `symmetry_yz`    | Symmetry plane normal to X                      | UX=0 (symmetric BC)      |
| `interface`       | Mating face between assembly components         | Tied contact / conforming |

### 4.4.3 Data Structures

```python
@dataclass
class BoundaryFace:
    """A face tagged with its boundary role."""
    face_id: int                         # Gmsh physical surface tag or CQ face index
    role: str                            # from the role table above
    area_mm2: float
    centroid_mm: tuple[float, float, float]
    normal_mm: tuple[float, float, float]  # outward normal
    confidence: float                    # 0-1, detection confidence
    user_override: bool = False          # True if user manually assigned this role
    metadata: dict = field(default_factory=dict)  # extra info per role


@dataclass
class BoundaryIdentificationResult:
    """Complete boundary identification for a component or assembly."""
    faces: list[BoundaryFace]
    symmetry_planes: list[dict]          # [{axis, position_mm, normal}]
    warnings: list[str]
    component_type: str                  # horn | booster | transducer
```

### 4.4.4 Class Interface

```python
class BoundaryIdentifier:
    """Identify and tag boundary surfaces on acoustic stack components."""

    def __init__(self):
        """Initialize. No persistent state."""

    # --- Public API ---

    def identify(
        self,
        geometry_result: GeometryResult,
        component_type: str = "auto",
    ) -> BoundaryIdentificationResult:
        """Auto-detect boundary surfaces on a single component.

        If component_type is "auto", infers from geometry_result.component_type.
        """

    def identify_assembly(
        self,
        geometry_results: list[GeometryResult],
    ) -> list[BoundaryIdentificationResult]:
        """Identify boundaries across an assembled stack.

        Additionally identifies interface faces between adjacent components.
        """

    def apply_user_override(
        self,
        result: BoundaryIdentificationResult,
        face_id: int,
        new_role: str,
    ) -> BoundaryIdentificationResult:
        """Override the auto-detected role of a specific face.

        Marks the face with user_override=True so it is preserved
        across re-identification.
        """

    def detect_symmetry_planes(
        self,
        geometry_result: GeometryResult,
        tolerance: float = 0.01,
    ) -> list[dict]:
        """Detect geometric symmetry planes.

        Tests reflection symmetry about XZ, YZ, and XY planes through
        the center of mass. A plane is symmetric if the reflected body
        overlaps the original within tolerance (measured as volume
        difference < tolerance * total_volume).

        Returns list of {axis: str, position_mm: float, normal: (x,y,z)}.
        """

    # --- Internal ---

    def _identify_horn_faces(self, geometry_result: GeometryResult) -> list[BoundaryFace]:
        """Horn-specific face identification.

        Algorithm:
          1. Find the two faces with normals most aligned with the principal
             axis and at the axial extremes.
          2. The face with larger area -> "input".
          3. The face with smaller area -> "output".
          4. Annular faces near the nodal plane -> "mounting_ring".
          5. Internal cylindrical bores -> "thread_bore".
          6. Remaining external faces -> "lateral".
        """

    def _identify_booster_faces(self, geometry_result: GeometryResult) -> list[BoundaryFace]:
        """Booster-specific: input, output, flange, thread bores."""

    def _identify_transducer_faces(self, geometry_result: GeometryResult) -> list[BoundaryFace]:
        """Transducer-specific: back face, front face, PZT interfaces, bolt bore."""

    def _identify_interface_faces(
        self,
        result_a: GeometryResult,
        result_b: GeometryResult,
        tolerance_mm: float = 0.1,
    ) -> list[tuple[int, int]]:
        """Find coincident face pairs between two adjacent components.

        For each face on body A's output end, find the closest face on
        body B's input end. If face centroid distance < tolerance and
        normal directions are anti-parallel (dot product < -0.9), mark
        as an interface pair.
        """

    def _classify_face_geometry(self, face) -> str:
        """Classify face as planar | cylindrical | conical | toroidal | bspline."""
```

### 4.4.5 Detection Algorithm for Input/Output Faces

```
Given a solid with principal axis P (computed from moments of inertia):

1. For each face F in the solid:
   a. Compute face centroid C_f and average outward normal N_f.
   b. Compute alignment = |dot(N_f, P)|.
   c. Compute axial_position = dot(C_f - CoM, P).

2. Candidate input faces: alignment > 0.95 AND axial_position < 0
   (faces perpendicular to axis at the input end).

3. Candidate output faces: alignment > 0.95 AND axial_position > 0
   (faces perpendicular to axis at the output end).

4. Among candidates, select the single face with largest area as
   "input" and the single face at the opposite end as "output".

5. Confidence = alignment * (area / total_axial_face_area).
```

---

## 4.5 Integration with Existing Codebase

### 4.5.1 Relationship to Current Code

The design extends the existing codebase as follows:

| Existing File                                | Relationship                                      |
|----------------------------------------------|---------------------------------------------------|
| `geometry_analyzer/horn_generator.py`        | **Superseded** by `parametric_geometry.py`. The existing `HornGenerator` and `HornParams` become a thin compatibility wrapper that delegates to `ParametricGeometryGenerator`. |
| `geometry_analyzer/fea/material_properties.py` | **Unchanged.** Wave speed computation uses `FEA_MATERIALS` directly. |
| `web/services/geometry_service.py`           | **Extended.** `GeometryService` gains methods to call `CADImportPipeline` and `BoundaryIdentifier`. The existing `_classify_from_dimensions()` heuristics are replaced by the richer `CADImportPipeline.classify_body()`. |
| `web/services/fea_service.py`                | **Refactored.** The inline `_generate_hex_mesh()` is replaced by calls to `GmshMesher`. The existing HEX8 mesh is upgraded to TET10/HEX20. Eigenvalue solver and stress computation remain. |
| `web/routers/geometry.py`                    | **Extended** with new endpoints for parametric generation, assembly, and convergence studies. |
| `web/schemas/horn.py`                        | **Extended** with new request/response models for booster, transducer, and stack assembly. |
| `core/models.py`                             | **Extended.** `SonotrodeInfo` and `BoosterInfo` gain additional geometry fields to match the new parametric params. |

### 4.5.2 New File Layout

```
ultrasonic_weld_master/plugins/geometry_analyzer/
    __init__.py
    parametric_geometry.py       # Section 4.1  (~1,400 lines)
    cad_import.py                # Section 4.2  (~650 lines)
    gmsh_mesher.py               # Section 4.3  (~1,100 lines)
    boundary_identifier.py       # Section 4.4  (~500 lines)
    horn_generator.py            # Existing, becomes compatibility wrapper (~100 lines)
    fea/
        __init__.py
        material_properties.py   # Existing, unchanged
```

**Total estimated new code: ~3,650 lines** (excluding tests).
**Estimated test code: ~1,200 lines** in `tests/test_geometry/`.

### 4.5.3 Dependency Installation

```toml
# Addition to pyproject.toml [project.optional-dependencies]
fea = [
    "cadquery>=2.4",
    "gmsh>=4.12",
]
```

Both CadQuery and Gmsh are optional. The system degrades gracefully:
- Without CadQuery: parametric generation uses the existing numpy fallback; CAD import
  falls back to text-based STEP parsing (existing code).
- Without Gmsh: meshing uses the existing inline structured hex mesh generator in
  `fea_service.py` (limited to HEX8, no adaptive refinement).
- With both: full production-grade geometry and meshing.

### 4.5.4 Data Flow

```
User Input (parametric params OR STEP file)
    |
    v
[ParametricGeometryGenerator]  OR  [CADImportPipeline]
    |                                  |
    v                                  v
GeometryResult (CQ solid + boundary face tags)
    |
    v
[BoundaryIdentifier]
    |  tags faces with roles: input, output, mounting, thread, symmetry
    v
BoundaryIdentificationResult
    |
    v
[GmshMesher]
    |  generates TET10/HEX20 mesh with adaptive refinement
    |  applies physical groups for BC faces
    v
MeshResult (nodes, elements, boundary groups, quality report)
    |
    v
[FEA Solver] (existing eigsh/spsolve in fea_service.py,
               upgraded to use MeshResult instead of inline mesh)
```
