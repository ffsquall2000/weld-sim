# FEA Enhancement System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-grade, engineering-validation-grade FEA system for ultrasonic welding horn/booster/transducer analysis, deployed on server (180.152.71.166:8001), replacing ANSYS/COMSOL for daily design workflows.

**Architecture:** Hybrid solver system -- Solver A (numpy/scipy, self-developed) for daily modal/harmonic/stress/fatigue analysis + optional Solver B (FEniCSx, Docker) for piezoelectric coupling, nonlinear contact, and cross-validation. All computation server-side, results returned via REST API to frontend.

**Tech Stack:** Python 3.9+ / numpy / scipy / gmsh (Python API) / CadQuery (optional) / FastAPI / FEniCSx+DOLFINx 0.8 (Docker)

**Design Documents:**
- [Unified Design](./2026-02-27-fea-enhancement-design.md)
- [Geometry & Meshing](./2026-02-27-geometry-meshing-design.md)
- [Core Solver A](./2026-02-27-fea-core-solver-a-design.md)
- [FEniCSx Plugin B](./2026-02-27-fenicsx-plugin-design.md)
- [Material Database](../design/material_database_fatigue_assessment.md)
- [API & Scenarios](./2026-02-27-fea-api-endpoints-analysis-scenarios-design.md)
- [Implementation Roadmap](./2026-02-27-fea-phased-implementation-roadmap.md)

---

## Prerequisites

```bash
# Activate the existing venv
source .venv/bin/activate  # or: source venv/bin/activate on server

# Install new FEA dependencies
pip install gmsh>=4.12 meshio>=5.3

# Verify Gmsh works headless (no OpenGL needed)
python -c "import gmsh; gmsh.initialize(); gmsh.finalize(); print('Gmsh OK')"

# Verify existing tests still pass
pytest tests/ -x -q
```

---

## Phase 1: Foundation (Weeks 1-2)

**Goal:** Gmsh-based meshing with TET10 quadratic elements, solver interface abstraction, enhanced material database. This is the foundation everything else depends on.

**Dependency graph:**
```
Task 1 (config+results) ──► Task 3 (mesher) ──► Task 5 (integration)
Task 2 (materials)       ──► Task 3 (mesher)
Task 1 (config+results) ──► Task 4 (elements) ──► Task 5 (integration)
Task 1 (config+results) ──► Task 5 (integration)
```

---

### Task 1: FEA Configuration and Result Dataclasses

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/__init__.py`
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/config.py`
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/results.py`
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_interface.py`
- Create: `tests/test_fea/__init__.py`
- Create: `tests/test_fea/test_config.py`

**Step 1: Create the fea package and __init__.py**

```python
# ultrasonic_weld_master/plugins/geometry_analyzer/fea/__init__.py
"""FEA module for ultrasonic welding component analysis."""
```

**Step 2: Write failing tests for config dataclasses**

```python
# tests/test_fea/__init__.py
```

```python
# tests/test_fea/test_config.py
"""Tests for FEA configuration and result dataclasses."""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    ModalConfig,
    HarmonicConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ModalResult,
    HarmonicResult,
    StaticResult,
    FatigueResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_interface import (
    SolverInterface,
)


class TestFEAMesh:
    def test_create_mesh(self):
        mesh = FEAMesh(
            nodes=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
            elements=np.array([[0, 1, 2, 3]]),
            element_type="TET4",
            node_sets={"bottom": np.array([0, 1])},
            element_sets={},
            surface_tris=np.array([[0, 1, 2]]),
            mesh_stats={"num_nodes": 4, "num_elements": 1},
        )
        assert mesh.nodes.shape == (4, 3)
        assert mesh.element_type == "TET4"
        assert len(mesh.node_sets["bottom"]) == 2

    def test_mesh_n_dof(self):
        mesh = FEAMesh(
            nodes=np.zeros((100, 3)),
            elements=np.zeros((50, 10), dtype=int),
            element_type="TET10",
            node_sets={},
            element_sets={},
            surface_tris=np.zeros((20, 3), dtype=int),
            mesh_stats={},
        )
        assert mesh.n_dof == 300  # 100 nodes * 3 DOF


class TestModalConfig:
    def test_defaults(self):
        mesh = FEAMesh(
            nodes=np.zeros((4, 3)),
            elements=np.zeros((1, 4), dtype=int),
            element_type="TET4",
            node_sets={}, element_sets={},
            surface_tris=np.zeros((1, 3), dtype=int),
            mesh_stats={},
        )
        config = ModalConfig(mesh=mesh, material_name="Titanium Ti-6Al-4V")
        assert config.n_modes == 20
        assert config.target_frequency_hz == 20000.0
        assert config.boundary_conditions == "free-free"


class TestModalResult:
    def test_create_result(self):
        result = ModalResult(
            frequencies_hz=np.array([19500.0, 20100.0]),
            mode_shapes=np.zeros((2, 30)),
            mode_types=["longitudinal", "flexural"],
            effective_mass_ratios=np.array([0.85, 0.05]),
            mesh=None,
            solve_time_s=1.5,
            solver_name="SolverA",
        )
        assert len(result.frequencies_hz) == 2
        assert result.mode_types[0] == "longitudinal"


class TestSolverInterface:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            SolverInterface()  # Cannot instantiate abstract class
```

**Step 3: Run tests to verify they fail**

Run: `pytest tests/test_fea/test_config.py -v`
Expected: FAIL (ImportError -- modules don't exist yet)

**Step 4: Implement config.py**

```python
# ultrasonic_weld_master/plugins/geometry_analyzer/fea/config.py
"""FEA configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FEAMesh:
    """Container for a finite element mesh."""
    nodes: np.ndarray              # (N, 3) coordinates in meters
    elements: np.ndarray           # (E, nodes_per_elem) connectivity
    element_type: str              # "TET4", "TET10", "HEX8", "HEX20"
    node_sets: dict[str, np.ndarray]
    element_sets: dict[str, np.ndarray]
    surface_tris: np.ndarray       # (F, 3) for visualization
    mesh_stats: dict

    @property
    def n_dof(self) -> int:
        """Total degrees of freedom (3 per node)."""
        return self.nodes.shape[0] * 3


@dataclass
class ModalConfig:
    """Configuration for modal (eigenvalue) analysis."""
    mesh: FEAMesh
    material_name: str
    n_modes: int = 20
    target_frequency_hz: float = 20000.0
    boundary_conditions: str = "free-free"  # "free-free" | "clamped" | "pre-stressed"
    fixed_node_sets: list[str] = field(default_factory=list)


@dataclass
class HarmonicConfig:
    """Configuration for harmonic response analysis."""
    mesh: FEAMesh
    material_name: str
    freq_min_hz: float = 16000.0
    freq_max_hz: float = 24000.0
    n_freq_points: int = 201
    damping_model: str = "hysteretic"  # "hysteretic" | "rayleigh" | "modal"
    damping_ratio: float = 0.005
    excitation_node_set: str = "bottom_face"
    response_node_set: str = "top_face"


@dataclass
class StaticConfig:
    """Configuration for static stress analysis."""
    mesh: FEAMesh
    material_name: str
    loads: list[dict] = field(default_factory=list)
    boundary_conditions: list[dict] = field(default_factory=list)
```

**Step 5: Implement results.py**

```python
# ultrasonic_weld_master/plugins/geometry_analyzer/fea/results.py
"""FEA result container dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ModalResult:
    """Modal analysis result."""
    frequencies_hz: np.ndarray          # (n_modes,)
    mode_shapes: np.ndarray             # (n_modes, n_dof)
    mode_types: list[str]               # "longitudinal", "flexural", "torsional"
    effective_mass_ratios: np.ndarray
    mesh: Optional[object]              # FEAMesh reference
    solve_time_s: float
    solver_name: str


@dataclass
class HarmonicResult:
    """Harmonic response result."""
    frequencies_hz: np.ndarray          # sweep frequencies
    displacement_amplitudes: np.ndarray # (n_freq, n_dof) complex
    contact_face_uniformity: float
    gain: float
    q_factor: float
    mesh: Optional[object]
    solve_time_s: float
    solver_name: str


@dataclass
class StaticResult:
    """Static analysis result."""
    displacement: np.ndarray            # (n_dof,)
    stress_vm: np.ndarray               # (n_elements,) Von Mises at centroids
    stress_tensor: np.ndarray           # (n_elements, 6) Voigt notation
    max_stress_mpa: float
    mesh: Optional[object]
    solve_time_s: float
    solver_name: str


@dataclass
class FatigueResult:
    """Fatigue assessment result."""
    safety_factors: np.ndarray          # per-element
    min_safety_factor: float
    critical_location: np.ndarray       # [x, y, z]
    estimated_life_cycles: float
    sn_curve_name: str
```

**Step 6: Implement solver_interface.py**

```python
# ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_interface.py
"""Abstract solver interface for FEA backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    ModalConfig,
    HarmonicConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ModalResult,
    HarmonicResult,
    StaticResult,
)


class SolverInterface(ABC):
    """Abstract base for FEA solvers."""

    @abstractmethod
    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Run eigenvalue analysis."""
        ...

    @abstractmethod
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Run harmonic response analysis."""
        ...

    @abstractmethod
    def static_analysis(self, config: StaticConfig) -> StaticResult:
        """Run static stress analysis."""
        ...
```

**Step 7: Run tests to verify they pass**

Run: `pytest tests/test_fea/test_config.py -v`
Expected: All 4 tests PASS

**Step 8: Commit**

```bash
git add ultrasonic_weld_master/plugins/geometry_analyzer/fea/__init__.py \
  ultrasonic_weld_master/plugins/geometry_analyzer/fea/config.py \
  ultrasonic_weld_master/plugins/geometry_analyzer/fea/results.py \
  ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_interface.py \
  tests/test_fea/__init__.py tests/test_fea/test_config.py
git commit -m "feat(fea): add config, results, and solver interface dataclasses"
```

---

### Task 2: Enhanced Material Database

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/material_properties.py`
- Create: `tests/test_fea/test_materials.py`

**Step 1: Write failing tests for new material properties**

```python
# tests/test_fea/test_materials.py
"""Tests for enhanced material database."""
from __future__ import annotations

import math
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
    list_materials,
    FEA_MATERIALS,
)


class TestExistingMaterials:
    """Ensure backward compatibility."""

    def test_titanium_exists(self):
        mat = get_material("Titanium Ti-6Al-4V")
        assert mat is not None
        assert mat["E_pa"] == 113.8e9

    def test_alias_lookup(self):
        mat = get_material("ti64")
        assert mat is not None
        assert mat["E_pa"] == 113.8e9


class TestNewProperties:
    """Test new properties added to existing materials."""

    @pytest.mark.parametrize("name", [
        "Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
    ])
    def test_damping_ratio_present(self, name):
        mat = get_material(name)
        assert "damping_ratio" in mat
        assert 0 < mat["damping_ratio"] < 0.1

    @pytest.mark.parametrize("name", [
        "Titanium Ti-6Al-4V", "Steel D2", "Aluminum 7075-T6",
    ])
    def test_acoustic_velocity_consistent(self, name):
        mat = get_material(name)
        assert "acoustic_velocity_m_s" in mat
        # Verify c = sqrt(E/rho) within 5%
        c_calc = math.sqrt(mat["E_pa"] / mat["rho_kg_m3"])
        assert abs(mat["acoustic_velocity_m_s"] - c_calc) / c_calc < 0.05

    def test_all_materials_have_damping(self):
        for name in list_materials():
            mat = get_material(name)
            assert "damping_ratio" in mat, f"{name} missing damping_ratio"


class TestNewMaterials:
    """Test newly added materials (PZT, Steel 4140, etc)."""

    def test_pzt4_exists(self):
        mat = get_material("PZT-4")
        assert mat is not None
        assert "d33" in mat

    def test_pzt8_exists(self):
        mat = get_material("PZT-8")
        assert mat is not None
        assert "d33" in mat

    def test_steel_4140_exists(self):
        mat = get_material("Steel 4140")
        assert mat is not None
        assert mat["E_pa"] > 190e9
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fea/test_materials.py -v`
Expected: FAIL (missing damping_ratio, acoustic_velocity_m_s, PZT materials)

**Step 3: Enhance material_properties.py**

Add to each existing material entry:
- `damping_ratio` (loss factor eta)
- `acoustic_velocity_m_s` (longitudinal bar wave speed)
- `fatigue_endurance_mpa` (endurance limit at 10^7 cycles)

Add new materials: PZT-4, PZT-8, Steel 4140, Ferro-Titanit WFN.

The exact values come from the material database design doc: `docs/design/material_database_fatigue_assessment.md`

Key values to add:
```python
# Add to "Titanium Ti-6Al-4V":
"damping_ratio": 0.003,  # eta = 1/Q, Q=5000 at low strain -> eta ≈ 0.0002, but effective eta in ultrasonic horns is ~0.003
"acoustic_velocity_m_s": 5068.0,  # sqrt(113.8e9 / 4430)
"fatigue_endurance_mpa": 510.0,  # at 1e7 cycles, R=-1

# Add to "Steel D2":
"damping_ratio": 0.005,
"acoustic_velocity_m_s": 5222.0,  # sqrt(210e9 / 7700)
"fatigue_endurance_mpa": 750.0,

# Add to "Aluminum 7075-T6":
"damping_ratio": 0.002,
"acoustic_velocity_m_s": 5050.0,  # sqrt(71.7e9 / 2810)
"fatigue_endurance_mpa": 159.0,

# New PZT-4 material:
"PZT-4": {
    "E_pa": 81.3e9,  # Young's modulus (average, from compliance matrix)
    "nu": 0.31,
    "rho_kg_m3": 7500.0,
    "k_w_mk": 2.1,
    "cp_j_kgk": 420.0,
    "yield_mpa": 80.0,  # compressive strength (brittle)
    "alpha_1_k": 4.0e-6,
    "damping_ratio": 0.004,  # 1/Qm, Qm ~ 500
    "acoustic_velocity_m_s": 3293.0,
    "fatigue_endurance_mpa": 25.0,
    "d33": 289e-12,  # piezoelectric constant [C/N]
    "d31": -123e-12,
    "eps_33": 1300.0,  # relative permittivity
    "k_t": 0.51,  # thickness coupling coefficient
    "is_piezoelectric": True,
},

# New PZT-8:
"PZT-8": {
    "E_pa": 86.9e9,
    "nu": 0.31,
    "rho_kg_m3": 7600.0,
    "k_w_mk": 2.1,
    "cp_j_kgk": 420.0,
    "yield_mpa": 80.0,
    "alpha_1_k": 4.0e-6,
    "damping_ratio": 0.001,  # 1/Qm, Qm ~ 1000
    "acoustic_velocity_m_s": 3381.0,
    "fatigue_endurance_mpa": 25.0,
    "d33": 225e-12,
    "d31": -97e-12,
    "eps_33": 1000.0,
    "k_t": 0.48,
    "is_piezoelectric": True,
},

# New Steel 4140:
"Steel 4140": {
    "E_pa": 200.0e9,
    "nu": 0.29,
    "rho_kg_m3": 7850.0,
    "k_w_mk": 42.6,
    "cp_j_kgk": 473.0,
    "yield_mpa": 1170.0,
    "alpha_1_k": 12.3e-6,
    "damping_ratio": 0.004,
    "acoustic_velocity_m_s": 5048.0,
    "fatigue_endurance_mpa": 550.0,
},
```

Also add aliases for new materials and update `damping_ratio` + `acoustic_velocity_m_s` + `fatigue_endurance_mpa` for ALL existing materials (Copper, Nickel, M2 HSS, CPM 10V, PM60, HAP40, HAP72).

**Step 4: Run tests**

Run: `pytest tests/test_fea/test_materials.py -v`
Expected: All PASS

**Step 5: Run existing tests to verify no regression**

Run: `pytest tests/ -x -q`
Expected: All existing tests PASS

**Step 6: Commit**

```bash
git add ultrasonic_weld_master/plugins/geometry_analyzer/fea/material_properties.py \
  tests/test_fea/test_materials.py
git commit -m "feat(fea): enhance material database with damping, acoustic velocity, PZT materials"
```

---

### Task 3: Gmsh Mesh Generation

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py`
- Create: `tests/test_fea/test_mesher.py`

**Step 1: Write failing tests**

```python
# tests/test_fea/test_mesher.py
"""Tests for Gmsh-based mesh generation."""
from __future__ import annotations

import numpy as np
import pytest

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.mesher import GmshMesher
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import FEAMesh

# Skip all tests if gmsh not installed
gmsh = pytest.importorskip("gmsh")


class TestGmshMesher:
    def setup_method(self):
        self.mesher = GmshMesher()

    def test_mesh_cylinder_tet4(self):
        """Generate a TET4 mesh of a simple cylinder."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=1,
        )
        assert isinstance(mesh, FEAMesh)
        assert mesh.element_type == "TET4"
        assert mesh.nodes.shape[1] == 3
        assert mesh.elements.shape[1] == 4
        assert mesh.nodes.shape[0] > 50  # should have reasonable node count
        assert mesh.mesh_stats["num_nodes"] == mesh.nodes.shape[0]

    def test_mesh_cylinder_tet10(self):
        """Generate a TET10 mesh (quadratic elements)."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.element_type == "TET10"
        assert mesh.elements.shape[1] == 10

    def test_mesh_box_horn(self):
        """Generate mesh for a rectangular block horn."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="flat",
            dimensions={"width_mm": 30.0, "depth_mm": 20.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.nodes.shape[0] > 100

    def test_node_sets_identified(self):
        """Top and bottom face node sets should be auto-detected."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=2,
        )
        assert "top_face" in mesh.node_sets
        assert "bottom_face" in mesh.node_sets
        assert len(mesh.node_sets["top_face"]) > 0
        assert len(mesh.node_sets["bottom_face"]) > 0

    def test_bounding_box_correct(self):
        """Mesh bounding box should match input dimensions."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=2,
        )
        # Nodes in meters, so 80mm = 0.08m
        bbox_min = mesh.nodes.min(axis=0)
        bbox_max = mesh.nodes.max(axis=0)
        length = bbox_max[1] - bbox_min[1]  # y-axis = longitudinal
        assert abs(length - 0.080) < 0.001

    def test_no_inverted_elements(self):
        """All elements should have positive volume (no inverted elements)."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=4.0,
            order=1,
        )
        # Check TET4 element volumes are all positive
        for elem in mesh.elements:
            v0, v1, v2, v3 = mesh.nodes[elem]
            vol = np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0
            assert vol > 0, "Inverted element detected"

    def test_surface_tris_generated(self):
        """Surface triangulation should be present for visualization."""
        mesh = self.mesher.mesh_parametric_horn(
            horn_type="cylindrical",
            dimensions={"diameter_mm": 25.0, "length_mm": 80.0},
            mesh_size=5.0,
            order=2,
        )
        assert mesh.surface_tris.shape[1] == 3
        assert mesh.surface_tris.shape[0] > 10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fea/test_mesher.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement mesher.py**

The mesher uses Gmsh Python API:
- `gmsh.model.occ.addCylinder` for cylindrical horns
- `gmsh.model.occ.addBox` for flat/block horns
- `gmsh.model.mesh.generate(3)` for 3D mesh generation
- `gmsh.model.mesh.setOrder(2)` for TET10 elements
- `gmsh.model.mesh.getNodes()` / `gmsh.model.mesh.getElements()` to extract mesh data
- Headless mode: `gmsh.option.setNumber("General.Terminal", 0)`

Key implementation details:
- Coordinates stored in meters (divide mm by 1000)
- Y-axis is the longitudinal axis (horn axis of revolution)
- Node sets identified geometrically: top_face = nodes where y == y_max ± tol, bottom_face = y == y_min ± tol
- Surface triangulation extracted from Gmsh's surface mesh (element type 2 = triangle)

See design doc `docs/plans/2026-02-27-geometry-meshing-design.md` Section 4.3 for complete Gmsh meshing strategy.

**Step 4: Run tests**

Run: `pytest tests/test_fea/test_mesher.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesher.py \
  tests/test_fea/test_mesher.py
git commit -m "feat(fea): add Gmsh-based mesh generation with TET4/TET10 support"
```

---

### Task 4: TET10 Quadratic Element

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/elements.py`
- Create: `tests/test_fea/test_elements.py`

**Step 1: Write failing tests**

```python
# tests/test_fea/test_elements.py
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
        """N_i(node_j) = delta_ij at each of the 10 nodes."""
        # Natural coordinates of the 10 TET10 nodes
        nat_coords = self.elem.NODE_NATURAL_COORDS  # (10, 3)
        for i, (xi, eta, zeta) in enumerate(nat_coords):
            N = self.elem.shape_functions(xi, eta, zeta)
            for j in range(10):
                expected = 1.0 if i == j else 0.0
                assert abs(N[j] - expected) < 1e-12, f"N_{j}({i}) = {N[j]}, expected {expected}"

    def test_shape_derivatives_sum_to_zero(self):
        """Sum of dN/d(xi) over all nodes = 0 (constant field has zero gradient)."""
        xi, eta, zeta = 0.25, 0.25, 0.25
        dN = self.elem.shape_derivatives(xi, eta, zeta)
        assert dN.shape == (3, 10)
        assert np.allclose(dN.sum(axis=1), 0.0, atol=1e-12)


class TestTET10StiffnessMatrix:
    def setup_method(self):
        self.elem = TET10Element()
        # Regular tetrahedron (10 nodes with mid-edge nodes)
        self.coords = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],  # corners
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],      # mid-edge
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],    # mid-edge
        ], dtype=float)
        # Titanium isotropic elasticity matrix (6x6 Voigt)
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
        """Element stiffness matrix must be symmetric."""
        Ke = self.elem.stiffness_matrix(self.coords, self.D)
        assert Ke.shape == (30, 30)
        assert np.allclose(Ke, Ke.T, atol=1e-6)

    def test_stiffness_positive_semidefinite(self):
        """K_e should have 6 zero eigenvalues (rigid body modes) and 24 positive."""
        Ke = self.elem.stiffness_matrix(self.coords, self.D)
        eigvals = np.linalg.eigvalsh(Ke)
        assert np.sum(eigvals < 1e-6 * eigvals.max()) == 6  # 6 rigid body modes
        assert np.all(eigvals[6:] > 0)


class TestTET10MassMatrix:
    def setup_method(self):
        self.elem = TET10Element()
        self.coords = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],
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
        """Sum of diagonal in x-direction should equal rho * V."""
        Me = self.elem.mass_matrix(self.coords, rho=4430.0)
        # Volume of unit tet = 1/6
        expected_mass = 4430.0 * (1.0 / 6.0)
        # Sum of M[3i,3i] for i=0..9 (x-DOF diagonal) = total mass
        total_mass = sum(Me[3*i, 3*i] for i in range(10))
        # For consistent mass, diagonal sum != total mass, but row sum does
        # Check trace/3 = total mass
        trace_third = np.trace(Me) / 3.0
        assert abs(trace_third - expected_mass) / expected_mass < 0.01


class TestTET10PatchTest:
    """The patch test: a constant strain field must be exactly reproduced."""

    def test_uniform_tension(self):
        """Under uniform x-tension, stress should be exactly sigma_x = E * epsilon_x."""
        elem = TET10Element()
        coords = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5],
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

        # Apply uniform strain epsilon_x = 0.001
        eps_x = 0.001
        # Displacement field: u_x = eps_x * x, u_y = -nu*eps_x*y, u_z = -nu*eps_x*z
        u_e = np.zeros(30)
        for i in range(10):
            u_e[3*i] = eps_x * coords[i, 0]
            u_e[3*i+1] = -nu * eps_x * coords[i, 1]
            u_e[3*i+2] = -nu * eps_x * coords[i, 2]

        # Check stress at centroid
        stress = elem.stress_at_point(coords, u_e, D, 0.25, 0.25, 0.25)
        expected_sxx = E_val * eps_x  # For uniaxial, sigma_x = E * eps_x (Poisson accounted in D)
        # Actually sigma = D * epsilon, so sigma_x = (lam+2mu)*eps_x + lam*(-nu*eps_x) + lam*(-nu*eps_x)
        expected = D @ np.array([eps_x, -nu*eps_x, -nu*eps_x, 0, 0, 0])
        assert np.allclose(stress, expected, rtol=1e-10)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fea/test_elements.py -v`
Expected: FAIL (ImportError)

**Step 3: Implement elements.py**

Implement the TET10 element with:
- 10 shape functions (4 corner + 6 mid-edge, quadratic in natural coords L1,L2,L3,L4)
- Shape function derivatives (3x10 matrix)
- 4-point Gauss quadrature (Keast rule, degree 2)
- Stiffness matrix: K_e = sum_gp(B^T D B det(J) w) -- 30x30
- Consistent mass matrix: M_e = sum_gp(rho N^T N det(J) w) -- 30x30
- B-matrix (strain-displacement): 6x30
- Stress computation: sigma = D * B * u_e

See design doc Section 2 (`fea-core-solver-a-design.md`), specifically sections on TET10 element formulation.

**Step 4: Run tests**

Run: `pytest tests/test_fea/test_elements.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add ultrasonic_weld_master/plugins/geometry_analyzer/fea/elements.py \
  tests/test_fea/test_elements.py
git commit -m "feat(fea): implement TET10 quadratic element with shape functions, K, M matrices"
```

---

### Task 5: Phase 1 Integration Test

**Files:**
- Create: `tests/test_fea/test_phase1_integration.py`

**Step 1: Write integration test**

```python
# tests/test_fea/test_phase1_integration.py
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
```

**Step 2: Run integration tests**

Run: `pytest tests/test_fea/test_phase1_integration.py -v`
Expected: All PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All PASS (including existing tests)

**Step 4: Commit**

```bash
git add tests/test_fea/test_phase1_integration.py
git commit -m "test(fea): add Phase 1 integration tests for mesh-element-material pipeline"
```

---

### Task 6: Update requirements.txt and deploy to server

**Files:**
- Modify: `requirements.txt`

**Step 1: Update requirements**

Add `gmsh>=4.12` to requirements.txt.

**Step 2: Verify local tests pass**

Run: `pytest tests/test_fea/ -v`
Expected: All PASS

**Step 3: Commit and push**

```bash
git add requirements.txt
git commit -m "chore: add gmsh dependency for FEA meshing"
git push origin claude/fervent-cannon
```

**Step 4: Deploy to server**

```bash
ssh root@180.152.71.166
cd /opt/weld-sim
git pull origin claude/fervent-cannon
source venv/bin/activate
pip install -r requirements.txt
systemctl restart weld-sim
```

---

## Phase 2: Core Modal Analysis (Weeks 3-4)

**Goal:** Global assembly + eigenvalue solver + mode classification. The most critical capability.

### Task 7: Global Matrix Assembly

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/assembler.py` (~350 LOC)
- Create: `tests/test_fea/test_assembler.py`

Key implementation:
- COO format accumulation → CSR conversion
- DOF numbering: `[u_x0, u_y0, u_z0, u_x1, u_y1, u_z1, ...]`
- Vectorized element loop for performance (target: 10K node mesh < 5s)
- Returns `(K_global, M_global)` as `scipy.sparse.csr_matrix`

Key tests:
- Symmetry of K and M
- Correct matrix dimensions (3*n_nodes x 3*n_nodes)
- Assembly of 2 elements sharing nodes: summed correctly
- Performance benchmark: 5K node mesh < 3 seconds

### Task 8: Modal Solver (SolverA.modal_analysis)

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` (~600 LOC)
- Create: `tests/test_fea/test_solver_a.py`

Key implementation:
- `scipy.sparse.linalg.eigsh` with shift-invert at sigma = (2*pi*f_target)^2
- Free-free: request n_modes+6, discard modes where f < 100 Hz
- Clamped: zero rows/cols for constrained DOFs using penalty method
- Returns `ModalResult` with sorted frequencies and classified modes

Key tests:
- Uniform cylinder free-free: f1 = c/(2L) within 1% for Ti-6Al-4V, Al 7075-T6, Steel D2
- 6 rigid-body modes at ~0 Hz discarded
- Modes sorted by frequency ascending

### Task 9: Mode Classifier

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mode_classifier.py` (~400 LOC)
- Create: `tests/test_fea/test_mode_classifier.py`

Key implementation:
- Displacement ratio classification: longitudinal (d_y dominant), flexural (d_x or d_z dominant), torsional (angular momentum dominant)
- Parasitic mode detection: non-longitudinal modes within 500 Hz of target
- Nodal plane identification: zero-crossings of axial displacement along y-axis

Key tests:
- Classify first mode of uniform cylinder as "longitudinal"
- Detect flexural parasitic mode near target frequency
- Find nodal plane at L/2 for half-wave resonator

### Task 10: Analytical Validation Benchmarks

**Files:**
- Create: `tests/test_fea/benchmarks/` directory with JSON golden data
- Create: `tests/test_fea/test_modal_validation.py`

Key tests (using analytical formula f_n = n * c / (2*L)):
- Ti-6Al-4V uniform bar @ 20 kHz: f1 error < 1%
- Al 7075-T6 uniform bar @ 20 kHz: f1 error < 1%
- Steel D2 uniform bar @ 20 kHz: f1 error < 1%
- Mesh convergence: 3 refinement levels → monotonic convergence

### Task 11: Web API Integration (use_gmsh flag)

**Files:**
- Modify: `web/services/fea_service.py` (~100 LOC change)
- Modify: `web/routers/acoustic.py` (add `use_gmsh` field to request)

Add `use_gmsh: bool = False` flag. When True, route through new Gmsh + TET10 + SolverA pipeline. Old HEX8 path unchanged.

---

## Phase 3: Harmonic Response + Amplitude (Weeks 5-6)

### Task 12: Damping Models

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/damping.py` (~200 LOC)
- Create: `tests/test_fea/test_damping.py`

Implement: HystereticDamping (constant eta), RayleighDamping (alpha*M + beta*K), ModalDamping.

### Task 13: Harmonic Response Solver

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` (add ~300 LOC)
- Create: `tests/test_fea/test_harmonic_validation.py`

Two methods: direct solve (Z^{-1}*F) for narrow-band, modal superposition for wide-band.

### Task 14: Post-Processing (Gain, Uniformity, FRF)

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/post_processing.py` (~400 LOC)
- Create: `tests/test_fea/test_post_processing.py`

AmplitudeAnalyzer: gain = output_amplitude/input_amplitude, uniformity = 1 - std/mean, FRF curve, Q-factor from 3dB bandwidth.

---

## Phase 4: Static Stress + Fatigue (Weeks 7-8)

### Task 15: Static Stress Solver

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` (add ~200 LOC)

Solve Ku=F with pressure, force, bolt_preload, gravity loads.

### Task 16: Stress Recovery (Von Mises, SPR, Hotspots)

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/stress_recovery.py` (~400 LOC)
- Create: `tests/test_fea/test_stress_validation.py`

### Task 17: Fatigue Assessment Module

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/fatigue.py` (~350 LOC)
- Create: `tests/test_fea/test_fatigue.py`

S-N interpolation, Goodman diagram, Marin correction factors, safety factor calculator.

### Task 18: Pre-Stressed Modal Analysis

**Files:**
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_a.py` (add ~150 LOC)

Geometric stiffness K_sigma from static preload → (K + K_sigma - w^2 M) phi = 0.

---

## Phase 5: Booster + Assembly (Weeks 9-10)

### Task 19: Booster Parametric Geometry

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/booster_generator.py` (~300 LOC)
- Create: `tests/test_fea/test_booster.py`

Stepped, exponential, catenoidal profiles. Auto half-wavelength length.

### Task 20: Assembly Builder (Multi-Body Coupling)

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/assembly_builder.py` (~500 LOC)
- Create: `tests/test_fea/test_assembly.py`

Bonded (shared DOF), tied (MPC penalty), component merging, material map.

### Task 21: Full-Stack Workflow + Gain Chain

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/workflow.py` (~300 LOC)
- Modify: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/post_processing.py` (~150 LOC)

End-to-end: assemble → modal → harmonic → stress → fatigue → gain chain report.

### Task 22: Assembly API Endpoints

**Files:**
- Create: `web/routers/assembly.py` (~200 LOC)
- Modify: `web/services/fea_service.py` (~150 LOC)

---

## Phase 6: FEniCSx Plugin (Weeks 11-12)

### Task 23: Docker Deployment for FEniCSx

**Files:**
- Create: `docker/Dockerfile.fenics`
- Create: `docker/docker-compose.yml`

Based on `dolfinx/dolfinx:v0.8.0` image. Internal API on port 8002.

### Task 24: SolverB Implementation (FEniCSx Backend)

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/solver_b.py` (~700 LOC)
- Create: `tests/test_fea/test_solver_b.py`

Modal + harmonic via FEniCSx/SLEPc/PETSc. Piezoelectric coupled analysis. Impedance spectrum.

### Task 25: Cross-Validation Harness

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/cross_validation.py` (~250 LOC)
- Create: `tests/test_fea/test_cross_validation.py`

MAC matrix, frequency deviation, stress peak comparison. PASS/WARNING/FAIL thresholds.

---

## Phase 7: Advanced Features + Polish (Weeks 13-14)

### Task 26: Thermal Coupling (SolverB)

Modify `solver_b.py` (+~250 LOC). Hysteretic + frictional heat sources, BDF2 time integration.

### Task 27: Contact Analysis (SolverB)

Modify `solver_b.py` (+~300 LOC). Augmented Lagrangian + Coulomb friction, Newton solver.

### Task 28: Mesh Convergence Automation

**Files:**
- Create: `ultrasonic_weld_master/plugins/geometry_analyzer/fea/mesh_convergence.py` (~300 LOC)

3-5 refinement levels, Richardson extrapolation, recommended mesh size.

### Task 29: STEP Import Enhancement

Modify `mesher.py` (+~100 LOC). Multi-body STEP, auto face identification, defeaturing.

### Task 30: Legacy Migration Completion

Modify `web/services/fea_service.py`. Switch `use_gmsh` default to True. Remove old HEX8 codepath (~300 LOC deleted).

### Task 31: Server Deployment (Full Stack)

Deploy Phases 1-5 via pip. Deploy Phase 6+ via Docker. Production docker-compose for weld-sim + fenicsx-worker.

```bash
# On server 180.152.71.166
cd /opt/weld-sim
git pull origin claude/fervent-cannon
source venv/bin/activate
pip install -r requirements.txt
systemctl restart weld-sim

# For Phase 6+ (Docker):
docker-compose -f docker/docker-compose.yml up -d solver-b
```
