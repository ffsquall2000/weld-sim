# FEniCSx Plugin (Solver B) -- Design Document

**Goal:** Provide advanced FEA capabilities (piezoelectric coupling, nonlinear contact, coupled thermomechanical analysis) as an optional plugin that complements the core numpy/scipy solver (Solver A), while enabling cross-validation between both backends.

**Architecture:** Optional plugin behind a `SolverBackend` abstraction. Graceful degradation when FEniCSx is not installed. Docker-based deployment for the FEniCSx runtime. Unified result objects across both solver backends.

**Tech Stack:** FEniCSx/DOLFINx 0.8+, PETSc 3.20+, petsc4py, Gmsh 4.12+ (via `gmsh` Python API), basix, UFL

**Dependencies:** Docker (recommended) or conda-forge environment with `fenics-dolfinx`, `petsc4py`, `gmsh`

---

## 1. Plugin Architecture

### 1.1 SolverBackend Abstraction

Both Solver A (numpy/scipy, the existing `FEAService`) and Solver B (FEniCSx) implement a common abstract interface. This allows the system to dispatch analysis requests to either backend transparently.

```python
# ultrasonic_weld_master/core/solver_backend.py

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np


class AnalysisType(Enum):
    MODAL = "modal"
    HARMONIC = "harmonic"
    STATIC = "static"
    PIEZOELECTRIC = "piezoelectric"
    CONTACT = "contact"
    THERMOMECHANICAL = "thermomechanical"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class MeshData:
    """Solver-agnostic mesh representation for result transfer."""
    vertices: np.ndarray          # (N, 3) node coordinates in mm
    elements: np.ndarray          # (M, nodes_per_elem) connectivity
    element_type: str             # "hex8", "tet4", "tet10", etc.
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    element_sets: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ModalResult:
    """Unified modal analysis result."""
    frequencies_hz: np.ndarray            # (n_modes,)
    mode_shapes: np.ndarray               # (n_modes, n_nodes, 3)
    mode_types: list[str]                 # ["longitudinal", "flexural", ...]
    effective_mass_ratios: np.ndarray     # (n_modes,)
    solver_backend: str                   # "numpy_scipy" or "fenicsx"
    solve_time_s: float
    mesh: MeshData
    metadata: dict = field(default_factory=dict)


@dataclass
class HarmonicResult:
    """Unified harmonic response result."""
    sweep_frequencies_hz: np.ndarray      # (n_freq,)
    sweep_amplitudes: np.ndarray          # (n_freq,) normalized peak amplitude
    displacement_field: np.ndarray        # (n_nodes, 3) complex at target freq
    stress_field: np.ndarray              # (n_elements, 6) Voigt stress at target
    contact_amplitudes: np.ndarray        # (n_contact_nodes,) amplitude at weld face
    amplitude_uniformity: float
    stress_max_mpa: float
    solver_backend: str
    solve_time_s: float
    mesh: MeshData
    metadata: dict = field(default_factory=dict)


@dataclass
class ImpedanceResult:
    """Electrical impedance spectrum from piezoelectric analysis."""
    frequencies_hz: np.ndarray            # (n_freq,)
    impedance_magnitude: np.ndarray       # (n_freq,) |Z| in Ohms
    impedance_phase_deg: np.ndarray       # (n_freq,) phase in degrees
    admittance_magnitude: np.ndarray      # (n_freq,) |Y| in Siemens
    resonant_freq_hz: float               # f_r (minimum impedance)
    antiresonant_freq_hz: float           # f_a (maximum impedance)
    k_eff: float                          # electromechanical coupling coefficient
    solver_backend: str
    solve_time_s: float
    metadata: dict = field(default_factory=dict)


@dataclass
class ContactResult:
    """Nonlinear contact analysis result."""
    contact_pressure: np.ndarray          # (n_contact_nodes,) in MPa
    gap: np.ndarray                       # (n_contact_nodes,) gap in mm (>0 = open)
    slip: np.ndarray                      # (n_contact_nodes,) tangential slip in mm
    bolt_force_n: float                   # resultant bolt preload
    stressed_frequencies_hz: Optional[np.ndarray] = None  # pre-stressed modal
    stressed_mode_shapes: Optional[np.ndarray] = None
    solver_backend: str = "fenicsx"
    solve_time_s: float = 0.0
    mesh: Optional[MeshData] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ThermoResult:
    """Coupled thermomechanical analysis result."""
    time_steps: np.ndarray                # (n_steps,) time in seconds
    temperature_field: np.ndarray         # (n_steps, n_nodes) in Celsius
    thermal_stress_field: np.ndarray      # (n_steps, n_nodes, 6) Voigt stress
    max_temperature_c: float
    frequency_shift_hz: float             # predicted shift from thermal effects
    thermal_expansion_strain: np.ndarray  # (n_nodes, 6) at final time
    solver_backend: str = "fenicsx"
    solve_time_s: float = 0.0
    mesh: Optional[MeshData] = None
    metadata: dict = field(default_factory=dict)


class SolverBackend(ABC):
    """Abstract interface implemented by both Solver A and Solver B."""

    @abstractmethod
    def get_name(self) -> str:
        """Return backend identifier: 'numpy_scipy' or 'fenicsx'."""
        ...

    @abstractmethod
    def get_capabilities(self) -> set[AnalysisType]:
        """Return the set of analysis types this backend supports."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if all runtime dependencies are satisfied."""
        ...

    @abstractmethod
    def run_modal(
        self,
        mesh: MeshData,
        material: dict,
        boundary_conditions: dict,
        target_frequency_hz: float,
        n_modes: int = 10,
    ) -> ModalResult:
        ...

    @abstractmethod
    def run_harmonic(
        self,
        mesh: MeshData,
        material: dict,
        boundary_conditions: dict,
        frequency_range_hz: tuple[float, float],
        n_sweep: int = 21,
        damping_ratio: float = 0.01,
    ) -> HarmonicResult:
        ...

    def run_piezoelectric(self, **kwargs) -> ImpedanceResult:
        raise NotImplementedError(
            f"{self.get_name()} does not support piezoelectric analysis"
        )

    def run_contact(self, **kwargs) -> ContactResult:
        raise NotImplementedError(
            f"{self.get_name()} does not support nonlinear contact analysis"
        )

    def run_thermomechanical(self, **kwargs) -> ThermoResult:
        raise NotImplementedError(
            f"{self.get_name()} does not support thermomechanical analysis"
        )
```

### 1.2 Solver A Adapter (Existing FEAService)

The existing `FEAService` in `web/services/fea_service.py` is wrapped to conform to the `SolverBackend` interface. No changes to the existing solver implementation are required.

```python
# ultrasonic_weld_master/solvers/numpy_scipy_backend.py

from __future__ import annotations
from ultrasonic_weld_master.core.solver_backend import (
    SolverBackend, AnalysisType, MeshData, ModalResult, HarmonicResult,
)
from web.services.fea_service import FEAService


class NumpyScipyBackend(SolverBackend):
    """Solver A: wraps the existing FEAService for the SolverBackend interface."""

    def __init__(self) -> None:
        self._fea = FEAService()

    def get_name(self) -> str:
        return "numpy_scipy"

    def get_capabilities(self) -> set[AnalysisType]:
        return {AnalysisType.MODAL, AnalysisType.HARMONIC, AnalysisType.STATIC}

    def is_available(self) -> bool:
        return True  # numpy/scipy are always installed

    def run_modal(self, mesh, material, boundary_conditions,
                  target_frequency_hz, n_modes=10) -> ModalResult:
        # Delegates to existing FEAService._prepare_model + _eigen_solve
        # Converts internal result dict to ModalResult dataclass
        ...

    def run_harmonic(self, mesh, material, boundary_conditions,
                     frequency_range_hz, n_sweep=21,
                     damping_ratio=0.01) -> HarmonicResult:
        # Delegates to existing FEAService.run_acoustic_analysis
        # Converts internal result dict to HarmonicResult dataclass
        ...
```

### 1.3 Solver B (FEniCSx Backend)

```python
# ultrasonic_weld_master/solvers/fenicsx_backend.py

from __future__ import annotations
import logging
from ultrasonic_weld_master.core.solver_backend import (
    SolverBackend, AnalysisType, MeshData, ModalResult, HarmonicResult,
    ImpedanceResult, ContactResult, ThermoResult,
)

logger = logging.getLogger(__name__)

# Lazy import guard -- FEniCSx may not be installed
_FENICSX_AVAILABLE = False
try:
    import dolfinx
    import dolfinx.fem
    import dolfinx.mesh
    import dolfinx.io
    import ufl
    import basix
    from petsc4py import PETSc
    from slepc4py import SLEPc
    _FENICSX_AVAILABLE = True
except ImportError:
    logger.info(
        "FEniCSx not available. Advanced analyses (piezoelectric, contact, "
        "thermomechanical) are disabled. Install via: "
        "conda install -c conda-forge fenics-dolfinx petsc4py slepc4py"
    )


class FEniCSxBackend(SolverBackend):
    """Solver B: FEniCSx/DOLFINx + PETSc for advanced physics."""

    def get_name(self) -> str:
        return "fenicsx"

    def get_capabilities(self) -> set[AnalysisType]:
        if not _FENICSX_AVAILABLE:
            return set()
        return {
            AnalysisType.MODAL,
            AnalysisType.HARMONIC,
            AnalysisType.PIEZOELECTRIC,
            AnalysisType.CONTACT,
            AnalysisType.THERMOMECHANICAL,
        }

    def is_available(self) -> bool:
        return _FENICSX_AVAILABLE

    # ... analysis methods defined in Sections 2-4 below ...
```

### 1.4 Graceful Degradation

The plugin registration in `EngineService` follows the same lazy-import pattern already established for the `geometry_analyzer` plugin:

```python
# In web/services/engine_service.py (or equivalent plugin registration)

# Optional FEniCSx solver backend
try:
    from ultrasonic_weld_master.solvers.fenicsx_backend import FEniCSxBackend
    fenicsx = FEniCSxBackend()
    if fenicsx.is_available():
        engine.register_solver_backend(fenicsx)
        logger.info("FEniCSx backend registered: %s", fenicsx.get_capabilities())
    else:
        logger.info("FEniCSx backend imported but runtime unavailable")
except ImportError:
    logger.info("FEniCSx backend not installed, advanced analyses disabled")
```

When a user requests an analysis type that requires Solver B but it is not installed, the system returns a structured error:

```python
@dataclass
class FeatureNotAvailable:
    feature: str
    reason: str
    install_instructions: str

# Example:
# FeatureNotAvailable(
#     feature="Piezoelectric Analysis",
#     reason="FEniCSx/DOLFINx is not installed",
#     install_instructions=(
#         "Option 1: docker run -p 8080:8080 uwm/fenicsx-solver:latest\n"
#         "Option 2: conda install -c conda-forge fenics-dolfinx petsc4py slepc4py"
#     ),
# )
```

### 1.5 Mesh Transfer: Gmsh to DOLFINx

Both solvers share the Gmsh meshing frontend. DOLFINx provides native Gmsh import through `dolfinx.io.gmshio`. The mesh pipeline is:

```
Gmsh Python API  -->  .msh file (or in-memory)  -->  dolfinx.io.gmshio.model_to_mesh()
                                                 -->  numpy arrays (for Solver A)
```

```python
# ultrasonic_weld_master/solvers/mesh_bridge.py

from __future__ import annotations
import numpy as np
import gmsh
from ultrasonic_weld_master.core.solver_backend import MeshData


def generate_gmsh_model(
    geometry_params: dict,
    mesh_size: float = 2.0,
    element_order: int = 1,
    element_type: str = "tet",  # "tet" or "hex"
) -> None:
    """Build the Gmsh model in-memory using the Gmsh Python API.

    This creates the transducer stack geometry:
    - Back mass (steel)
    - PZT ceramic rings (with electrode surfaces tagged)
    - Front mass (titanium/aluminum)
    - Booster
    - Horn

    All volumes are tagged with physical groups for material assignment.
    Interface surfaces are tagged for contact/coupling boundary conditions.
    """
    gmsh.initialize()
    gmsh.model.add("transducer_stack")

    # ... geometry construction using gmsh.model.occ ...
    # Physical groups: "pzt_ceramic", "back_mass", "front_mass",
    #                  "booster", "horn", "electrode_top", "electrode_bottom",
    #                  "contact_horn_booster", "contact_booster_front_mass",
    #                  "weld_face"

    gmsh.model.occ.synchronize()

    # Mesh size control
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.ElementOrder", element_order)

    if element_type == "hex":
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 9)  # HXT for hex
    else:
        gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT for tet

    gmsh.model.mesh.generate(3)


def gmsh_to_dolfinx(comm=None):
    """Convert the current Gmsh model to a DOLFINx mesh.

    Uses dolfinx.io.gmshio.model_to_mesh() which natively reads
    Gmsh physical groups, facet tags, and cell tags.

    Returns:
        (dolfinx.mesh.Mesh, cell_tags, facet_tags)
    """
    from mpi4py import MPI
    import dolfinx.io.gmshio

    if comm is None:
        comm = MPI.COMM_WORLD

    mesh, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, comm, rank=0, gdim=3
    )
    return mesh, cell_tags, facet_tags


def gmsh_to_numpy() -> MeshData:
    """Extract the current Gmsh model as numpy arrays for Solver A.

    Returns a MeshData with vertices, elements, and named node/element sets
    extracted from Gmsh physical groups.
    """
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    vertices = coords.reshape(-1, 3)

    # Build tag -> sequential index mapping
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)
    # Assuming single element type for simplicity
    etype = elem_types[0]
    props = gmsh.model.mesh.getElementProperties(etype)
    nodes_per_elem = props[3]
    raw_conn = elem_node_tags[0].reshape(-1, nodes_per_elem)
    elements = np.array(
        [[tag_to_idx[int(n)] for n in row] for row in raw_conn],
        dtype=np.int32,
    )

    etype_name = {4: "tet4", 5: "hex8", 11: "tet10", 12: "hex20"}.get(
        etype, f"type_{etype}"
    )

    # Extract physical group memberships
    node_sets: dict[str, np.ndarray] = {}
    element_sets: dict[str, np.ndarray] = {}
    for dim_tag in gmsh.model.getPhysicalGroups():
        dim, tag = dim_tag
        name = gmsh.model.getPhysicalName(dim, tag)
        if not name:
            continue
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
        if dim == 3:
            # Volume -> element set
            eidx = []
            for ent in entities:
                _, etags, _ = gmsh.model.mesh.getElements(dim, ent)
                if len(etags) > 0:
                    eidx.extend(etags[0].tolist())
            element_sets[name] = np.array(eidx, dtype=np.int32)
        elif dim == 2:
            # Surface -> node set
            nidx = set()
            for ent in entities:
                ntags, _, _ = gmsh.model.mesh.getNodes(dim, ent, True)
                nidx.update(tag_to_idx[int(t)] for t in ntags)
            node_sets[name] = np.array(sorted(nidx), dtype=np.int32)

    return MeshData(
        vertices=vertices,
        elements=elements,
        element_type=etype_name,
        node_sets=node_sets,
        element_sets=element_sets,
    )
```

### 1.6 Solver Dispatch

```python
# ultrasonic_weld_master/solvers/dispatcher.py

from __future__ import annotations
import logging
from ultrasonic_weld_master.core.solver_backend import (
    SolverBackend, AnalysisType, FeatureNotAvailable,
)

logger = logging.getLogger(__name__)


class SolverDispatcher:
    """Routes analysis requests to the appropriate backend."""

    def __init__(self) -> None:
        self._backends: dict[str, SolverBackend] = {}
        self._preferred_order: list[str] = []

    def register(self, backend: SolverBackend) -> None:
        name = backend.get_name()
        self._backends[name] = backend
        self._preferred_order.append(name)
        logger.info("Registered solver backend: %s (capabilities: %s)",
                     name, backend.get_capabilities())

    def get_backend(
        self,
        analysis_type: AnalysisType,
        preferred: str | None = None,
    ) -> SolverBackend:
        """Return the best available backend for the requested analysis type.

        Priority:
        1. Explicitly preferred backend (if capable and available)
        2. FEniCSx (for advanced analysis types)
        3. numpy_scipy (for standard analysis types)

        Raises FeatureNotAvailable if no backend can handle the request.
        """
        if preferred and preferred in self._backends:
            b = self._backends[preferred]
            if b.is_available() and analysis_type in b.get_capabilities():
                return b

        # For advanced types, prefer fenicsx
        advanced = {
            AnalysisType.PIEZOELECTRIC,
            AnalysisType.CONTACT,
            AnalysisType.THERMOMECHANICAL,
        }

        if analysis_type in advanced:
            if "fenicsx" in self._backends:
                b = self._backends["fenicsx"]
                if b.is_available():
                    return b
            raise FeatureNotAvailable(
                feature=analysis_type.value,
                reason="FEniCSx backend is not available",
                install_instructions=(
                    "Option 1: docker run -p 8080:8080 uwm/fenicsx-solver:latest\n"
                    "Option 2: conda install -c conda-forge "
                    "fenics-dolfinx petsc4py slepc4py"
                ),
            )

        # For standard types, prefer numpy_scipy (faster for small models)
        for name in ["numpy_scipy", "fenicsx"]:
            if name in self._backends:
                b = self._backends[name]
                if b.is_available() and analysis_type in b.get_capabilities():
                    return b

        raise FeatureNotAvailable(
            feature=analysis_type.value,
            reason="No solver backend available for this analysis type",
            install_instructions="Ensure numpy/scipy are installed",
        )

    def list_capabilities(self) -> dict[str, set[AnalysisType]]:
        """Return capabilities of all registered backends."""
        return {
            name: b.get_capabilities()
            for name, b in self._backends.items()
            if b.is_available()
        }
```

---

## 2. Piezoelectric Coupled Analysis

### 2.1 Governing Equations

Ultrasonic transducers use PZT ceramic elements (PZT-4 or PZT-8) sandwiched between metallic masses. The piezoelectric constitutive equations in stress-charge form are:

**Mechanical equilibrium (stress form):**

$$\sigma_{ij} = c^E_{ijkl} \, \varepsilon_{kl} - e_{kij} \, E_k$$

**Electrical flux (charge form):**

$$D_i = e_{ikl} \, \varepsilon_{kl} + \varepsilon^S_{ik} \, E_k$$

where:
- $\sigma_{ij}$ -- stress tensor [Pa]
- $\varepsilon_{kl}$ -- strain tensor [-]
- $E_k$ -- electric field vector [V/m]
- $D_i$ -- electric displacement vector [C/m^2]
- $c^E_{ijkl}$ -- elastic stiffness at constant electric field [Pa]
- $e_{kij}$ -- piezoelectric coupling tensor [C/m^2]
- $\varepsilon^S_{ik}$ -- permittivity at constant strain [F/m]

In Voigt notation (6-component stress/strain, 3-component field), the coupled system becomes the matrix equation:

$$\begin{bmatrix} \boldsymbol{\sigma} \\ \mathbf{D} \end{bmatrix} = \begin{bmatrix} \mathbf{c}^E & -\mathbf{e}^T \\ \mathbf{e} & \boldsymbol{\varepsilon}^S \end{bmatrix} \begin{bmatrix} \boldsymbol{\varepsilon} \\ \mathbf{E} \end{bmatrix}$$

### 2.2 Material Tensors

For PZT ceramics (transversely isotropic, poling in z-direction), the material tensors in Voigt notation are:

**PZT-4 properties:**

```python
PZT4_PROPERTIES = {
    # Elastic stiffness at constant E [Pa] (6x6 Voigt)
    "c_E": [
        [139.0e9, 77.8e9,  74.3e9,  0,       0,       0      ],
        [77.8e9,  139.0e9, 74.3e9,  0,       0,       0      ],
        [74.3e9,  74.3e9,  115.0e9, 0,       0,       0      ],
        [0,       0,       0,       25.6e9,  0,       0      ],
        [0,       0,       0,       0,       25.6e9,  0      ],
        [0,       0,       0,       0,       0,       30.6e9 ],
    ],
    # Piezoelectric coupling [C/m^2] (3x6 Voigt: e_ij maps strain_j to D_i)
    "e": [
        [0,     0,     0,     0,    12.7,   0   ],   # e_15
        [0,     0,     0,     12.7,  0,     0   ],   # e_24 = e_15
        [-5.2,  -5.2,  15.1,  0,     0,     0   ],   # e_31, e_33
    ],
    # Permittivity at constant strain [F/m] (3x3 diagonal)
    "eps_S": [
        [6.45e-9, 0,       0      ],
        [0,       6.45e-9, 0      ],
        [0,       0,       5.62e-9],
    ],
    "rho": 7500.0,  # kg/m^3
}

PZT8_PROPERTIES = {
    "c_E": [
        [146.9e9, 81.1e9,  81.1e9,  0,       0,       0      ],
        [81.1e9,  146.9e9, 81.1e9,  0,       0,       0      ],
        [81.1e9,  81.1e9,  131.7e9, 0,       0,       0      ],
        [0,       0,       0,       31.4e9,  0,       0      ],
        [0,       0,       0,       0,       31.4e9,  0      ],
        [0,       0,       0,       0,       0,       32.9e9 ],
    ],
    "e": [
        [0,     0,     0,     0,    10.4,   0   ],
        [0,     0,     0,     10.4,  0,     0   ],
        [-4.0,  -4.0,  13.2,  0,     0,     0   ],
    ],
    "eps_S": [
        [1.14e-8, 0,       0      ],
        [0,       1.14e-8, 0      ],
        [0,       0,       8.85e-9],
    ],
    "rho": 7600.0,
}
```

### 2.3 FEniCSx Weak Form

The piezoelectric problem has two primary unknowns: displacement **u** (vector, 3 DOF/node) and electric potential phi (scalar, 1 DOF/node). Using UFL, the weak form is:

```python
def define_piezoelectric_problem(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
    facet_tags: dolfinx.mesh.MeshTags,
    pzt_material: dict,
    metal_materials: dict[int, dict],  # cell_tag -> material props
    electrode_tags: dict[str, int],     # "top"/"bottom" -> facet tag
    frequency_hz: float,
):
    """Set up the piezoelectric variational problem in DOLFINx.

    Uses a mixed function space: (vector Lagrange P1 for u, scalar Lagrange P1 for phi).
    """
    import dolfinx.fem
    import ufl
    import basix

    # Mixed function space: V = [displacement (3-vector), potential (scalar)]
    elem_u = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1,
                                shape=(3,))
    elem_phi = basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
    mixed_elem = basix.ufl.mixed_element([elem_u, elem_phi])
    V = dolfinx.fem.functionspace(mesh, mixed_elem)

    # Trial and test functions
    (u, phi) = ufl.TrialFunctions(V)
    (v, psi) = ufl.TestFunctions(V)

    # Strain and electric field from primary variables
    def strain(w):
        return ufl.sym(ufl.grad(w))

    def electric_field(p):
        return -ufl.grad(p)

    # Material tensors as UFL constants
    # For PZT regions (identified by cell_tags), use piezoelectric constitutive law
    # For metallic regions, use standard elasticity (e=0, eps_S irrelevant)
    c_E = ufl.as_tensor(pzt_material["c_E"])   # (6,6) Voigt -> (3,3,3,3) tensor
    e_tensor = ufl.as_tensor(pzt_material["e"])  # (3,6) Voigt -> (3,3,3) tensor
    eps_S = ufl.as_tensor(pzt_material["eps_S"])  # (3,3)

    # The Voigt-to-tensor mapping for UFL:
    # sigma_ij = c_E_ijkl * eps_kl - e_kij * E_k
    # D_i = e_ikl * eps_kl + eps_S_ik * E_k

    # Bilinear forms (stiffness contributions):
    # Mechanical: integral of (c_E : strain(u)) : strain(v)
    # Coupling (mech->elec): integral of (e * strain(u)) . grad(psi)
    # Coupling (elec->mech): integral of (e^T * E(phi)) : strain(v)
    # Electrical: integral of (eps_S * E(phi)) . E(psi)

    # Using subdomain integration with cell_tags for different materials:
    dx_pzt = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags,
                          subdomain_id=cell_tags_pzt_id)
    dx_metal = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags,
                            subdomain_id=cell_tags_metal_id)

    # Mechanical stiffness (PZT region)
    a_uu_pzt = ufl.inner(
        voigt_to_stress(c_E, strain(u)), strain(v)
    ) * dx_pzt

    # Mechanical stiffness (metal regions)
    # Standard isotropic elasticity for each metal
    a_uu_metal = sum(
        ufl.inner(
            isotropic_stress(mat["E_pa"], mat["nu"], strain(u)), strain(v)
        ) * ufl.Measure("dx", subdomain_data=cell_tags, subdomain_id=tag)
        for tag, mat in metal_materials.items()
    )

    # Piezoelectric coupling: u -> phi
    a_u_phi = ufl.inner(
        voigt_e_contract(e_tensor, strain(u)), electric_field(psi)
    ) * dx_pzt

    # Piezoelectric coupling: phi -> u (transpose of above)
    a_phi_u = -ufl.inner(
        voigt_eT_contract(e_tensor, electric_field(phi)), strain(v)
    ) * dx_pzt

    # Dielectric stiffness
    a_phi_phi = ufl.inner(
        eps_S * electric_field(phi), electric_field(psi)
    ) * dx_pzt

    # Mass matrix (all regions)
    rho_pzt = dolfinx.fem.Constant(mesh, pzt_material["rho"])
    a_mass_pzt = rho_pzt * ufl.inner(u, v) * dx_pzt
    a_mass_metal = sum(
        dolfinx.fem.Constant(mesh, mat["rho_kg_m3"]) * ufl.inner(u, v)
        * ufl.Measure("dx", subdomain_data=cell_tags, subdomain_id=tag)
        for tag, mat in metal_materials.items()
    )

    # Combined stiffness and mass
    a_stiffness = a_uu_pzt + a_uu_metal + a_u_phi + a_phi_u + a_phi_phi
    a_mass = a_mass_pzt + a_mass_metal

    return V, a_stiffness, a_mass
```

### 2.4 Impedance Frequency Sweep

To compute the electrical impedance spectrum, we drive the transducer with a unit voltage on the top electrode and sweep across frequency:

```python
def run_impedance_sweep(
    mesh, cell_tags, facet_tags,
    pzt_material: dict,
    metal_materials: dict[int, dict],
    electrode_tags: dict[str, int],
    freq_range_hz: tuple[float, float],
    n_freq: int = 200,
) -> ImpedanceResult:
    """Compute electrical impedance Z(f) and admittance Y(f) = 1/Z(f).

    At each frequency f:
    1. Assemble dynamic stiffness: K_dyn = K - (2*pi*f)^2 * M + i*(2*pi*f)*C
    2. Apply electrode BC: phi = V0 on top electrode, phi = 0 on bottom electrode
    3. Solve for (u, phi)
    4. Compute charge Q = integral(D . n) over top electrode surface
    5. Impedance: Z = V0 / (i*omega*Q), Admittance: Y = 1/Z

    The resonant frequency f_r is where |Z| is minimum.
    The anti-resonant frequency f_a is where |Z| is maximum.
    The effective coupling coefficient: k_eff = sqrt(1 - (f_r/f_a)^2)
    """
    import numpy as np
    from petsc4py import PETSc

    V, a_stiffness, a_mass = define_piezoelectric_problem(
        mesh, cell_tags, facet_tags, pzt_material, metal_materials,
        electrode_tags, freq_range_hz[0],
    )

    # Assemble stiffness and mass matrices
    K = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_stiffness))
    K.assemble()
    M = dolfinx.fem.petsc.assemble_matrix(dolfinx.fem.form(a_mass))
    M.assemble()

    # Rayleigh damping: C = alpha*M + beta*K
    # alpha, beta chosen for ~1% damping at sweep center
    f_center = (freq_range_hz[0] + freq_range_hz[1]) / 2
    omega_center = 2 * np.pi * f_center
    zeta = 0.005  # 0.5% structural damping for PZT
    alpha = 2 * zeta * omega_center
    beta = 2 * zeta / omega_center

    frequencies = np.linspace(freq_range_hz[0], freq_range_hz[1], n_freq)
    impedance_mag = np.zeros(n_freq)
    impedance_phase = np.zeros(n_freq)

    V0 = 1.0  # Unit driving voltage

    for i, f in enumerate(frequencies):
        omega = 2 * np.pi * f

        # Z_dyn = K - omega^2 * M + i*omega*(alpha*M + beta*K)
        # Z_dyn = (1 + i*omega*beta)*K + (-omega^2 + i*omega*alpha)*M
        # Solve: Z_dyn * x = F (with electrode BC)
        # ...PETSc KSP solve...

        # Extract charge on electrode: Q = integral over electrode of (D . n)
        # Z(f) = V0 / (i * omega * Q)
        # impedance_mag[i] = |Z(f)|
        # impedance_phase[i] = angle(Z(f)) in degrees
        pass

    # Post-process: find resonance and anti-resonance
    i_r = np.argmin(impedance_mag)
    i_a = np.argmax(impedance_mag)
    f_r = frequencies[i_r]
    f_a = frequencies[i_a]
    k_eff = np.sqrt(1.0 - (f_r / f_a) ** 2) if f_a > f_r else 0.0

    admittance_mag = 1.0 / np.maximum(impedance_mag, 1e-12)

    return ImpedanceResult(
        frequencies_hz=frequencies,
        impedance_magnitude=impedance_mag,
        impedance_phase_deg=impedance_phase,
        admittance_magnitude=admittance_mag,
        resonant_freq_hz=f_r,
        antiresonant_freq_hz=f_a,
        k_eff=k_eff,
        solver_backend="fenicsx",
        solve_time_s=0.0,  # filled by timing wrapper
    )
```

### 2.5 Outputs

| Output | Description | Units |
|--------|-------------|-------|
| Impedance spectrum Z(f) | Magnitude and phase vs. frequency | Ohm, degrees |
| Admittance spectrum Y(f) | 1/Z(f) | Siemens |
| Resonant frequency f_r | Frequency of minimum impedance | Hz |
| Anti-resonant frequency f_a | Frequency of maximum impedance | Hz |
| Coupling coefficient k_eff | sqrt(1 - (f_r/f_a)^2) | dimensionless |
| Mode shapes at resonance | Displacement field at f_r | m |
| Electric potential distribution | phi field at f_r | V |

---

## 3. Nonlinear Contact Analysis

### 3.1 Problem Description

The ultrasonic welding stack consists of multiple bolted components: transducer (PZT + masses) -- booster -- horn. These joints are held together by a central bolt (typical preload 20--40 kN). Under high-amplitude vibration, the contact interfaces can experience:
- Partial separation (gap opening) at low-pressure regions
- Frictional sliding at the interface
- Local stress concentrations at contact edges

Understanding these effects is critical for predicting:
- Energy loss at joints (reducing amplitude at horn tip)
- Fatigue failure at bolt/contact locations
- Frequency shift under varying bolt torque

### 3.2 Contact Formulation

We use an augmented Lagrangian contact formulation, which is robust and avoids the ill-conditioning of pure penalty methods.

**Normal contact (Signorini conditions):**

$$g_n \geq 0, \quad p_n \leq 0, \quad p_n \cdot g_n = 0$$

where $g_n$ is the gap function and $p_n$ is the normal contact pressure.

**Augmented Lagrangian enforcement:**

$$p_n = \lambda_n + r \cdot g_n \quad \text{if } \lambda_n + r \cdot g_n < 0$$
$$p_n = 0 \quad \text{otherwise (gap open)}$$

where $\lambda_n$ is the Lagrange multiplier (iterated) and $r > 0$ is the augmentation parameter.

**Tangential contact (Coulomb friction):**

$$\|\mathbf{p}_t\| \leq \mu |p_n| \quad \text{(stick)}$$
$$\mathbf{p}_t = -\mu |p_n| \frac{\dot{\mathbf{g}}_t}{\|\dot{\mathbf{g}}_t\|} \quad \text{(slip)}$$

where $\mu$ is the Coulomb friction coefficient (typical 0.15--0.3 for steel-titanium interfaces with bolt grease).

### 3.3 DOLFINx Implementation

```python
def run_contact_analysis(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
    facet_tags: dolfinx.mesh.MeshTags,
    materials: dict[int, dict],           # cell_tag -> material properties
    contact_pairs: list[dict],            # [{master: facet_tag, slave: facet_tag, mu: float}]
    bolt_preload_n: float = 30000.0,      # bolt preload force
    bolt_facet_tag: int = 0,              # facet tag for bolt bearing surface
) -> ContactResult:
    """Nonlinear contact analysis with bolt preload and Coulomb friction.

    Steps:
    1. Apply bolt preload as distributed pressure on bolt bearing surface
    2. Solve nonlinear contact problem iteratively (Newton + augmented Lagrangian)
    3. Extract contact pressure, gap, and slip distributions
    4. Optionally: extract pre-stressed stiffness for subsequent modal analysis
    """
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl
    from petsc4py import PETSc

    # Function space: vector Lagrange P1
    V = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1, shape=(3,))
    )

    u = dolfinx.fem.Function(V)       # displacement solution
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # Material assignment by cell tag
    # Build piecewise constitutive law
    def sigma(u_field, tag_id, mat):
        """Isotropic linear elastic stress for a tagged subdomain."""
        E = mat["E_pa"]
        nu = mat["nu"]
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu_e = E / (2 * (1 + nu))
        eps = ufl.sym(ufl.grad(u_field))
        return lam * ufl.tr(eps) * ufl.Identity(3) + 2 * mu_e * eps

    # Internal virtual work (sum over material subdomains)
    F_int = sum(
        ufl.inner(sigma(u, tag, mat), ufl.sym(ufl.grad(v)))
        * ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags,
                       subdomain_id=tag)
        for tag, mat in materials.items()
    )

    # Bolt preload: distributed traction on bolt bearing surface
    bolt_area = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(
            1.0 * ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                              subdomain_id=bolt_facet_tag)
        )
    )
    bolt_pressure = bolt_preload_n / max(bolt_area, 1e-12)
    n = ufl.FacetNormal(mesh)
    ds_bolt = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                          subdomain_id=bolt_facet_tag)
    F_bolt = -bolt_pressure * ufl.inner(n, v) * ds_bolt

    # Contact contribution (augmented Lagrangian)
    # For each contact pair, compute gap and apply contact traction
    # This requires custom iteration loop:
    #
    # for k in range(max_augmentation_iters):
    #     1. Solve: K*du = R (residual) using Newton's method
    #     2. Update gap g_n, slip g_t from current displacement
    #     3. Update Lagrange multipliers:
    #        lambda_n <- max(0, lambda_n + r * g_n)  [compressive positive]
    #        lambda_t: Coulomb friction return mapping
    #     4. Check convergence: |delta_lambda| < tol
    #
    # The augmentation parameter r is chosen as:
    #   r ~ E_avg / h_min  (element stiffness / mesh size)

    # Newton solver configuration
    problem = dolfinx.fem.petsc.NonlinearProblem(F_int - F_bolt, u)
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.convergence_criterion = "incremental"
    solver.rtol = 1e-8
    solver.atol = 1e-10
    solver.max_it = 50

    # Configure PETSc KSP (linear solver within Newton)
    ksp = solver.krylov_solver
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")

    # Solve
    n_iters, converged = solver.solve(u)

    # Post-process: extract contact pressure, gap, slip on contact surfaces
    # ...

    return ContactResult(
        contact_pressure=contact_pressure_array,
        gap=gap_array,
        slip=slip_array,
        bolt_force_n=bolt_preload_n,
        solver_backend="fenicsx",
        solve_time_s=solve_time,
        mesh=mesh_data,
    )
```

### 3.4 Pre-Stressed Modal Analysis

After the contact equilibrium is found, the tangent stiffness matrix incorporates the stress-stiffening effect. We then solve the eigenvalue problem on this pre-stressed state:

$$(\mathbf{K}_T - \omega^2 \mathbf{M}) \, \boldsymbol{\phi} = \mathbf{0}$$

where $\mathbf{K}_T$ is the tangent stiffness from the converged nonlinear solution (includes geometric stiffness from the pre-stress). This predicts how bolt preload shifts the natural frequencies.

```python
def run_prestressed_modal(
    mesh, u_contact, K_tangent, M,
    target_frequency_hz: float,
    n_modes: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalue solve on the pre-stressed tangent stiffness.

    Uses SLEPc shift-invert spectral transformation for efficient
    extraction of modes near the target frequency.
    """
    from slepc4py import SLEPc

    eigensolver = SLEPc.EPS().create(mesh.comm)
    eigensolver.setOperators(K_tangent, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)

    # Shift-invert near target
    sigma = (2 * np.pi * target_frequency_hz) ** 2
    st = eigensolver.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(sigma)

    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eigensolver.setTarget(sigma)
    eigensolver.setDimensions(nev=n_modes)
    eigensolver.setTolerances(tol=1e-8, max_it=500)

    eigensolver.solve()

    nconv = eigensolver.getConverged()
    eigenvalues = np.zeros(nconv)
    # eigenvectors extracted via eigensolver.getEigenpair(...)

    return eigenvalues, eigenvectors
```

### 3.5 Outputs

| Output | Description | Units |
|--------|-------------|-------|
| Contact pressure map | Normal pressure at each contact node | MPa |
| Gap distribution | Normal gap (>0 means open) | mm |
| Slip distribution | Tangential sliding distance | mm |
| Contact status | Stick / Slip / Open at each node | enum |
| Pre-stressed natural frequencies | Eigenfrequencies with bolt preload | Hz |
| Pre-stressed mode shapes | Eigenvectors from K_T | m |
| Bolt force resultant | Verification of applied preload | N |

---

## 4. Coupled Thermomechanical Analysis

### 4.1 Heat Sources in Ultrasonic Welding

Two primary heat generation mechanisms:

**1. Internal friction / hysteretic heating (within the horn body):**

$$Q_{hyst}(\mathbf{x}) = \pi f \, \eta \, \sigma_{ij}(\mathbf{x}) \, \varepsilon_{ij}(\mathbf{x})$$

where $f$ is the operating frequency, $\eta$ is the material loss factor (typically 0.001--0.01 for metals), and $\sigma$, $\varepsilon$ are the stress/strain amplitudes from the harmonic analysis.

**2. Frictional heating at the weld interface:**

$$Q_{fric}(t) = \mu \, p_n \, v_{rel}(t)$$

where $\mu$ is the friction coefficient, $p_n$ is the contact normal pressure, and $v_{rel}(t) = 2\pi f \cdot A \cdot |\cos(2\pi f t)|$ is the relative sliding velocity (with amplitude $A$ at the weld face).

The time-averaged frictional power density at the interface is:

$$\overline{Q}_{fric} = \frac{2}{\pi} \mu \, p_n \, (2\pi f \, A)$$

### 4.2 Transient Heat Equation

$$\rho(\mathbf{x}) \, c_p(\mathbf{x}) \, \frac{\partial T}{\partial t} = \nabla \cdot \bigl(k(\mathbf{x}) \, \nabla T\bigr) + Q(\mathbf{x}, t)$$

with initial condition $T(\mathbf{x}, 0) = T_0$ (ambient, typically 25 C) and boundary conditions:
- Convection on free surfaces: $-k \nabla T \cdot \mathbf{n} = h(T - T_\infty)$ with $h \approx 10$ W/(m^2 K) (natural convection)
- Prescribed temperature or heat flux at clamp surfaces

**Temperature-dependent material properties:**

The elastic modulus and Poisson's ratio vary with temperature:

$$E(T) = E_0 \bigl(1 - \alpha_E (T - T_0)\bigr)$$
$$\nu(T) \approx \nu_0 \quad \text{(weak dependence, typically held constant)}$$

where $\alpha_E$ is the modulus temperature coefficient. For titanium Ti-6Al-4V, $\alpha_E \approx 3 \times 10^{-4}$ /K (modulus decreases approximately 0.03%/K).

### 4.3 FEniCSx Thermal Solver

```python
def run_thermal_analysis(
    mesh: dolfinx.mesh.Mesh,
    cell_tags: dolfinx.mesh.MeshTags,
    facet_tags: dolfinx.mesh.MeshTags,
    materials: dict[int, dict],           # cell_tag -> material with thermal props
    heat_source_field: dolfinx.fem.Function,  # Q(x) from harmonic analysis
    interface_heat_flux: float,           # W/m^2 at weld interface
    weld_time_s: float = 0.5,            # total welding time
    dt: float = 0.005,                   # time step (5 ms)
    T_ambient: float = 25.0,             # ambient temperature (C)
    convection_h: float = 10.0,          # convection coefficient W/(m^2 K)
    time_scheme: str = "bdf2",           # "backward_euler" or "bdf2"
) -> ThermoResult:
    """Transient thermal analysis with volumetric and surface heat sources.

    Time integration:
    - Backward Euler: (T^{n+1} - T^n)/dt = RHS^{n+1}  (first-order, unconditionally stable)
    - BDF2: (3*T^{n+1} - 4*T^n + T^{n-1})/(2*dt) = RHS^{n+1}  (second-order)

    Returns temperature field at each time step and derived thermal stresses.
    """
    import dolfinx.fem
    import dolfinx.fem.petsc
    import ufl

    # Scalar function space for temperature
    V_T = dolfinx.fem.functionspace(
        mesh,
        basix.ufl.element("Lagrange", mesh.topology.cell_name(), 1)
    )

    T = dolfinx.fem.Function(V_T, name="Temperature")
    T_n = dolfinx.fem.Function(V_T, name="T_prev")      # T at time n
    T_nm1 = dolfinx.fem.Function(V_T, name="T_prev2")    # T at time n-1 (for BDF2)
    q = ufl.TestFunction(V_T)

    # Initialize to ambient
    T.x.array[:] = T_ambient
    T_n.x.array[:] = T_ambient
    T_nm1.x.array[:] = T_ambient

    # Material properties (piecewise constant per subdomain)
    # rho, cp, k from material database (already in FEA_MATERIALS)
    # Example for a single-material domain:
    rho = materials[1]["rho_kg_m3"]
    cp = materials[1]["cp_j_kgk"]
    k_cond = materials[1]["k_w_mk"]

    rho_c = dolfinx.fem.Constant(mesh, rho * cp)
    k_const = dolfinx.fem.Constant(mesh, k_cond)
    h_conv = dolfinx.fem.Constant(mesh, convection_h)
    T_inf = dolfinx.fem.Constant(mesh, T_ambient)
    dt_const = dolfinx.fem.Constant(mesh, dt)

    # Measures
    dx = ufl.Measure("dx", domain=mesh)
    ds_free = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                          subdomain_id=FREE_SURFACE_TAG)
    ds_weld = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags,
                          subdomain_id=WELD_INTERFACE_TAG)

    # Time derivative approximation
    if time_scheme == "bdf2":
        dT_dt = (3 * T - 4 * T_n + T_nm1) / (2 * dt_const)
    else:  # backward_euler
        dT_dt = (T - T_n) / dt_const

    # Weak form: rho*cp * dT/dt * q + k * grad(T) . grad(q) = Q*q + ...
    F_thermal = (
        rho_c * dT_dt * q * dx
        + k_const * ufl.inner(ufl.grad(T), ufl.grad(q)) * dx
        - heat_source_field * q * dx                              # volumetric source
        - interface_heat_flux * q * ds_weld                       # weld interface flux
        + h_conv * (T - T_inf) * q * ds_free                     # convection loss
    )

    # Newton solver for nonlinear thermal problem
    # (nonlinear because material properties can depend on T)
    problem = dolfinx.fem.petsc.NonlinearProblem(F_thermal, T)
    solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
    solver.rtol = 1e-6
    solver.max_it = 20

    # Time-stepping loop
    n_steps = int(weld_time_s / dt)
    time_array = np.zeros(n_steps + 1)
    T_history = np.zeros((n_steps + 1, len(T.x.array)))
    T_history[0, :] = T.x.array[:]

    for step in range(1, n_steps + 1):
        t = step * dt
        time_array[step] = t

        # Update heat source if time-dependent
        # (e.g., ramp up during weld, zero during hold)
        # update_heat_source(heat_source_field, t, weld_time_s)

        solver.solve(T)

        T_history[step, :] = T.x.array[:]

        # Shift time levels
        T_nm1.x.array[:] = T_n.x.array[:]
        T_n.x.array[:] = T.x.array[:]

    max_temp = float(np.max(T_history))

    return ThermoResult(
        time_steps=time_array,
        temperature_field=T_history,
        thermal_stress_field=np.array([]),  # computed in post-processing step
        max_temperature_c=max_temp,
        frequency_shift_hz=0.0,  # computed from thermal stress
        thermal_expansion_strain=np.array([]),
        solver_backend="fenicsx",
        solve_time_s=0.0,
        mesh=None,
    )
```

### 4.4 Thermal Stress and Frequency Shift Post-Processing

After computing $T(\mathbf{x}, t)$, we compute the thermal-induced frequency shift:

**Step 1: Thermal strain**

$$\varepsilon^{th}_{ij} = \alpha(T - T_0) \, \delta_{ij}$$

where $\alpha$ is the coefficient of thermal expansion (from `FEA_MATERIALS`).

**Step 2: Thermal stress** (from non-uniform temperature distribution)

$$\sigma^{th}_{ij} = c_{ijkl} \bigl(\varepsilon_{kl} - \varepsilon^{th}_{kl}\bigr)$$

**Step 3: Frequency shift prediction**

The frequency shift has two contributions:

1. **Geometric stiffening from thermal stress** (stress-stiffening effect):
$$\Delta f_{stress} = \frac{f_0}{2} \frac{\int \sigma^{th}_{ij} \, \partial \phi_i / \partial x_j \, dV}{\int \rho \, \phi_i \phi_i \, dV}$$

2. **Modulus change from temperature rise:**
$$\Delta f_{modulus} = -\frac{f_0}{2} \frac{\alpha_E \, \Delta T_{avg}}{1}$$
where $\Delta T_{avg}$ is the volume-averaged temperature rise.

The total predicted frequency shift:

$$\Delta f = \Delta f_{stress} + \Delta f_{modulus}$$

```python
def compute_frequency_shift(
    mesh: dolfinx.mesh.Mesh,
    T_field: np.ndarray,           # final temperature distribution
    mode_shape: np.ndarray,        # displacement eigenvector (n_nodes, 3)
    material: dict,                # includes alpha_1_k, E_pa
    f0_hz: float,                  # baseline natural frequency
    T_ref: float = 25.0,          # reference temperature (C)
) -> float:
    """Estimate frequency shift from non-uniform thermal field.

    Returns delta_f in Hz (negative = frequency decreases).
    """
    alpha = material["alpha_1_k"]       # CTE [1/K]
    E0 = material["E_pa"]              # baseline modulus [Pa]
    alpha_E = 3e-4                      # modulus temperature coeff [1/K]
    rho = material["rho_kg_m3"]

    # Volume-averaged temperature rise
    delta_T_avg = np.mean(T_field) - T_ref

    # Modulus-driven shift: df/f = -0.5 * alpha_E * delta_T
    df_modulus = -0.5 * f0_hz * alpha_E * delta_T_avg

    # Thermal expansion shifts geometry -> affects frequency
    # (positive expansion -> lower frequency for longitudinal modes)
    df_expansion = -0.5 * f0_hz * alpha * delta_T_avg

    # Total first-order estimate
    delta_f = df_modulus + df_expansion

    return delta_f
```

### 4.5 Outputs

| Output | Description | Units |
|--------|-------------|-------|
| Temperature field T(x,t) | Transient temperature at every node | C |
| Max temperature | Peak temperature anywhere in the model | C |
| Temperature at weld interface | Interface temperature evolution | C |
| Thermal stress field | Von Mises thermal stress at final time | MPa |
| Frequency shift | Predicted shift from thermal effects | Hz |
| Time to thermal equilibrium | Time constant of temperature rise | s |

---

## 5. Cross-Validation Mode

### 5.1 Purpose

Cross-validation runs the same analysis on both Solver A (numpy/scipy) and Solver B (FEniCSx) and compares the results. This serves three purposes:
1. **Solver verification** -- confirms both implementations are correct
2. **Mesh convergence validation** -- verifies results are mesh-independent
3. **Confidence scoring** -- quantifies agreement for production use

### 5.2 Comparison Metrics

**5.2.1 Natural Frequency Comparison**

$$\delta_i = \frac{|f_i^A - f_i^B|}{f_i^A} \times 100\%$$

Tolerance thresholds:
- PASS: $\delta_i < 1\%$ for all modes
- WARNING: $1\% \leq \delta_i < 5\%$ for any mode
- FAIL: $\delta_i \geq 5\%$ for any mode

**5.2.2 Mode Shape Correlation (MAC Matrix)**

The Modal Assurance Criterion compares mode shapes between solvers:

$$\text{MAC}_{ij} = \frac{|\boldsymbol{\phi}_i^{A\,T} \boldsymbol{\phi}_j^B|^2}{(\boldsymbol{\phi}_i^{A\,T} \boldsymbol{\phi}_i^A)(\boldsymbol{\phi}_j^{B\,T} \boldsymbol{\phi}_j^B)}$$

where $\boldsymbol{\phi}_i^A$ is the i-th mode shape from Solver A and $\boldsymbol{\phi}_j^B$ is the j-th mode shape from Solver B.

Interpretation:
- MAC = 1.0: identical mode shapes
- MAC > 0.9: strong correlation (modes match)
- MAC < 0.7: weak correlation (mode mismatch or mode swapping)

The diagonal of the MAC matrix (MAC_ii) indicates how well corresponding modes agree.

**5.2.3 Stress Peak Comparison**

$$\delta_{\sigma} = \frac{|\sigma_{max}^A - \sigma_{max}^B|}{\max(\sigma_{max}^A, \sigma_{max}^B)} \times 100\%$$

**5.2.4 Amplitude Uniformity Comparison**

$$\delta_U = |U^A - U^B|$$

where $U$ is the amplitude uniformity metric (0 to 1).

### 5.3 Implementation

```python
# ultrasonic_weld_master/solvers/cross_validator.py

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from ultrasonic_weld_master.core.solver_backend import (
    SolverBackend, ModalResult, HarmonicResult, MeshData,
)

logger = logging.getLogger(__name__)


@dataclass
class MACMatrix:
    """Modal Assurance Criterion matrix between two sets of mode shapes."""
    values: np.ndarray              # (n_modes_A, n_modes_B)
    diagonal: np.ndarray            # (min(n_A, n_B),)  MAC_ii
    mode_pairing: list[tuple[int, int]]  # best-match pairs (i_A, j_B)

    @property
    def min_diagonal(self) -> float:
        return float(np.min(self.diagonal)) if len(self.diagonal) > 0 else 0.0

    @property
    def mean_diagonal(self) -> float:
        return float(np.mean(self.diagonal)) if len(self.diagonal) > 0 else 0.0


@dataclass
class CrossValidationReport:
    """Comprehensive comparison report between Solver A and Solver B."""
    solver_a_name: str
    solver_b_name: str

    # Frequency comparison
    freq_a_hz: np.ndarray
    freq_b_hz: np.ndarray
    freq_deviation_percent: np.ndarray
    freq_max_deviation_percent: float

    # Mode shape comparison
    mac: MACMatrix

    # Stress comparison
    stress_max_a_mpa: float
    stress_max_b_mpa: float
    stress_deviation_percent: float

    # Amplitude comparison
    uniformity_a: float
    uniformity_b: float
    uniformity_deviation: float

    # Overall verdict
    status: str  # "PASS", "WARNING", "FAIL"
    messages: list[str] = field(default_factory=list)
    solve_time_a_s: float = 0.0
    solve_time_b_s: float = 0.0

    # Thresholds used
    thresholds: dict = field(default_factory=lambda: {
        "freq_pass": 1.0,       # %
        "freq_warn": 5.0,       # %
        "mac_pass": 0.9,
        "mac_warn": 0.7,
        "stress_pass": 10.0,    # %
        "stress_warn": 25.0,    # %
    })


def compute_mac_matrix(
    modes_a: np.ndarray,    # (n_modes_A, n_nodes, 3)
    modes_b: np.ndarray,    # (n_modes_B, n_nodes, 3)
    node_mapping: np.ndarray | None = None,  # if meshes differ
) -> MACMatrix:
    """Compute the MAC matrix between two sets of mode shapes.

    If the meshes are different, node_mapping maps Solver B nodes to
    the nearest Solver A nodes (computed via spatial interpolation).

    MAC_ij = |phi_A_i^T * phi_B_j|^2 / ((phi_A_i^T * phi_A_i) * (phi_B_j^T * phi_B_j))
    """
    n_a = modes_a.shape[0]
    n_b = modes_b.shape[0]

    # Flatten mode shapes to (n_modes, 3*n_nodes)
    phi_a = modes_a.reshape(n_a, -1)  # (n_a, 3N)
    phi_b = modes_b.reshape(n_b, -1)  # (n_b, 3N)

    if node_mapping is not None:
        # Reorder Solver B nodes to match Solver A ordering
        n_nodes = modes_a.shape[1]
        phi_b_mapped = np.zeros_like(phi_a[:n_b])
        for j in range(n_b):
            mode_3d = modes_b[j]  # (n_nodes_b, 3)
            mapped = mode_3d[node_mapping]  # (n_nodes_a, 3)
            phi_b_mapped[j] = mapped.flatten()
        phi_b = phi_b_mapped

    # MAC matrix computation
    mac_vals = np.zeros((n_a, n_b))
    for i in range(n_a):
        for j in range(n_b):
            cross = np.dot(phi_a[i], phi_b[j])
            auto_a = np.dot(phi_a[i], phi_a[i])
            auto_b = np.dot(phi_b[j], phi_b[j])
            denom = auto_a * auto_b
            if denom > 1e-30:
                mac_vals[i, j] = cross ** 2 / denom
            else:
                mac_vals[i, j] = 0.0

    # Diagonal (corresponding modes)
    n_diag = min(n_a, n_b)
    diagonal = np.array([mac_vals[i, i] for i in range(n_diag)])

    # Best-match pairing (greedy, based on MAC value)
    pairing: list[tuple[int, int]] = []
    used_b = set()
    for i in range(n_a):
        best_j = -1
        best_mac = -1.0
        for j in range(n_b):
            if j not in used_b and mac_vals[i, j] > best_mac:
                best_mac = mac_vals[i, j]
                best_j = j
        if best_j >= 0:
            pairing.append((i, best_j))
            used_b.add(best_j)

    return MACMatrix(values=mac_vals, diagonal=diagonal, mode_pairing=pairing)


def run_cross_validation(
    solver_a: SolverBackend,
    solver_b: SolverBackend,
    mesh: MeshData,
    material: dict,
    boundary_conditions: dict,
    target_frequency_hz: float,
    n_modes: int = 10,
    thresholds: dict | None = None,
) -> CrossValidationReport:
    """Run identical modal + harmonic analysis on both solvers and compare.

    The same MeshData is provided to both solvers. Solver A interprets it
    as numpy arrays directly; Solver B converts it to a DOLFINx mesh
    via the mesh bridge.
    """
    # --- Run Solver A ---
    t0 = time.perf_counter()
    modal_a = solver_a.run_modal(
        mesh, material, boundary_conditions, target_frequency_hz, n_modes
    )
    t_a = time.perf_counter() - t0

    # --- Run Solver B ---
    t0 = time.perf_counter()
    modal_b = solver_b.run_modal(
        mesh, material, boundary_conditions, target_frequency_hz, n_modes
    )
    t_b = time.perf_counter() - t0

    # --- Compare frequencies ---
    n_compare = min(len(modal_a.frequencies_hz), len(modal_b.frequencies_hz))
    freq_a = modal_a.frequencies_hz[:n_compare]
    freq_b = modal_b.frequencies_hz[:n_compare]
    freq_dev = np.abs(freq_a - freq_b) / np.maximum(freq_a, 1e-6) * 100
    max_freq_dev = float(np.max(freq_dev))

    # --- Compare mode shapes ---
    mac = compute_mac_matrix(
        modal_a.mode_shapes[:n_compare],
        modal_b.mode_shapes[:n_compare],
    )

    # --- Compare stress ---
    stress_a = modal_a.metadata.get("stress_max_mpa", 0.0)
    stress_b = modal_b.metadata.get("stress_max_mpa", 0.0)
    stress_denom = max(stress_a, stress_b, 1e-6)
    stress_dev = abs(stress_a - stress_b) / stress_denom * 100

    # --- Determine verdict ---
    thresh = thresholds or {
        "freq_pass": 1.0, "freq_warn": 5.0,
        "mac_pass": 0.9, "mac_warn": 0.7,
        "stress_pass": 10.0, "stress_warn": 25.0,
    }

    messages = []
    status = "PASS"

    if max_freq_dev >= thresh["freq_warn"]:
        status = "FAIL"
        messages.append(
            f"Frequency deviation {max_freq_dev:.2f}% exceeds {thresh['freq_warn']}% threshold"
        )
    elif max_freq_dev >= thresh["freq_pass"]:
        status = "WARNING"
        messages.append(
            f"Frequency deviation {max_freq_dev:.2f}% exceeds {thresh['freq_pass']}% threshold"
        )

    if mac.min_diagonal < thresh["mac_warn"]:
        status = "FAIL"
        messages.append(
            f"Minimum MAC diagonal {mac.min_diagonal:.3f} below {thresh['mac_warn']} threshold"
        )
    elif mac.min_diagonal < thresh["mac_pass"]:
        if status != "FAIL":
            status = "WARNING"
        messages.append(
            f"Minimum MAC diagonal {mac.min_diagonal:.3f} below {thresh['mac_pass']} threshold"
        )

    if stress_dev >= thresh["stress_warn"]:
        status = "FAIL"
        messages.append(
            f"Stress deviation {stress_dev:.1f}% exceeds {thresh['stress_warn']}% threshold"
        )
    elif stress_dev >= thresh["stress_pass"]:
        if status != "FAIL":
            status = "WARNING"
        messages.append(
            f"Stress deviation {stress_dev:.1f}% exceeds {thresh['stress_pass']}% threshold"
        )

    if not messages:
        messages.append("All metrics within acceptance thresholds")

    return CrossValidationReport(
        solver_a_name=solver_a.get_name(),
        solver_b_name=solver_b.get_name(),
        freq_a_hz=freq_a,
        freq_b_hz=freq_b,
        freq_deviation_percent=freq_dev,
        freq_max_deviation_percent=max_freq_dev,
        mac=mac,
        stress_max_a_mpa=stress_a,
        stress_max_b_mpa=stress_b,
        stress_deviation_percent=stress_dev,
        uniformity_a=0.0,  # filled when harmonic comparison is run
        uniformity_b=0.0,
        uniformity_deviation=0.0,
        status=status,
        messages=messages,
        solve_time_a_s=t_a,
        solve_time_b_s=t_b,
        thresholds=thresh,
    )
```

### 5.4 Cross-Validation Workflow

```
1. User selects "Cross-Validation" analysis mode
2. System generates shared Gmsh mesh
3. Mesh converted to both numpy arrays (Solver A) and DOLFINx mesh (Solver B)
4. Both solvers run modal analysis with identical parameters
5. Results compared: frequencies, MAC matrix, stress peaks
6. If harmonic analysis requested, both solvers run frequency sweep
7. Additional comparison: amplitude uniformity, response peaks
8. Report generated with PASS/WARNING/FAIL verdict per metric
9. Results stored for historical tracking of solver agreement
```

---

## 6. Deployment

### 6.1 Docker Container

The recommended deployment packages FEniCSx, PETSc, SLEPc, Gmsh, and all Python dependencies into a single Docker image:

```dockerfile
# Dockerfile.fenicsx-solver
FROM ghcr.io/fenics/dolfinx/dolfinx:v0.8.0

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    gmsh==4.12.2 \
    meshio>=5.3 \
    pyyaml>=6.0 \
    fastapi>=0.110 \
    uvicorn[standard]>=0.27 \
    httpx>=0.27

# Copy the solver plugin code
COPY ultrasonic_weld_master/ /app/ultrasonic_weld_master/
COPY web/ /app/web/

WORKDIR /app

# REST API entrypoint
EXPOSE 8080
CMD ["uvicorn", "web.fenicsx_api:app", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```bash
docker build -t uwm/fenicsx-solver:latest -f Dockerfile.fenicsx-solver .
docker run -d -p 8080:8080 --name uwm-fenicsx uwm/fenicsx-solver:latest
```

### 6.2 REST API Wrapper

When running as a Docker service, the FEniCSx backend exposes a REST API for the main application to call:

```python
# web/fenicsx_api.py

from __future__ import annotations
import asyncio
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="UWM FEniCSx Solver API", version="0.1.0")
logger = logging.getLogger(__name__)

# Process pool for CPU-bound FEA solves
_executor = ProcessPoolExecutor(max_workers=2)

# In-memory result cache (production: use Redis)
_results: dict[str, dict] = {}
_status: dict[str, str] = {}  # job_id -> "pending" | "running" | "completed" | "failed"


class AnalysisRequest(BaseModel):
    analysis_type: str           # "piezoelectric", "contact", "thermomechanical", "modal"
    geometry_params: dict        # passed to Gmsh mesh builder
    material_params: dict        # material properties
    boundary_conditions: dict    # BCs
    solver_params: dict = {}     # frequency range, damping, etc.


class JobStatus(BaseModel):
    job_id: str
    status: str                  # "pending", "running", "completed", "failed"
    progress: float = 0.0       # 0.0 to 1.0
    result: dict | None = None
    error: str | None = None


@app.post("/api/v1/analyze", response_model=dict)
async def submit_analysis(request: AnalysisRequest) -> dict:
    """Submit an analysis job. Returns a job_id for polling."""
    job_id = str(uuid.uuid4())
    _status[job_id] = "pending"

    # Run in background process pool to avoid blocking
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_analysis_sync,
        job_id,
        request.model_dump(),
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/v1/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str) -> JobStatus:
    """Poll analysis job status."""
    status = _status.get(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = _results.get(job_id) if status == "completed" else None
    return JobStatus(
        job_id=job_id,
        status=status,
        result=result,
    )


@app.get("/api/v1/result/{job_id}")
async def get_result(job_id: str) -> dict:
    """Retrieve completed analysis results."""
    if _status.get(job_id) != "completed":
        raise HTTPException(status_code=404,
                            detail=f"Job {job_id} not completed")
    return _results[job_id]


@app.get("/api/v1/capabilities")
async def get_capabilities() -> dict:
    """Return available analysis types and solver version info."""
    from ultrasonic_weld_master.solvers.fenicsx_backend import FEniCSxBackend
    backend = FEniCSxBackend()
    return {
        "available": backend.is_available(),
        "capabilities": [c.value for c in backend.get_capabilities()],
        "dolfinx_version": getattr(__import__("dolfinx"), "__version__", "unknown"),
        "petsc_version": "unknown",
    }


def _run_analysis_sync(job_id: str, params: dict) -> None:
    """Synchronous analysis runner (executed in process pool)."""
    from ultrasonic_weld_master.solvers.fenicsx_backend import FEniCSxBackend

    _status[job_id] = "running"
    try:
        backend = FEniCSxBackend()
        analysis_type = params["analysis_type"]

        if analysis_type == "piezoelectric":
            result = backend.run_piezoelectric(**params)
        elif analysis_type == "contact":
            result = backend.run_contact(**params)
        elif analysis_type == "thermomechanical":
            result = backend.run_thermomechanical(**params)
        elif analysis_type == "modal":
            result = backend.run_modal(**params)
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        _results[job_id] = _serialize_result(result)
        _status[job_id] = "completed"

    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        _results[job_id] = {"error": str(exc)}
        _status[job_id] = "failed"


def _serialize_result(result: Any) -> dict:
    """Convert dataclass result with numpy arrays to JSON-serializable dict."""
    import numpy as np
    from dataclasses import asdict

    d = asdict(result)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    return convert(d)
```

### 6.3 Client-Side Integration

When the main application needs to call the Docker-based solver, it uses an async HTTP client:

```python
# ultrasonic_weld_master/solvers/fenicsx_client.py

from __future__ import annotations
import asyncio
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080"
POLL_INTERVAL_S = 2.0
MAX_WAIT_S = 600.0  # 10 minutes timeout


class FEniCSxClient:
    """HTTP client for the FEniCSx Docker solver service."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        self._base_url = base_url.rstrip("/")

    async def check_health(self) -> bool:
        """Check if the FEniCSx solver service is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/api/v1/capabilities")
                return resp.status_code == 200
        except httpx.RequestError:
            return False

    async def submit_and_wait(
        self,
        analysis_type: str,
        geometry_params: dict,
        material_params: dict,
        boundary_conditions: dict,
        solver_params: dict | None = None,
        progress_callback: Any = None,
    ) -> dict:
        """Submit an analysis, poll for completion, and return results.

        Raises TimeoutError if the analysis does not complete within MAX_WAIT_S.
        Raises RuntimeError if the analysis fails.
        """
        payload = {
            "analysis_type": analysis_type,
            "geometry_params": geometry_params,
            "material_params": material_params,
            "boundary_conditions": boundary_conditions,
            "solver_params": solver_params or {},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Submit
            resp = await client.post(
                f"{self._base_url}/api/v1/analyze",
                json=payload,
            )
            resp.raise_for_status()
            job_id = resp.json()["job_id"]
            logger.info("Submitted FEniCSx job: %s", job_id)

            # Poll
            t0 = time.monotonic()
            while time.monotonic() - t0 < MAX_WAIT_S:
                await asyncio.sleep(POLL_INTERVAL_S)
                resp = await client.get(
                    f"{self._base_url}/api/v1/status/{job_id}"
                )
                resp.raise_for_status()
                status_data = resp.json()
                status = status_data["status"]

                if progress_callback:
                    progress_callback(status_data.get("progress", 0.0))

                if status == "completed":
                    resp = await client.get(
                        f"{self._base_url}/api/v1/result/{job_id}"
                    )
                    resp.raise_for_status()
                    return resp.json()

                if status == "failed":
                    error = status_data.get("error", "Unknown error")
                    raise RuntimeError(
                        f"FEniCSx analysis failed: {error}"
                    )

            raise TimeoutError(
                f"FEniCSx analysis {job_id} did not complete within "
                f"{MAX_WAIT_S}s"
            )
```

### 6.4 Direct Python Import (Alternative)

When FEniCSx is installed locally (e.g., in a conda environment), the backend is used directly without HTTP:

```python
# In SolverDispatcher registration:

try:
    from ultrasonic_weld_master.solvers.fenicsx_backend import FEniCSxBackend
    backend = FEniCSxBackend()
    if backend.is_available():
        dispatcher.register(backend)
except ImportError:
    # Try Docker-based remote backend
    from ultrasonic_weld_master.solvers.fenicsx_client import FEniCSxClient
    client = FEniCSxClient()
    # Wrap client as SolverBackend adapter (async-to-sync bridge)
    # ...
```

### 6.5 Result Caching

Analysis results are cached to avoid redundant computation:

```python
# Cache key: hash of (analysis_type, geometry_params, material_params, BCs, solver_params)
# Cache storage: SQLite table or filesystem JSON files
# Cache invalidation: on parameter change or manual clear

import hashlib
import json

def compute_cache_key(analysis_type: str, params: dict) -> str:
    """Deterministic cache key from analysis parameters."""
    canonical = json.dumps(
        {"type": analysis_type, **params},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]
```

Cache entries are stored in the existing SQLite database in a new table:

```sql
CREATE TABLE IF NOT EXISTS fea_cache (
    cache_key TEXT PRIMARY KEY,
    analysis_type TEXT NOT NULL,
    params_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    solver_backend TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    solve_time_s REAL,
    mesh_node_count INTEGER,
    hit_count INTEGER DEFAULT 0
);
```

### 6.6 Architecture Summary

```
┌──────────────────────────────────────────────────────────┐
│                    Main Application                       │
│                                                           │
│  ┌───────────────┐     ┌────────────────────────────┐    │
│  │ SolverDispatch│     │  Cross-Validation Engine    │    │
│  │               │     │  (runs both, compares)      │    │
│  └───────┬───────┘     └─────────────┬──────────────┘    │
│          │                           │                    │
│   ┌──────┴──────┐            ┌───────┴───────┐           │
│   │             │            │               │           │
│   ▼             ▼            ▼               ▼           │
│ ┌──────────┐ ┌──────────────────────────────────┐        │
│ │ Solver A │ │         Solver B Interface        │        │
│ │ numpy/   │ │                                   │        │
│ │ scipy    │ │  ┌─────────┐  ┌───────────────┐  │        │
│ │ (local)  │ │  │ Direct  │  │ Docker/HTTP   │  │        │
│ │          │ │  │ import  │  │ client        │  │        │
│ └──────────┘ │  │ (conda) │  │ (FEniCSxClient│  │        │
│              │  └────┬────┘  └──────┬────────┘  │        │
│              └───────┼──────────────┼───────────┘        │
│                      │              │                     │
└──────────────────────┼──────────────┼─────────────────────┘
                       │              │
            ┌──────────┘      ┌───────┘
            ▼                 ▼
     ┌────────────┐   ┌─────────────────────┐
     │ Local      │   │ Docker Container     │
     │ FEniCSx    │   │ ┌─────────────────┐ │
     │ (conda     │   │ │ FEniCSx/DOLFINx │ │
     │  install)  │   │ │ PETSc/SLEPc     │ │
     │            │   │ │ Gmsh            │ │
     └────────────┘   │ │ FastAPI server  │ │
                      │ └─────────────────┘ │
                      │   Port 8080         │
                      └─────────────────────┘
```

### 6.7 Shared Gmsh Mesh Pipeline

```
                    ┌──────────────────┐
                    │   Gmsh Python    │
                    │   API (shared)   │
                    └────────┬─────────┘
                             │
                    ┌────────┴─────────┐
                    │  .msh / in-mem   │
                    └────────┬─────────┘
                             │
               ┌─────────────┴──────────────┐
               │                            │
               ▼                            ▼
      ┌────────────────┐          ┌─────────────────┐
      │ gmsh_to_numpy  │          │ gmsh_to_dolfinx │
      │                │          │ (gmshio)        │
      │  → MeshData    │          │ → dolfinx.Mesh  │
      │    (vertices,  │          │   (cell_tags,   │
      │     elements)  │          │    facet_tags)   │
      └───────┬────────┘          └────────┬────────┘
              │                            │
              ▼                            ▼
         Solver A                     Solver B
       (numpy/scipy)               (FEniCSx/PETSc)
```

---

## Appendix A: File Listing

| File | Purpose | New/Modified |
|------|---------|-------------|
| `ultrasonic_weld_master/core/solver_backend.py` | SolverBackend ABC + result dataclasses | New |
| `ultrasonic_weld_master/solvers/__init__.py` | Package init | New |
| `ultrasonic_weld_master/solvers/numpy_scipy_backend.py` | Solver A adapter wrapping FEAService | New |
| `ultrasonic_weld_master/solvers/fenicsx_backend.py` | Solver B: FEniCSx implementation | New |
| `ultrasonic_weld_master/solvers/mesh_bridge.py` | Gmsh <-> DOLFINx <-> numpy mesh conversion | New |
| `ultrasonic_weld_master/solvers/dispatcher.py` | Routes analysis to appropriate backend | New |
| `ultrasonic_weld_master/solvers/cross_validator.py` | Cross-validation engine | New |
| `ultrasonic_weld_master/solvers/fenicsx_client.py` | HTTP client for Docker-based solver | New |
| `ultrasonic_weld_master/solvers/piezo_materials.py` | PZT-4/PZT-8 material tensor database | New |
| `web/fenicsx_api.py` | FastAPI REST wrapper for Docker deployment | New |
| `web/services/engine_service.py` | Register FEniCSx backend at startup | Modified |
| `Dockerfile.fenicsx-solver` | Docker image for FEniCSx environment | New |
| `tests/test_solvers/test_cross_validation.py` | Cross-validation unit tests | New |
| `tests/test_solvers/test_fenicsx_backend.py` | FEniCSx backend tests (skip if not installed) | New |

## Appendix B: Capability Matrix

| Analysis Type | Solver A (numpy/scipy) | Solver B (FEniCSx) | Cross-Validation |
|--------------|----------------------|-------------------|------------------|
| Modal analysis | Yes | Yes | Yes |
| Harmonic response | Yes | Yes | Yes |
| Static stress | Yes | Yes | Yes |
| Piezoelectric coupling | No | Yes | N/A (Solver B only) |
| Nonlinear contact | No | Yes | N/A (Solver B only) |
| Thermomechanical coupling | No | Yes | N/A (Solver B only) |
| Pre-stressed modal | No | Yes | N/A (Solver B only) |
| Fatigue life estimate | Yes (post-process) | Yes (post-process) | Yes |

## Appendix C: Performance Expectations

| Analysis | Typical Model Size | Solver A Time | Solver B Time |
|----------|-------------------|--------------|--------------|
| Modal (10 modes) | 5k--50k nodes | 2--15 s | 5--30 s |
| Harmonic sweep (21 pts) | 5k--50k nodes | 10--60 s | 30--180 s |
| Piezoelectric impedance (200 pts) | 20k--100k nodes | N/A | 5--30 min |
| Nonlinear contact | 50k--200k nodes | N/A | 10--60 min |
| Thermomechanical transient (100 steps) | 20k--100k nodes | N/A | 15--90 min |
| Cross-validation (modal) | 5k--50k nodes | 2x modal time | -- |

Solver A is significantly faster for standard analyses because it uses structured hex meshes, lumped mass matrices, and direct sparse solvers, making it the preferred choice for daily design iteration. Solver B is reserved for detailed verification and physics that Solver A cannot model.
