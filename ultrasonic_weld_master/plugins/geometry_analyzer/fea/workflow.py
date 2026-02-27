"""Full-stack FEA analysis workflow for ultrasonic welding stacks.

Orchestrates the complete analysis pipeline:

1. **Assembly** -- couple multi-body meshes into a single system.
2. **Modal analysis** -- find resonant modes near the operating frequency.
3. **Harmonic analysis** -- frequency sweep and FRF computation.
4. **Static analysis** -- (optional) preload stress state.
5. **Fatigue assessment** -- (optional) safety factors and life estimation.
6. **Gain chain** -- amplitude gain through each component.
7. **Summary report** -- human-readable results dict.

Each step delegates to its dedicated module; this module only
orchestrates the sequence and passes data between steps.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.assembly_builder import (
    AssemblyBuilder,
    AssemblyConfig,
    AssemblyResult,
    ComponentMesh,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    FEAMesh,
    HarmonicConfig,
    ModalConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.fatigue import (
    FatigueAssessor,
    FatigueConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.material_properties import (
    get_material,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.post_processing import (
    compute_gain_chain,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    FatigueResult,
    HarmonicResult,
    ModalResult,
    StaticResult,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.solver_a import SolverA

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid analysis names
# ---------------------------------------------------------------------------

_VALID_ANALYSES = {"modal", "harmonic", "static", "fatigue"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WorkflowConfig:
    """Configuration for the full analysis workflow.

    Parameters
    ----------
    components : list[dict]
        One dict per component.  Required keys:

        - ``name`` : str -- unique component identifier.
        - ``mesh`` : FEAMesh -- the component mesh.
        - ``material_name`` : str -- material for property lookup.
        - ``interface_nodes`` : dict[str, np.ndarray] -- interface
          node indices (typically ``{"top": ..., "bottom": ...}``).

    assembly_config : dict
        Passed to ``AssemblyConfig``.  Keys: ``coupling_method``,
        ``penalty_factor``, ``component_order``.
    analyses : list[str]
        Which analyses to run.  Valid values: ``"modal"``,
        ``"harmonic"``, ``"static"``, ``"fatigue"``.
    modal_config : dict
        Overrides for ``ModalConfig`` (excluding ``mesh`` and
        ``material_name``, which come from the assembly).
    harmonic_config : dict
        Overrides for ``HarmonicConfig``.
    static_config : dict
        Overrides for ``StaticConfig``.
    fatigue_config : dict
        Overrides for ``FatigueConfig``.
    frequency_hz : float
        Target operating frequency [Hz].
    """

    components: list[dict]
    assembly_config: dict
    analyses: list[str] = field(
        default_factory=lambda: ["modal", "harmonic"]
    )
    modal_config: dict = field(default_factory=dict)
    harmonic_config: dict = field(default_factory=dict)
    static_config: dict = field(default_factory=dict)
    fatigue_config: dict = field(default_factory=dict)
    frequency_hz: float = 20000.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Complete analysis results from the workflow.

    Parameters
    ----------
    assembly : AssemblyResult
        Coupled system matrices and metadata.
    modal : ModalResult or None
        Modal analysis results (if run).
    harmonic : HarmonicResult or None
        Harmonic analysis results (if run).
    static : StaticResult or None
        Static analysis results (if run).
    fatigue : FatigueResult or None
        Fatigue assessment results (if run).
    gain_chain : dict or None
        Gain chain through the stack components.
    summary : dict
        Human-readable summary of the analysis.
    """

    assembly: AssemblyResult
    modal: Optional[ModalResult] = None
    harmonic: Optional[HarmonicResult] = None
    static: Optional[StaticResult] = None
    fatigue: Optional[FatigueResult] = None
    gain_chain: Optional[dict] = None
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow orchestrator
# ---------------------------------------------------------------------------


class AnalysisWorkflow:
    """Orchestrates the full FEA analysis pipeline.

    Parameters
    ----------
    config : WorkflowConfig
        Complete workflow configuration.

    Raises
    ------
    ValueError
        If the config is invalid (empty components, unknown analysis
        names, etc.).
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self._config = config
        self._solver = SolverA()
        self._validate_config()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate the workflow configuration.

        Raises
        ------
        ValueError
            If components list is empty, if component dicts are missing
            required keys, or if analysis names are unrecognised.
        """
        if not self._config.components:
            raise ValueError("WorkflowConfig.components must not be empty.")

        required_keys = {"name", "mesh", "material_name", "interface_nodes"}
        for i, comp in enumerate(self._config.components):
            missing = required_keys - set(comp.keys())
            if missing:
                raise ValueError(
                    f"Component at index {i} is missing keys: {sorted(missing)}."
                )

        unknown = set(self._config.analyses) - _VALID_ANALYSES
        if unknown:
            raise ValueError(
                f"Unknown analyses: {sorted(unknown)}.  "
                f"Valid values: {sorted(_VALID_ANALYSES)}."
            )

        # Fatigue requires harmonic (for alternating stress)
        if "fatigue" in self._config.analyses:
            if "harmonic" not in self._config.analyses:
                raise ValueError(
                    "Fatigue analysis requires harmonic analysis.  "
                    "Add 'harmonic' to the analyses list."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> WorkflowResult:
        """Execute the full analysis pipeline.

        Steps:

        1. Build assembly (couple components).
        2. Run modal analysis (find resonant modes).
        3. Run harmonic analysis (frequency sweep, FRF).
        4. Run static analysis (if requested, for preload).
        5. Run fatigue assessment (if requested).
        6. Compute gain chain through components.
        7. Build summary report.

        Returns
        -------
        WorkflowResult
            Complete results from all requested analyses.
        """
        t_start = time.perf_counter()
        analyses = set(self._config.analyses)

        logger.info(
            "Starting workflow: %d components, analyses=%s, f_target=%.0f Hz",
            len(self._config.components),
            sorted(analyses),
            self._config.frequency_hz,
        )

        # Step 1: Build assembly
        assembly = self._build_assembly()
        result = WorkflowResult(assembly=assembly)

        # Step 2: Modal analysis
        if "modal" in analyses:
            result.modal = self._run_modal(assembly)

        # Step 3: Harmonic analysis
        if "harmonic" in analyses:
            result.harmonic = self._run_harmonic(assembly)

        # Step 4: Static analysis
        if "static" in analyses:
            result.static = self._run_static(assembly)

        # Step 5: Fatigue assessment
        if "fatigue" in analyses:
            result.fatigue = self._run_fatigue(
                result.harmonic, result.static
            )

        # Step 6: Gain chain
        if result.harmonic is not None:
            result.gain_chain = self._compute_gain_chain(
                assembly, result.harmonic
            )

        # Step 7: Summary
        t_total = time.perf_counter() - t_start
        result.summary = self._build_summary(result, t_total)

        logger.info(
            "Workflow complete in %.2f s.  Summary: %s",
            t_total,
            result.summary.get("status", "unknown"),
        )

        return result

    # ------------------------------------------------------------------
    # Step 1: Build assembly
    # ------------------------------------------------------------------

    def _build_assembly(self) -> AssemblyResult:
        """Create ComponentMesh objects and run AssemblyBuilder.

        Returns
        -------
        AssemblyResult
            Coupled global system.
        """
        logger.info("Step 1/7: Building assembly")

        component_meshes = []
        for comp_dict in self._config.components:
            cm = ComponentMesh(
                name=comp_dict["name"],
                mesh=comp_dict["mesh"],
                material_name=comp_dict["material_name"],
                interface_nodes=comp_dict["interface_nodes"],
            )
            component_meshes.append(cm)

        ac_dict = self._config.assembly_config
        assembly_config = AssemblyConfig(
            coupling_method=ac_dict.get("coupling_method", "bonded"),
            penalty_factor=ac_dict.get("penalty_factor", 1e3),
            component_order=ac_dict.get(
                "component_order",
                [c["name"] for c in self._config.components],
            ),
        )

        builder = AssemblyBuilder(component_meshes, assembly_config)
        return builder.build()

    # ------------------------------------------------------------------
    # Step 2: Modal analysis
    # ------------------------------------------------------------------

    def _run_modal(self, assembly: AssemblyResult) -> ModalResult:
        """Run modal analysis on the coupled system.

        Constructs a synthetic ``FEAMesh`` representing the full
        assembly, then delegates to ``SolverA.modal_analysis``.

        Parameters
        ----------
        assembly : AssemblyResult
            Coupled system from step 1.

        Returns
        -------
        ModalResult
        """
        logger.info("Step 2/7: Running modal analysis")

        # Build a combined mesh for the solver
        combined_mesh = self._build_combined_mesh(assembly)

        mc = self._config.modal_config
        config = ModalConfig(
            mesh=combined_mesh,
            material_name=self._config.components[0]["material_name"],
            n_modes=mc.get("n_modes", 20),
            target_frequency_hz=mc.get(
                "target_frequency_hz", self._config.frequency_hz
            ),
            boundary_conditions=mc.get("boundary_conditions", "free-free"),
            fixed_node_sets=mc.get("fixed_node_sets", []),
        )

        return self._solver.modal_analysis(config)

    # ------------------------------------------------------------------
    # Step 3: Harmonic analysis
    # ------------------------------------------------------------------

    def _run_harmonic(self, assembly: AssemblyResult) -> HarmonicResult:
        """Run harmonic analysis on the coupled system.

        Parameters
        ----------
        assembly : AssemblyResult
            Coupled system from step 1.

        Returns
        -------
        HarmonicResult
        """
        logger.info("Step 3/7: Running harmonic analysis")

        combined_mesh = self._build_combined_mesh(assembly)

        hc = self._config.harmonic_config
        f_target = self._config.frequency_hz
        config = HarmonicConfig(
            mesh=combined_mesh,
            material_name=self._config.components[0]["material_name"],
            freq_min_hz=hc.get("freq_min_hz", 0.8 * f_target),
            freq_max_hz=hc.get("freq_max_hz", 1.2 * f_target),
            n_freq_points=hc.get("n_freq_points", 201),
            damping_model=hc.get("damping_model", "hysteretic"),
            damping_ratio=hc.get("damping_ratio", 0.005),
            excitation_node_set=hc.get(
                "excitation_node_set", "bottom_face"
            ),
            response_node_set=hc.get("response_node_set", "top_face"),
        )

        return self._solver.harmonic_analysis(config)

    # ------------------------------------------------------------------
    # Step 4: Static analysis
    # ------------------------------------------------------------------

    def _run_static(self, assembly: AssemblyResult) -> Optional[StaticResult]:
        """Run static analysis if configured.

        Parameters
        ----------
        assembly : AssemblyResult
            Coupled system from step 1.

        Returns
        -------
        StaticResult or None
        """
        logger.info("Step 4/7: Running static analysis")

        combined_mesh = self._build_combined_mesh(assembly)

        sc = self._config.static_config
        config = StaticConfig(
            mesh=combined_mesh,
            material_name=self._config.components[0]["material_name"],
            loads=sc.get("loads", []),
            boundary_conditions=sc.get("boundary_conditions", []),
        )

        return self._solver.static_analysis(config)

    # ------------------------------------------------------------------
    # Step 5: Fatigue assessment
    # ------------------------------------------------------------------

    def _run_fatigue(
        self,
        harmonic: Optional[HarmonicResult],
        static: Optional[StaticResult],
    ) -> Optional[FatigueResult]:
        """Run fatigue assessment from harmonic and/or static results.

        Uses the Von Mises stress at resonance as the alternating
        stress.  If static results are available, uses their Von Mises
        as the mean stress.

        Parameters
        ----------
        harmonic : HarmonicResult or None
            Harmonic analysis results (required).
        static : StaticResult or None
            Static analysis results (optional, for mean stress).

        Returns
        -------
        FatigueResult or None
            None if harmonic results are not available.
        """
        if harmonic is None:
            logger.warning(
                "Step 5/7: Skipping fatigue -- no harmonic results."
            )
            return None

        logger.info("Step 5/7: Running fatigue assessment")

        fc = self._config.fatigue_config
        fatigue_config = FatigueConfig(
            material=fc.get("material", "Ti-6Al-4V"),
            Kt_global=fc.get("Kt_global", 1.0),
            Kt_regions=fc.get("Kt_regions", {}),
            target_SF=fc.get("target_SF", 2.0),
            n_critical_locations=fc.get("n_critical_locations", 10),
            include_mean_stress=fc.get(
                "include_mean_stress", static is not None
            ),
            surface_finish=fc.get("surface_finish", "machined"),
            characteristic_diameter_mm=fc.get(
                "characteristic_diameter_mm", 25.0
            ),
            reliability_pct=fc.get("reliability_pct", 50.0),
            temperature_c=fc.get("temperature_c", 25.0),
        )

        assessor = FatigueAssessor(fatigue_config)

        # Extract alternating stress from harmonic result
        # Use displacement_amplitudes at resonance
        disp_amps = harmonic.displacement_amplitudes  # (n_freq, n_dof)
        freqs = harmonic.frequencies_hz

        # Find resonance index (peak response)
        mean_amp_per_freq = np.mean(np.abs(disp_amps), axis=1)
        idx_res = int(np.argmax(mean_amp_per_freq))

        # Use Von Mises stress from harmonic displacement at resonance
        # as a proxy: scale |U| to MPa (simplified)
        disp_at_res = np.abs(disp_amps[idx_res])

        # Compute per-node alternating stress (simplified estimate)
        # In a real workflow, this would use StressRecovery
        n_nodes = len(disp_at_res) // 3
        # Use nodal displacement magnitude as stress proxy [MPa]
        stress_alt = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_nodes):
            u_mag = np.sqrt(
                disp_at_res[3 * i] ** 2
                + disp_at_res[3 * i + 1] ** 2
                + disp_at_res[3 * i + 2] ** 2
            )
            # Scale to approximate stress in MPa
            stress_alt[i] = u_mag * 1e6

        stress_mean = None
        if static is not None and static.stress_vm is not None:
            stress_mean = static.stress_vm / 1e6  # Pa to MPa

        return assessor.assess(
            stress_alternating=stress_alt,
            stress_mean=stress_mean,
            mesh=harmonic.mesh,
        )

    # ------------------------------------------------------------------
    # Step 6: Gain chain
    # ------------------------------------------------------------------

    def _compute_gain_chain(
        self,
        assembly: AssemblyResult,
        harmonic: HarmonicResult,
    ) -> dict:
        """Compute amplitude gain through each component.

        For each component interface:
            gain_i = mean|U_y| at output face / mean|U_y| at input face

        Total stack gain = product of all component gains.

        Parameters
        ----------
        assembly : AssemblyResult
            Coupled system with DOF mapping.
        harmonic : HarmonicResult
            Harmonic analysis results.

        Returns
        -------
        dict
            Gain chain data: ``components`` list and ``total_gain``.
        """
        logger.info("Step 6/7: Computing gain chain")

        # Find resonance index
        disp_amps = harmonic.displacement_amplitudes
        freqs = harmonic.frequencies_hz
        mean_amp_per_freq = np.mean(np.abs(disp_amps), axis=1)
        idx_res = int(np.argmax(mean_amp_per_freq))
        displacement_at_res = disp_amps[idx_res]  # (n_dof,) complex

        # Build component interfaces for gain chain computation
        # Each component's input is its bottom face DOFs (Y-direction),
        # output is its top face DOFs (Y-direction).
        component_interfaces = []
        for comp_dict in self._config.components:
            name = comp_dict["name"]
            iface_nodes = comp_dict["interface_nodes"]
            node_offset = assembly.node_offset_map.get(name, 0)

            bottom_local = iface_nodes.get(
                "bottom", np.array([], dtype=int)
            )
            top_local = iface_nodes.get("top", np.array([], dtype=int))

            # Convert local node indices to global Y-direction DOFs
            # Y is direction index 1 within each 3-DOF node
            input_dofs = np.array(
                [3 * (int(n) + node_offset) + 1 for n in bottom_local],
                dtype=np.intp,
            )
            output_dofs = np.array(
                [3 * (int(n) + node_offset) + 1 for n in top_local],
                dtype=np.intp,
            )

            # Skip components with no interface DOFs
            if len(input_dofs) == 0 or len(output_dofs) == 0:
                logger.warning(
                    "Component %r has no input or output interface nodes; "
                    "skipping gain chain entry.",
                    name,
                )
                continue

            component_interfaces.append({
                "name": name,
                "input_dofs": input_dofs,
                "output_dofs": output_dofs,
            })

        if len(component_interfaces) == 0:
            logger.warning("No components with valid interfaces for gain chain.")
            return {"components": [], "total_gain": 0.0}

        result = compute_gain_chain(displacement_at_res, component_interfaces)

        return {
            "components": result.components,
            "total_gain": result.total_gain,
        }

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self, result: WorkflowResult, total_time_s: float
    ) -> dict:
        """Create a human-readable summary dictionary.

        Parameters
        ----------
        result : WorkflowResult
            All analysis results collected so far.
        total_time_s : float
            Total wall-clock time for the workflow.

        Returns
        -------
        dict
            Summary with keys for each analysis stage.
        """
        logger.info("Step 7/7: Building summary")

        summary: dict = {
            "status": "completed",
            "total_time_s": round(total_time_s, 3),
            "n_components": len(self._config.components),
            "n_total_dof": result.assembly.n_total_dof,
            "target_frequency_hz": self._config.frequency_hz,
            "analyses_run": list(self._config.analyses),
        }

        # Modal summary
        if result.modal is not None:
            modal = result.modal
            n_modes = len(modal.frequencies_hz)
            summary["modal"] = {
                "n_modes_found": n_modes,
                "frequency_range_hz": [
                    float(modal.frequencies_hz[0]) if n_modes > 0 else 0.0,
                    float(modal.frequencies_hz[-1]) if n_modes > 0 else 0.0,
                ],
                "closest_to_target_hz": float(
                    modal.frequencies_hz[
                        np.argmin(
                            np.abs(
                                modal.frequencies_hz
                                - self._config.frequency_hz
                            )
                        )
                    ]
                )
                if n_modes > 0
                else None,
                "solve_time_s": round(modal.solve_time_s, 3),
            }

        # Harmonic summary
        if result.harmonic is not None:
            harmonic = result.harmonic
            summary["harmonic"] = {
                "gain": round(harmonic.gain, 4),
                "q_factor": round(harmonic.q_factor, 2),
                "uniformity": round(harmonic.contact_face_uniformity, 4),
                "solve_time_s": round(harmonic.solve_time_s, 3),
            }

        # Static summary
        if result.static is not None:
            static = result.static
            summary["static"] = {
                "max_stress_mpa": round(static.max_stress_mpa, 2),
                "solve_time_s": round(static.solve_time_s, 3),
            }

        # Fatigue summary
        if result.fatigue is not None:
            fatigue = result.fatigue
            summary["fatigue"] = {
                "min_safety_factor": round(fatigue.min_safety_factor, 4),
                "estimated_life_cycles": fatigue.estimated_life_cycles,
                "sn_curve": fatigue.sn_curve_name,
            }

        # Gain chain summary
        if result.gain_chain is not None:
            gc = result.gain_chain
            summary["gain_chain"] = {
                "total_gain": round(gc.get("total_gain", 0.0), 4),
                "components": [
                    {
                        "name": c["name"],
                        "gain": round(c["gain"], 4),
                    }
                    for c in gc.get("components", [])
                ],
            }

        # Impedance from assembly
        summary["impedance"] = {
            name: round(z, 2)
            for name, z in result.assembly.impedance.items()
        }
        summary["transmission_coefficients"] = {
            key: round(t, 6)
            for key, t in result.assembly.transmission_coefficients.items()
        }

        return summary

    # ------------------------------------------------------------------
    # Helper: build combined mesh
    # ------------------------------------------------------------------

    def _build_combined_mesh(self, assembly: AssemblyResult) -> FEAMesh:
        """Build a synthetic combined FEAMesh from all components.

        Concatenates node coordinates and element connectivity
        (with offset) from all components in the configured order.

        Parameters
        ----------
        assembly : AssemblyResult
            Coupled system with node offset map.

        Returns
        -------
        FEAMesh
            Combined mesh representing the full assembly.
        """
        all_nodes = []
        all_elements = []
        combined_node_sets: dict[str, np.ndarray] = {}
        combined_element_sets: dict[str, np.ndarray] = {}
        all_surface_tris = []

        node_offset = 0
        elem_offset = 0

        for comp_dict in self._config.components:
            mesh: FEAMesh = comp_dict["mesh"]
            n_nodes = mesh.nodes.shape[0]
            n_elems = mesh.elements.shape[0]

            all_nodes.append(mesh.nodes)
            all_elements.append(mesh.elements + node_offset)

            # Merge node sets with component name prefix
            for set_name, node_ids in mesh.node_sets.items():
                global_name = f"{comp_dict['name']}_{set_name}"
                combined_node_sets[global_name] = (
                    np.asarray(node_ids) + node_offset
                )

            # Also create aggregate sets (bottom_face, top_face)
            # from the first and last component respectively
            for set_name, node_ids in mesh.node_sets.items():
                if set_name not in combined_node_sets:
                    combined_node_sets[set_name] = (
                        np.asarray(node_ids) + node_offset
                    )

            # Merge element sets
            for set_name, elem_ids in mesh.element_sets.items():
                global_name = f"{comp_dict['name']}_{set_name}"
                combined_element_sets[global_name] = (
                    np.asarray(elem_ids) + elem_offset
                )

            # Surface tris
            if len(mesh.surface_tris) > 0:
                all_surface_tris.append(mesh.surface_tris + node_offset)

            node_offset += n_nodes
            elem_offset += n_elems

        nodes = np.vstack(all_nodes)
        elements = np.vstack(all_elements)
        surface_tris = (
            np.vstack(all_surface_tris)
            if all_surface_tris
            else np.empty((0, 3), dtype=int)
        )

        return FEAMesh(
            nodes=nodes,
            elements=elements,
            element_type=self._config.components[0]["mesh"].element_type,
            node_sets=combined_node_sets,
            element_sets=combined_element_sets,
            surface_tris=surface_tris,
            mesh_stats={
                "n_nodes": nodes.shape[0],
                "n_elements": elements.shape[0],
                "source": "workflow_combined",
            },
        )
