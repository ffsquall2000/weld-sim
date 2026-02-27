# Section 6: Material Database and Fatigue Assessment Module

## 6.0 Overview and Motivation

The current material database (`material_properties.py`) stores seven basic properties per
material (E, nu, rho, k, cp, yield, CTE) for 10 materials.  This is insufficient for
production-grade FEA of ultrasonic welding horns, which requires:

- **Acoustic properties** for wave-speed-based tuning calculations
- **Full strength data** (yield, ultimate, fatigue endurance) for life prediction
- **S-N fatigue curves** at 10^7 through 10^9+ cycles for high-cycle fatigue assessment
- **Temperature-dependent properties** for thermal-structural coupling
- **Damping characterization** for accurate harmonic response amplitudes
- **Piezoelectric tensors** for transducer stack modeling (PZT elements)
- **Stress concentration** and **surface finish** correction factor databases

This section specifies the enhanced material schema, the comprehensive material database
with actual numeric values, the fatigue assessment module, the damping model, and the
YAML-based data storage format.

---

## 6.1 Enhanced Material Property Schema

### 6.1.1 Pydantic Validation Models

All material data is validated at load time through a hierarchy of pydantic models.
The top-level `FEAMaterial` model is the single source of truth for what constitutes a
complete material definition.

```python
"""ultrasonic_weld_master/plugins/material_db/schemas.py"""
from __future__ import annotations

import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ── Enumerations ──────────────────────────────────────────────────────────

class MaterialCategory(str, Enum):
    TITANIUM_ALLOY = "titanium_alloy"
    ALUMINUM_ALLOY = "aluminum_alloy"
    TOOL_STEEL = "tool_steel"
    HIGH_SPEED_STEEL = "high_speed_steel"
    POWDER_METALLURGY_STEEL = "powder_metallurgy_steel"
    CEMENTED_CARBIDE = "cemented_carbide"
    PIEZOELECTRIC_CERAMIC = "piezoelectric_ceramic"
    STRUCTURAL_STEEL = "structural_steel"


class HornSuitability(str, Enum):
    PRIMARY = "primary"           # First-choice horn material
    SECONDARY = "secondary"       # Viable with trade-offs
    SPECIAL_PURPOSE = "special"   # Wear / extreme duty only
    NOT_RECOMMENDED = "no"        # Not suitable for horns


# ── Sub-models ────────────────────────────────────────────────────────────

class ElasticProperties(BaseModel):
    """Isotropic elastic constants."""
    E_GPa: float = Field(..., gt=0, description="Young's modulus [GPa]")
    nu: float = Field(..., gt=0, lt=0.5, description="Poisson's ratio [-]")
    rho_kg_m3: float = Field(..., gt=0, description="Density [kg/m^3]")

    @property
    def G_GPa(self) -> float:
        """Shear modulus from E and nu."""
        return self.E_GPa / (2.0 * (1.0 + self.nu))

    @property
    def K_GPa(self) -> float:
        """Bulk modulus."""
        return self.E_GPa / (3.0 * (1.0 - 2.0 * self.nu))


class AcousticProperties(BaseModel):
    """Properties governing wave propagation and energy dissipation."""
    c_longitudinal_m_s: float = Field(
        ..., gt=0, description="Longitudinal wave speed [m/s]"
    )
    c_shear_m_s: Optional[float] = Field(
        None, gt=0, description="Shear wave speed [m/s]"
    )
    Q_mechanical: float = Field(
        ..., gt=0, description="Mechanical quality factor at 20 kHz [-]"
    )
    loss_tangent: Optional[float] = Field(
        None, ge=0, description="Loss tangent eta = 1/Q [-]"
    )

    @model_validator(mode="after")
    def _sync_loss_tangent(self):
        if self.loss_tangent is None and self.Q_mechanical > 0:
            self.loss_tangent = 1.0 / self.Q_mechanical
        return self


class ThermalProperties(BaseModel):
    """Thermal constants for thermal-structural coupling."""
    k_W_mK: float = Field(..., gt=0, description="Thermal conductivity [W/(m*K)]")
    cp_J_kgK: float = Field(..., gt=0, description="Specific heat capacity [J/(kg*K)]")
    alpha_1_K: float = Field(
        ..., gt=0, description="Coefficient of thermal expansion [1/K]"
    )
    T_melt_C: float = Field(..., gt=0, description="Melting/solidus temperature [deg C]")
    T_max_service_C: Optional[float] = Field(
        None, gt=0, description="Maximum recommended service temperature [deg C]"
    )


class StrengthProperties(BaseModel):
    """Static and fatigue strength data."""
    sigma_y_MPa: float = Field(..., gt=0, description="0.2% offset yield strength [MPa]")
    sigma_u_MPa: float = Field(..., gt=0, description="Ultimate tensile strength [MPa]")
    elongation_pct: Optional[float] = Field(
        None, ge=0, description="Elongation at break [%]"
    )
    hardness_HRC: Optional[float] = Field(None, description="Rockwell C hardness")
    # Fatigue endurance limits (unnotched, R = -1, fully reversed bending)
    sigma_e_MPa_1e7: float = Field(
        ..., gt=0, description="Fatigue endurance limit at 10^7 cycles [MPa]"
    )
    sigma_e_MPa_1e9: Optional[float] = Field(
        None, gt=0, description="Fatigue endurance limit at 10^9 cycles [MPa]"
    )
    # Notched endurance limits at specific Kt values
    sigma_e_notched_MPa: Optional[dict[float, float]] = Field(
        None,
        description="Map of Kt -> endurance limit [MPa] at 10^7 cycles"
    )

    @model_validator(mode="after")
    def _yield_lt_ultimate(self):
        if self.sigma_u_MPa < self.sigma_y_MPa:
            raise ValueError("UTS must be >= yield strength")
        return self


class SNDataPoint(BaseModel):
    """A single point on an S-N curve."""
    N_cycles: float = Field(..., gt=0, description="Number of cycles to failure")
    S_MPa: float = Field(..., gt=0, description="Stress amplitude [MPa]")


class SNcurve(BaseModel):
    """Complete S-N (Woehler) curve definition."""
    R_ratio: float = Field(
        -1.0, description="Stress ratio R = sigma_min / sigma_max"
    )
    test_frequency_Hz: Optional[float] = Field(
        None, description="Test frequency [Hz] (if ultrasonic, ~20 kHz)"
    )
    data_points: list[SNDataPoint] = Field(
        ..., min_length=3, description="S-N data points (minimum 3)"
    )
    source: Optional[str] = Field(
        None, description="Literature / test report reference"
    )


class TemperatureDependence(BaseModel):
    """Linear coefficients for property variation with temperature.

    Property(T) = Property(T_ref) * (1 + coefficient * (T - T_ref))
    """
    T_ref_C: float = Field(20.0, description="Reference temperature [deg C]")
    dE_dT_frac_per_C: float = Field(
        ..., description="Fractional change in E per degree C [1/C]"
    )
    dsigma_y_dT_frac_per_C: float = Field(
        ..., description="Fractional change in yield per degree C [1/C]"
    )
    dsigma_e_dT_frac_per_C: Optional[float] = Field(
        None, description="Fractional change in endurance limit per deg C [1/C]"
    )


class DampingModel(BaseModel):
    """Frequency- and amplitude-dependent damping characterization."""
    Q_at_20kHz: float = Field(..., gt=0, description="Q at 20 kHz, low amplitude")
    Q_at_40kHz: Optional[float] = Field(None, gt=0, description="Q at 40 kHz")
    Q_at_15kHz: Optional[float] = Field(None, gt=0, description="Q at 15 kHz")
    # Amplitude dependence:  Q(eps) = Q_0 for eps < eps_threshold,
    #                        Q(eps) = Q_0 * (eps_threshold / eps)^n  for eps >= eps_threshold
    strain_threshold: Optional[float] = Field(
        None, gt=0,
        description="Strain amplitude threshold above which Q degrades [-]"
    )
    amplitude_exponent_n: Optional[float] = Field(
        None, gt=0,
        description="Exponent n for Q degradation with strain amplitude"
    )
    # Temperature dependence:  Q(T) = Q_0 * (1 + dQ_dT_frac * (T - T_ref))
    dQ_dT_frac_per_C: Optional[float] = Field(
        None,
        description="Fractional change in Q per degree C"
    )


class PiezoelectricProperties(BaseModel):
    """IEEE-standard piezoelectric material constants (Voigt notation).

    Tensors are stored as flat lists in Voigt order for YAML readability.
    cE is the 6x6 elastic stiffness at constant electric field.
    e is the 3x6 piezoelectric stress tensor.
    eps_S is the 3x3 permittivity at constant strain.
    """
    cE_GPa: list[float] = Field(
        ..., min_length=6, max_length=36,
        description="Independent elastic stiffness constants cE_ij [GPa] "
                    "(c11, c12, c13, c33, c44, c66) for 6mm symmetry"
    )
    e_C_m2: list[float] = Field(
        ..., min_length=3, max_length=18,
        description="Piezoelectric stress constants (e15, e31, e33) [C/m^2]"
    )
    eps_S_relative: list[float] = Field(
        ..., min_length=2, max_length=9,
        description="Relative permittivity at constant strain (eps11/eps0, eps33/eps0)"
    )
    d33_pC_N: Optional[float] = Field(None, description="d33 [pC/N]")
    d31_pC_N: Optional[float] = Field(None, description="d31 [pC/N]")
    k33: Optional[float] = Field(None, description="Coupling factor k33 [-]")
    k31: Optional[float] = Field(None, description="Coupling factor k31 [-]")
    Qm: float = Field(..., gt=0, description="Mechanical quality factor")
    T_curie_C: float = Field(..., gt=0, description="Curie temperature [deg C]")


# ── Top-level Material Model ─────────────────────────────────────────────

class FEAMaterial(BaseModel):
    """Complete material definition for FEA-grade ultrasonic welding analysis."""
    name: str = Field(..., description="Canonical material name")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative lookup names"
    )
    category: MaterialCategory
    horn_suitability: HornSuitability = HornSuitability.NOT_RECOMMENDED
    description: Optional[str] = None

    # Property groups
    elastic: ElasticProperties
    acoustic: AcousticProperties
    thermal: ThermalProperties
    strength: StrengthProperties

    # Optional extended data
    sn_curves: list[SNcurve] = Field(default_factory=list)
    temperature_dependence: Optional[TemperatureDependence] = None
    damping: Optional[DampingModel] = None
    piezoelectric: Optional[PiezoelectricProperties] = None

    # Metadata
    data_sources: list[str] = Field(
        default_factory=list, description="References for property values"
    )
    notes: Optional[str] = None

    @property
    def acoustic_impedance_MRayl(self) -> float:
        """Z = rho * c  [MRayl = 10^6 kg/(m^2*s)]"""
        return (self.elastic.rho_kg_m3 * self.acoustic.c_longitudinal_m_s) / 1e6
```

### 6.1.2 Property Key Mapping (Backward Compatibility)

To maintain backward compatibility with the existing `FEA_MATERIALS` dict format,
a flattening function is provided:

```python
def flatten_to_legacy(mat: FEAMaterial) -> dict:
    """Convert FEAMaterial to the legacy flat-dict format used by existing FEA code."""
    return {
        "E_pa": mat.elastic.E_GPa * 1e9,
        "nu": mat.elastic.nu,
        "rho_kg_m3": mat.elastic.rho_kg_m3,
        "k_w_mk": mat.thermal.k_W_mK,
        "cp_j_kgk": mat.thermal.cp_J_kgK,
        "yield_mpa": mat.strength.sigma_y_MPa,
        "alpha_1_k": mat.thermal.alpha_1_K,
    }
```

---

## 6.2 Material Database -- Comprehensive Numeric Values

### 6.2.1 Ti-6Al-4V (Grade 5 Titanium)

The primary horn material for ultrasonic welding due to its excellent fatigue-to-density
ratio, high Q factor, and corrosion resistance.

```yaml
# materials/ti6al4v_annealed.yaml
name: "Ti-6Al-4V (Annealed)"
aliases: ["titanium", "ti64", "ti-6al-4v", "grade5"]
category: titanium_alloy
horn_suitability: primary
description: "Primary ultrasonic horn material. Best fatigue/density ratio. Annealed condition."

elastic:
  E_GPa: 113.8
  nu: 0.342
  rho_kg_m3: 4430.0

acoustic:
  c_longitudinal_m_s: 6070        # sqrt(E/rho) theoretical=5070, measured bar velocity
  c_shear_m_s: 3120
  Q_mechanical: 5000              # At 20 kHz, low amplitude. Ti64 has very high Q.
  loss_tangent: 0.0002            # eta = 1/Q

thermal:
  k_W_mK: 6.7
  cp_J_kgK: 526.3
  alpha_1_K: 8.6e-6
  T_melt_C: 1660                  # Solidus
  T_max_service_C: 400

strength:
  sigma_y_MPa: 880
  sigma_u_MPa: 950
  elongation_pct: 14.0
  hardness_HRC: 36
  sigma_e_MPa_1e7: 510            # Unnotched, R=-1, rotating bending
  sigma_e_MPa_1e9: 350            # Gigacycle fatigue limit (ultrasonic test data)
  sigma_e_notched_MPa:
    1.5: 380                      # Kt=1.5
    2.0: 310                      # Kt=2.0
    3.0: 230                      # Kt=3.0
    4.0: 190                      # Kt=4.0

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 20000      # Ultrasonic fatigue test
    source: "Bathias & Paris, Gigacycle Fatigue in Mechanical Practice, 2005"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 800}
      - {N_cycles: 1.0e+5, S_MPa: 680}
      - {N_cycles: 1.0e+6, S_MPa: 560}
      - {N_cycles: 1.0e+7, S_MPa: 510}
      - {N_cycles: 1.0e+8, S_MPa: 430}
      - {N_cycles: 1.0e+9, S_MPa: 350}
      - {N_cycles: 1.0e+10, S_MPa: 320}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -3.2e-4      # E drops ~3.2% per 100 C
  dsigma_y_dT_frac_per_C: -5.5e-4 # Yield drops ~5.5% per 100 C
  dsigma_e_dT_frac_per_C: -4.0e-4

damping:
  Q_at_20kHz: 5000
  Q_at_15kHz: 5500
  Q_at_40kHz: 4200
  strain_threshold: 0.003         # Critical: Q drops sharply above 0.3% strain
  amplitude_exponent_n: 0.8       # Q ~ Q_0*(eps_th/eps)^0.8 above threshold
  dQ_dT_frac_per_C: -1.5e-3      # Q degrades ~0.15% per degree C

data_sources:
  - "ASM Handbook Vol. 2: Properties and Selection"
  - "Bathias & Paris, Gigacycle Fatigue in Mechanical Practice, 2005"
  - "Branson Ultrasonics, Horn Design Guide"
notes: "Annealed condition. STA variant has sigma_y=1100 MPa, sigma_e_1e7=580 MPa."
```

```yaml
# materials/ti6al4v_sta.yaml
name: "Ti-6Al-4V (STA)"
aliases: ["ti64-sta", "ti-6al-4v-sta"]
category: titanium_alloy
horn_suitability: primary
description: "Solution treated and aged. Higher strength, slightly lower ductility."

elastic:
  E_GPa: 114.0
  nu: 0.342
  rho_kg_m3: 4430.0

acoustic:
  c_longitudinal_m_s: 6070
  c_shear_m_s: 3120
  Q_mechanical: 4500
  loss_tangent: 0.000222

thermal:
  k_W_mK: 6.7
  cp_J_kgK: 526.3
  alpha_1_K: 8.6e-6
  T_melt_C: 1660
  T_max_service_C: 350

strength:
  sigma_y_MPa: 1100
  sigma_u_MPa: 1170
  elongation_pct: 10.0
  hardness_HRC: 39
  sigma_e_MPa_1e7: 580
  sigma_e_MPa_1e9: 400
  sigma_e_notched_MPa:
    1.5: 430
    2.0: 350
    3.0: 260
    4.0: 210

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 20000
    source: "Bathias & Paris, 2005; Marines et al., Int J Fatigue 2003"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 950}
      - {N_cycles: 1.0e+5, S_MPa: 790}
      - {N_cycles: 1.0e+6, S_MPa: 650}
      - {N_cycles: 1.0e+7, S_MPa: 580}
      - {N_cycles: 1.0e+8, S_MPa: 480}
      - {N_cycles: 1.0e+9, S_MPa: 400}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -3.2e-4
  dsigma_y_dT_frac_per_C: -6.0e-4
  dsigma_e_dT_frac_per_C: -4.5e-4

damping:
  Q_at_20kHz: 4500
  Q_at_15kHz: 5000
  Q_at_40kHz: 3800
  strain_threshold: 0.0025
  amplitude_exponent_n: 0.9
  dQ_dT_frac_per_C: -1.5e-3

data_sources:
  - "ASM Handbook Vol. 2"
  - "Marines, I. et al., Int J Fatigue 25 (2003) 1037-1046"
```

### 6.2.2 Aluminum 7075-T6

Lightweight horn material for low-force applications. Excellent machinability. Limited
fatigue life compared to titanium.

```yaml
name: "Aluminum 7075-T6"
aliases: ["al7075", "7075-t6", "7075", "aluminum"]
category: aluminum_alloy
horn_suitability: secondary
description: "Lightweight horn material. Good for low-amplitude, short-run applications."

elastic:
  E_GPa: 71.7
  nu: 0.33
  rho_kg_m3: 2810.0

acoustic:
  c_longitudinal_m_s: 6320
  c_shear_m_s: 3130
  Q_mechanical: 9000             # Aluminum has very high Q (low damping)
  loss_tangent: 0.000111

thermal:
  k_W_mK: 130.0
  cp_J_kgK: 960.0
  alpha_1_K: 23.6e-6
  T_melt_C: 635                  # Solidus
  T_max_service_C: 120           # Aging overaging threshold

strength:
  sigma_y_MPa: 503
  sigma_u_MPa: 572
  elongation_pct: 11.0
  hardness_HRC: null              # Typically measured as HB 150
  sigma_e_MPa_1e7: 159           # R=-1, rotating bending
  sigma_e_MPa_1e9: 105           # Continued decline in gigacycle regime
  sigma_e_notched_MPa:
    1.5: 120
    2.0: 100
    3.0: 75

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 20000
    source: "Stanzl-Tschegg et al., Int J Fatigue 2007"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 420}
      - {N_cycles: 1.0e+5, S_MPa: 310}
      - {N_cycles: 1.0e+6, S_MPa: 220}
      - {N_cycles: 1.0e+7, S_MPa: 159}
      - {N_cycles: 1.0e+8, S_MPa: 130}
      - {N_cycles: 1.0e+9, S_MPa: 105}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -4.8e-4
  dsigma_y_dT_frac_per_C: -7.0e-4
  dsigma_e_dT_frac_per_C: -6.0e-4

damping:
  Q_at_20kHz: 9000
  Q_at_15kHz: 10000
  Q_at_40kHz: 7500
  strain_threshold: 0.002
  amplitude_exponent_n: 0.5
  dQ_dT_frac_per_C: -1.0e-3

data_sources:
  - "ASM Handbook Vol. 2"
  - "Stanzl-Tschegg, S.E., Int J Fatigue 29 (2007) 2050-2059"
```

### 6.2.3 Steel D2 (Tool Steel)

Wear-resistant horn material for abrasive workpiece applications. Hardened to 58-62 HRC.

```yaml
name: "Steel D2"
aliases: ["d2", "d2-steel", "aisi-d2"]
category: tool_steel
horn_suitability: special
description: "High-chrome tool steel. Wear-resistant horns for abrasive applications."

elastic:
  E_GPa: 210.0
  nu: 0.30
  rho_kg_m3: 7700.0

acoustic:
  c_longitudinal_m_s: 5870
  c_shear_m_s: 3200
  Q_mechanical: 3000
  loss_tangent: 0.000333

thermal:
  k_W_mK: 20.0
  cp_J_kgK: 460.0
  alpha_1_K: 10.4e-6
  T_melt_C: 1421
  T_max_service_C: 425

strength:
  sigma_y_MPa: 1620
  sigma_u_MPa: 1930
  elongation_pct: 1.0
  hardness_HRC: 60
  sigma_e_MPa_1e7: 520
  sigma_e_MPa_1e9: 380
  sigma_e_notched_MPa:
    1.5: 380
    2.0: 300
    3.0: 220

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100        # Conventional fatigue test
    source: "Carpenter Technology datasheet; ASM tool steel handbook"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1200}
      - {N_cycles: 1.0e+5, S_MPa: 900}
      - {N_cycles: 1.0e+6, S_MPa: 680}
      - {N_cycles: 1.0e+7, S_MPa: 520}
      - {N_cycles: 1.0e+8, S_MPa: 440}
      - {N_cycles: 1.0e+9, S_MPa: 380}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -3.0e-4
  dsigma_y_dT_frac_per_C: -4.0e-4
  dsigma_e_dT_frac_per_C: -3.5e-4

damping:
  Q_at_20kHz: 3000
  Q_at_15kHz: 3200
  Q_at_40kHz: 2600
  strain_threshold: 0.001
  amplitude_exponent_n: 0.6
  dQ_dT_frac_per_C: -2.0e-3
```

### 6.2.4 M2 High Speed Steel (AISI M2)

```yaml
name: "M2 High Speed Steel"
aliases: ["m2", "m2-hss", "aisi-m2"]
category: high_speed_steel
horn_suitability: secondary
description: "General-purpose HSS for ultrasonic tooling. Good balance of toughness and wear."

elastic:
  E_GPa: 220.0
  nu: 0.28
  rho_kg_m3: 8160.0

acoustic:
  c_longitudinal_m_s: 5190
  c_shear_m_s: 3100
  Q_mechanical: 2800
  loss_tangent: 0.000357

thermal:
  k_W_mK: 25.9
  cp_J_kgK: 420.0
  alpha_1_K: 11.0e-6
  T_melt_C: 1430
  T_max_service_C: 550

strength:
  sigma_y_MPa: 2200
  sigma_u_MPa: 3000
  elongation_pct: 1.5
  hardness_HRC: 64
  sigma_e_MPa_1e7: 690
  sigma_e_MPa_1e9: 490
  sigma_e_notched_MPa:
    1.5: 510
    2.0: 400
    3.0: 290

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 20000
    source: "Furuya et al., Scripta Materialia 2005"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1800}
      - {N_cycles: 1.0e+5, S_MPa: 1400}
      - {N_cycles: 1.0e+6, S_MPa: 1000}
      - {N_cycles: 1.0e+7, S_MPa: 690}
      - {N_cycles: 1.0e+8, S_MPa: 560}
      - {N_cycles: 1.0e+9, S_MPa: 490}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.5e-4
  dsigma_y_dT_frac_per_C: -3.5e-4
  dsigma_e_dT_frac_per_C: -3.0e-4

damping:
  Q_at_20kHz: 2800
  Q_at_15kHz: 3000
  Q_at_40kHz: 2400
  strain_threshold: 0.001
  amplitude_exponent_n: 0.5
  dQ_dT_frac_per_C: -1.8e-3
```

### 6.2.5 CPM 10V (Crucible Particle Metallurgy)

```yaml
name: "CPM 10V"
aliases: ["cpm10v", "cpm-10v", "10v"]
category: powder_metallurgy_steel
horn_suitability: special
description: "Extreme wear resistance (9.75% V). For highly abrasive welding applications."

elastic:
  E_GPa: 222.0
  nu: 0.29
  rho_kg_m3: 7690.0

acoustic:
  c_longitudinal_m_s: 5375
  c_shear_m_s: 3140
  Q_mechanical: 2500
  loss_tangent: 0.0004

thermal:
  k_W_mK: 20.4
  cp_J_kgK: 430.0
  alpha_1_K: 10.8e-6
  T_melt_C: 1400
  T_max_service_C: 540

strength:
  sigma_y_MPa: 2100
  sigma_u_MPa: 2900
  elongation_pct: 1.0
  hardness_HRC: 62
  sigma_e_MPa_1e7: 640
  sigma_e_MPa_1e9: 450
  sigma_e_notched_MPa:
    1.5: 470
    2.0: 370
    3.0: 270

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "Crucible Industries CPM 10V datasheet"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1700}
      - {N_cycles: 1.0e+5, S_MPa: 1300}
      - {N_cycles: 1.0e+6, S_MPa: 940}
      - {N_cycles: 1.0e+7, S_MPa: 640}
      - {N_cycles: 1.0e+8, S_MPa: 520}
      - {N_cycles: 1.0e+9, S_MPa: 450}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.6e-4
  dsigma_y_dT_frac_per_C: -3.8e-4
  dsigma_e_dT_frac_per_C: -3.2e-4

damping:
  Q_at_20kHz: 2500
  Q_at_15kHz: 2700
  Q_at_40kHz: 2100
  strain_threshold: 0.001
  amplitude_exponent_n: 0.55
  dQ_dT_frac_per_C: -2.0e-3
```

### 6.2.6 PM60 Powder Metallurgy Steel

```yaml
name: "PM60 Powder Steel"
aliases: ["pm60"]
category: powder_metallurgy_steel
horn_suitability: secondary
description: "High-Cr-Mo PM steel. Good combination of strength and toughness."

elastic:
  E_GPa: 230.0
  nu: 0.28
  rho_kg_m3: 8100.0

acoustic:
  c_longitudinal_m_s: 5328
  c_shear_m_s: 3180
  Q_mechanical: 2600
  loss_tangent: 0.000385

thermal:
  k_W_mK: 24.0
  cp_J_kgK: 430.0
  alpha_1_K: 10.5e-6
  T_melt_C: 1420
  T_max_service_C: 550

strength:
  sigma_y_MPa: 2400
  sigma_u_MPa: 3200
  elongation_pct: 1.2
  hardness_HRC: 65
  sigma_e_MPa_1e7: 720
  sigma_e_MPa_1e9: 510
  sigma_e_notched_MPa:
    1.5: 530
    2.0: 420
    3.0: 310

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "Hitachi Metals PM60 technical datasheet"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1900}
      - {N_cycles: 1.0e+5, S_MPa: 1480}
      - {N_cycles: 1.0e+6, S_MPa: 1050}
      - {N_cycles: 1.0e+7, S_MPa: 720}
      - {N_cycles: 1.0e+8, S_MPa: 590}
      - {N_cycles: 1.0e+9, S_MPa: 510}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.5e-4
  dsigma_y_dT_frac_per_C: -3.5e-4
  dsigma_e_dT_frac_per_C: -3.0e-4

damping:
  Q_at_20kHz: 2600
  Q_at_15kHz: 2800
  Q_at_40kHz: 2200
  strain_threshold: 0.001
  amplitude_exponent_n: 0.55
  dQ_dT_frac_per_C: -1.8e-3
```

### 6.2.7 HAP40 and HAP72 Powder High Speed Steel

```yaml
# HAP40
name: "HAP40 Powder HSS"
aliases: ["hap40"]
category: powder_metallurgy_steel
horn_suitability: secondary
description: "Hitachi HAP40. Balanced W-Co-V composition for heavy-duty ultrasonic tooling."

elastic:
  E_GPa: 228.0
  nu: 0.28
  rho_kg_m3: 8050.0

acoustic:
  c_longitudinal_m_s: 5319
  c_shear_m_s: 3160
  Q_mechanical: 2700
  loss_tangent: 0.00037

thermal:
  k_W_mK: 23.0
  cp_J_kgK: 425.0
  alpha_1_K: 10.7e-6
  T_melt_C: 1410
  T_max_service_C: 560

strength:
  sigma_y_MPa: 2500
  sigma_u_MPa: 3300
  elongation_pct: 1.0
  hardness_HRC: 66
  sigma_e_MPa_1e7: 740
  sigma_e_MPa_1e9: 520
  sigma_e_notched_MPa:
    1.5: 550
    2.0: 430
    3.0: 320

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "Hitachi Metals HAP40 technical datasheet"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 2000}
      - {N_cycles: 1.0e+5, S_MPa: 1550}
      - {N_cycles: 1.0e+6, S_MPa: 1100}
      - {N_cycles: 1.0e+7, S_MPa: 740}
      - {N_cycles: 1.0e+8, S_MPa: 610}
      - {N_cycles: 1.0e+9, S_MPa: 520}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.4e-4
  dsigma_y_dT_frac_per_C: -3.4e-4
  dsigma_e_dT_frac_per_C: -2.8e-4

damping:
  Q_at_20kHz: 2700
  Q_at_15kHz: 2900
  Q_at_40kHz: 2300
  strain_threshold: 0.001
  amplitude_exponent_n: 0.5
  dQ_dT_frac_per_C: -1.8e-3
```

```yaml
# HAP72
name: "HAP72 Powder HSS"
aliases: ["hap72"]
category: powder_metallurgy_steel
horn_suitability: special
description: "Hitachi HAP72. Ultra-hard PM HSS for extreme-duty tooling."

elastic:
  E_GPa: 235.0
  nu: 0.27
  rho_kg_m3: 8200.0

acoustic:
  c_longitudinal_m_s: 5352
  c_shear_m_s: 3200
  Q_mechanical: 2400
  loss_tangent: 0.000417

thermal:
  k_W_mK: 21.0
  cp_J_kgK: 420.0
  alpha_1_K: 10.3e-6
  T_melt_C: 1400
  T_max_service_C: 570

strength:
  sigma_y_MPa: 2800
  sigma_u_MPa: 3600
  elongation_pct: 0.8
  hardness_HRC: 68
  sigma_e_MPa_1e7: 800
  sigma_e_MPa_1e9: 560
  sigma_e_notched_MPa:
    1.5: 590
    2.0: 460
    3.0: 340

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "Hitachi Metals HAP72 technical datasheet"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 2200}
      - {N_cycles: 1.0e+5, S_MPa: 1700}
      - {N_cycles: 1.0e+6, S_MPa: 1200}
      - {N_cycles: 1.0e+7, S_MPa: 800}
      - {N_cycles: 1.0e+8, S_MPa: 650}
      - {N_cycles: 1.0e+9, S_MPa: 560}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.3e-4
  dsigma_y_dT_frac_per_C: -3.3e-4
  dsigma_e_dT_frac_per_C: -2.7e-4

damping:
  Q_at_20kHz: 2400
  Q_at_15kHz: 2600
  Q_at_40kHz: 2000
  strain_threshold: 0.0008
  amplitude_exponent_n: 0.6
  dQ_dT_frac_per_C: -2.0e-3
```

### 6.2.8 AISI 4140 Steel (Quenched & Tempered)

Used for transducer back mass and front mass components (not as horn tip material).

```yaml
name: "Steel 4140 (Q&T)"
aliases: ["4140", "aisi-4140", "4140-steel"]
category: structural_steel
horn_suitability: not_recommended
description: "Cr-Mo structural steel for transducer back mass / front mass. Not for horn tips."

elastic:
  E_GPa: 205.0
  nu: 0.29
  rho_kg_m3: 7850.0

acoustic:
  c_longitudinal_m_s: 5900
  c_shear_m_s: 3240
  Q_mechanical: 3500
  loss_tangent: 0.000286

thermal:
  k_W_mK: 42.6
  cp_J_kgK: 473.0
  alpha_1_K: 12.3e-6
  T_melt_C: 1416
  T_max_service_C: 400

strength:
  sigma_y_MPa: 655
  sigma_u_MPa: 900
  elongation_pct: 17.7
  hardness_HRC: 28
  sigma_e_MPa_1e7: 420
  sigma_e_MPa_1e9: 310
  sigma_e_notched_MPa:
    1.5: 310
    2.0: 250
    3.0: 185

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 50
    source: "MIL-HDBK-5J; ASM Fatigue Data Book"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 720}
      - {N_cycles: 1.0e+5, S_MPa: 600}
      - {N_cycles: 1.0e+6, S_MPa: 500}
      - {N_cycles: 1.0e+7, S_MPa: 420}
      - {N_cycles: 1.0e+8, S_MPa: 360}
      - {N_cycles: 1.0e+9, S_MPa: 310}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -3.5e-4
  dsigma_y_dT_frac_per_C: -5.0e-4
  dsigma_e_dT_frac_per_C: -4.0e-4

damping:
  Q_at_20kHz: 3500
  Q_at_15kHz: 3800
  Q_at_40kHz: 3000
  strain_threshold: 0.002
  amplitude_exponent_n: 0.4
  dQ_dT_frac_per_C: -1.5e-3
```

### 6.2.9 PZT-4 and PZT-8 (Piezoelectric Ceramics)

These are the transducer elements. Properties defined using IEEE standard notation.

```yaml
# PZT-4 (Navy Type I)
name: "PZT-4"
aliases: ["pzt4", "navy-type-i", "pzt-4"]
category: piezoelectric_ceramic
horn_suitability: not_recommended
description: "Hard PZT. Standard power transducer material. High Q, moderate coupling."

elastic:
  E_GPa: 81.3                    # Y33^E (along poling direction)
  nu: 0.31
  rho_kg_m3: 7500.0

acoustic:
  c_longitudinal_m_s: 4600       # Along thickness (poling) direction
  c_shear_m_s: 2600
  Q_mechanical: 500               # Mechanical Q of PZT-4
  loss_tangent: 0.004              # Relatively low for PZT

thermal:
  k_W_mK: 1.5
  cp_J_kgK: 420.0
  alpha_1_K: 4.0e-6
  T_melt_C: 1350                  # Sintering temperature
  T_max_service_C: 200            # Curie temperature limit

strength:
  sigma_y_MPa: 80                 # Compressive strength (ceramics)
  sigma_u_MPa: 80                 # Compressive (tensile strength ~55 MPa)
  sigma_e_MPa_1e7: 25             # Conservative fatigue limit for ceramics
  sigma_e_MPa_1e9: 20

sn_curves:
  - R_ratio: 0.0                  # PZT always under compression bias
    test_frequency_Hz: 20000
    source: "Morgan Technical Ceramics PZT-4 datasheet"
    data_points:
      - {N_cycles: 1.0e+6, S_MPa: 40}
      - {N_cycles: 1.0e+7, S_MPa: 25}
      - {N_cycles: 1.0e+9, S_MPa: 20}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -5.0e-4
  dsigma_y_dT_frac_per_C: -3.0e-4
  dsigma_e_dT_frac_per_C: -4.0e-4

damping:
  Q_at_20kHz: 500
  Q_at_15kHz: 520
  Q_at_40kHz: 450
  dQ_dT_frac_per_C: -5.0e-3

piezoelectric:
  # Elastic stiffness cE [GPa] for 6mm symmetry: c11, c12, c13, c33, c44, c66
  cE_GPa: [139.0, 77.8, 74.3, 115.0, 25.6, 30.6]
  # Piezoelectric stress constants [C/m^2]: e15, e31, e33
  e_C_m2: [12.7, -5.2, 15.1]
  # Relative permittivity at constant strain: eps11/eps0, eps33/eps0
  eps_S_relative: [730, 635]
  d33_pC_N: 289
  d31_pC_N: -123
  k33: 0.70
  k31: 0.334
  Qm: 500
  T_curie_C: 328

data_sources:
  - "Morgan Electro Ceramics, PZT-4 datasheet"
  - "IEEE Standard on Piezoelectricity, ANSI/IEEE Std 176-1987"
  - "Berlincourt, D., J. Acoust. Soc. Am. 1971"
```

```yaml
# PZT-8 (Navy Type III)
name: "PZT-8"
aliases: ["pzt8", "navy-type-iii", "pzt-8"]
category: piezoelectric_ceramic
horn_suitability: not_recommended
description: "Hard PZT. Higher Q than PZT-4. Preferred for high-power ultrasonic transducers."

elastic:
  E_GPa: 86.9
  nu: 0.31
  rho_kg_m3: 7600.0

acoustic:
  c_longitudinal_m_s: 4560
  c_shear_m_s: 2620
  Q_mechanical: 1000              # PZT-8 has significantly higher Q than PZT-4
  loss_tangent: 0.001

thermal:
  k_W_mK: 1.5
  cp_J_kgK: 420.0
  alpha_1_K: 4.0e-6
  T_melt_C: 1350
  T_max_service_C: 225

strength:
  sigma_y_MPa: 85
  sigma_u_MPa: 85
  sigma_e_MPa_1e7: 28
  sigma_e_MPa_1e9: 22

sn_curves:
  - R_ratio: 0.0
    test_frequency_Hz: 20000
    source: "Morgan Technical Ceramics PZT-8 datasheet"
    data_points:
      - {N_cycles: 1.0e+6, S_MPa: 45}
      - {N_cycles: 1.0e+7, S_MPa: 28}
      - {N_cycles: 1.0e+9, S_MPa: 22}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -4.0e-4
  dsigma_y_dT_frac_per_C: -3.0e-4
  dsigma_e_dT_frac_per_C: -3.5e-4

damping:
  Q_at_20kHz: 1000
  Q_at_15kHz: 1050
  Q_at_40kHz: 900
  dQ_dT_frac_per_C: -4.0e-3

piezoelectric:
  cE_GPa: [146.9, 81.1, 81.0, 131.7, 31.4, 32.9]
  e_C_m2: [10.4, -4.0, 13.8]
  eps_S_relative: [900, 600]
  d33_pC_N: 225
  d31_pC_N: -97
  k33: 0.64
  k31: 0.30
  Qm: 1000
  T_curie_C: 300

data_sources:
  - "Morgan Electro Ceramics, PZT-8 datasheet"
  - "IEEE Standard on Piezoelectricity, ANSI/IEEE Std 176-1987"
```

### 6.2.10 Ferro-Titanit WFN

```yaml
name: "Ferro-Titanit WFN"
aliases: ["ferro-titanit", "wfn", "ferro-titanit-wfn"]
category: cemented_carbide
horn_suitability: special
description: >
  Titanium-carbide-based cermet with steel binder. Extremely wear-resistant horn material.
  Manufactured by Deutsche Edelstahlwerke (DEW). Used for high-volume production where
  horn wear is the limiting factor. Difficult to machine; requires EDM and grinding.

elastic:
  E_GPa: 290.0                   # Very stiff due to TiC content (~50 vol%)
  nu: 0.25
  rho_kg_m3: 6550.0              # Lower density than tungsten carbide cermets

acoustic:
  c_longitudinal_m_s: 6650       # High wave speed from high E and moderate density
  c_shear_m_s: 4080
  Q_mechanical: 2000             # Lower Q than pure metals due to cermet microstructure
  loss_tangent: 0.0005

thermal:
  k_W_mK: 15.0                   # Low conductivity due to TiC phase
  cp_J_kgK: 400.0
  alpha_1_K: 8.8e-6
  T_melt_C: 1350
  T_max_service_C: 500

strength:
  sigma_y_MPa: 1800              # Compressive yield; tensile behavior is brittle
  sigma_u_MPa: 2200              # Transverse rupture strength (TRS)
  elongation_pct: 0.5
  hardness_HRC: 63
  sigma_e_MPa_1e7: 600
  sigma_e_MPa_1e9: 420
  sigma_e_notched_MPa:
    1.5: 440
    2.0: 350
    3.0: 260

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "DEW Ferro-Titanit WFN datasheet; internal Branson validation data"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1500}
      - {N_cycles: 1.0e+5, S_MPa: 1150}
      - {N_cycles: 1.0e+6, S_MPa: 850}
      - {N_cycles: 1.0e+7, S_MPa: 600}
      - {N_cycles: 1.0e+8, S_MPa: 490}
      - {N_cycles: 1.0e+9, S_MPa: 420}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -1.8e-4      # Cermets less temperature-sensitive
  dsigma_y_dT_frac_per_C: -2.5e-4
  dsigma_e_dT_frac_per_C: -2.2e-4

damping:
  Q_at_20kHz: 2000
  Q_at_15kHz: 2200
  Q_at_40kHz: 1700
  strain_threshold: 0.0005        # Very low strain threshold -- brittle
  amplitude_exponent_n: 0.7
  dQ_dT_frac_per_C: -2.5e-3

data_sources:
  - "Deutsche Edelstahlwerke (DEW), Ferro-Titanit WFN datasheet"
  - "Dukane Ultrasonics, Horn Material Selection Guide"
notes: "Requires EDM and precision grinding. Cannot be conventional-machined."
```

### 6.2.11 CPM Rex M4 (High-Performance PM HSS)

```yaml
name: "CPM Rex M4"
aliases: ["cpm-m4", "rex-m4", "cpm-rex-m4", "m4"]
category: powder_metallurgy_steel
horn_suitability: secondary
description: >
  Crucible CPM Rex M4. Super-high-speed PM steel with high V content.
  Excellent wear resistance with better toughness than CPM 10V.
  Popular choice for high-volume ultrasonic welding horns.

elastic:
  E_GPa: 225.0
  nu: 0.28
  rho_kg_m3: 7970.0

acoustic:
  c_longitudinal_m_s: 5313
  c_shear_m_s: 3150
  Q_mechanical: 2600
  loss_tangent: 0.000385

thermal:
  k_W_mK: 22.0
  cp_J_kgK: 420.0
  alpha_1_K: 10.9e-6
  T_melt_C: 1410
  T_max_service_C: 550

strength:
  sigma_y_MPa: 2350
  sigma_u_MPa: 3100
  elongation_pct: 1.2
  hardness_HRC: 64
  sigma_e_MPa_1e7: 710
  sigma_e_MPa_1e9: 500
  sigma_e_notched_MPa:
    1.5: 520
    2.0: 410
    3.0: 300

sn_curves:
  - R_ratio: -1.0
    test_frequency_Hz: 100
    source: "Crucible Industries CPM Rex M4 datasheet"
    data_points:
      - {N_cycles: 1.0e+4, S_MPa: 1850}
      - {N_cycles: 1.0e+5, S_MPa: 1430}
      - {N_cycles: 1.0e+6, S_MPa: 1030}
      - {N_cycles: 1.0e+7, S_MPa: 710}
      - {N_cycles: 1.0e+8, S_MPa: 580}
      - {N_cycles: 1.0e+9, S_MPa: 500}

temperature_dependence:
  T_ref_C: 20
  dE_dT_frac_per_C: -2.5e-4
  dsigma_y_dT_frac_per_C: -3.5e-4
  dsigma_e_dT_frac_per_C: -3.0e-4

damping:
  Q_at_20kHz: 2600
  Q_at_15kHz: 2800
  Q_at_40kHz: 2200
  strain_threshold: 0.001
  amplitude_exponent_n: 0.55
  dQ_dT_frac_per_C: -1.9e-3

data_sources:
  - "Crucible Industries CPM Rex M4 datasheet"
  - "Dukane Ultrasonics material qualification data"
```

### 6.2.12 Material Property Summary Table

| Material | E [GPa] | rho [kg/m3] | c_L [m/s] | Q (20kHz) | sigma_y [MPa] | sigma_e 1e7 [MPa] | sigma_e 1e9 [MPa] |
|---|---|---|---|---|---|---|---|
| Ti-6Al-4V (Ann) | 113.8 | 4430 | 6070 | 5000 | 880 | 510 | 350 |
| Ti-6Al-4V (STA) | 114.0 | 4430 | 6070 | 4500 | 1100 | 580 | 400 |
| Al 7075-T6 | 71.7 | 2810 | 6320 | 9000 | 503 | 159 | 105 |
| Steel D2 | 210.0 | 7700 | 5870 | 3000 | 1620 | 520 | 380 |
| M2 HSS | 220.0 | 8160 | 5190 | 2800 | 2200 | 690 | 490 |
| CPM 10V | 222.0 | 7690 | 5375 | 2500 | 2100 | 640 | 450 |
| PM60 | 230.0 | 8100 | 5328 | 2600 | 2400 | 720 | 510 |
| HAP40 | 228.0 | 8050 | 5319 | 2700 | 2500 | 740 | 520 |
| HAP72 | 235.0 | 8200 | 5352 | 2400 | 2800 | 800 | 560 |
| Steel 4140 | 205.0 | 7850 | 5900 | 3500 | 655 | 420 | 310 |
| PZT-4 | 81.3 | 7500 | 4600 | 500 | 80 | 25 | 20 |
| PZT-8 | 86.9 | 7600 | 4560 | 1000 | 85 | 28 | 22 |
| Ferro-Titanit WFN | 290.0 | 6550 | 6650 | 2000 | 1800 | 600 | 420 |
| CPM Rex M4 | 225.0 | 7970 | 5313 | 2600 | 2350 | 710 | 500 |

---

## 6.3 Fatigue Assessment Module

### 6.3.1 Architecture

```
FatigueAssessor
 |
 +-- SNInterpolator          # Log-log S-N curve interpolation
 +-- StressConcentrationDB   # Kt factors for common geometric features
 +-- GoodmanDiagram          # Modified Goodman mean-stress correction
 +-- SafetyFactorCalculator  # Combined SF with all correction factors
 +-- RainflowCounter         # ASTM E1049 rainflow cycle counting (optional)
 +-- CorrectionFactors       # Surface finish, size, temperature, reliability
```

### 6.3.2 S-N Curve Interpolation

S-N data is interpolated in log-log space. For stress amplitudes between data points,
linear interpolation on (log10 N, log10 S) is used. Extrapolation beyond the last data
point uses the slope of the final two points, with a minimum endurance limit floor.

```python
"""ultrasonic_weld_master/fea/fatigue/sn_interpolator.py"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


class SNInterpolator:
    """Log-log interpolation of S-N fatigue data.

    Stores data as (log10(N), log10(S)) for stable numeric interpolation.
    """

    def __init__(
        self,
        N_data: list[float],
        S_data: list[float],
        endurance_limit_MPa: Optional[float] = None,
        R_ratio: float = -1.0,
    ):
        """
        Parameters
        ----------
        N_data : list[float]
            Cycle counts (must be sorted ascending).
        S_data : list[float]
            Corresponding stress amplitudes [MPa] (must be sorted descending).
        endurance_limit_MPa : float, optional
            If provided, S never drops below this value.
        R_ratio : float
            Stress ratio for this curve.
        """
        assert len(N_data) == len(S_data) >= 3, "Need at least 3 S-N data points"
        self._logN = np.array([math.log10(n) for n in N_data])
        self._logS = np.array([math.log10(s) for s in S_data])
        self._R = R_ratio
        self._Se = endurance_limit_MPa

        # Pre-compute piecewise slopes for each segment
        self._slopes = np.diff(self._logS) / np.diff(self._logN)

    def cycles_to_failure(self, S_amplitude_MPa: float) -> float:
        """Predict N_f for a given fully-reversed stress amplitude.

        Returns
        -------
        float
            Predicted cycles to failure. Returns math.inf if S is at or below
            the endurance limit.
        """
        if S_amplitude_MPa <= 0:
            return math.inf
        if self._Se is not None and S_amplitude_MPa <= self._Se:
            return math.inf

        logS = math.log10(S_amplitude_MPa)

        # Above the highest data point stress => low-cycle extrapolation
        if logS >= self._logS[0]:
            slope = self._slopes[0]
            if slope == 0:
                return 10 ** self._logN[0]
            logN = self._logN[0] + (logS - self._logS[0]) / slope
            return max(1.0, 10 ** logN)

        # Below the lowest data point stress => high-cycle extrapolation
        if logS <= self._logS[-1]:
            slope = self._slopes[-1]
            if slope == 0:
                return math.inf
            logN = self._logN[-1] + (logS - self._logS[-1]) / slope
            return 10 ** logN

        # Interior interpolation
        for i in range(len(self._logS) - 1):
            if self._logS[i] >= logS >= self._logS[i + 1]:
                slope = self._slopes[i]
                if slope == 0:
                    return 10 ** self._logN[i]
                logN = self._logN[i] + (logS - self._logS[i]) / slope
                return 10 ** logN

        return math.inf  # Should not reach here

    def stress_at_cycles(self, N_cycles: float) -> float:
        """Predict allowable stress amplitude for a target life.

        Returns
        -------
        float
            Stress amplitude [MPa] at the given cycle count.
        """
        if N_cycles <= 0:
            raise ValueError("N_cycles must be positive")

        logN = math.log10(N_cycles)

        # Clamp to data range with extrapolation
        if logN <= self._logN[0]:
            slope = self._slopes[0]
            logS = self._logS[0] + slope * (logN - self._logN[0])
        elif logN >= self._logN[-1]:
            slope = self._slopes[-1]
            logS = self._logS[-1] + slope * (logN - self._logN[-1])
        else:
            # Interior interpolation
            idx = np.searchsorted(self._logN, logN) - 1
            idx = max(0, min(idx, len(self._slopes) - 1))
            logS = self._logS[idx] + self._slopes[idx] * (logN - self._logN[idx])

        S = 10 ** logS

        # Floor at endurance limit
        if self._Se is not None:
            S = max(S, self._Se)

        return S
```

### 6.3.3 Stress Concentration Factor Database

Kt values for common geometric features encountered in ultrasonic horn design.

```python
"""ultrasonic_weld_master/fea/fatigue/stress_concentration.py"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class KtResult:
    """Result of stress concentration calculation."""
    Kt: float
    feature_type: str
    formula_source: str
    Kf: Optional[float] = None    # Fatigue notch factor (if q is known)
    q: Optional[float] = None     # Notch sensitivity


class StressConcentrationDB:
    """Kt calculations for common ultrasonic horn geometric features.

    All formulas from Peterson's Stress Concentration Factors (Pilkey & Pilkey).
    """

    @staticmethod
    def shoulder_fillet_tension(D: float, d: float, r: float) -> KtResult:
        """Stepped shaft with shoulder fillet under axial tension.

        Parameters
        ----------
        D : float   Larger diameter [mm]
        d : float   Smaller diameter [mm]
        r : float   Fillet radius [mm]

        Returns Kt per Peterson's Charts 3.1
        """
        t = (D - d) / 2.0
        rho = r / t
        Dd = D / d

        # Neuber's formula approximation for stepped flat bar / shaft
        # Kt = C1 + C2*(r/t) + C3*(r/t)^2 + C4*(r/t)^3
        # Coefficients depend on D/d ratio
        if Dd <= 1.1:
            C1, C2, C3, C4 = 0.926, 1.157, -0.099, 0.016
        elif Dd <= 1.5:
            C1, C2, C3, C4 = 1.005, 0.925, -0.404, 0.124
        elif Dd <= 2.0:
            C1, C2, C3, C4 = 1.032, 0.832, -0.586, 0.195
        else:
            C1, C2, C3, C4 = 1.049, 0.768, -0.698, 0.239

        rho_clamped = max(0.01, min(rho, 4.0))
        Kt = C1 + C2 * rho_clamped + C3 * rho_clamped**2 + C4 * rho_clamped**3
        Kt = max(1.0, Kt)

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="shoulder_fillet_tension",
            formula_source="Peterson's SCF, Chart 3.1"
        )

    @staticmethod
    def shoulder_fillet_bending(D: float, d: float, r: float) -> KtResult:
        """Stepped shaft with shoulder fillet under bending.

        Parameters matching shoulder_fillet_tension.
        """
        t = (D - d) / 2.0
        rho = r / t
        Dd = D / d

        if Dd <= 1.1:
            C1, C2, C3, C4 = 0.927, 1.149, -0.086, 0.010
        elif Dd <= 1.5:
            C1, C2, C3, C4 = 1.007, 0.886, -0.360, 0.100
        elif Dd <= 2.0:
            C1, C2, C3, C4 = 1.027, 0.817, -0.517, 0.160
        else:
            C1, C2, C3, C4 = 1.038, 0.770, -0.618, 0.200

        rho_clamped = max(0.01, min(rho, 4.0))
        Kt = C1 + C2 * rho_clamped + C3 * rho_clamped**2 + C4 * rho_clamped**3
        Kt = max(1.0, Kt)

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="shoulder_fillet_bending",
            formula_source="Peterson's SCF, Chart 3.2"
        )

    @staticmethod
    def circumferential_groove(D: float, d: float, r: float) -> KtResult:
        """Circumferential U-groove in a shaft (tension).

        Parameters
        ----------
        D : float   Outer diameter [mm]
        d : float   Root diameter [mm]
        r : float   Groove root radius [mm]
        """
        t = (D - d) / 2.0
        if r <= 0 or t <= 0:
            return KtResult(Kt=1.0, feature_type="circumferential_groove",
                            formula_source="N/A (degenerate geometry)")

        # Neuber approximation
        Kt = 1.0 + 2.0 * math.sqrt(t / r)
        Kt = min(Kt, 6.0)  # Physical upper bound

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="circumferential_groove",
            formula_source="Neuber's formula (approx)"
        )

    @staticmethod
    def transverse_hole(d_hole: float, D_shaft: float) -> KtResult:
        """Transverse hole through a shaft (tension).

        Parameters
        ----------
        d_hole : float   Hole diameter [mm]
        D_shaft : float  Shaft diameter [mm]
        """
        ratio = d_hole / D_shaft
        # For small d/D, Kt approaches 3.0 (stress concentration at hole)
        # Modified formula from Peterson's
        Kt = 3.0 - 3.13 * ratio + 3.66 * ratio**2 - 1.53 * ratio**3
        Kt = max(1.0, Kt)

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="transverse_hole",
            formula_source="Peterson's SCF, Chart 4.1"
        )

    @staticmethod
    def thread_root(pitch_mm: float, root_radius_mm: float) -> KtResult:
        """Thread root stress concentration.

        Parameters
        ----------
        pitch_mm : float       Thread pitch [mm]
        root_radius_mm : float Root radius [mm]
        """
        if root_radius_mm <= 0:
            Kt = 5.0  # Sharp root approximation
        else:
            # Heywood's formula for metric threads
            Kt = 1.0 + 2.0 * math.sqrt(pitch_mm / (4.0 * root_radius_mm))
            Kt = min(Kt, 6.0)

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="thread_root",
            formula_source="Heywood's thread formula"
        )

    @staticmethod
    def slot_keyway(w: float, d: float, r: float) -> KtResult:
        """Slot or keyway in a shaft.

        Parameters
        ----------
        w : float   Slot width [mm]
        d : float   Shaft diameter [mm]
        r : float   Corner radius of slot [mm]
        """
        if r <= 0:
            Kt = 4.0  # Sharp corner
        else:
            Kt = 1.0 + 1.5 * math.sqrt(w / (2.0 * r))
            Kt = min(Kt, 5.0)

        return KtResult(
            Kt=round(Kt, 3),
            feature_type="slot_keyway",
            formula_source="Peterson's SCF, keyway approximation"
        )

    @staticmethod
    def notch_sensitivity(
        Kt: float,
        r_mm: float,
        sigma_u_MPa: float,
        method: str = "peterson"
    ) -> tuple[float, float]:
        """Compute fatigue notch factor Kf from Kt using notch sensitivity.

        Parameters
        ----------
        Kt : float          Elastic stress concentration factor
        r_mm : float         Notch root radius [mm]
        sigma_u_MPa : float  Ultimate tensile strength [MPa]
        method : str         "peterson" or "neuber"

        Returns
        -------
        tuple[float, float]  (q, Kf) where q is notch sensitivity and Kf = 1 + q*(Kt-1)
        """
        if method == "neuber":
            # Neuber's constant (for steel, empirical fit)
            # a = (300/sigma_u)^1.8 in mm (Neuber's material constant)
            a_mm = (300.0 / sigma_u_MPa) ** 1.8
            q = 1.0 / (1.0 + math.sqrt(a_mm / max(r_mm, 0.01)))
        else:
            # Peterson's formula
            # a = 0.0254 * (2070/sigma_u)^1.8  [mm]
            a_mm = 0.0254 * (2070.0 / sigma_u_MPa) ** 1.8
            q = 1.0 / (1.0 + a_mm / max(r_mm, 0.01))

        q = max(0.0, min(q, 1.0))
        Kf = 1.0 + q * (Kt - 1.0)
        return (round(q, 4), round(Kf, 3))
```

### 6.3.4 Modified Goodman Diagram

For ultrasonic horn operation, the primary loading is fully reversed (R = -1), but
thermal stresses and bolt preloads introduce a mean stress component.

```python
"""ultrasonic_weld_master/fea/fatigue/goodman.py"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class MeanStressTheory(str, Enum):
    GOODMAN = "goodman"
    GERBER = "gerber"
    SODERBERG = "soderberg"
    MORROW = "morrow"


@dataclass
class GoodmanResult:
    """Result of mean stress correction."""
    sigma_a_corrected_MPa: float   # Equivalent fully-reversed amplitude
    safety_factor: float
    theory: MeanStressTheory
    on_yield_envelope: bool        # True if yield criterion is more limiting


class GoodmanDiagram:
    """Modified Goodman diagram for mean-stress correction.

    Given a combined (sigma_mean, sigma_alternating) state, computes the
    equivalent fully-reversed stress amplitude for use with R=-1 S-N data.
    """

    def __init__(
        self,
        sigma_u_MPa: float,
        sigma_y_MPa: float,
        sigma_e_MPa: float,
    ):
        self._Su = sigma_u_MPa
        self._Sy = sigma_y_MPa
        self._Se = sigma_e_MPa

    def evaluate(
        self,
        sigma_mean_MPa: float,
        sigma_alt_MPa: float,
        theory: MeanStressTheory = MeanStressTheory.GOODMAN,
    ) -> GoodmanResult:
        """Evaluate fatigue safety factor under combined mean + alternating stress.

        Parameters
        ----------
        sigma_mean_MPa : float
            Mean stress [MPa] (positive = tensile).
        sigma_alt_MPa : float
            Alternating stress amplitude [MPa] (always positive).
        theory : MeanStressTheory
            Which mean-stress correction to use.

        Returns
        -------
        GoodmanResult
        """
        Sa = abs(sigma_alt_MPa)
        Sm = sigma_mean_MPa  # Can be negative (compressive mean)

        # Equivalent fully reversed amplitude
        if theory == MeanStressTheory.GOODMAN:
            # Modified Goodman: Sa/Se + Sm/Su = 1/SF
            if Sm >= 0:
                denom = 1.0 - Sm / self._Su if self._Su > Sm else 0.001
                Sa_eq = Sa / max(denom, 0.001)
                SF = self._Se / Sa_eq if Sa_eq > 0 else math.inf
            else:
                # Compressive mean stress is beneficial (conservative: ignore it)
                Sa_eq = Sa
                SF = self._Se / Sa if Sa > 0 else math.inf

        elif theory == MeanStressTheory.GERBER:
            # Gerber parabola: Sa/Se + (Sm/Su)^2 = 1/SF
            if Sm >= 0:
                denom = 1.0 - (Sm / self._Su) ** 2 if self._Su > 0 else 0.001
                Sa_eq = Sa / max(denom, 0.001)
                SF = self._Se / Sa_eq if Sa_eq > 0 else math.inf
            else:
                Sa_eq = Sa
                SF = self._Se / Sa if Sa > 0 else math.inf

        elif theory == MeanStressTheory.SODERBERG:
            # Soderberg (conservative, uses yield): Sa/Se + Sm/Sy = 1/SF
            if Sm >= 0:
                denom = 1.0 - Sm / self._Sy if self._Sy > Sm else 0.001
                Sa_eq = Sa / max(denom, 0.001)
                SF = self._Se / Sa_eq if Sa_eq > 0 else math.inf
            else:
                Sa_eq = Sa
                SF = self._Se / Sa if Sa > 0 else math.inf

        elif theory == MeanStressTheory.MORROW:
            # Morrow (uses true fracture strength, approximated as 1.1*Su)
            sigma_f = 1.1 * self._Su
            if Sm >= 0:
                denom = 1.0 - Sm / sigma_f if sigma_f > Sm else 0.001
                Sa_eq = Sa / max(denom, 0.001)
                SF = self._Se / Sa_eq if Sa_eq > 0 else math.inf
            else:
                Sa_eq = Sa
                SF = self._Se / Sa if Sa > 0 else math.inf
        else:
            raise ValueError(f"Unknown theory: {theory}")

        # Check yield envelope:  Sa + Sm <= Sy
        yield_SF = self._Sy / (Sa + abs(Sm)) if (Sa + abs(Sm)) > 0 else math.inf
        on_yield = yield_SF < SF

        return GoodmanResult(
            sigma_a_corrected_MPa=round(Sa_eq, 2),
            safety_factor=round(min(SF, yield_SF), 3),
            theory=theory,
            on_yield_envelope=on_yield,
        )
```

### 6.3.5 Safety Factor Calculator

The master safety factor calculation combines all correction factors into a single
fatigue assessment result.

```python
"""ultrasonic_weld_master/fea/fatigue/safety_factor.py"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CorrectionFactors:
    """Aggregate of all fatigue life correction factors."""
    k_surface: float = 1.0       # Surface finish factor (0.2 - 1.0)
    k_size: float = 1.0          # Size factor (0.6 - 1.0)
    k_reliability: float = 1.0   # Reliability factor (0.814 for 99%, 0.753 for 99.9%)
    k_temperature: float = 1.0   # Temperature correction (<=1.0 for T > T_ref)
    k_miscellaneous: float = 1.0 # Other effects (corrosion, residual stress, etc.)

    @property
    def k_total(self) -> float:
        return (self.k_surface * self.k_size * self.k_reliability
                * self.k_temperature * self.k_miscellaneous)


@dataclass
class FatigueAssessmentResult:
    """Complete fatigue assessment output."""
    # Input summary
    sigma_vm_alternating_MPa: float
    sigma_vm_mean_MPa: float
    Kt: float
    Kf: float

    # Material baseline
    sigma_e_unnotched_MPa: float
    sigma_e_corrected_MPa: float   # After all correction factors

    # Safety factor
    safety_factor: float
    safety_factor_yield: float

    # Life prediction
    predicted_cycles: float
    target_cycles: float
    life_adequate: bool

    # Correction factor breakdown
    corrections: CorrectionFactors
    goodman_theory: str = "goodman"

    # Warnings
    warnings: list[str] = field(default_factory=list)


class SafetyFactorCalculator:
    """Compute fatigue safety factor with all engineering corrections.

    Methodology
    -----------
    1. Start with material's unnotched endurance limit sigma_e (at target N)
    2. Apply Marin correction factors: sigma_e_corrected = sigma_e * k_total
    3. Apply stress concentration: sigma_eff = Kf * sigma_alternating
    4. Apply Goodman mean-stress correction
    5. SF = sigma_e_corrected / sigma_a_equivalent
    """

    # ── Surface finish factors (Marin) ──

    SURFACE_FINISH_COEFFICIENTS = {
        # (a, b) where k_surface = a * Su^b   (Su in MPa)
        "ground": (1.58, -0.085),
        "machined": (4.51, -0.265),
        "hot_rolled": (57.7, -0.718),
        "forged": (272.0, -0.995),
        "polished": (1.0, 0.0),       # k = 1.0 (reference condition)
        "mirror": (1.0, 0.0),
        "edm": (3.04, -0.217),        # EDM-finished surfaces
    }

    @classmethod
    def surface_finish_factor(cls, finish: str, sigma_u_MPa: float) -> float:
        """Marin surface finish correction factor.

        Parameters
        ----------
        finish : str
            One of: "polished", "ground", "machined", "hot_rolled", "forged", "edm".
        sigma_u_MPa : float
            Material ultimate tensile strength [MPa].
        """
        if finish not in cls.SURFACE_FINISH_COEFFICIENTS:
            return 1.0
        a, b = cls.SURFACE_FINISH_COEFFICIENTS[finish]
        if b == 0:
            return a
        return a * sigma_u_MPa ** b

    @staticmethod
    def size_factor(d_mm: float) -> float:
        """Marin size correction factor for round cross-sections.

        Parameters
        ----------
        d_mm : float
            Equivalent diameter [mm]. For non-round sections, compute
            d_eq = sqrt(A_95 / 0.0766) where A_95 is the 95% stressed area.
        """
        if d_mm <= 8.0:
            return 1.0
        elif d_mm <= 250.0:
            return 1.189 * d_mm ** (-0.097)
        else:
            return 0.6  # Conservative for very large sections

    @staticmethod
    def reliability_factor(reliability: float = 0.99) -> float:
        """Reliability correction factor.

        Parameters
        ----------
        reliability : float
            Target reliability (e.g., 0.90, 0.99, 0.999, 0.9999).
        """
        # Based on log-normal distribution of fatigue data
        RELIABILITY_TABLE = {
            0.50: 1.000,
            0.90: 0.897,
            0.95: 0.868,
            0.99: 0.814,
            0.999: 0.753,
            0.9999: 0.702,
            0.99999: 0.659,
        }
        # Find closest entry
        closest = min(RELIABILITY_TABLE.keys(), key=lambda x: abs(x - reliability))
        return RELIABILITY_TABLE[closest]

    @staticmethod
    def temperature_factor(
        T_operating_C: float,
        T_ref_C: float = 20.0,
        dSe_dT_frac: float = -4.0e-4,
    ) -> float:
        """Temperature derating factor for fatigue endurance limit.

        Parameters
        ----------
        T_operating_C : float
            Actual operating temperature [deg C].
        T_ref_C : float
            Reference temperature for material data [deg C].
        dSe_dT_frac : float
            Fractional change per degree C (negative means derating).
        """
        delta_T = T_operating_C - T_ref_C
        k = 1.0 + dSe_dT_frac * delta_T
        return max(0.1, min(k, 1.2))  # Clamp to reasonable range

    def assess(
        self,
        sigma_vm_alt_MPa: float,
        sigma_vm_mean_MPa: float,
        sigma_e_MPa: float,
        sigma_u_MPa: float,
        sigma_y_MPa: float,
        Kt: float = 1.0,
        Kf: Optional[float] = None,
        target_cycles: float = 1e9,
        corrections: Optional[CorrectionFactors] = None,
        sn_interpolator=None,
    ) -> FatigueAssessmentResult:
        """Perform complete fatigue assessment.

        Parameters
        ----------
        sigma_vm_alt_MPa : float
            Von Mises alternating stress from FEA [MPa].
        sigma_vm_mean_MPa : float
            Von Mises mean stress from FEA [MPa].
        sigma_e_MPa : float
            Material endurance limit at target life [MPa].
        sigma_u_MPa : float
            Ultimate tensile strength [MPa].
        sigma_y_MPa : float
            Yield strength [MPa].
        Kt : float
            Geometric stress concentration factor.
        Kf : float, optional
            Fatigue notch factor. If None, uses Kf = Kt (conservative).
        target_cycles : float
            Target fatigue life [cycles].
        corrections : CorrectionFactors, optional
            Pre-computed correction factors.
        sn_interpolator : SNInterpolator, optional
            For life prediction from S-N curve.
        """
        if corrections is None:
            corrections = CorrectionFactors()

        if Kf is None:
            Kf = Kt  # Conservative: full theoretical SCF

        warnings = []

        # Step 1: Corrected endurance limit
        Se_corrected = sigma_e_MPa * corrections.k_total

        # Step 2: Effective alternating stress (with notch)
        sigma_a_eff = Kf * sigma_vm_alt_MPa

        # Step 3: Goodman mean-stress correction
        from .goodman import GoodmanDiagram, MeanStressTheory
        gd = GoodmanDiagram(sigma_u_MPa, sigma_y_MPa, Se_corrected)
        gr = gd.evaluate(sigma_vm_mean_MPa, sigma_a_eff, MeanStressTheory.GOODMAN)

        SF = gr.safety_factor
        SF_yield = sigma_y_MPa / (sigma_a_eff + abs(sigma_vm_mean_MPa)) \
            if (sigma_a_eff + abs(sigma_vm_mean_MPa)) > 0 else math.inf

        # Step 4: Life prediction
        if sn_interpolator is not None:
            predicted_N = sn_interpolator.cycles_to_failure(sigma_a_eff)
        else:
            # Basquin-type estimate: N = (Se/S)^m * N_e
            m = 8.0  # Typical high-cycle exponent for steels
            if sigma_a_eff > 0 and Se_corrected > 0:
                predicted_N = target_cycles * (Se_corrected / sigma_a_eff) ** m
            else:
                predicted_N = math.inf

        life_ok = predicted_N >= target_cycles

        # Step 5: Generate warnings
        if SF < 1.0:
            warnings.append(
                f"CRITICAL: Safety factor {SF:.2f} < 1.0. Fatigue failure expected."
            )
        elif SF < 1.5:
            warnings.append(
                f"WARNING: Safety factor {SF:.2f} < 1.5. Marginal design."
            )
        elif SF < 2.0:
            warnings.append(
                f"CAUTION: Safety factor {SF:.2f} < 2.0. Acceptable for "
                f"controlled conditions only."
            )

        if corrections.k_surface < 0.5:
            warnings.append(
                "Surface finish significantly reduces fatigue life. "
                "Consider polishing critical areas."
            )

        if Kt > 3.0:
            warnings.append(
                f"High stress concentration Kt={Kt:.1f}. "
                "Redesign fillet/radius to reduce Kt."
            )

        return FatigueAssessmentResult(
            sigma_vm_alternating_MPa=round(sigma_vm_alt_MPa, 2),
            sigma_vm_mean_MPa=round(sigma_vm_mean_MPa, 2),
            Kt=round(Kt, 3),
            Kf=round(Kf, 3),
            sigma_e_unnotched_MPa=round(sigma_e_MPa, 2),
            sigma_e_corrected_MPa=round(Se_corrected, 2),
            safety_factor=round(SF, 3),
            safety_factor_yield=round(SF_yield, 3),
            predicted_cycles=predicted_N,
            target_cycles=target_cycles,
            life_adequate=life_ok,
            corrections=corrections,
            goodman_theory="goodman",
            warnings=warnings,
        )
```

### 6.3.6 Rainflow Cycle Counting (Optional Module)

For complex load histories (e.g., during horn ramp-up/ramp-down or frequency sweeps),
ASTM E1049 rainflow counting extracts individual fatigue cycles.

```python
"""ultrasonic_weld_master/fea/fatigue/rainflow.py"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FatigueCycle:
    """A single extracted fatigue cycle."""
    range_MPa: float
    mean_MPa: float
    count: float  # 0.5 for half-cycles, 1.0 for full cycles


def rainflow_count(stress_history: list[float]) -> list[FatigueCycle]:
    """ASTM E1049-85 three-point rainflow cycle counting.

    Parameters
    ----------
    stress_history : list[float]
        Time-ordered stress values [MPa].

    Returns
    -------
    list[FatigueCycle]
        Extracted cycles with range, mean, and count.
    """
    # Extract peaks and valleys (reversals)
    reversals = _extract_reversals(stress_history)

    cycles: list[FatigueCycle] = []
    stack: list[float] = []

    for point in reversals:
        stack.append(point)

        while len(stack) >= 3:
            # Three-point rule
            s0, s1, s2 = stack[-3], stack[-2], stack[-1]
            range_inner = abs(s1 - s0)
            range_outer = abs(s2 - s1)

            if range_inner <= range_outer:
                # Inner range forms a complete cycle
                cycles.append(FatigueCycle(
                    range_MPa=range_inner,
                    mean_MPa=(s0 + s1) / 2.0,
                    count=1.0,
                ))
                # Remove the two points that formed the cycle
                stack.pop(-2)
                stack.pop(-2)
            else:
                break

    # Remaining points form half-cycles
    for i in range(len(stack) - 1):
        rng = abs(stack[i + 1] - stack[i])
        mean = (stack[i] + stack[i + 1]) / 2.0
        cycles.append(FatigueCycle(
            range_MPa=rng,
            mean_MPa=mean,
            count=0.5,
        ))

    return cycles


def palmgren_miner_damage(
    cycles: list[FatigueCycle],
    sn_interpolator,
) -> float:
    """Compute cumulative Palmgren-Miner damage from extracted cycles.

    Parameters
    ----------
    cycles : list[FatigueCycle]
        Output from rainflow_count.
    sn_interpolator : SNInterpolator
        S-N curve for the material.

    Returns
    -------
    float
        Cumulative damage D. Failure predicted when D >= 1.0.
    """
    D = 0.0
    for cycle in cycles:
        S_amp = cycle.range_MPa / 2.0  # Amplitude = range / 2
        N_f = sn_interpolator.cycles_to_failure(S_amp)
        if N_f < float("inf"):
            D += cycle.count / N_f
    return D


def _extract_reversals(history: list[float]) -> list[float]:
    """Extract peaks and valleys from a stress history."""
    if len(history) < 3:
        return list(history)

    reversals = [history[0]]
    for i in range(1, len(history) - 1):
        prev, curr, nxt = history[i - 1], history[i], history[i + 1]
        if (curr - prev) * (nxt - curr) < 0:  # Sign change in slope
            reversals.append(curr)
    reversals.append(history[-1])
    return reversals
```

---

## 6.4 Damping Model

### 6.4.1 Damping Parameter Conversions

The FEA system must convert between different damping representations depending on
the solver being used.

```python
"""ultrasonic_weld_master/fea/damping/damping_model.py"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class DampingParameters:
    """All equivalent damping representations for a given material state."""
    Q: float                      # Mechanical quality factor
    loss_tangent: float           # eta = 1/Q = tan(delta)
    damping_ratio: float          # zeta = 1/(2Q)
    log_decrement: float          # delta = pi/Q
    rayleigh_alpha: float         # Mass-proportional Rayleigh coefficient [1/s]
    rayleigh_beta: float          # Stiffness-proportional Rayleigh coefficient [s]
    specific_damping_capacity: float  # psi = 2*pi/Q  (energy ratio per cycle)


class MaterialDamping:
    """Compute damping parameters for ultrasonic horn materials.

    Handles three dependencies:
    1. Frequency dependence (Q varies with frequency)
    2. Strain-amplitude dependence (critical for Ti-6Al-4V above ~0.3% strain)
    3. Temperature dependence

    Rayleigh Damping
    ----------------
    For a frequency-band FEA analysis, Rayleigh damping is specified as:
        C = alpha * M + beta * K

    where alpha and beta are chosen to match the target damping ratio zeta
    at two frequencies f1 and f2:

        alpha = 2 * zeta * (omega1 * omega2) / (omega1 + omega2)
        beta  = 2 * zeta / (omega1 + omega2)

    For single-frequency harmonic analysis (the primary use case), structural
    damping via the loss tangent eta is more appropriate.
    """

    def __init__(
        self,
        Q_ref: float,
        freq_ref_Hz: float = 20000.0,
        Q_at_freq: Optional[dict[float, float]] = None,
        strain_threshold: Optional[float] = None,
        amplitude_exponent: float = 0.5,
        dQ_dT_frac: float = 0.0,
        T_ref_C: float = 20.0,
    ):
        """
        Parameters
        ----------
        Q_ref : float
            Quality factor at reference frequency and low amplitude.
        freq_ref_Hz : float
            Reference frequency [Hz].
        Q_at_freq : dict, optional
            Map of frequency [Hz] -> Q value for frequency interpolation.
        strain_threshold : float, optional
            Strain amplitude above which Q degrades.
        amplitude_exponent : float
            Exponent n in Q(eps) = Q_0 * (eps_th/eps)^n.
        dQ_dT_frac : float
            Fractional Q change per degree C.
        T_ref_C : float
            Reference temperature [deg C].
        """
        self._Q_ref = Q_ref
        self._f_ref = freq_ref_Hz
        self._Q_freq = Q_at_freq or {}
        self._eps_th = strain_threshold
        self._n = amplitude_exponent
        self._dQ_dT = dQ_dT_frac
        self._T_ref = T_ref_C

    def get_Q(
        self,
        frequency_Hz: float = 20000.0,
        strain_amplitude: float = 0.0,
        temperature_C: float = 20.0,
    ) -> float:
        """Compute effective Q for given operating conditions.

        Parameters
        ----------
        frequency_Hz : float
            Operating frequency [Hz].
        strain_amplitude : float
            Peak dynamic strain amplitude [-].
        temperature_C : float
            Material temperature [deg C].

        Returns
        -------
        float
            Effective mechanical quality factor.
        """
        # 1. Frequency dependence: interpolate from known Q-frequency pairs
        Q = self._interpolate_Q_freq(frequency_Hz)

        # 2. Strain amplitude dependence
        if (self._eps_th is not None
                and strain_amplitude > self._eps_th
                and strain_amplitude > 0):
            Q *= (self._eps_th / strain_amplitude) ** self._n

        # 3. Temperature dependence
        dT = temperature_C - self._T_ref
        Q *= (1.0 + self._dQ_dT * dT)

        return max(Q, 10.0)  # Physical lower bound

    def get_all_parameters(
        self,
        frequency_Hz: float = 20000.0,
        strain_amplitude: float = 0.0,
        temperature_C: float = 20.0,
        f1_Hz: Optional[float] = None,
        f2_Hz: Optional[float] = None,
    ) -> DampingParameters:
        """Compute all damping representations for current state.

        Parameters
        ----------
        frequency_Hz : float
            Primary operating frequency [Hz].
        strain_amplitude, temperature_C : float
            Operating conditions.
        f1_Hz, f2_Hz : float, optional
            Frequency band for Rayleigh damping. Defaults to 0.8*f and 1.2*f.
        """
        Q = self.get_Q(frequency_Hz, strain_amplitude, temperature_C)
        eta = 1.0 / Q
        zeta = 1.0 / (2.0 * Q)
        delta = math.pi / Q
        psi = 2.0 * math.pi / Q

        # Rayleigh coefficients
        if f1_Hz is None:
            f1_Hz = 0.8 * frequency_Hz
        if f2_Hz is None:
            f2_Hz = 1.2 * frequency_Hz
        omega1 = 2.0 * math.pi * f1_Hz
        omega2 = 2.0 * math.pi * f2_Hz
        alpha = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
        beta = 2.0 * zeta / (omega1 + omega2)

        return DampingParameters(
            Q=round(Q, 1),
            loss_tangent=eta,
            damping_ratio=zeta,
            log_decrement=delta,
            rayleigh_alpha=alpha,
            rayleigh_beta=beta,
            specific_damping_capacity=psi,
        )

    def _interpolate_Q_freq(self, f_Hz: float) -> float:
        """Log-linear interpolation of Q over frequency."""
        if not self._Q_freq:
            return self._Q_ref

        freqs = sorted(self._Q_freq.keys())
        Qs = [self._Q_freq[f] for f in freqs]

        if f_Hz <= freqs[0]:
            return Qs[0]
        if f_Hz >= freqs[-1]:
            return Qs[-1]

        # Find bracketing frequencies
        for i in range(len(freqs) - 1):
            if freqs[i] <= f_Hz <= freqs[i + 1]:
                # Log-linear interpolation
                log_f = math.log10(f_Hz)
                log_f1 = math.log10(freqs[i])
                log_f2 = math.log10(freqs[i + 1])
                t = (log_f - log_f1) / (log_f2 - log_f1)
                return Qs[i] + t * (Qs[i + 1] - Qs[i])

        return self._Q_ref
```

### 6.4.2 Material-Specific Damping Notes

| Material | Q (20kHz) | Strain Threshold | Critical Behavior |
|---|---|---|---|
| Ti-6Al-4V | 5000 | 0.003 (0.3%) | Q drops sharply above threshold; primary cause of horn heating at high amplitude |
| Al 7075-T6 | 9000 | 0.002 | Very low damping; horns run cool but fatigue-limited |
| Steel D2 | 3000 | 0.001 | Moderate damping; adequate for most applications |
| M2 HSS | 2800 | 0.001 | Similar to D2 |
| PZT-4 | 500 | N/A | Dominant loss source in transducer stack; generates most heat |
| PZT-8 | 1000 | N/A | Preferred over PZT-4 for high-power due to 2x Q |

**Critical Design Rule**: For Ti-6Al-4V horns operating above 50 microns amplitude at
20 kHz, the strain in the nodal plane region typically exceeds the 0.003 threshold.
The damping model MUST account for the amplitude-dependent Q degradation, or thermal
runaway risk will be underestimated.

---

## 6.5 Data Storage Format

### 6.5.1 YAML File Structure

```
ultrasonic_weld_master/
  plugins/
    material_db/
      materials/                     # Per-material YAML files
        ti6al4v_annealed.yaml
        ti6al4v_sta.yaml
        al_7075_t6.yaml
        steel_d2.yaml
        m2_hss.yaml
        cpm_10v.yaml
        pm60.yaml
        hap40.yaml
        hap72.yaml
        steel_4140.yaml
        pzt4.yaml
        pzt8.yaml
        ferro_titanit_wfn.yaml
        cpm_rex_m4.yaml
      combinations/
        workpiece_combinations.yaml  # Cu-Al, Cu-Cu, etc. (existing)
      schema.py                      # Pydantic models (Section 6.1.1)
      plugin.py                      # Enhanced MaterialDBPlugin
      materials.yaml                 # Legacy file (retained for backward compat)
```

### 6.5.2 YAML Validation

Materials are validated at load time. Invalid files produce clear error messages and
are skipped (with a warning) rather than crashing the system.

```python
"""ultrasonic_weld_master/plugins/material_db/loader.py"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .schemas import FEAMaterial

logger = logging.getLogger(__name__)


class MaterialLoader:
    """Load and validate material YAML files."""

    def __init__(self, materials_dir: Optional[str] = None):
        if materials_dir is None:
            materials_dir = os.path.join(os.path.dirname(__file__), "materials")
        self._dir = Path(materials_dir)

    def load_all(self) -> dict[str, FEAMaterial]:
        """Load all .yaml files from the materials directory.

        Returns
        -------
        dict[str, FEAMaterial]
            Map of canonical name -> validated material.
        """
        materials: dict[str, FEAMaterial] = {}

        if not self._dir.exists():
            logger.warning("Materials directory not found: %s", self._dir)
            return materials

        for yaml_path in sorted(self._dir.glob("*.yaml")):
            try:
                mat = self.load_file(yaml_path)
                if mat is not None:
                    materials[mat.name] = mat
                    logger.debug("Loaded material: %s from %s", mat.name, yaml_path.name)
            except Exception as e:
                logger.error("Failed to load %s: %s", yaml_path.name, e)

        logger.info("Loaded %d materials from %s", len(materials), self._dir)
        return materials

    def load_file(self, path: Path) -> Optional[FEAMaterial]:
        """Load and validate a single material YAML file.

        Returns None if validation fails (with logged error).
        """
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            logger.warning("Empty YAML file: %s", path)
            return None

        try:
            return FEAMaterial(**raw)
        except ValidationError as e:
            logger.error("Validation failed for %s:\n%s", path.name, e)
            return None
```

### 6.5.3 Enhanced MaterialDBPlugin

```python
"""ultrasonic_weld_master/plugins/material_db/plugin.py  (enhanced)"""
from __future__ import annotations

import os
from typing import Any, Optional

import yaml

from ultrasonic_weld_master.core.plugin_api import PluginBase, PluginInfo
from .loader import MaterialLoader
from .schemas import FEAMaterial, MaterialCategory, HornSuitability


class MaterialDBPlugin(PluginBase):
    """Enhanced material database with FEA-grade properties and fatigue data."""

    def __init__(self):
        self._materials: dict[str, FEAMaterial] = {}
        self._alias_map: dict[str, str] = {}
        self._combinations: dict = {}
        self._legacy_materials: dict = {}  # Backward compat

    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="material_db", version="2.0.0",
            description="FEA-grade material database with fatigue assessment",
            author="UltrasonicWeldMaster", dependencies=[],
        )

    def activate(self, context: Any) -> None:
        # Load new YAML-per-material files
        loader = MaterialLoader()
        self._materials = loader.load_all()

        # Build alias index
        self._alias_map.clear()
        for name, mat in self._materials.items():
            self._alias_map[name.lower()] = name
            for alias in mat.aliases:
                self._alias_map[alias.lower()] = name

        # Load legacy materials.yaml for backward compat
        legacy_path = os.path.join(os.path.dirname(__file__), "materials.yaml")
        if os.path.exists(legacy_path):
            with open(legacy_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self._legacy_materials = data.get("materials", {})
            self._combinations = data.get("combinations", {})

    def deactivate(self) -> None:
        self._materials.clear()
        self._alias_map.clear()
        self._combinations.clear()
        self._legacy_materials.clear()

    # ── Query API ─────────────────────────────────────────────────────

    def get_fea_material(self, name: str) -> Optional[FEAMaterial]:
        """Look up full FEA material by name or alias (case-insensitive)."""
        key = self._alias_map.get(name.lower().strip())
        if key is not None:
            return self._materials.get(key)
        return None

    def get_material(self, material_type: str) -> Optional[dict]:
        """Legacy API: return flat property dict for backward compatibility."""
        mat = self.get_fea_material(material_type)
        if mat is not None:
            from .schemas import FEAMaterial
            return _flatten_to_legacy(mat)
        # Fall back to legacy YAML
        return self._legacy_materials.get(material_type)

    def list_materials(self) -> list[str]:
        """Return canonical names of all loaded materials."""
        return sorted(self._materials.keys())

    def find_by_category(self, category: MaterialCategory) -> list[FEAMaterial]:
        """Return all materials of a given category."""
        return [m for m in self._materials.values() if m.category == category]

    def find_horn_materials(
        self, suitability: Optional[HornSuitability] = None
    ) -> list[FEAMaterial]:
        """Return materials suitable for horn fabrication."""
        results = [
            m for m in self._materials.values()
            if m.horn_suitability != HornSuitability.NOT_RECOMMENDED
        ]
        if suitability is not None:
            results = [m for m in results if m.horn_suitability == suitability]
        return sorted(results, key=lambda m: m.name)

    def find_by_property(
        self,
        min_sigma_e_1e9_MPa: Optional[float] = None,
        max_density: Optional[float] = None,
        min_Q: Optional[float] = None,
    ) -> list[FEAMaterial]:
        """Find materials matching property constraints."""
        results = list(self._materials.values())
        if min_sigma_e_1e9_MPa is not None:
            results = [
                m for m in results
                if (m.strength.sigma_e_MPa_1e9 or 0) >= min_sigma_e_1e9_MPa
            ]
        if max_density is not None:
            results = [m for m in results if m.elastic.rho_kg_m3 <= max_density]
        if min_Q is not None:
            results = [m for m in results if m.acoustic.Q_mechanical >= min_Q]
        return results

    def get_combination_properties(self, mat_a: str, mat_b: str) -> dict:
        """Return workpiece combination properties (legacy API)."""
        key = f"{mat_a}-{mat_b}"
        if key in self._combinations:
            return self._combinations[key]
        reverse = f"{mat_b}-{mat_a}"
        if reverse in self._combinations:
            return self._combinations[reverse]
        return {}


def _flatten_to_legacy(mat: FEAMaterial) -> dict:
    """Convert FEAMaterial to legacy flat-dict format."""
    return {
        "E_pa": mat.elastic.E_GPa * 1e9,
        "nu": mat.elastic.nu,
        "rho_kg_m3": mat.elastic.rho_kg_m3,
        "k_w_mk": mat.thermal.k_W_mK,
        "cp_j_kgk": mat.thermal.cp_J_kgK,
        "yield_mpa": mat.strength.sigma_y_MPa,
        "alpha_1_k": mat.thermal.alpha_1_K,
        # Extended properties available in new format
        "ultimate_mpa": mat.strength.sigma_u_MPa,
        "sigma_e_1e7_mpa": mat.strength.sigma_e_MPa_1e7,
        "sigma_e_1e9_mpa": mat.strength.sigma_e_MPa_1e9,
        "Q_mechanical": mat.acoustic.Q_mechanical,
        "c_longitudinal_m_s": mat.acoustic.c_longitudinal_m_s,
    }
```

### 6.5.4 REST API Extensions

The existing FastAPI materials router is extended with endpoints for the enhanced schema:

```python
# web/routers/materials.py  (new endpoints)

@router.get("/materials/{name}/fatigue")
async def get_fatigue_data(name: str, svc=Depends(get_engine_service)) -> dict:
    """Return S-N curve data and fatigue properties for a material."""
    ...

@router.get("/materials/{name}/damping")
async def get_damping_data(
    name: str,
    frequency_hz: float = 20000,
    strain_amplitude: float = 0.0,
    temperature_c: float = 20.0,
    svc=Depends(get_engine_service),
) -> dict:
    """Return computed damping parameters at specified operating conditions."""
    ...

@router.get("/materials/search")
async def search_materials(
    min_sigma_e_1e9: Optional[float] = None,
    max_density: Optional[float] = None,
    min_Q: Optional[float] = None,
    horn_only: bool = False,
    svc=Depends(get_engine_service),
) -> dict:
    """Search materials by property constraints."""
    ...

@router.post("/fatigue/assess")
async def assess_fatigue(request: FatigueAssessmentRequest) -> dict:
    """Run full fatigue assessment on FEA stress results."""
    ...
```

---

## 6.6 Integration Points

### 6.6.1 FEA Solver Integration

The material database feeds directly into three FEA analysis stages:

1. **Modal Analysis**: `elastic` properties (E, nu, rho) define the stiffness and mass
   matrices. The material's acoustic wave speed `c_longitudinal` is used as a sanity
   check on computed natural frequencies.

2. **Harmonic Response**: `damping` model provides frequency- and amplitude-dependent
   loss tangent for the structural damping matrix. This is critical for computing
   accurate tip displacement amplitudes and identifying the peak stress in the horn.

3. **Thermal-Structural Coupling**: `thermal` properties (k, cp, alpha) combined with
   the damping-based heat generation rate `Q_heat = eta * sigma^2 * V / (2*E)` drive
   the steady-state thermal analysis. Temperature results feed back to update E(T),
   sigma_y(T), and Q(T) through the `temperature_dependence` model.

### 6.6.2 Fatigue Post-Processing Pipeline

```
FEA harmonic response
    |
    v
Extract sigma_vm_alternating at each node
    |
    v
Identify hot spots (top 1% stressed nodes)
    |
    v
For each hot spot:
    +-- Look up Kt from geometry (StressConcentrationDB)
    +-- Compute Kf via notch sensitivity (Peterson/Neuber)
    +-- Apply Marin correction factors (surface, size, reliability, temperature)
    +-- Goodman mean-stress correction (thermal mean stress from thermal FEA)
    +-- Interpolate S-N curve for predicted life
    +-- Compute safety factor
    |
    v
FatigueAssessmentResult per hot spot
    |
    v
Aggregate into report with:
    - Minimum SF location and value
    - Predicted life at critical location
    - Recommended design changes (if SF < 2.0)
    - Damping-induced heating estimate at critical strain
```

### 6.6.3 Piezoelectric Transducer Stack Modeling

For PZT-4 and PZT-8 materials, the `piezoelectric` property group provides the full
tensor data needed for coupled electro-mechanical FEA:

- **Mechanical domain**: cE elastic stiffness (6x6, Voigt) + rho
- **Electrical domain**: epsilon_S permittivity (3x3)
- **Coupling**: e piezoelectric stress tensor (3x6)

The transducer model computes the electrical impedance spectrum and electromechanical
coupling efficiency, which determines the power delivery to the horn.

---

## 6.7 Design Decisions and Rationale

| Decision | Rationale |
|---|---|
| One YAML file per material | Enables independent version control, code review of property changes, and easy addition of new materials |
| Pydantic validation at load time | Catches data entry errors immediately; enforces physical constraints (E > 0, 0 < nu < 0.5, etc.) |
| Separate S-N curves from endurance limits | S-N interpolation provides full life prediction; endurance limits provide quick screening |
| Amplitude-dependent damping model | Ti-6Al-4V Q drops 10-50x above 0.3% strain; ignoring this causes dangerous underestimation of horn heating |
| Goodman + Marin + Kt/Kf pipeline | Industry-standard fatigue methodology (per Shigley/Norton) adapted for ultrasonic-specific conditions |
| Legacy flat-dict compatibility layer | Existing FEA code and tests continue to work unchanged during migration |
| 10^9 cycle endurance limit column | Ultrasonic horns operate at 20 kHz continuously; 10^9 cycles = ~14 hours of operation. The traditional 10^7 "endurance limit" is non-conservative for this application. |
| Piezoelectric tensors in Voigt notation | Standard compact representation; directly consumable by ANSYS, COMSOL, and custom FEA solvers |
