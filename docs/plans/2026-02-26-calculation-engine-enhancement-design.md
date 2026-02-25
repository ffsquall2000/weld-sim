# Calculation Engine Enhancement — Design Document

**Goal:** Improve calculation accuracy by adding cylinder parameters, knurl pattern corrections, horn gain/mode, and material-knurl friction interaction. Add visual QPainter diagrams to the input wizard. Add a "Device Settings Card" to the output.

**Architecture:** Extend the existing 4-step wizard to 5 steps, add 4 QPainter diagram widgets, modify the 3-layer calculator to incorporate 6 correction factors, and add a Device Settings Card to the result panel.

**Tech Stack:** PySide6 QPainter, existing plugin calculator architecture

---

## Part 1: GUI Wizard Changes

### 5-Step Wizard Structure

| Step | Title | Content |
|------|-------|---------|
| 1 | Application Type | Unchanged |
| 2 | Materials | Existing fields + **MaterialStackWidget** (QPainter dynamic, updates with layers/thickness) |
| 3 | Horn/Anvil/Knurl | Weld geometry + Horn type + Anvil type + Knurl params + **3 diagram widgets** |
| 4 | Equipment | Cylinder (bore/pressure/efficiency) + Booster gain + Horn gain/mode |
| 5 | Constraints + Summary | Frequency, max power, input summary |

### Step 2 — Material Stack Diagram

New `MaterialStackWidget` (QPainter): cross-section view of upper foil stack + lower substrate. Dynamic — layer count and thickness update in real-time. Material-specific colors (Al=silver-gray, Cu=copper, Ni=light-gray, Steel=dark-gray).

### Step 3 — Horn/Anvil/Knurl (major expansion)

**Horn Type Section:**
- Types: Flat / Curved / Segmented / Blade / Heavy-duty / Branson DP / Custom
- Horn gain: user input (range 0.5–4.0)
- Mode: Longitudinal(L) / Flexural(F) / Torsional(T)
- Resonant frequency: user input (kHz)
- `HornTypeWidget` (QPainter): side-view of 7 preset horn shapes, switches on selection

**Anvil Type Section:**
- Types: Fixed Flat / Knurled / Contoured / Rotary / Multi-station / Resonant Cannon / Custom
- `AnvilTypeWidget` (QPainter): side-view of 7 preset anvil shapes, resonant anvil shows frequency annotation

**Knurl Parameters Section:**
- Knurl type: Linear / Cross-hatch / Diamond / Conical / Spherical / Custom
- Pitch: 0.3–3.0 mm (default 1.0)
- Tooth width: 0.1–2.0 mm (default 0.5)
- Depth: 0.1–1.0 mm (default 0.3)
- Vibration direction relative to linear knurl: Perpendicular / Parallel (only for linear type)
- Computed display: effective contact ratio (%), effective contact area (mm²)
- When "Custom" selected: user directly inputs contact ratio (%), hide pitch/tooth-width/depth
- `KnurlPatternWidget` (QPainter): top-view pattern + cross-section, dynamic with pitch/tooth-width parameters

**Weld Geometry Section (existing):**
- Width, Length, Nominal Area (unchanged)

### Step 4 — Equipment Parameters (new step)

**Cylinder Section:**
- Bore diameter: 20–100 mm (default 50)
- Working air pressure range: min/max bar (default 1.0–6.0)
- Mechanical efficiency: 0.70–1.00 (default 0.90)

**Booster Section:**
- Gain ratio: 1:1 / 1:1.5 / 1:2 / 1:2.5 / Custom
- Rated amplitude: 20–120 μm (max amplitude at this gain)

### QPainter Diagram Widgets (4 new)

1. **MaterialStackWidget** — material layer cross-section (dynamic: layers, thickness, material color)
2. **HornTypeWidget** — horn type side-view (7 preset shapes, selection switches display)
3. **AnvilTypeWidget** — anvil type side-view (7 preset shapes + resonant anvil with frequency label)
4. **KnurlPatternWidget** — knurl top-view + cross-section (6 pattern types, dynamic pitch/tooth-width)

All widgets use the same dark-industrial visual style as existing GaugeWidget (orange #ff9800 on near-black #0d1117).

---

## Part 2: Calculation Engine — 6 Corrections

### Correction 1: Effective Contact Area

```
contact_ratio = tooth_width / knurl_pitch                    # Linear
contact_ratio = (tooth_width / pitch)^2                      # Cross-hatch / Diamond
contact_ratio = π × (tip_radius)^2 / pitch^2                 # Conical / Spherical
contact_ratio = user_input                                    # Custom

effective_area = nominal_area × contact_ratio
```

All downstream calculations (power density, pressure, energy) use `effective_area` instead of `nominal_area`.

### Correction 2: Knurl-Material Effective Friction Coefficient

```
μ_ploughing = min(knurl_depth / (hardness_HV × 0.01) × 0.15, 0.2)

direction_coupling = {
    linear_perpendicular: 0.90,
    linear_parallel:      0.65,
    cross_hatch:          0.85,
    diamond:              0.85,
    conical:              0.80,
    spherical:            0.75,
}

μ_effective = (μ_base + μ_ploughing) × direction_coupling
```

- `μ_base` from `materials.yaml` combinations (existing)
- `hardness_HV` from `materials.yaml` material properties (existing, currently unused)
- `knurl_depth` from user input (new)
- `direction_coupling` from knurl type + vibration direction (new)

### Correction 3: Cylinder Force ↔ Air Pressure Back-calculation

```
cylinder_area_mm2 = π × (bore_mm / 2)^2
target_force_N = target_pressure_mpa × effective_area_mm2
required_air_bar = target_force_N / (cylinder_area_mm2 × 0.1 × efficiency)

# Reverse validation
max_force = max_air_bar × cylinder_area_mm2 × 0.1 × efficiency
achievable_pressure_range = [min_force / effective_area, max_force / effective_area]
```

Warning if required_air_bar exceeds user-specified air pressure range.

### Correction 4: Amplitude Chain (Booster × Horn Gain)

```
system_gain = booster_gain × horn_gain
actual_amplitude_um = target_amplitude_um  (from layer-2 empirical)
amplitude_percent = actual_amplitude_um / rated_amplitude_um × 100
```

Use `actual_amplitude_um` in the physics model (power density formula). Output `amplitude_percent` in Device Settings Card.

### Correction 5: Mode Warning

```
freq_diff = abs(horn_resonant_freq_khz - working_freq_khz)
if freq_diff > 0.5:
    warning("Horn resonant frequency deviation too large: ±X.X kHz")
if mode != "longitudinal":
    warning("Non-longitudinal mode may affect weld quality")
```

Added to recommendations list.

### Correction 6: Device Settings Card Output

New QGroupBox in ResultPanel — "Device Settings Card":

```
══ Device Settings Card ══
Air Pressure:     3.2 bar
Amplitude:        30 μm (42%)
Weld Time:        0.35 s
Delay Time:       0.20 s
Hold Time:        0.50 s
Trigger Mode:     Energy Mode 45 J
──────────────────────────
Effective Area:   62.5 mm²
Effective μ:      0.42
Actual Pressure:  0.60 MPa
System Gain:      2.25x
══════════════════════════
```

---

## Data Model Changes

### models.py — SonotrodeInfo update

```python
@dataclass
class SonotrodeInfo:
    sonotrode_type: str = "flat"        # flat/curved/segmented/blade/heavy/branson_dp/custom
    horn_gain: float = 1.0              # horn amplification ratio
    mode: str = "longitudinal"          # longitudinal/flexural/torsional
    resonant_freq_khz: float = 20.0     # horn resonant frequency
    knurl_type: str = "linear"          # linear/cross_hatch/diamond/conical/spherical/custom
    knurl_pitch_mm: float = 1.0
    knurl_tooth_width_mm: float = 0.5
    knurl_depth_mm: float = 0.3
    knurl_direction: str = "perpendicular"  # perpendicular/parallel (for linear only)
    custom_contact_ratio: float = 0.5   # used when knurl_type == "custom"
    contact_width_mm: float = 5.0
    contact_length_mm: float = 25.0
```

### models.py — New AnvilInfo

```python
@dataclass
class AnvilInfo:
    anvil_type: str = "fixed_flat"      # fixed_flat/knurled/contoured/rotary/multi_station/resonant/custom
    resonant_freq_khz: float = 0.0      # only for resonant type
```

### models.py — New CylinderInfo

```python
@dataclass
class CylinderInfo:
    bore_mm: float = 50.0
    min_air_bar: float = 1.0
    max_air_bar: float = 6.0
    efficiency: float = 0.90
```

### models.py — New BoosterInfo

```python
@dataclass
class BoosterInfo:
    gain_ratio: float = 1.5            # 1:1, 1:1.5, 1:2, 1:2.5
    rated_amplitude_um: float = 70.0   # max amplitude at this gain
```

### WeldInputs — add new fields

```python
@dataclass
class WeldInputs:
    # ... existing fields ...
    sonotrode: Optional[SonotrodeInfo] = None   # existing, now populated
    anvil: Optional[AnvilInfo] = None            # new
    cylinder: Optional[CylinderInfo] = None      # new
    booster: Optional[BoosterInfo] = None        # new
```

---

## i18n

All new strings must use `self.tr()` and be added to `translations/app_zh_CN.ts`. Estimated ~40 new translatable strings.

---

## Files Affected

### Create:
- `ultrasonic_weld_master/gui/widgets/material_stack_widget.py`
- `ultrasonic_weld_master/gui/widgets/horn_type_widget.py`
- `ultrasonic_weld_master/gui/widgets/anvil_type_widget.py`
- `ultrasonic_weld_master/gui/widgets/knurl_pattern_widget.py`

### Modify:
- `ultrasonic_weld_master/core/models.py` — add AnvilInfo, CylinderInfo, BoosterInfo; update SonotrodeInfo, WeldInputs
- `ultrasonic_weld_master/gui/panels/input_wizard.py` — 5-step wizard, new fields, diagram widgets
- `ultrasonic_weld_master/gui/panels/result_panel.py` — Device Settings Card QGroupBox
- `ultrasonic_weld_master/plugins/li_battery/calculator.py` — 6 corrections
- `ultrasonic_weld_master/plugins/li_battery/physics.py` — effective friction model
- `ultrasonic_weld_master/plugins/general_metal/calculator.py` — same corrections
- `ultrasonic_weld_master/plugins/li_battery/plugin.py` — pass new input fields
- `ultrasonic_weld_master/plugins/general_metal/plugin.py` — pass new input fields
- `translations/app_zh_CN.ts` — ~40 new strings
- `UltrasonicWeldMaster.spec` — add new widget hiddenimports
