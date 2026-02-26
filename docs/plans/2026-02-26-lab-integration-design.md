# LAB-WeldSim-JIANKONG Integration Design

**Date**: 2026-02-26
**Status**: Approved
**Approach**: LAB as central controller (Plan A)

## Overview

Integrate UltrasonicWeldMaster (weld-sim) simulation capabilities into the LAB experiment management platform. LAB serves as the single entry point for all operations. JIANKONG continues as the data collection layer. weld-sim provides computation via its existing REST API.

## Architecture

```
                    LAB (Main Controller)
                    Next.js + Prisma + AI
                    ┌─────────────────────────────────────┐
                    │                                     │
                    │  Experiment Management (existing)    │
                    │  Weld Session Tracking (existing)    │
                    │  AI Analysis + LLM Reports (existing)│
                    │                                     │
                    │  ┌─────────────────────────────┐    │
                    │  │ NEW: Simulation Integration  │    │
                    │  │  - Parameter mapping         │    │
                    │  │  - Run simulation via API    │    │
                    │  │  - Store results             │    │
                    │  │  - Compare actual vs sim     │    │
                    │  │  - Embed in reports          │    │
                    │  └─────────────────────────────┘    │
                    └──────┬──────────────┬───────────────┘
                           │              │
              ┌────────────▼──┐    ┌──────▼──────────┐
              │  JIANKONG     │    │  weld-sim       │
              │  :9900        │    │  :8001          │
              │  (no changes) │    │  (no changes)   │
              │               │    │                 │
              │  TCP collect  │    │  FastAPI         │
              │  → LAB ingest │    │  Simulate API   │
              └───────────────┘    │  Geometry API   │
                                   │  FEA API        │
                                   └─────────────────┘
```

## Data Flow

### Flow 1: Pre-experiment Simulation (predict)

```
Engineer creates experiment in LAB
  → fills material info (ExperimentMaterial)
  → clicks "Run Simulation"
  → LAB API maps material fields to weld-sim SimulateRequest
  → POST http://127.0.0.1:8001/api/v1/simulate
  → Response: recipe with recommended params + safety windows + risk
  → Store in AnalysisResult (type: "simulation")
  → Display in experiment detail page
```

### Flow 2: Post-experiment Comparison (verify)

```
JIANKONG collects weld data during experiment
  → POST /api/ingest/weld-sessions → LAB
  → LAB auto-resolves to experiment (existing)
  → WeldSession + WeldingParameter + WeldingResult created

Engineer clicks "Generate Report"
  → LAB fetches simulation AnalysisResult (type: "simulation")
  → LAB computes actual averages from WeldingParameter/Result
  → Compares: recommended vs actual for each parameter
  → Deviation analysis: within/outside safety window
  → New report section: "Simulation Comparison"
```

## Changes Required

### LAB Backend (3 files to modify/create)

#### 1. NEW: `src/app/api/experiments/[id]/simulate/route.ts`

POST handler that:
- Reads experiment materials from DB (ExperimentMaterial where position="upper"/"lower")
- Reads latest WeldingParameter for equipment params (frequency, power)
- Maps to weld-sim SimulateRequest format:

```typescript
// Material mapping: ExperimentMaterial → SimulateRequest
{
  application: deriveApplication(materials),  // "li_battery_tab" etc.
  upper_material_type: upper.material,        // e.g. "Nickel 201"
  upper_thickness_mm: upper.thickness,
  upper_layers: 1,                            // default or from form
  lower_material_type: lower.material,
  lower_thickness_mm: lower.thickness,
  weld_width_mm: upper.width || 3.0,
  weld_length_mm: upper.length || 25.0,
  frequency_khz: params.frequency || 20.0,
  max_power_w: params.power || 3500,
  // Optional fields from form overrides
  horn_type, knurl_type, booster_gain, etc.
}
```

- Calls `http://127.0.0.1:8001/api/v1/simulate` with mapped request
- Stores full response in `AnalysisResult`:
  - type: `"simulation"`
  - title: "Simulation: {application}"
  - summary: one-line summary of key params
  - details: full SimulateResponse JSON
  - confidence: 0.8 (fixed, indicates simulation-grade)

#### 2. MODIFY: `src/lib/experiment-report.ts`

Add section to `buildExperimentReportModel()`:

```typescript
// After existing sections, before recommendations:
const simAnalysis = analyses.find(a => a.type === "simulation");
if (simAnalysis) {
  const simParams = simAnalysis.details;
  const actualAvg = computeActualAverages(parameters);

  report.simulationComparison = {
    recommended: {
      amplitude_um: simParams.parameters.amplitude_um,
      pressure_mpa: simParams.parameters.pressure_mpa,
      energy_j: simParams.parameters.energy_j,
      time_ms: simParams.parameters.time_ms,
    },
    actual: actualAvg,
    safetyWindow: simParams.safety_window,
    deviations: computeDeviations(simParams, actualAvg),
    riskAssessment: simParams.risk_assessment,
  };
}
```

Markdown output adds section "## 6. Simulation vs Actual Comparison":
- Table: Parameter | Recommended | Actual | Deviation | In Window?
- Risk badges from simulation
- Recommendations from simulation

#### 3. MODIFY: `src/app/api/experiments/[id]/report/route.ts`

Pass simulation comparison data to report builder. Minimal change - the report builder already reads from AnalysisResult.

### LAB Frontend (1 new component)

#### 4. NEW: Simulation panel in experiment detail page

Add to `src/app/experiments/[id]/page.tsx`:
- New tab or section: "Simulation Analysis"
- Form with pre-filled fields from experiment materials
- "Run Simulation" button → POST /api/experiments/{id}/simulate
- Results display: parameter cards, safety windows, risk badges
- If weld sessions exist: comparison table (sim vs actual)

### weld-sim Changes: NONE

The existing API at `:8001/api/v1/simulate` already handles everything needed. LAB calls it through the Next.js rewrite proxy.

### JIANKONG Changes: NONE

The existing ingest flow is already working.

## Material Mapping Logic

The key challenge is mapping LAB's material names to weld-sim's material_type identifiers:

```typescript
const MATERIAL_MAP: Record<string, string> = {
  // LAB ExperimentMaterial.material → weld-sim upper/lower_material_type
  "铜": "Copper C110",
  "镍": "Nickel 201",
  "铝": "Al",
  "铜箔": "Copper C110",
  "镍片": "Nickel 201",
  "铝箔": "Al",
  // English variants
  "copper": "Copper C110",
  "nickel": "Nickel 201",
  "aluminum": "Al",
  // Direct pass-through for known weld-sim types
  "Copper C110": "Copper C110",
  "Nickel 201": "Nickel 201",
  "Al": "Al",
  "Cu": "Cu",
};

function deriveApplication(materials: ExperimentMaterial[]): string {
  // If any material mentions "battery" or "tab" → li_battery_tab
  // If mentions "busbar" → li_battery_busbar
  // Default → li_battery_tab (most common use case)
  return "li_battery_tab";
}
```

## AnalysisResult Storage Format

```json
{
  "type": "simulation",
  "title": "Weld Simulation - li_battery_tab",
  "summary": "Recommended: 43.6 um amplitude, 10.9 MPa, 764.6 J, 567 ms",
  "confidence": 0.8,
  "details": {
    "recipe_id": "57de9d1bc594",
    "application": "li_battery_tab",
    "parameters": { ... },
    "safety_window": { ... },
    "risk_assessment": { ... },
    "recommendations": [ ... ],
    "validation": { ... },
    "request_inputs": { ... }
  }
}
```

## Auth Consideration

The LAB proxy (`/api/weld-sim/*`) goes through Next.js rewrites. Since the simulate call is made server-side from LAB's API route handler (not from the browser), it bypasses any client-side auth. The server-side fetch to `127.0.0.1:8001` is internal and auth-free.

## Testing Strategy

1. Unit test the material mapping function
2. Integration test: create experiment → run simulate → verify AnalysisResult stored
3. Report test: experiment with simulation + weld sessions → verify comparison section in markdown
4. E2E: full flow from experiment creation through report generation

## Out of Scope

- Real-time simulation during welding (latency too high)
- Automatic parameter push to welding machines
- JIANKONG modification
- weld-sim modification
