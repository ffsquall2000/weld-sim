# Web Frontend + API Service Design

## Context

UltrasonicWeldMaster is a desktop PySide6 application for ultrasonic welding parameter calculation. It needs to be deployed on a Linux server (180.152.71.166) as an API service with a web frontend, integrating with two existing systems:

- **JIANKONG** (`/opt/jiankong`, port 9900): Flask + Gunicorn TCP weld data collector with fleet management
- **LAB** (`/opt/lab`, port 3000 + python_service on 8000): Next.js experiment tracking + FastAPI analysis service

## Architecture

```
Mac Browser
    |
    +-- LAB (port 3000)           -- existing Next.js frontend
    |     rewrites /api/weld-sim/* --> port 8001
    |     rewrites /api/fleet/*   --> port 9900 (existing)
    |
    +-- WeldSim (port 8001)       -- NEW: this project
          |-- FastAPI backend (calculation engine)
          |-- Vue 3 SPA (static files served by FastAPI)
          `-- Integration APIs for JIANKONG/LAB

Internal calls:
  JIANKONG (9900)  -->  LAB (3000)  -->  WeldSim API (8001)
  JIANKONG (9900)  ------------------->  WeldSim API (8001)
  LAB python_service (8000) ---------->  WeldSim API (8001)
```

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend | FastAPI + uvicorn | Matches LAB python_service pattern |
| Frontend | Vue 3 + Vite + TypeScript | Lightweight, strong i18n, good for dashboards |
| 3D Viewer | Three.js | WebGL-based CAD model rendering |
| Charts | ECharts | Rich industrial chart library |
| State | Pinia | Vue 3 standard state management |
| Styling | Tailwind CSS | Matches LAB frontend pattern |
| i18n | vue-i18n | Chinese/English |
| HTTP | axios | API client |
| Realtime | WebSocket | FEA progress streaming |

## Backend API Design

### Deployment: `/opt/weld-sim/` on server

```
/opt/weld-sim/
  ultrasonic_weld_master/     # core engine (symlink or copy)
  web/
    __init__.py
    app.py                    # FastAPI app entry
    config.py                 # env-based config
    dependencies.py           # Engine singleton DI
    routers/
      calculation.py          # POST /simulate
      materials.py            # GET /materials
      recipes.py              # GET /recipes
      reports.py              # POST /reports/export
      geometry.py             # POST /geometry/upload
      fea.py                  # WS /fea/run
      integration.py          # simplified API for JIANKONG/LAB
    schemas/
      calculation.py          # Pydantic request/response models
      materials.py
      reports.py
    services/
      engine_service.py       # Engine wrapper (thread-safe)
      file_service.py         # Upload/download management
  frontend/                   # Vue 3 SPA (build output in dist/)
  run.sh                      # startup script
  .env                        # configuration
  requirements.txt
```

### Endpoints

```
# Core calculation
POST   /api/v1/simulate                 # single calculation
POST   /api/v1/simulate/batch           # batch calculation
GET    /api/v1/simulate/schema/{app}    # input schema for dynamic forms

# Materials
GET    /api/v1/materials                # list all materials
GET    /api/v1/materials/{type}         # material properties
GET    /api/v1/materials/combination/{a}/{b}  # pair properties

# Recipes (history)
GET    /api/v1/recipes                  # list recent recipes
GET    /api/v1/recipes/{id}             # single recipe detail

# Reports
POST   /api/v1/reports/export           # generate report file
GET    /api/v1/reports/download/{name}  # download report

# Geometry (optional)
POST   /api/v1/geometry/upload          # upload STEP/PDF
POST   /api/v1/geometry/analyze         # run geometry classification
WS     /api/v1/fea/run                  # FEA with progress

# Health
GET    /api/v1/health                   # service status
```

### Key Schema: POST /api/v1/simulate

Request:
```json
{
  "application": "li_battery_tab",
  "upper_material_type": "Nickel 201",
  "upper_thickness_mm": 0.1,
  "upper_layers": 40,
  "lower_material_type": "Copper C110",
  "lower_thickness_mm": 0.3,
  "weld_width_mm": 3.0,
  "weld_length_mm": 25.0,
  "frequency_khz": 20.0,
  "max_power_w": 3500,
  "horn_type": "flat",
  "knurl_type": "cross_hatch",
  "knurl_pitch_mm": 1.2,
  "knurl_tooth_width_mm": 0.6,
  "knurl_depth_mm": 0.35,
  "anvil_type": "knurled",
  "booster_gain": 1.5
}
```

Response:
```json
{
  "recipe_id": "abc123def456",
  "application": "li_battery_tab",
  "parameters": {
    "amplitude_um": 28.5,
    "pressure_mpa": 0.45,
    "energy_j": 850,
    "weld_time_ms": 320,
    "force_n": 1200
  },
  "safety_window": {
    "amplitude_um": [22.0, 35.0],
    "pressure_mpa": [0.3, 0.6]
  },
  "risk_assessment": {
    "overweld_risk": "low",
    "perforation_risk": "medium",
    "thermal_risk": "low"
  },
  "validation": {
    "status": "PASS",
    "messages": ["All parameters within safe range"]
  },
  "recommendations": ["Consider reducing pressure for thinner foils"],
  "quality_estimate": { "energy_density_j_mm2": 11.3 }
}
```

## Frontend Design

### Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | Dashboard | Overview with recent calculations, quick actions |
| `/calculate` | CalculationWizard | 6-step wizard (same as desktop) |
| `/results/:id` | ResultsView | Parameter cards + risk badges + charts |
| `/reports/:id` | ReportView | Preview + export buttons |
| `/history` | HistoryView | Searchable table of past calculations |
| `/settings` | SettingsView | Theme, language, defaults |
| `/geometry` | GeometryView | 3D import + FEA (Three.js) |

### Wizard Steps (matching desktop)

1. Application Type (li_battery_tab/busbar/collector, general_metal)
2. Materials (upper: material+thickness+layers, lower: material+thickness)
3. Horn/Anvil/Knurl (type selectors + parameter inputs)
4. Geometry Import (drag-drop STEP/PDF, optional)
5. Equipment (frequency, power, booster, cylinder)
6. Constraints (safety windows, risk thresholds)

### Component Architecture

```
src/
  views/
    CalculationWizard.vue       # step container + navigation
    ResultsView.vue             # results display
    ReportView.vue              # report preview/export
    HistoryView.vue             # history table
    SettingsView.vue            # settings form
    GeometryView.vue            # 3D + FEA
  components/
    wizard/
      StepIndicator.vue         # progress circles
      ApplicationStep.vue       # step 1
      MaterialStep.vue          # step 2 + stack visualization
      HornAnvilStep.vue         # step 3 + SVG diagrams
      GeometryStep.vue          # step 4 (Three.js)
      EquipmentStep.vue         # step 5
      ConstraintStep.vue        # step 6
    charts/
      ParameterCard.vue         # key metric card with accent border
      RiskBadge.vue             # risk level indicator
      SafetyGauge.vue           # ECharts gauge for parameter ranges
      MaterialStack.vue         # SVG cross-section diagram
    three/
      ModelViewer.vue           # Three.js 3D viewport
      StressHeatmap.vue         # FEA result overlay
    common/
      AppLayout.vue             # sidebar + content layout
      ThemeToggle.vue           # dark/light switch
      LanguageSwitcher.vue      # zh/en
  stores/
    calculation.ts              # wizard state + API calls
    materials.ts                # material database cache
    history.ts                  # recipe history
    settings.ts                 # user preferences
  composables/
    useWebSocket.ts             # FEA progress WebSocket
    useTheme.ts                 # dark/light theme
  api/
    client.ts                   # axios instance with base URL
    simulation.ts               # /simulate endpoints
    materials.ts                # /materials endpoints
    reports.ts                  # /reports endpoints
  i18n/
    zh-CN.json                  # Chinese translations
    en.json                     # English translations
  styles/
    variables.css               # CSS custom properties
    dark.css                    # dark theme overrides
```

## LAB Integration

### next.config.ts addition

```typescript
// Add to rewrites array:
{ source: "/api/weld-sim/:path*", destination: "http://127.0.0.1:8001/api/v1/:path*" }
```

### LAB calling WeldSim

From LAB's Next.js API routes or frontend:
```typescript
const result = await fetch("/api/weld-sim/simulate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ application: "li_battery_tab", ... })
});
```

### JIANKONG calling WeldSim

From JIANKONG's Flask backend:
```python
import requests
result = requests.post("http://127.0.0.1:8001/api/v1/simulate", json={...}).json()
```

## Deployment

### Systemd service: `/etc/systemd/system/weld-sim.service`

```ini
[Unit]
Description=UltrasonicWeldMaster Simulation Service
After=network.target

[Service]
Type=simple
User=squall
WorkingDirectory=/opt/weld-sim
ExecStart=/opt/weld-sim/venv/bin/uvicorn web.app:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=5
Environment=PYTHONPATH=/opt/weld-sim

[Install]
WantedBy=multi-user.target
```

### Build and deploy script

```bash
# On dev machine: build frontend
cd frontend && npm run build

# Deploy to server
rsync -avz . squall@180.152.71.166:/opt/weld-sim/

# On server: setup
cd /opt/weld-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo systemctl enable --now weld-sim
```

## Testing Strategy

- Backend: pytest with TestClient (FastAPI)
- Frontend: Vitest + Vue Test Utils
- Integration: curl/httpie against running service
- E2E: Playwright (optional)
