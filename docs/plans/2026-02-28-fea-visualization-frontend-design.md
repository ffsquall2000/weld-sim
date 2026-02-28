# FEA Scientific Visualization Frontend Design

> **Goal:** Upgrade the frontend from Canvas 2D to a production-grade WebGL scientific visualization system comparable to ANSYS Workbench, supporting 200K+ node meshes with real-time interaction.

**Tech Stack:**
- Three.js (WebGL 3D rendering)
- ECharts 6 + vue-echarts 8 (2D scientific charts)
- WebSocket (real-time analysis progress)
- WebWorker (off-thread mesh processing)
- Vue 3 Composition API + TypeScript

---

## Module 1: WebGL 3D Engine (`FEAViewer.vue`)

### Replace Canvas 2D ThreeViewer with Three.js WebGL

**Architecture:**
```
FEAViewer.vue
├── Three.js Scene
│   ├── WebGLRenderer (antialias, preserveDrawingBuffer)
│   ├── PerspectiveCamera + OrbitControls
│   ├── MeshGroup (BufferGeometry + ShaderMaterial)
│   ├── EdgesGroup (wireframe overlay)
│   ├── ClippingPlanes[] (cross-section)
│   ├── IsosurfaceGroup (marching tet)
│   ├── ArrowGroup (vector display)
│   └── Lights (ambient + directional + hemisphere)
├── ColorBar.vue (gradient legend with min/max labels)
├── ViewerToolbar.vue (display mode controls)
└── NodeInfoPopup.vue (hover/click node data)
```

**Key Implementation:**
- `THREE.BufferGeometry` with Float32Array for positions, normals, colors
- Custom GLSL ShaderMaterial for scalar field coloring:
  - Vertex shader: pass scalar value as varying
  - Fragment shader: sample 1D colormap texture (jet/viridis/rainbow)
- Support 200K+ nodes via:
  - Indexed BufferGeometry (shared vertices)
  - GPU-side colormap lookup (no CPU color computation)
  - Frustum culling (built into Three.js)
  - Optional LOD: EdgesGeometry at distance > threshold

**Visualization Modes:**
1. **Geometry** — wireframe / solid / transparent / wireframe+solid
2. **Scalar Cloud** — Von Mises stress, displacement magnitude, temperature, safety factor
3. **Vector Display** — displacement arrows (ArrowHelper), configurable scale
4. **Deformed Shape** — u_deformed = u_original + scale * displacement
5. **Cross-Section** — GPU clipping planes (X/Y/Z), interactive drag
6. **Isosurface** — Marching Tetrahedra in WebWorker, adjustable threshold slider

**Colormap System:**
- 5 built-in colormaps: jet, viridis, coolwarm, rainbow, grayscale
- Generated as 256×1 textures, uploaded to GPU once
- ColorBar component shows gradient + min/max/unit labels
- User can switch colormap via dropdown

**Interaction:**
- OrbitControls: left-drag rotate, right-drag pan, scroll zoom
- Raycaster: hover shows node ID + scalar value in tooltip
- Click: selects node, shows full stress tensor in side panel
- Double-click: center view on clicked point

---

## Module 2: Modal Animation System

**Animation equation:** `position(t) = base_position + amplitude_scale × mode_shape × sin(2π × phase / 360)`

**Controls:**
```
┌─────────────────────────────────────────────┐
│ ◄◄  ▶/⏸  ►►  │━━━━━●━━━━━│  🔄 Loop  ⚡1x  │
│ Phase: 0° ─────────────── 360°              │
│ Amplitude: ━━━━━●━━━━━━━ (deformation scale) │
│ Mode: [▼ Mode 1: 19,856 Hz (longitudinal)]  │
└─────────────────────────────────────────────┘
```

**Implementation:**
- Store all mode shapes in Float32Array buffers
- On each animation frame: update `geometry.attributes.position` directly
- Recompute normals per-frame for correct lighting on deformed shape
- Auto-scale: `default_amplitude = 0.05 * bbox_diagonal / max_displacement`
- Color mode during animation: displacement magnitude (real-time update)
- Mode selector dropdown shows: frequency, mode type (longitudinal/flexural/torsional), effective mass ratio

**Performance for 200K nodes:**
- BufferAttribute with `needsUpdate = true` (GPU re-upload per frame)
- At 200K vertices × 3 floats × 4 bytes = 2.4 MB per frame — well within GPU bandwidth
- Target: 60 FPS animation

---

## Module 3: ECharts Scientific Charts

### Chart Components

**1. FRFChart.vue — Harmonic Response (Frequency Response Function)**
- Page: AcousticView
- Type: Line chart, log Y-axis
- X: Frequency (Hz), Y: Amplitude (μm)
- Features: zoom, 3dB bandwidth markers, Q-factor annotation, resonance peak labels
- Data: `HarmonicResult.frequencies_hz`, `HarmonicResult.displacements`

**2. GainChart.vue — Gain vs Frequency**
- Page: AcousticView
- Type: Line chart
- X: Frequency (Hz), Y: Gain ratio
- Features: target frequency vertical line, gain > 1 region highlighted

**3. ModalBarChart.vue — Modal Frequencies**
- Page: GeometryView
- Type: Horizontal bar chart
- Each bar = one mode, colored by type (longitudinal=blue, flexural=orange, torsional=green)
- Target frequency shown as vertical dashed line
- Parasitic modes highlighted in red

**4. SafetyGauge.vue — Parameter Safety Dashboard**
- Page: ResultsView
- Type: ECharts gauge
- Shows: amplitude, pressure, energy, power within safe ranges
- Colors: green (safe), yellow (warning), red (danger)

**5. SNChart.vue — Fatigue S-N Curve**
- Page: New FatigueView
- Type: Log-log line chart
- X: Cycles to failure (N), Y: Stress amplitude (MPa)
- Shows: material S-N curve, operating point marker, safety factor annotation

**6. ConvergenceChart.vue — Mesh Convergence**
- Type: Line chart with error bars
- X: DOF count, Y: Target quantity (frequency)
- Shows: Richardson extrapolation line, converged region shading

**7. ParetoChart.vue — Knurl Optimization Pareto Front**
- Page: KnurlDesignView
- Type: Scatter plot
- X: Energy coupling, Y: Material damage
- Pareto-optimal points highlighted

---

## Module 4: Real-Time Analysis Progress (WebSocket)

### Backend: FastAPI WebSocket Endpoint

```python
# web/routers/ws.py
@router.websocket("/api/v1/ws/analysis/{task_id}")
async def analysis_progress(websocket: WebSocket, task_id: str):
    await websocket.accept()
    async for progress in analysis_manager.subscribe(task_id):
        await websocket.send_json(progress)
```

**Progress message format:**
```json
{
  "task_id": "abc123",
  "step": 2,
  "total_steps": 6,
  "step_name": "Eigenvalue solve",
  "progress_pct": 65,
  "message": "Found 12/20 modes",
  "elapsed_s": 8.3,
  "estimated_remaining_s": 4.5
}
```

### Frontend: ProgressOverlay.vue

- Full-screen semi-transparent overlay during analysis
- Animated progress bar with percentage
- Step indicators (mesh → assemble → solve → postprocess → complete)
- Real-time message updates
- Estimated remaining time
- Cancel button (sends cancel signal via WebSocket)
- Auto-dismiss on completion, triggers result loading

### Steps tracked:
1. Mesh generation (Gmsh)
2. Matrix assembly (K, M)
3. Eigenvalue solve / Linear solve
4. Post-processing (stress recovery, mode classification)
5. Gain chain computation
6. Complete — results ready

---

## Module 5: Cross-Section & Isosurface

### Cross-Section (Clipping Planes)

**Three.js native clipping:**
```typescript
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0)
renderer.clippingPlanes = [clipPlane]
```

**UI Controls:**
- Three toggle buttons: Clip X / Clip Y / Clip Z
- Slider to move clip plane position along selected axis
- Show/hide cap fill (stencil buffer technique for solid cross-section)
- Cross-section outline highlight

**Performance:** Zero GPU cost — hardware clipping is native to WebGL.

### Isosurface (Marching Tetrahedra)

**Algorithm:** For each TET10 element, evaluate scalar at vertices, extract triangles where scalar crosses threshold value.

**Implementation:**
- Marching Tetrahedra lookup table (16 cases per tet)
- Run in WebWorker to avoid UI blocking
- Input: element connectivity + nodal scalar values + threshold
- Output: triangle vertices + interpolated scalar values
- Render as separate transparent BufferGeometry

**UI Controls:**
- Threshold slider (min to max of scalar field)
- Opacity slider
- Multiple isosurfaces: add/remove with different thresholds and colors

---

## New API Endpoints Required

### WebSocket endpoint
- `WS /api/v1/ws/analysis/{task_id}` — real-time progress

### Mesh data endpoint (optimized for frontend)
- `GET /api/v1/mesh/{task_id}/geometry` — vertices + faces as binary (ArrayBuffer)
- `GET /api/v1/mesh/{task_id}/scalars?field=von_mises` — scalar field as Float32Array
- `GET /api/v1/mesh/{task_id}/modes/{mode_index}` — mode shape as Float32Array

Using binary transfer (ArrayBuffer) instead of JSON for mesh data — critical for 200K nodes:
- JSON: ~50 MB for 200K nodes × 3 coords → slow parse
- Binary Float32: ~2.4 MB → fast, direct to GPU

---

## File Structure

```
frontend/src/
├── components/
│   ├── viewer/
│   │   ├── FEAViewer.vue           # Main 3D viewport (Three.js)
│   │   ├── ColorBar.vue            # Gradient legend
│   │   ├── ViewerToolbar.vue       # Display mode controls
│   │   ├── AnimationControls.vue   # Mode animation playback
│   │   ├── ClippingControls.vue    # Cross-section UI
│   │   ├── IsosurfaceControls.vue  # Isosurface threshold UI
│   │   └── NodeInfoPopup.vue       # Hover/click node data
│   ├── charts/
│   │   ├── FRFChart.vue            # Harmonic response line chart
│   │   ├── GainChart.vue           # Gain vs frequency
│   │   ├── ModalBarChart.vue       # Modal frequency bars
│   │   ├── SafetyGauge.vue         # Parameter safety gauges
│   │   ├── SNChart.vue             # S-N fatigue curve
│   │   ├── ConvergenceChart.vue    # Mesh convergence
│   │   └── ParetoChart.vue         # Knurl Pareto front
│   └── progress/
│       └── ProgressOverlay.vue     # WebSocket analysis progress
├── composables/
│   ├── useThreeScene.ts            # Three.js scene setup/teardown
│   ├── useColormap.ts              # Colormap texture generation
│   ├── useMeshLoader.ts            # Binary mesh data loading
│   ├── useAnimation.ts             # Mode shape animation loop
│   ├── useClipping.ts              # Clipping plane management
│   ├── useIsosurface.ts            # Marching tet worker interface
│   └── useAnalysisProgress.ts      # WebSocket progress subscription
├── workers/
│   ├── meshProcessor.worker.ts     # Off-thread mesh processing
│   └── isosurface.worker.ts        # Marching tetrahedra computation
└── shaders/
    ├── colormap.vert.glsl          # Vertex shader (scalar → varying)
    └── colormap.frag.glsl          # Fragment shader (colormap lookup)
```

---

## Performance Targets

| Metric | Target | How |
|--------|--------|-----|
| 200K node mesh load | < 2s | Binary transfer + direct BufferAttribute |
| 60 FPS rotation | ✓ | GPU-side rendering, no CPU per-frame work |
| 60 FPS animation | ✓ | BufferAttribute update only (~2.4 MB/frame) |
| Colormap switch | < 100ms | GPU texture swap, no re-upload geometry |
| Isosurface compute | < 1s | WebWorker parallelism |
| Cross-section | instant | Hardware clipping planes |

---

## Migration Plan

1. Install Three.js: `npm install three @types/three`
2. Build FEAViewer.vue incrementally alongside existing ThreeViewer.vue
3. Route by route, replace ThreeViewer with FEAViewer
4. Remove old Canvas 2D code after all views migrated
5. Activate ECharts in each view (already installed, just need components)
6. Add WebSocket progress overlay
7. Add binary mesh API endpoints
