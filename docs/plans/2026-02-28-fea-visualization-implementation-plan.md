# FEA Scientific Visualization Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the web frontend from Canvas 2D to a production-grade WebGL scientific visualization system with Three.js 3D engine, ECharts charts, modal animation, cross-section/isosurface rendering, and real-time WebSocket analysis progress.

**Architecture:** Vue 3 + Three.js (WebGL 3D) + ECharts 6 (2D charts) + WebSocket (progress). The FEAViewer component replaces the existing Canvas 2D ThreeViewer. Composables encapsulate Three.js/WebSocket logic. WebWorkers handle off-thread mesh processing. Binary mesh API endpoints avoid JSON overhead for 200K+ node meshes.

**Tech Stack:** Vue 3 / TypeScript 5.9 / Vite 7 / Three.js / ECharts 6 / vue-echarts 8 / WebSocket / WebWorker / GLSL shaders / Tailwind CSS 4

**Design Document:** `docs/plans/2026-02-28-fea-visualization-frontend-design.md`

---

## Prerequisites

```bash
cd frontend
npm install three @types/three
# Verify: node -e "require('three')" should not error
```

---

## Phase A: Three.js Foundation (Tasks 1-4)

**Goal:** Core Three.js composables and basic FEAViewer that can render a TET10 mesh with rotation/zoom/pan.

**Dependency graph:**
```
Task 1 (useThreeScene) ──► Task 3 (FEAViewer basic)
Task 2 (useColormap)   ──► Task 4 (scalar coloring)
Task 3 (FEAViewer basic) ──► Task 4 (scalar coloring)
```

---

### Task 1: Three.js Scene Composable

**Files:**
- Create: `frontend/src/composables/useThreeScene.ts`

**Step 1: Write the composable**

```typescript
// frontend/src/composables/useThreeScene.ts
import { ref, onMounted, onBeforeUnmount, type Ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

export interface ThreeSceneOptions {
  antialias?: boolean
  background?: string
  enableClipping?: boolean
}

export function useThreeScene(
  containerRef: Ref<HTMLDivElement | null>,
  options: ThreeSceneOptions = {},
) {
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000)
  let renderer: THREE.WebGLRenderer | null = null
  let controls: OrbitControls | null = null
  let animationId: number | null = null
  const isReady = ref(false)

  // Clipping planes (for cross-section feature)
  const clippingPlanes: THREE.Plane[] = []

  function init() {
    const container = containerRef.value
    if (!container) return

    const { antialias = true, background = '#1a1a2e', enableClipping = false } = options

    renderer = new THREE.WebGLRenderer({
      antialias,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.setClearColor(new THREE.Color(background))
    if (enableClipping) {
      renderer.localClippingEnabled = true
    }
    container.appendChild(renderer.domElement)

    // Camera
    camera.aspect = container.clientWidth / container.clientHeight
    camera.position.set(0, 0.15, 0.3)
    camera.updateProjectionMatrix()

    // Controls
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(1, 2, 3)
    scene.add(dirLight)
    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x0f0e0d, 0.3)
    scene.add(hemiLight)

    isReady.value = true
  }

  function animate(customUpdate?: () => void) {
    function loop() {
      animationId = requestAnimationFrame(loop)
      controls?.update()
      if (customUpdate) customUpdate()
      if (renderer) renderer.render(scene, camera)
    }
    loop()
  }

  function resize() {
    const container = containerRef.value
    if (!container || !renderer) return
    const w = container.clientWidth
    const h = container.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }

  function fitToObject(object: THREE.Object3D) {
    const box = new THREE.Box3().setFromObject(object)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z)
    const dist = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360))
    camera.position.copy(center).add(new THREE.Vector3(0, dist * 0.3, dist * 1.2))
    controls?.target.copy(center)
    controls?.update()
  }

  function dispose() {
    if (animationId !== null) cancelAnimationFrame(animationId)
    controls?.dispose()
    renderer?.dispose()
    scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose()
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
        else obj.material.dispose()
      }
    })
    if (renderer?.domElement.parentElement) {
      renderer.domElement.parentElement.removeChild(renderer.domElement)
    }
    isReady.value = false
  }

  let resizeObserver: ResizeObserver | null = null

  onMounted(() => {
    init()
    resizeObserver = new ResizeObserver(() => resize())
    if (containerRef.value) resizeObserver.observe(containerRef.value)
  })

  onBeforeUnmount(() => {
    resizeObserver?.disconnect()
    dispose()
  })

  return {
    scene,
    camera,
    renderer: renderer as THREE.WebGLRenderer,
    controls: controls as OrbitControls,
    clippingPlanes,
    isReady,
    animate,
    resize,
    fitToObject,
    dispose,
  }
}
```

**Step 2: Verify TypeScript compiles**

Run: `cd frontend && npx vue-tsc --noEmit 2>&1 | head -20`

**Step 3: Commit**

```bash
git add frontend/src/composables/useThreeScene.ts
git commit -m "feat(frontend): add Three.js scene composable with OrbitControls and lighting"
```

---

### Task 2: Colormap System Composable

**Files:**
- Create: `frontend/src/composables/useColormap.ts`
- Create: `frontend/src/shaders/colormap.vert.glsl`
- Create: `frontend/src/shaders/colormap.frag.glsl`

**Step 1: Create GLSL shaders**

```glsl
// frontend/src/shaders/colormap.vert.glsl
varying float vScalar;
varying vec3 vNormal;
varying vec3 vViewPosition;
attribute float scalar;

void main() {
  vScalar = scalar;
  vNormal = normalize(normalMatrix * normal);
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  vViewPosition = -mvPosition.xyz;
  gl_Position = projectionMatrix * mvPosition;
}
```

```glsl
// frontend/src/shaders/colormap.frag.glsl
uniform sampler2D colormapTexture;
uniform float scalarMin;
uniform float scalarMax;
uniform float opacity;
varying float vScalar;
varying vec3 vNormal;
varying vec3 vViewPosition;

void main() {
  // Normalize scalar to [0, 1]
  float t = clamp((vScalar - scalarMin) / (scalarMax - scalarMin + 1e-10), 0.0, 1.0);

  // Sample colormap
  vec3 color = texture2D(colormapTexture, vec2(t, 0.5)).rgb;

  // Simple Lambertian shading
  vec3 lightDir = normalize(vec3(1.0, 2.0, 3.0));
  float diffuse = max(dot(normalize(vNormal), lightDir), 0.0);
  vec3 shadedColor = color * (0.3 + 0.7 * diffuse);

  gl_FragColor = vec4(shadedColor, opacity);
}
```

**Step 2: Create colormap composable**

```typescript
// frontend/src/composables/useColormap.ts
import * as THREE from 'three'
import vertexShader from '@/shaders/colormap.vert.glsl?raw'
import fragmentShader from '@/shaders/colormap.frag.glsl?raw'

export type ColormapName = 'jet' | 'viridis' | 'coolwarm' | 'rainbow' | 'grayscale'

// Generate colormap lookup tables (256 RGB values)
function jetLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let r: number, g: number, b: number
    if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4 }
    else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1 }
    else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4 }
    else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0 }
    else { r = 1 - (t - 0.875) * 4; g = 0; b = 0 }
    data[i * 4] = Math.round(r * 255)
    data[i * 4 + 1] = Math.round(g * 255)
    data[i * 4 + 2] = Math.round(b * 255)
    data[i * 4 + 3] = 255
  }
  return data
}

function viridisLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  // Simplified viridis: purple → teal → yellow
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const r = Math.round(255 * (0.267 + t * (0.993 - 0.267)))
    const g = Math.round(255 * (0.004 + t * 0.906 * (1 - 0.4 * (t - 0.5) ** 2)))
    const b = Math.round(255 * (0.329 + 0.5 * Math.sin(Math.PI * t) * (1 - t)))
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255
  }
  return data
}

function coolwarmLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    // Blue → White → Red
    const r = Math.round(255 * Math.min(1, 0.2 + 1.6 * t))
    const g = Math.round(255 * (1 - 2 * Math.abs(t - 0.5)))
    const b = Math.round(255 * Math.min(1, 1.8 - 1.6 * t))
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255
  }
  return data
}

function rainbowLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const h = t * 300 / 360 // hue 0-300 degrees
    const s = 1, l = 0.5
    // HSL to RGB
    const c = (1 - Math.abs(2 * l - 1)) * s
    const x = c * (1 - Math.abs((h * 6) % 2 - 1))
    const m = l - c / 2
    let r = 0, g = 0, b = 0
    const h6 = h * 6
    if (h6 < 1) { r = c; g = x }
    else if (h6 < 2) { r = x; g = c }
    else if (h6 < 3) { g = c; b = x }
    else if (h6 < 4) { g = x; b = c }
    else if (h6 < 5) { r = x; b = c }
    else { r = c; b = x }
    data[i * 4] = Math.round((r + m) * 255)
    data[i * 4 + 1] = Math.round((g + m) * 255)
    data[i * 4 + 2] = Math.round((b + m) * 255)
    data[i * 4 + 3] = 255
  }
  return data
}

function grayscaleLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    data[i * 4] = i; data[i * 4 + 1] = i; data[i * 4 + 2] = i; data[i * 4 + 3] = 255
  }
  return data
}

const LUT_GENERATORS: Record<ColormapName, () => Uint8Array> = {
  jet: jetLUT,
  viridis: viridisLUT,
  coolwarm: coolwarmLUT,
  rainbow: rainbowLUT,
  grayscale: grayscaleLUT,
}

export function useColormap() {
  const textureCache = new Map<ColormapName, THREE.DataTexture>()

  function getTexture(name: ColormapName): THREE.DataTexture {
    if (textureCache.has(name)) return textureCache.get(name)!
    const lut = LUT_GENERATORS[name]()
    const tex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat)
    tex.needsUpdate = true
    tex.minFilter = THREE.LinearFilter
    tex.magFilter = THREE.LinearFilter
    textureCache.set(name, tex)
    return tex
  }

  function createColormapMaterial(
    colormapName: ColormapName = 'jet',
    scalarMin = 0,
    scalarMax = 1,
    opacity = 1.0,
  ): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        colormapTexture: { value: getTexture(colormapName) },
        scalarMin: { value: scalarMin },
        scalarMax: { value: scalarMax },
        opacity: { value: opacity },
      },
      transparent: opacity < 1.0,
      side: THREE.DoubleSide,
      clipping: true,
    })
  }

  function updateMaterial(
    material: THREE.ShaderMaterial,
    opts: { colormap?: ColormapName; min?: number; max?: number; opacity?: number },
  ) {
    if (opts.colormap) material.uniforms.colormapTexture.value = getTexture(opts.colormap)
    if (opts.min !== undefined) material.uniforms.scalarMin.value = opts.min
    if (opts.max !== undefined) material.uniforms.scalarMax.value = opts.max
    if (opts.opacity !== undefined) {
      material.uniforms.opacity.value = opts.opacity
      material.transparent = opts.opacity < 1.0
    }
  }

  /** Get RGB array [r,g,b] for a normalized value t in [0,1] using given colormap */
  function sampleColor(name: ColormapName, t: number): [number, number, number] {
    const lut = LUT_GENERATORS[name]()
    const idx = Math.round(Math.max(0, Math.min(1, t)) * 255) * 4
    return [lut[idx] / 255, lut[idx + 1] / 255, lut[idx + 2] / 255]
  }

  function disposeAll() {
    textureCache.forEach((tex) => tex.dispose())
    textureCache.clear()
  }

  return { getTexture, createColormapMaterial, updateMaterial, sampleColor, disposeAll }
}
```

**Step 2: Add Vite raw import for GLSL files**

Vite supports `?raw` imports natively — no config needed. But add type declaration:

```typescript
// frontend/src/shaders/glsl.d.ts
declare module '*.glsl?raw' {
  const content: string
  export default content
}
```

**Step 3: Commit**

```bash
git add frontend/src/composables/useColormap.ts frontend/src/shaders/
git commit -m "feat(frontend): add GPU colormap system with 5 colormaps and GLSL shaders"
```

---

### Task 3: Basic FEAViewer Component

**Files:**
- Create: `frontend/src/components/viewer/FEAViewer.vue`
- Create: `frontend/src/components/viewer/ColorBar.vue`

**Step 1: Create FEAViewer with basic mesh rendering**

```vue
<!-- frontend/src/components/viewer/FEAViewer.vue -->
<template>
  <div class="fea-viewer" ref="containerRef">
    <!-- Toolbar -->
    <div class="viewer-toolbar">
      <button
        v-for="mode in displayModes"
        :key="mode.value"
        :class="['toolbar-btn', { active: displayMode === mode.value }]"
        :title="mode.label"
        @click="displayMode = mode.value"
      >
        {{ mode.icon }}
      </button>
      <span class="toolbar-separator" />
      <select v-model="currentColormap" class="toolbar-select" title="Colormap">
        <option v-for="cm in colormaps" :key="cm" :value="cm">{{ cm }}</option>
      </select>
    </div>

    <!-- 3D Canvas (Three.js renders here) -->
    <div ref="viewportRef" class="viewport" />

    <!-- Color Bar -->
    <ColorBar
      v-if="hasScalarField"
      :colormap="currentColormap"
      :min="scalarMin"
      :max="scalarMax"
      :label="scalarLabel"
    />

    <!-- Loading overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="spinner" />
      <span>Loading mesh...</span>
    </div>

    <!-- Empty state -->
    <div v-if="!hasMesh && !loading" class="empty-state">
      <span class="empty-icon">📐</span>
      <span>{{ placeholder }}</span>
    </div>

    <!-- Node info tooltip -->
    <div
      v-if="hoveredNode"
      class="node-tooltip"
      :style="{ left: tooltipPos.x + 'px', top: tooltipPos.y + 'px' }"
    >
      <div>Node {{ hoveredNode.id }}</div>
      <div v-if="hoveredNode.scalar !== undefined">
        {{ scalarLabel }}: {{ hoveredNode.scalar.toFixed(4) }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount, shallowRef } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { useColormap, type ColormapName } from '@/composables/useColormap'
import ColorBar from './ColorBar.vue'

export interface FEAMeshData {
  vertices: number[][] | Float32Array  // (N, 3) or flat
  faces: number[][] | Uint32Array      // (F, 3) or flat
  normals?: Float32Array               // optional pre-computed
}

const props = withDefaults(defineProps<{
  mesh?: FEAMeshData | null
  scalarField?: Float32Array | number[] | null
  scalarLabel?: string
  wireframe?: boolean
  placeholder?: string
  deformation?: Float32Array | null
  deformationScale?: number
}>(), {
  mesh: null,
  scalarField: null,
  scalarLabel: 'Value',
  wireframe: false,
  placeholder: 'No mesh loaded',
  deformation: null,
  deformationScale: 1.0,
})

const emit = defineEmits<{
  (e: 'nodeHover', nodeId: number | null): void
  (e: 'nodeClick', nodeId: number): void
}>()

// Refs
const containerRef = ref<HTMLDivElement | null>(null)
const viewportRef = ref<HTMLDivElement | null>(null)
const loading = ref(false)
const displayMode = ref<'solid' | 'wireframe' | 'solid+wire' | 'transparent'>('solid')
const currentColormap = ref<ColormapName>('jet')
const hoveredNode = ref<{ id: number; scalar?: number } | null>(null)
const tooltipPos = ref({ x: 0, y: 0 })

const displayModes = [
  { value: 'solid' as const, label: 'Solid', icon: '◼' },
  { value: 'wireframe' as const, label: 'Wireframe', icon: '◻' },
  { value: 'solid+wire' as const, label: 'Solid + Wire', icon: '▣' },
  { value: 'transparent' as const, label: 'Transparent', icon: '◇' },
]
const colormaps: ColormapName[] = ['jet', 'viridis', 'coolwarm', 'rainbow', 'grayscale']

// Three.js objects
let scene: THREE.Scene
let camera: THREE.PerspectiveCamera
let renderer: THREE.WebGLRenderer
let controls: OrbitControls
let animationId: number | null = null
let meshGroup: THREE.Group
let wireframeGroup: THREE.Group
let raycaster: THREE.Raycaster
let mouse: THREE.Vector2

// Colormap system
const { createColormapMaterial, updateMaterial, disposeAll: disposeColormaps } = useColormap()
let colormapMaterial: THREE.ShaderMaterial | null = null

// Computed
const hasMesh = computed(() => props.mesh && (
  Array.isArray(props.mesh.vertices) ? props.mesh.vertices.length > 0 :
  props.mesh.vertices.length > 0
))
const hasScalarField = computed(() => props.scalarField && props.scalarField.length > 0)
const scalarMin = ref(0)
const scalarMax = ref(1)

// Initialize Three.js
function initScene() {
  const el = viewportRef.value
  if (!el) return

  scene = new THREE.Scene()
  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.0001, 1000)
  camera.position.set(0, 0.15, 0.3)

  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.setSize(el.clientWidth, el.clientHeight)
  renderer.setClearColor(new THREE.Color('#1a1a2e'))
  renderer.localClippingEnabled = true
  el.appendChild(renderer.domElement)

  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.4))
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
  dirLight.position.set(1, 2, 3)
  scene.add(dirLight)
  scene.add(new THREE.HemisphereLight(0xddeeff, 0x0f0e0d, 0.3))

  meshGroup = new THREE.Group()
  wireframeGroup = new THREE.Group()
  scene.add(meshGroup)
  scene.add(wireframeGroup)

  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  // Events
  renderer.domElement.addEventListener('mousemove', onMouseMove)
  renderer.domElement.addEventListener('click', onClick)

  // Start render loop
  animate()
}

function animate() {
  animationId = requestAnimationFrame(animate)
  controls?.update()
  renderer?.render(scene, camera)
}

function resize() {
  const el = viewportRef.value
  if (!el || !renderer) return
  camera.aspect = el.clientWidth / el.clientHeight
  camera.updateProjectionMatrix()
  renderer.setSize(el.clientWidth, el.clientHeight)
}

// Build mesh geometry from props
function buildMesh() {
  if (!props.mesh) return
  loading.value = true

  // Clear previous
  meshGroup.clear()
  wireframeGroup.clear()

  const { vertices, faces } = props.mesh

  // Convert to flat Float32Arrays if needed
  let positions: Float32Array
  let indices: Uint32Array

  if (Array.isArray(vertices)) {
    positions = new Float32Array(vertices.length * 3)
    for (let i = 0; i < vertices.length; i++) {
      positions[i * 3] = vertices[i][0]
      positions[i * 3 + 1] = vertices[i][1]
      positions[i * 3 + 2] = vertices[i][2]
    }
  } else {
    positions = vertices as Float32Array
  }

  if (Array.isArray(faces)) {
    indices = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      indices[i * 3] = faces[i][0]
      indices[i * 3 + 1] = faces[i][1]
      indices[i * 3 + 2] = faces[i][2]
    }
  } else {
    indices = faces as Uint32Array
  }

  // Apply deformation if provided
  if (props.deformation) {
    const deformed = new Float32Array(positions.length)
    for (let i = 0; i < positions.length; i++) {
      deformed[i] = positions[i] + props.deformationScale * props.deformation[i]
    }
    positions = deformed
  }

  // Create BufferGeometry
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))
  geometry.computeVertexNormals()

  // Add scalar field as vertex attribute
  if (hasScalarField.value && props.scalarField) {
    const scalars = props.scalarField instanceof Float32Array
      ? props.scalarField
      : new Float32Array(props.scalarField)
    const min = scalars.reduce((a, b) => Math.min(a, b), Infinity)
    const max = scalars.reduce((a, b) => Math.max(a, b), -Infinity)
    scalarMin.value = min
    scalarMax.value = max
    geometry.setAttribute('scalar', new THREE.BufferAttribute(scalars, 1))

    colormapMaterial = createColormapMaterial(currentColormap.value, min, max)
    const solidMesh = new THREE.Mesh(geometry, colormapMaterial)
    meshGroup.add(solidMesh)
  } else {
    // Default material (no scalar)
    const material = new THREE.MeshPhongMaterial({
      color: 0x4488cc,
      side: THREE.DoubleSide,
      flatShading: false,
    })
    const solidMesh = new THREE.Mesh(geometry, material)
    meshGroup.add(solidMesh)
  }

  // Wireframe overlay
  const edgeGeo = new THREE.EdgesGeometry(geometry, 15)
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 1 })
  const edges = new THREE.LineSegments(edgeGeo, edgeMat)
  wireframeGroup.add(edges)

  // Fit camera
  fitCamera()
  loading.value = false
}

function fitCamera() {
  const box = new THREE.Box3().setFromObject(meshGroup)
  if (box.isEmpty()) return
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const maxDim = Math.max(size.x, size.y, size.z)
  const dist = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360))
  camera.position.copy(center).add(new THREE.Vector3(0, dist * 0.3, dist * 1.2))
  controls.target.copy(center)
  controls.update()
}

function onMouseMove(e: MouseEvent) {
  if (!renderer) return
  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1
  tooltipPos.value = { x: e.clientX - rect.left + 15, y: e.clientY - rect.top + 15 }

  // Raycast
  raycaster.setFromCamera(mouse, camera)
  const meshes = meshGroup.children.filter((c): c is THREE.Mesh => c instanceof THREE.Mesh)
  const intersects = raycaster.intersectObjects(meshes)
  if (intersects.length > 0) {
    const face = intersects[0].face
    if (face) {
      const nodeId = face.a
      const scalar = props.scalarField
        ? (props.scalarField instanceof Float32Array ? props.scalarField[nodeId] : props.scalarField[nodeId])
        : undefined
      hoveredNode.value = { id: nodeId, scalar }
      emit('nodeHover', nodeId)
    }
  } else {
    hoveredNode.value = null
    emit('nodeHover', null)
  }
}

function onClick() {
  if (hoveredNode.value) {
    emit('nodeClick', hoveredNode.value.id)
  }
}

// Watch display mode
watch(displayMode, (mode) => {
  meshGroup.visible = mode !== 'wireframe'
  wireframeGroup.visible = mode === 'wireframe' || mode === 'solid+wire'
  meshGroup.children.forEach((child) => {
    if (child instanceof THREE.Mesh && child.material) {
      const mat = child.material as THREE.MeshPhongMaterial | THREE.ShaderMaterial
      if ('opacity' in mat) {
        mat.opacity = mode === 'transparent' ? 0.6 : 1.0
        mat.transparent = mode === 'transparent'
      }
    }
  })
})

// Watch colormap change
watch(currentColormap, (name) => {
  if (colormapMaterial) {
    updateMaterial(colormapMaterial, { colormap: name })
  }
})

// Watch mesh data
watch(() => props.mesh, () => buildMesh(), { deep: false })
watch(() => props.scalarField, () => buildMesh(), { deep: false })
watch(() => props.deformation, () => buildMesh(), { deep: false })
watch(() => props.deformationScale, () => buildMesh())

// Lifecycle
let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  initScene()
  if (hasMesh.value) buildMesh()
  resizeObserver = new ResizeObserver(resize)
  if (viewportRef.value) resizeObserver.observe(viewportRef.value)
})

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  if (animationId !== null) cancelAnimationFrame(animationId)
  controls?.dispose()
  renderer?.dispose()
  disposeColormaps()
})
</script>

<style scoped>
.fea-viewer {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 400px;
  border-radius: 8px;
  overflow: hidden;
  background: #1a1a2e;
}

.viewport {
  width: 100%;
  height: 100%;
}

.viewer-toolbar {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 10;
  display: flex;
  gap: 4px;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 6px;
  padding: 4px;
}

.toolbar-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  background: transparent;
  color: #ccc;
  transition: all 0.15s;
}
.toolbar-btn:hover { background: rgba(255,255,255,0.1) }
.toolbar-btn.active { background: var(--color-accent-orange, #ff9800); color: #fff }

.toolbar-separator {
  width: 1px;
  background: rgba(255,255,255,0.2);
  margin: 4px 2px;
}

.toolbar-select {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 4px;
  color: #ccc;
  padding: 2px 6px;
  font-size: 12px;
}

.loading-overlay, .empty-state {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: rgba(255,255,255,0.5);
  gap: 12px;
}

.spinner {
  width: 32px;
  height: 32px;
  border: 3px solid rgba(255,255,255,0.2);
  border-top-color: var(--color-accent-orange, #ff9800);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg) } }

.empty-icon { font-size: 48px }

.node-tooltip {
  position: absolute;
  z-index: 20;
  background: rgba(0,0,0,0.85);
  color: #fff;
  padding: 6px 10px;
  border-radius: 4px;
  font-size: 12px;
  pointer-events: none;
  white-space: nowrap;
}
</style>
```

**Step 2: Create ColorBar component**

```vue
<!-- frontend/src/components/viewer/ColorBar.vue -->
<template>
  <div class="colorbar">
    <div class="colorbar-label">{{ label }}</div>
    <div class="colorbar-gradient" ref="gradientRef">
      <canvas ref="canvasRef" width="20" height="200" />
    </div>
    <div class="colorbar-ticks">
      <span>{{ max.toExponential(2) }}</span>
      <span>{{ ((max + min) / 2).toExponential(2) }}</span>
      <span>{{ min.toExponential(2) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useColormap, type ColormapName } from '@/composables/useColormap'

const props = withDefaults(defineProps<{
  colormap?: ColormapName
  min?: number
  max?: number
  label?: string
}>(), {
  colormap: 'jet',
  min: 0,
  max: 1,
  label: 'Value',
})

const canvasRef = ref<HTMLCanvasElement | null>(null)
const { sampleColor } = useColormap()

function drawGradient() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const h = canvas.height
  for (let y = 0; y < h; y++) {
    const t = 1 - y / h // top = max, bottom = min
    const [r, g, b] = sampleColor(props.colormap, t)
    ctx.fillStyle = `rgb(${r * 255}, ${g * 255}, ${b * 255})`
    ctx.fillRect(0, y, canvas.width, 1)
  }
}

watch(() => props.colormap, drawGradient)
onMounted(drawGradient)
</script>

<style scoped>
.colorbar {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  gap: 4px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 6px;
  padding: 8px;
}

.colorbar-label {
  writing-mode: vertical-rl;
  text-orientation: mixed;
  font-size: 11px;
  color: #aaa;
  transform: rotate(180deg);
}

.colorbar-gradient canvas {
  border-radius: 3px;
}

.colorbar-ticks {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  font-size: 10px;
  color: #aaa;
  min-width: 60px;
}
</style>
```

**Step 3: Verify build**

Run: `cd frontend && npx vue-tsc --noEmit 2>&1 | head -20`

**Step 4: Commit**

```bash
git add frontend/src/components/viewer/FEAViewer.vue frontend/src/components/viewer/ColorBar.vue
git commit -m "feat(frontend): add FEAViewer with Three.js WebGL rendering, colormap, and ColorBar"
```

---

### Task 4: Modal Animation Controls

**Files:**
- Create: `frontend/src/components/viewer/AnimationControls.vue`
- Create: `frontend/src/composables/useAnimation.ts`

**Step 1: Create animation composable**

```typescript
// frontend/src/composables/useAnimation.ts
import { ref, computed } from 'vue'

export interface ModeData {
  index: number
  frequency_hz: number
  mode_type: string
  shape: Float32Array // (N*3) displacement vector
}

export function useAnimation() {
  const isPlaying = ref(false)
  const phase = ref(0) // 0-360 degrees
  const amplitudeScale = ref(1.0)
  const speed = ref(1.0)
  const loop = ref(true)
  const currentModeIndex = ref(0)
  const modes = ref<ModeData[]>([])

  let lastTime = 0
  let animationId: number | null = null

  const currentMode = computed(() => modes.value[currentModeIndex.value] ?? null)

  function setModes(modeList: ModeData[]) {
    modes.value = modeList
    currentModeIndex.value = 0
    phase.value = 0
  }

  function play() {
    isPlaying.value = true
    lastTime = performance.now()
    tick()
  }

  function pause() {
    isPlaying.value = false
    if (animationId !== null) {
      cancelAnimationFrame(animationId)
      animationId = null
    }
  }

  function togglePlay() {
    isPlaying.value ? pause() : play()
  }

  function stepForward() {
    phase.value = (phase.value + 10) % 360
  }

  function stepBackward() {
    phase.value = (phase.value - 10 + 360) % 360
  }

  function tick() {
    if (!isPlaying.value) return
    animationId = requestAnimationFrame(tick)

    const now = performance.now()
    const dt = (now - lastTime) / 1000 // seconds
    lastTime = now

    // Advance phase: speed = 1 means one full cycle per second
    phase.value = (phase.value + dt * speed.value * 360) % 360
    if (!loop.value && phase.value < dt * speed.value * 360 - 360) {
      pause()
      phase.value = 0
    }
  }

  /** Get deformed positions: pos + scale * mode_shape * sin(phase) */
  function getDeformation(): Float32Array | null {
    const mode = currentMode.value
    if (!mode) return null
    const factor = amplitudeScale.value * Math.sin((phase.value * Math.PI) / 180)
    const deformed = new Float32Array(mode.shape.length)
    for (let i = 0; i < mode.shape.length; i++) {
      deformed[i] = mode.shape[i] * factor
    }
    return deformed
  }

  function dispose() {
    pause()
    modes.value = []
  }

  return {
    isPlaying,
    phase,
    amplitudeScale,
    speed,
    loop,
    currentModeIndex,
    modes,
    currentMode,
    setModes,
    play,
    pause,
    togglePlay,
    stepForward,
    stepBackward,
    getDeformation,
    dispose,
  }
}
```

**Step 2: Create AnimationControls component**

```vue
<!-- frontend/src/components/viewer/AnimationControls.vue -->
<template>
  <div class="animation-controls" v-if="modes.length > 0">
    <!-- Mode selector -->
    <div class="mode-selector">
      <label>Mode:</label>
      <select v-model="currentModeIndex" class="mode-select">
        <option v-for="(mode, idx) in modes" :key="idx" :value="idx">
          Mode {{ idx + 1 }}: {{ mode.frequency_hz.toFixed(1) }} Hz ({{ mode.mode_type }})
        </option>
      </select>
    </div>

    <!-- Playback controls -->
    <div class="playback">
      <button class="ctrl-btn" @click="stepBackward" title="Step Back">◄◄</button>
      <button class="ctrl-btn play-btn" @click="togglePlay" :title="isPlaying ? 'Pause' : 'Play'">
        {{ isPlaying ? '⏸' : '▶' }}
      </button>
      <button class="ctrl-btn" @click="stepForward" title="Step Forward">►►</button>

      <!-- Phase slider -->
      <input
        type="range"
        min="0"
        max="360"
        :value="phase"
        @input="phase = Number(($event.target as HTMLInputElement).value)"
        class="phase-slider"
        title="Phase"
      />

      <button
        :class="['ctrl-btn', { active: loop }]"
        @click="loop = !loop"
        title="Loop"
      >🔄</button>

      <select v-model.number="speed" class="speed-select" title="Speed">
        <option :value="0.25">0.25x</option>
        <option :value="0.5">0.5x</option>
        <option :value="1">1x</option>
        <option :value="2">2x</option>
        <option :value="4">4x</option>
      </select>
    </div>

    <!-- Amplitude slider -->
    <div class="amplitude-row">
      <label>Amplitude:</label>
      <input
        type="range"
        min="0.1"
        max="10"
        step="0.1"
        v-model.number="amplitudeScale"
        class="amplitude-slider"
      />
      <span class="amplitude-value">{{ amplitudeScale.toFixed(1) }}x</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { toRefs } from 'vue'
import type { ModeData } from '@/composables/useAnimation'

const props = defineProps<{
  modes: ModeData[]
  isPlaying: boolean
  phase: number
  amplitudeScale: number
  speed: number
  loop: boolean
  currentModeIndex: number
}>()

const emit = defineEmits<{
  (e: 'update:isPlaying', v: boolean): void
  (e: 'update:phase', v: number): void
  (e: 'update:amplitudeScale', v: number): void
  (e: 'update:speed', v: number): void
  (e: 'update:loop', v: boolean): void
  (e: 'update:currentModeIndex', v: number): void
  (e: 'togglePlay'): void
  (e: 'stepForward'): void
  (e: 'stepBackward'): void
}>()

// Two-way binding helpers
const isPlaying = computed({
  get: () => props.isPlaying,
  set: (v) => emit('update:isPlaying', v),
})
const phase = computed({
  get: () => props.phase,
  set: (v) => emit('update:phase', v),
})
const amplitudeScale = computed({
  get: () => props.amplitudeScale,
  set: (v) => emit('update:amplitudeScale', v),
})
const speed = computed({
  get: () => props.speed,
  set: (v) => emit('update:speed', v),
})
const loop = computed({
  get: () => props.loop,
  set: (v) => emit('update:loop', v),
})
const currentModeIndex = computed({
  get: () => props.currentModeIndex,
  set: (v) => emit('update:currentModeIndex', v),
})

function togglePlay() { emit('togglePlay') }
function stepForward() { emit('stepForward') }
function stepBackward() { emit('stepBackward') }

import { computed } from 'vue'
</script>

<style scoped>
.animation-controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 10;
}

.mode-selector, .playback, .amplitude-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #ccc;
}

.mode-select, .speed-select {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 4px;
  color: #ccc;
  padding: 2px 6px;
  font-size: 12px;
}

.ctrl-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: rgba(255,255,255,0.1);
  color: #ccc;
  cursor: pointer;
  font-size: 12px;
}
.ctrl-btn:hover { background: rgba(255,255,255,0.2) }
.ctrl-btn.active { background: var(--color-accent-orange, #ff9800); color: #fff }
.play-btn { width: 36px; font-size: 16px }

.phase-slider, .amplitude-slider {
  flex: 1;
  accent-color: var(--color-accent-orange, #ff9800);
}
.amplitude-value { min-width: 40px; text-align: right }
</style>
```

**Step 3: Commit**

```bash
git add frontend/src/composables/useAnimation.ts frontend/src/components/viewer/AnimationControls.vue
git commit -m "feat(frontend): add modal animation system with playback controls and deformation rendering"
```

---

## Phase B: ECharts Scientific Charts (Tasks 5-8)

**Goal:** Activate the already-installed ECharts for all 2D scientific charts. These are independent of each other and of Phase A.

---

### Task 5: FRF Chart (Harmonic Response)

**Files:**
- Create: `frontend/src/components/charts/FRFChart.vue`
- Modify: `frontend/src/views/AcousticView.vue` — replace raw table with chart

The FRFChart renders frequency vs amplitude as a line chart with log Y-axis, 3dB bandwidth markers, and resonance peak annotation.

```vue
<!-- frontend/src/components/charts/FRFChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="frf-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent, CanvasRenderer])

const props = defineProps<{
  frequencies: number[]
  amplitudes: number[]
  targetFrequency?: number
  peakFrequency?: number
  peakAmplitude?: number
}>()

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    formatter: (params: any) => {
      const p = params[0]
      return `Frequency: ${p.value[0].toFixed(1)} Hz<br/>Amplitude: ${p.value[1].toExponential(3)} m`
    },
  },
  grid: { left: 80, right: 30, top: 40, bottom: 60 },
  xAxis: {
    type: 'value',
    name: 'Frequency (Hz)',
    nameLocation: 'center',
    nameGap: 35,
    axisLabel: { color: '#8b949e' },
    axisLine: { lineStyle: { color: '#30363d' } },
    nameTextStyle: { color: '#8b949e' },
  },
  yAxis: {
    type: 'log',
    name: 'Amplitude (m)',
    nameLocation: 'center',
    nameGap: 55,
    axisLabel: { color: '#8b949e', formatter: (v: number) => v.toExponential(1) },
    axisLine: { lineStyle: { color: '#30363d' } },
    nameTextStyle: { color: '#8b949e' },
  },
  dataZoom: [
    { type: 'inside', xAxisIndex: 0 },
    { type: 'slider', xAxisIndex: 0, bottom: 5, height: 20 },
  ],
  series: [{
    type: 'line',
    data: props.frequencies.map((f, i) => [f, props.amplitudes[i]]),
    smooth: true,
    lineStyle: { color: '#ff9800', width: 2 },
    itemStyle: { color: '#ff9800' },
    symbol: 'none',
    markLine: props.targetFrequency ? {
      silent: true,
      data: [{ xAxis: props.targetFrequency, label: { formatter: 'Target', color: '#58a6ff' }, lineStyle: { color: '#58a6ff', type: 'dashed' } }],
    } : undefined,
    markPoint: props.peakFrequency ? {
      data: [{
        coord: [props.peakFrequency, props.peakAmplitude],
        symbol: 'circle',
        symbolSize: 10,
        itemStyle: { color: '#f44336' },
        label: { formatter: `Peak: ${props.peakFrequency?.toFixed(0)} Hz`, position: 'top', color: '#f44336' },
      }],
    } : undefined,
  }],
}))
</script>

<style scoped>
.frf-chart { width: 100%; height: 350px; }
</style>
```

In AcousticView.vue, find the harmonic response table section and add the chart component above it. Import and register `FRFChart`.

**Step 2: Commit**

```bash
git add frontend/src/components/charts/FRFChart.vue
git commit -m "feat(frontend): add FRF harmonic response line chart with ECharts"
```

---

### Task 6: Modal Frequency Bar Chart

**Files:**
- Create: `frontend/src/components/charts/ModalBarChart.vue`

Horizontal bar chart showing each mode's frequency, colored by type (longitudinal=blue, flexural=orange, torsional=green), with target frequency as a vertical dashed line. Parasitic modes (non-longitudinal within 500 Hz of target) highlighted in red.

**Step 1: Implement component (follow same ECharts pattern as Task 5)**

**Step 2: Commit**

```bash
git add frontend/src/components/charts/ModalBarChart.vue
git commit -m "feat(frontend): add modal frequency bar chart with mode type coloring"
```

---

### Task 7: Safety Gauge Dashboard

**Files:**
- Create: `frontend/src/components/charts/SafetyGauge.vue`

ECharts gauge type showing parameter value within safe range. Green/yellow/red segments.

**Step 1: Implement (follow ECharts gauge type)**

**Step 2: Commit**

```bash
git add frontend/src/components/charts/SafetyGauge.vue
git commit -m "feat(frontend): add safety parameter gauge dashboard with ECharts"
```

---

### Task 8: Additional Charts (S-N, Convergence, Pareto)

**Files:**
- Create: `frontend/src/components/charts/SNChart.vue` — log-log S-N fatigue curve
- Create: `frontend/src/components/charts/ConvergenceChart.vue` — mesh convergence with Richardson extrapolation
- Create: `frontend/src/components/charts/ParetoChart.vue` — Pareto front scatter plot
- Create: `frontend/src/components/charts/GainChart.vue` — gain vs frequency line chart

Each follows the same vue-echarts pattern. Independent implementations.

**Step 1: Implement all four**

**Step 2: Commit**

```bash
git add frontend/src/components/charts/SNChart.vue frontend/src/components/charts/ConvergenceChart.vue frontend/src/components/charts/ParetoChart.vue frontend/src/components/charts/GainChart.vue
git commit -m "feat(frontend): add S-N, convergence, Pareto, and gain ECharts components"
```

---

## Phase C: WebSocket Progress (Tasks 9-10)

**Goal:** Real-time analysis progress feedback instead of dead-looking spinner.

---

### Task 9: Backend WebSocket Endpoint

**Files:**
- Create: `web/routers/ws.py`
- Modify: `web/app.py` — register WebSocket router
- Create: `web/services/analysis_manager.py` — task tracking

**Key implementation:**
- `AnalysisManager` class with `create_task()`, `update_progress()`, `subscribe()` methods
- Uses `asyncio.Queue` per subscriber
- WebSocket endpoint at `/api/v1/ws/analysis/{task_id}`
- Modify `fea_service.py` analysis functions to emit progress updates

**Step 1: Implement backend**

**Step 2: Test with `websocat ws://localhost:8001/api/v1/ws/analysis/test`**

**Step 3: Commit**

```bash
git add web/routers/ws.py web/services/analysis_manager.py
git commit -m "feat(web): add WebSocket endpoint for real-time analysis progress"
```

---

### Task 10: Frontend Progress Overlay

**Files:**
- Create: `frontend/src/components/progress/ProgressOverlay.vue`
- Create: `frontend/src/composables/useAnalysisProgress.ts`

**Key implementation:**
- `useAnalysisProgress(taskId)` composable opens WebSocket, returns reactive progress state
- `ProgressOverlay` shows animated progress bar, step indicators, estimated time, cancel button
- Auto-dismiss on completion

**Step 1: Implement composable + component**

**Step 2: Commit**

```bash
git add frontend/src/components/progress/ProgressOverlay.vue frontend/src/composables/useAnalysisProgress.ts
git commit -m "feat(frontend): add WebSocket progress overlay with step tracking"
```

---

## Phase D: Advanced 3D Features (Tasks 11-12)

**Goal:** Cross-section and isosurface rendering.

---

### Task 11: Cross-Section (Clipping Planes)

**Files:**
- Create: `frontend/src/components/viewer/ClippingControls.vue`
- Create: `frontend/src/composables/useClipping.ts`

**Key implementation:**
- `useClipping(renderer, scene)` manages THREE.Plane objects
- Three toggle buttons (X/Y/Z), slider for plane position
- Uses `renderer.clippingPlanes` — GPU-native, zero performance cost
- Optional stencil buffer cap fill for solid cross-section appearance

**Step 1: Implement**

**Step 2: Commit**

```bash
git add frontend/src/components/viewer/ClippingControls.vue frontend/src/composables/useClipping.ts
git commit -m "feat(frontend): add cross-section clipping planes with interactive controls"
```

---

### Task 12: Isosurface (Marching Tetrahedra)

**Files:**
- Create: `frontend/src/workers/isosurface.worker.ts`
- Create: `frontend/src/components/viewer/IsosurfaceControls.vue`
- Create: `frontend/src/composables/useIsosurface.ts`

**Key implementation:**
- Marching Tetrahedra lookup table in WebWorker
- Worker receives: connectivity, scalar values, threshold → outputs triangle vertices
- Main thread renders result as transparent BufferGeometry
- Controls: threshold slider, opacity slider, add/remove isosurfaces

**Step 1: Implement worker + composable + controls**

**Step 2: Commit**

```bash
git add frontend/src/workers/ frontend/src/components/viewer/IsosurfaceControls.vue frontend/src/composables/useIsosurface.ts
git commit -m "feat(frontend): add isosurface rendering with Marching Tetrahedra in WebWorker"
```

---

## Phase E: Binary Mesh API (Tasks 13-14)

**Goal:** Efficient binary data transfer for 200K+ node meshes.

---

### Task 13: Backend Binary Mesh Endpoints

**Files:**
- Create: `web/routers/mesh_data.py`
- Modify: `web/app.py` — register router

**Key implementation:**
- `GET /api/v1/mesh/{task_id}/geometry` — returns positions + indices as raw bytes (Content-Type: application/octet-stream)
- `GET /api/v1/mesh/{task_id}/scalars?field=von_mises` — returns Float32Array
- `GET /api/v1/mesh/{task_id}/modes/{mode_index}` — returns mode shape as Float32Array
- Uses `Response(content=array.tobytes(), media_type="application/octet-stream")`

**Step 1: Implement**

**Step 2: Test with `curl -o /tmp/mesh.bin http://localhost:8001/api/v1/mesh/test/geometry`**

**Step 3: Commit**

```bash
git add web/routers/mesh_data.py
git commit -m "feat(web): add binary mesh data endpoints for efficient frontend transfer"
```

---

### Task 14: Frontend Binary Mesh Loader

**Files:**
- Create: `frontend/src/composables/useMeshLoader.ts`
- Create: `frontend/src/api/mesh.ts`

**Key implementation:**
- `fetchMeshBinary(taskId)` uses `axios` with `responseType: 'arraybuffer'`
- Parses header (node_count, face_count) then extracts Float32Array/Uint32Array views
- `useMeshLoader()` composable manages loading state and caching

**Step 1: Implement**

**Step 2: Commit**

```bash
git add frontend/src/composables/useMeshLoader.ts frontend/src/api/mesh.ts
git commit -m "feat(frontend): add binary mesh loader for efficient GPU data upload"
```

---

## Phase F: View Integration (Tasks 15-18)

**Goal:** Wire everything together into the actual views.

---

### Task 15: Integrate FEAViewer into GeometryView

**Files:**
- Modify: `frontend/src/views/GeometryView.vue`

**Key changes:**
- Replace inline Canvas 2D renderer (lines 560-772) with `<FEAViewer>`
- Pass mesh data and scalar fields from FEA results
- Add AnimationControls below viewer when modal results available
- Add ClippingControls panel
- Add ModalBarChart for frequency display
- Keep file upload and parameter panels unchanged

---

### Task 16: Integrate FEAViewer into AcousticView

**Files:**
- Modify: `frontend/src/views/AcousticView.vue`

**Key changes:**
- Replace harmonic response table with `<FRFChart>` + `<GainChart>`
- Add `<FEAViewer>` 3D panel for mesh visualization
- Add AnimationControls for mode shape animation
- Add ProgressOverlay during analysis

---

### Task 17: Integrate into HornDesignView

**Files:**
- Modify: `frontend/src/views/HornDesignView.vue`

**Key changes:**
- Replace ThreeViewer usage with `<FEAViewer>`
- Pass horn mesh data to new viewer

---

### Task 18: Create FatigueView

**Files:**
- Create: `frontend/src/views/FatigueView.vue`
- Modify: `frontend/src/router/index.ts` — add `/fatigue` route
- Modify: `frontend/src/components/layout/Sidebar.vue` — add nav link

**Key implementation:**
- S-N curve chart (SNChart)
- Fatigue results display (safety factor, critical locations)
- 3D viewer showing fatigue hotspots as scalar field
- Goodman diagram visualization

---

## Phase G: Cleanup (Task 19)

### Task 19: Remove Legacy Canvas 2D Code

**Files:**
- Delete duplicated Canvas 2D renderer from GeometryView.vue
- Update ThreeViewer.vue to redirect to FEAViewer (or delete if fully replaced)
- Update package.json name from "frontend" to "ultrasonic-weld-master"
- Update index.html title

---

## Execution Summary

| Phase | Tasks | Can Parallelize | Dependencies |
|-------|-------|-----------------|--------------|
| A: Three.js Foundation | 1-4 | Tasks 1,2 parallel; 3 after 1; 4 after 2,3 | None |
| B: ECharts Charts | 5-8 | All parallel | None |
| C: WebSocket Progress | 9-10 | Sequential (10 needs 9) | None |
| D: Advanced 3D | 11-12 | Both parallel | Phase A |
| E: Binary Mesh API | 13-14 | Sequential (14 needs 13) | None |
| F: View Integration | 15-18 | All parallel | Phases A, B, C |
| G: Cleanup | 19 | Single | Phase F |

**Maximum parallelism:** Phases A, B, C, E can all run simultaneously (4 parallel tracks). Phase D after A completes. Phase F after A+B+C complete. Phase G last.
