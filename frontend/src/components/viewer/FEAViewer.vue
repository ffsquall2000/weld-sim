<!-- frontend/src/components/viewer/FEAViewer.vue -->
<template>
  <div class="fea-viewer">
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
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
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
      const v = vertices[i]!
      positions[i * 3] = v[0]!
      positions[i * 3 + 1] = v[1]!
      positions[i * 3 + 2] = v[2]!
    }
  } else {
    positions = vertices as Float32Array
  }

  if (Array.isArray(faces)) {
    indices = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      const f = faces[i]!
      indices[i * 3] = f[0]!
      indices[i * 3 + 1] = f[1]!
      indices[i * 3 + 2] = f[2]!
    }
  } else {
    indices = faces as Uint32Array
  }

  // Apply deformation if provided
  if (props.deformation) {
    const deformed = new Float32Array(positions.length)
    for (let i = 0; i < positions.length; i++) {
      deformed[i] = positions[i]! + props.deformationScale * props.deformation[i]!
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
    const face = intersects[0]!.face
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
