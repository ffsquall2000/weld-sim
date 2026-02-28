<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('geometry.title') }}</h1>

    <!-- Top: File Upload + 3D Viewer -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <!-- Left: File Upload -->
      <div class="card">
        <h2 class="text-lg font-semibold mb-4">{{ $t('geometry.fileImport') }}</h2>

        <!-- Drag & Drop Zone -->
        <div
          class="border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer"
          :style="{
            borderColor: isDragging ? 'var(--color-accent-orange)' : 'var(--color-border)',
            backgroundColor: isDragging ? 'rgba(234,88,12,0.05)' : 'transparent',
          }"
          @dragover.prevent="isDragging = true"
          @dragleave="isDragging = false"
          @drop.prevent="handleDrop"
          @click="triggerFileInput"
        >
          <div class="text-4xl mb-3">&#x1F4C1;</div>
          <p class="font-medium mb-1">{{ $t('geometry.dropHint') }}</p>
          <p class="text-sm" style="color: var(--color-text-secondary)">
            {{ $t('geometry.supportedFormats') }}
          </p>
          <input
            ref="fileInput"
            type="file"
            class="hidden"
            accept=".step,.stp,.x_t,.x_b,.pdf"
            @change="handleFileSelect"
          />
        </div>

        <!-- Upload Progress -->
        <div v-if="uploading" class="mt-4 flex items-center gap-3">
          <div
            class="animate-spin w-5 h-5 border-2 border-t-transparent rounded-full"
            style="border-color: var(--color-accent-orange); border-top-color: transparent"
          />
          <span class="text-sm">{{ $t('geometry.analyzing') }}</span>
        </div>

        <!-- Geometry Analysis Result -->
        <div v-if="cadResult" class="mt-4 space-y-3">
          <h3 class="font-semibold text-sm" style="color: var(--color-accent-orange)">
            {{ $t('geometry.detectionResult') }}
          </h3>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('geometry.hornType') }}</span>
              <div class="font-bold">{{ cadResult.horn_type }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('geometry.gainEstimate') }}</span>
              <div class="font-bold">{{ cadResult.gain_estimate }}x</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('geometry.volume') }}</span>
              <div class="font-bold">{{ cadResult.volume_mm3.toFixed(1) }} mm&sup3;</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('geometry.confidence') }}</span>
              <div class="font-bold">{{ (cadResult.confidence * 100).toFixed(0) }}%</div>
            </div>
          </div>
          <div class="text-sm">
            <span style="color: var(--color-text-secondary)">{{ $t('geometry.dimensions') }}:</span>
            {{
              Object.entries(cadResult.dimensions)
                .map(([k, v]) => `${k}: ${(v as number).toFixed(1)}mm`)
                .join(' | ')
            }}
          </div>
          <div v-if="cadResult.contact_dimensions" class="text-sm">
            <span style="color: var(--color-accent-orange); font-weight: 600;">{{ $t('geometry.contactArea') }}:</span>
            {{ cadResult.contact_dimensions.width_mm.toFixed(1) }} mm &times;
            {{ cadResult.contact_dimensions.length_mm.toFixed(1) }} mm
            = {{ (cadResult.contact_dimensions.width_mm * cadResult.contact_dimensions.length_mm).toFixed(1) }} mm&sup2;
          </div>
          <button class="btn-primary text-sm mt-2" @click="applyToWizard">
            {{ $t('geometry.applyToWizard') }}
          </button>
        </div>

        <!-- PDF Analysis Result -->
        <div v-if="pdfResult" class="mt-4 space-y-3">
          <h3 class="font-semibold text-sm" style="color: var(--color-accent-orange)">
            {{ $t('geometry.pdfAnalysis') }}
          </h3>
          <div class="text-sm" style="color: var(--color-text-secondary)">
            {{ $t('geometry.detectedDimensions', { count: pdfResult.detected_dimensions.length }) }},
            {{ $t('geometry.detectedNotes', { count: pdfResult.notes.length }) }}
          </div>
          <div v-if="pdfResult.detected_dimensions.length > 0" class="max-h-48 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">{{ $t('geometry.label') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('geometry.valueMm') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('geometry.tolerance') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('geometry.confidence') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(dim, i) in pdfResult.detected_dimensions" :key="i">
                  <td class="py-1 px-2">{{ dim.label }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ dim.value_mm }}</td>
                  <td class="py-1 px-2 text-right font-mono">
                    {{ dim.tolerance_mm > 0 ? `\u00B1${dim.tolerance_mm}` : '\u2014' }}
                  </td>
                  <td class="py-1 px-2 text-right">{{ (dim.confidence * 100).toFixed(0) }}%</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-if="pdfResult.notes.length > 0" class="text-sm">
            <strong>{{ $t('geometry.technicalNotes') }}:</strong>
            <ul class="mt-1 space-y-1">
              <li
                v-for="(note, i) in pdfResult.notes"
                :key="i"
                style="color: var(--color-text-secondary)"
              >
                &bull; {{ note }}
              </li>
            </ul>
          </div>
        </div>

        <!-- Error -->
        <div
          v-if="uploadError"
          class="mt-4 p-3 rounded text-sm"
          style="background-color: rgba(220, 38, 38, 0.1); color: #dc2626"
        >
          {{ uploadError }}
        </div>
      </div>

      <!-- Right: 3D Viewer -->
      <div class="card">
        <h2 class="text-lg font-semibold mb-4">{{ $t('geometry.preview3d') }}</h2>

        <!-- WebGL Three.js Viewer (lazy loaded) -->
        <FEAViewer
          v-if="feaViewerReady"
          :mesh="feaViewerMesh"
          :scalar-field="feaViewerScalar"
          scalar-label="Displacement"
          :placeholder="$t('geometry.viewerPlaceholder')"
          style="height: 400px"
        />

        <!-- Canvas 2D Fallback -->
        <div v-else>
          <div
            ref="viewerContainer"
            class="w-full rounded-lg overflow-hidden"
            style="height: 400px; background-color: #1a1a2e"
          >
            <canvas ref="threeCanvas" class="w-full h-full" />
          </div>
          <div class="flex gap-2 mt-3">
            <button class="btn-small" @click="resetCamera">{{ $t('geometry.resetView') }}</button>
            <button class="btn-small" @click="toggleWireframe">
              {{ wireframe ? $t('geometry.solid') : $t('geometry.wireframe') }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom: FEA Simulation Panel -->
    <div class="card">
      <h2 class="text-lg font-semibold mb-4">{{ $t('geometry.feaTitle') }}</h2>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- FEA Parameters -->
        <div class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('geometry.feaParams') }}</h3>

          <div>
            <label class="text-sm" style="color: var(--color-text-secondary)">{{
              $t('geometry.hornType')
            }}</label>
            <select v-model="feaForm.horn_type" class="input-field w-full">
              <option value="cylindrical">{{ $t('geometry.hornCylindrical') }}</option>
              <option value="flat">{{ $t('geometry.hornFlat') }}</option>
              <option value="blade">{{ $t('geometry.hornBlade') }}</option>
              <option value="exponential">{{ $t('geometry.hornExponential') }}</option>
              <option value="block">{{ $t('geometry.hornBlock') }}</option>
            </select>
          </div>

          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{
                $t('geometry.widthMm')
              }}</label>
              <input
                v-model.number="feaForm.width_mm"
                type="number"
                class="input-field w-full"
                min="1"
                step="0.5"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{
                $t('geometry.heightMm')
              }}</label>
              <input
                v-model.number="feaForm.height_mm"
                type="number"
                class="input-field w-full"
                min="1"
                step="0.5"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{
                $t('geometry.lengthMm')
              }}</label>
              <input
                v-model.number="feaForm.length_mm"
                type="number"
                class="input-field w-full"
                min="1"
                step="0.5"
              />
            </div>
          </div>

          <div>
            <label class="text-sm" style="color: var(--color-text-secondary)">{{
              $t('geometry.material')
            }}</label>
            <select v-model="feaForm.material" class="input-field w-full">
              <option v-for="mat in feaMaterials" :key="mat.name" :value="mat.name">
                {{ mat.name }} ({{ mat.E_gpa }} GPa)
              </option>
            </select>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{
                $t('geometry.targetFreqKhz')
              }}</label>
              <input
                v-model.number="feaForm.frequency_khz"
                type="number"
                class="input-field w-full"
                min="1"
                step="0.5"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{
                $t('geometry.meshDensity')
              }}</label>
              <select v-model="feaForm.mesh_density" class="input-field w-full">
                <option value="coarse">{{ $t('geometry.meshCoarse') }}</option>
                <option value="medium">{{ $t('geometry.meshMedium') }}</option>
                <option value="fine">{{ $t('geometry.meshFine') }}</option>
              </select>
            </div>
          </div>

          <button class="btn-primary w-full" :disabled="feaRunning" @click="runFEA">
            {{ feaRunning ? $t('geometry.feaRunning') : $t('geometry.feaRun') }}
          </button>
        </div>

        <!-- FEA Results: Mode Table -->
        <div v-if="feaResult" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('geometry.resonantModes') }}</h3>
          <div class="text-sm" style="color: var(--color-text-secondary)">
            {{ $t('geometry.solveTime') }}: {{ feaResult.solve_time_s }}s |
            {{ $t('geometry.nodes') }}: {{ feaResult.node_count }} |
            {{ $t('geometry.elements') }}: {{ feaResult.element_count }}
          </div>

          <!-- Modal Bar Chart -->
          <ModalBarChart
            v-if="feaResult.mode_shapes?.length"
            :modes="feaResult.mode_shapes.map((m, i) => ({ modeNumber: i + 1, frequency: m.frequency_hz, type: (m.mode_type as 'longitudinal' | 'flexural' | 'torsional' | 'unknown') }))"
            :target-frequency="feaResult.target_frequency_hz"
            style="height: 180px"
          />

          <div class="max-h-64 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">#</th>
                  <th class="text-right py-1 px-2">{{ $t('geometry.freqHz') }}</th>
                  <th class="text-left py-1 px-2">{{ $t('geometry.modeType') }}</th>
                  <th class="text-center py-1 px-2">{{ $t('geometry.deviation') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(mode, i) in feaResult.mode_shapes"
                  :key="i"
                  :style="{
                    backgroundColor:
                      mode.frequency_hz === feaResult!.closest_mode_hz
                        ? 'rgba(234,88,12,0.1)'
                        : 'transparent',
                    fontWeight:
                      mode.frequency_hz === feaResult!.closest_mode_hz ? '700' : '400',
                  }"
                >
                  <td class="py-1 px-2">{{ i + 1 }}</td>
                  <td class="py-1 px-2 text-right font-mono">
                    {{ mode.frequency_hz.toLocaleString() }}
                  </td>
                  <td class="py-1 px-2">
                    <span
                      class="px-2 py-0.5 rounded text-xs"
                      :style="{
                        backgroundColor: modeColor(mode.mode_type),
                        color: '#fff',
                      }"
                      >{{ modeLabel(mode.mode_type) }}</span
                    >
                  </td>
                  <td class="py-1 px-2 text-center font-mono">
                    {{
                      (
                        ((mode.frequency_hz - feaResult!.target_frequency_hz) /
                          feaResult!.target_frequency_hz) *
                        100
                      ).toFixed(1)
                    }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- FEA Results: Summary -->
        <div v-if="feaResult" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('geometry.analysisSummary') }}</h3>

          <div
            class="p-3 rounded-lg"
            :style="{ backgroundColor: deviationBg, border: `1px solid ${deviationBorder}` }"
          >
            <div class="text-sm font-bold" :style="{ color: deviationColor }">
              {{ $t('geometry.closestMode') }}: {{ feaResult.closest_mode_hz.toLocaleString() }} Hz
            </div>
            <div class="text-sm mt-1" :style="{ color: deviationColor }">
              {{ $t('geometry.deviation') }}: {{ feaResult.frequency_deviation_percent.toFixed(2) }}%
              {{ deviationQuality }}
            </div>
          </div>

          <div
            v-if="feaResult.stress_max_mpa"
            class="p-3 rounded"
            style="background-color: var(--color-bg-card)"
          >
            <div class="text-sm" style="color: var(--color-text-secondary)">
              {{ $t('geometry.maxStress') }}
            </div>
            <div class="text-lg font-bold">{{ feaResult.stress_max_mpa }} MPa</div>
          </div>

          <div class="p-3 rounded" style="background-color: var(--color-bg-card)">
            <div class="text-sm" style="color: var(--color-text-secondary)">
              {{ $t('geometry.targetFreq') }}
            </div>
            <div class="text-lg font-bold">
              {{ (feaResult.target_frequency_hz / 1000).toFixed(1) }} kHz
            </div>
          </div>
        </div>

        <!-- FEA Error -->
        <div
          v-if="feaError"
          class="col-span-full p-3 rounded text-sm"
          style="background-color: rgba(220, 38, 38, 0.1); color: #dc2626"
        >
          {{ feaError }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, defineAsyncComponent, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'
import {
  geometryApi,
  type GeometryAnalysisResponse,
  type FEAResponse,
  type PDFAnalysisResponse,
  type FEAMaterial,
  type FEARequest,
} from '@/api/geometry'
import ModalBarChart from '@/components/charts/ModalBarChart.vue'

// Lazy-load FEAViewer (Three.js) so it's in a separate chunk
const FEAViewer = defineAsyncComponent(() =>
  import('@/components/viewer/FEAViewer.vue'),
)

const router = useRouter()
const { t } = useI18n()
const calcStore = useCalculationStore()

// File upload state
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const uploading = ref(false)
const uploadError = ref<string | null>(null)
const cadResult = ref<GeometryAnalysisResponse | null>(null)
const pdfResult = ref<PDFAnalysisResponse | null>(null)

// 3D viewer state
const threeCanvas = ref<HTMLCanvasElement | null>(null)
const viewerContainer = ref<HTMLDivElement | null>(null)
const wireframe = ref(false)

// FEAViewer (Three.js WebGL) state
const feaViewerReady = ref(false)
const feaViewerMesh = ref<{ vertices: number[][]; faces: number[][] } | null>(null)
const feaViewerScalar = ref<number[] | null>(null)

interface MeshData {
  vertices: number[][]
  faces: number[][]
}

let meshObj: MeshData | null = null
let isDraggingMouse = false
let prevMouse = { x: 0, y: 0 }
let rotationState = { x: -0.5, y: 0.5 }
let zoom = 1.0

// FEA state
const feaMaterials = ref<FEAMaterial[]>([])
const feaForm = ref<FEARequest>({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium',
})
const feaRunning = ref(false)
const feaResult = ref<FEAResponse | null>(null)
const feaError = ref<string | null>(null)

// Computed
const deviationColor = computed(() => {
  if (!feaResult.value) return ''
  const d = feaResult.value.frequency_deviation_percent
  if (d < 2) return '#22c55e'
  if (d < 5) return '#eab308'
  return '#ef4444'
})

const deviationBg = computed(() => {
  if (!feaResult.value) return ''
  const d = feaResult.value.frequency_deviation_percent
  if (d < 2) return 'rgba(34,197,94,0.1)'
  if (d < 5) return 'rgba(234,179,8,0.1)'
  return 'rgba(239,68,68,0.1)'
})

const deviationBorder = computed(() => {
  if (!feaResult.value) return ''
  const d = feaResult.value.frequency_deviation_percent
  if (d < 2) return 'rgba(34,197,94,0.3)'
  if (d < 5) return 'rgba(234,179,8,0.3)'
  return 'rgba(239,68,68,0.3)'
})

const deviationQuality = computed(() => {
  if (!feaResult.value) return ''
  const d = feaResult.value.frequency_deviation_percent
  if (d < 2) return t('geometry.excellent')
  if (d < 5) return t('geometry.acceptable')
  return t('geometry.tooLarge')
})

// Methods
function triggerFileInput() {
  fileInput.value?.click()
}

function handleDrop(e: DragEvent) {
  isDragging.value = false
  const files = e.dataTransfer?.files
  const first = files?.[0]
  if (first) processFile(first)
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const first = input.files?.[0]
  if (first) processFile(first)
}

async function processFile(file: File) {
  const ext = file.name.split('.').pop()?.toLowerCase()
  uploading.value = true
  uploadError.value = null
  cadResult.value = null
  pdfResult.value = null

  try {
    if (ext === 'pdf') {
      const res = await geometryApi.uploadPDF(file)
      pdfResult.value = res.data
    } else {
      const res = await geometryApi.uploadCAD(file)
      cadResult.value = res.data
      if (res.data.mesh) {
        renderMesh(res.data.mesh)
      }
      // Auto-fill FEA form from CAD analysis
      feaForm.value.horn_type = res.data.horn_type
      feaForm.value.width_mm = Math.round(res.data.dimensions['width_mm'] ?? 25)
      feaForm.value.height_mm = Math.round(res.data.dimensions['height_mm'] ?? 80)
      feaForm.value.length_mm = Math.round(res.data.dimensions['length_mm'] ?? 25)
    }
  } catch (err: any) {
    uploadError.value = err.response?.data?.detail || err.message || t('common.uploadFailed')
  } finally {
    uploading.value = false
  }
}

function applyToWizard() {
  if (!cadResult.value) return
  const r = cadResult.value
  calcStore.hornType = r.horn_type
  // Prefer contact face dimensions (welding tip) over overall bounding box
  const cd = r.contact_dimensions
  if (cd && cd.width_mm > 0 && cd.length_mm > 0) {
    calcStore.weldWidth = cd.width_mm
    calcStore.weldLength = cd.length_mm
  } else {
    calcStore.weldWidth = r.dimensions['width_mm'] ?? 3.0
    calcStore.weldLength = r.dimensions['length_mm'] ?? 25.0
  }
  router.push('/calculate')
}

async function runFEA() {
  feaRunning.value = true
  feaError.value = null
  feaResult.value = null
  try {
    const res = await geometryApi.runFEA(feaForm.value)
    feaResult.value = res.data
    if (res.data.mesh) {
      renderMesh(res.data.mesh)
    }
  } catch (err: any) {
    feaError.value = err.response?.data?.detail || err.message || t('common.feaFailed')
  } finally {
    feaRunning.value = false
  }
}

function modeColor(type: string): string {
  switch (type) {
    case 'longitudinal':
      return '#22c55e'
    case 'flexural':
      return '#3b82f6'
    case 'torsional':
      return '#a855f7'
    default:
      return '#6b7280'
  }
}

function modeLabel(type: string): string {
  switch (type) {
    case 'longitudinal':
      return t('geometry.modeLongitudinal')
    case 'flexural':
      return t('geometry.modeFlexural')
    case 'torsional':
      return t('geometry.modeTorsional')
    default:
      return type
  }
}

// ---------- Canvas 2D 3D Renderer ----------
// A lightweight wireframe/solid renderer using Canvas 2D with perspective projection,
// mouse rotation, scroll zoom, painter's algorithm face sorting, and axis indicators.

function onMouseDown(e: MouseEvent) {
  isDraggingMouse = true
  prevMouse = { x: e.clientX, y: e.clientY }
}

function onMouseMove(e: MouseEvent) {
  if (!isDraggingMouse) return
  const dx = e.clientX - prevMouse.x
  const dy = e.clientY - prevMouse.y
  rotationState.x += dy * 0.01
  rotationState.y += dx * 0.01
  prevMouse = { x: e.clientX, y: e.clientY }
  if (meshObj) renderFrame()
}

function onMouseUp() {
  isDraggingMouse = false
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  zoom *= e.deltaY > 0 ? 0.9 : 1.1
  zoom = Math.max(0.2, Math.min(5, zoom))
  if (meshObj) renderFrame()
}

function initViewer() {
  const canvas = threeCanvas.value
  if (!canvas) return

  const container = viewerContainer.value
  if (container) {
    canvas.width = container.clientWidth
    canvas.height = container.clientHeight
  }

  // Draw initial empty state
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.fillStyle = '#1a1a2e'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = '#555'
    ctx.font = '14px sans-serif'
    ctx.textAlign = 'center'
    ctx.fillText(
      t('geometry.viewerPlaceholder'),
      canvas.width / 2,
      canvas.height / 2,
    )
  }

  // Mouse handlers for rotation
  canvas.addEventListener('mousedown', onMouseDown)
  canvas.addEventListener('mousemove', onMouseMove)
  canvas.addEventListener('mouseup', onMouseUp)
  canvas.addEventListener('mouseleave', onMouseUp)
  canvas.addEventListener('wheel', onWheel, { passive: false })
}

function renderMesh(mesh: MeshData) {
  meshObj = mesh
  // Feed mesh to FEAViewer if ready
  feaViewerMesh.value = mesh
  // Also render Canvas 2D fallback
  renderFrame()
}

function renderFrame() {
  const canvas = threeCanvas.value
  if (!canvas || !meshObj) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = canvas.width
  const H = canvas.height
  const verts = meshObj.vertices
  const faces = meshObj.faces

  // Find bounds for auto-scaling
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  let minZ = Infinity
  let maxZ = -Infinity
  for (const v of verts) {
    const vx = v[0] ?? 0
    const vy = v[1] ?? 0
    const vz = v[2] ?? 0
    if (vx < minX) minX = vx
    if (vx > maxX) maxX = vx
    if (vy < minY) minY = vy
    if (vy > maxY) maxY = vy
    if (vz < minZ) minZ = vz
    if (vz > maxZ) maxZ = vz
  }
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1)
  const scale = Math.min(W, H) * 0.35 * zoom / maxDim

  // Rotation matrices
  const cosX = Math.cos(rotationState.x)
  const sinX = Math.sin(rotationState.x)
  const cosY = Math.cos(rotationState.y)
  const sinY = Math.sin(rotationState.y)

  function project(x: number, y: number, z: number): [number, number, number] {
    // Center
    x -= cx
    y -= cy
    z -= cz
    // Rotate Y
    const x1 = x * cosY - z * sinY
    const z1 = x * sinY + z * cosY
    // Rotate X
    const y1 = y * cosX - z1 * sinX
    const z2 = y * sinX + z1 * cosX
    // Perspective
    const d = 500
    const factor = d / (d + z2)
    return [W / 2 + x1 * scale * factor, H / 2 - y1 * scale * factor, z2]
  }

  // Clear
  ctx.fillStyle = '#1a1a2e'
  ctx.fillRect(0, 0, W, H)

  // Project all vertices
  const projected = verts.map((v) => project(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0))

  // Sort faces by depth for painter's algorithm
  const faceDepths = faces.map((f, idx) => {
    const avgZ = f.reduce((sum, vi) => sum + (projected[vi]?.[2] ?? 0), 0) / f.length
    return { idx, avgZ }
  })
  faceDepths.sort((a, b) => b.avgZ - a.avgZ)

  // Draw faces
  for (const { idx } of faceDepths) {
    const face = faces[idx]
    if (!face) continue
    const pts = face.map((vi) => projected[vi] ?? [0, 0, 0] as [number, number, number])

    // Face normal for lighting
    if (pts.length >= 3) {
      const p0 = pts[0]!
      const p1 = pts[1]!
      const p2 = pts[2]!
      const ax = p1[0] - p0[0]
      const ay = p1[1] - p0[1]
      const bx = p2[0] - p0[0]
      const by = p2[1] - p0[1]
      const cross = ax * by - ay * bx

      ctx.beginPath()
      ctx.moveTo(p0[0], p0[1])
      for (let i = 1; i < pts.length; i++) {
        const pt = pts[i]!
        ctx.lineTo(pt[0], pt[1])
      }
      ctx.closePath()

      if (!wireframe.value) {
        // Shade based on normal direction (simple Lambertian)
        const brightness = Math.min(200, Math.max(50, 100 + cross * 0.01))
        ctx.fillStyle = `rgb(${Math.round(brightness * 0.5)}, ${Math.round(brightness * 0.7)}, ${Math.round(brightness)})`
        ctx.fill()
      }
      ctx.strokeStyle = wireframe.value
        ? 'rgba(100,180,255,0.6)'
        : 'rgba(100,180,255,0.15)'
      ctx.lineWidth = wireframe.value ? 1 : 0.5
      ctx.stroke()
    }
  }

  // Draw axes
  const axisLen = maxDim * 0.3
  const originPt = project(cx - maxDim * 0.45, cy - maxDim * 0.45, cz - maxDim * 0.45)
  const xEnd = project(cx - maxDim * 0.45 + axisLen, cy - maxDim * 0.45, cz - maxDim * 0.45)
  const yEnd = project(cx - maxDim * 0.45, cy - maxDim * 0.45 + axisLen, cz - maxDim * 0.45)
  const zEnd = project(cx - maxDim * 0.45, cy - maxDim * 0.45, cz - maxDim * 0.45 + axisLen)

  ctx.lineWidth = 2
  // X axis (red)
  ctx.strokeStyle = '#ef4444'
  ctx.beginPath()
  ctx.moveTo(originPt[0], originPt[1])
  ctx.lineTo(xEnd[0], xEnd[1])
  ctx.stroke()
  ctx.fillStyle = '#ef4444'
  ctx.font = '12px sans-serif'
  ctx.fillText('X', xEnd[0] + 5, xEnd[1])
  // Y axis (green)
  ctx.strokeStyle = '#22c55e'
  ctx.beginPath()
  ctx.moveTo(originPt[0], originPt[1])
  ctx.lineTo(yEnd[0], yEnd[1])
  ctx.stroke()
  ctx.fillStyle = '#22c55e'
  ctx.fillText('Y', yEnd[0] + 5, yEnd[1])
  // Z axis (blue)
  ctx.strokeStyle = '#3b82f6'
  ctx.beginPath()
  ctx.moveTo(originPt[0], originPt[1])
  ctx.lineTo(zEnd[0], zEnd[1])
  ctx.stroke()
  ctx.fillStyle = '#3b82f6'
  ctx.fillText('Z', zEnd[0] + 5, zEnd[1])
}

function resetCamera() {
  rotationState = { x: -0.5, y: 0.5 }
  zoom = 1.0
  if (meshObj) {
    renderFrame()
  } else {
    // Re-draw placeholder
    const canvas = threeCanvas.value
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.fillStyle = '#1a1a2e'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#555'
      ctx.font = '14px sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(t('geometry.viewerPlaceholder'), canvas.width / 2, canvas.height / 2)
    }
  }
}

function toggleWireframe() {
  wireframe.value = !wireframe.value
  if (meshObj) renderFrame()
}

// Lifecycle
onMounted(async () => {
  // Detect WebGL support and enable Three.js FEAViewer
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
    feaViewerReady.value = !!gl
  } catch {
    feaViewerReady.value = false
  }

  if (!feaViewerReady.value) {
    initViewer()
  }

  try {
    const res = await geometryApi.getMaterials()
    feaMaterials.value = res.data
  } catch {
    feaMaterials.value = [
      { name: 'Titanium Ti-6Al-4V', E_gpa: 113.8, density_kg_m3: 4430, poisson_ratio: 0.342 },
      { name: 'Steel D2', E_gpa: 210, density_kg_m3: 7700, poisson_ratio: 0.3 },
      { name: 'Aluminum 7075-T6', E_gpa: 71.7, density_kg_m3: 2810, poisson_ratio: 0.33 },
    ]
  }
})

onUnmounted(() => {
  const canvas = threeCanvas.value
  if (canvas) {
    canvas.removeEventListener('mousedown', onMouseDown)
    canvas.removeEventListener('mousemove', onMouseMove)
    canvas.removeEventListener('mouseup', onMouseUp)
    canvas.removeEventListener('mouseleave', onMouseUp)
    canvas.removeEventListener('wheel', onWheel)
  }
})
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
}
.btn-primary {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: var(--color-accent-orange);
  color: #fff;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn-primary:hover:not(:disabled) {
  opacity: 0.9;
}
.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.btn-small {
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  background-color: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
}
.btn-small:hover {
  border-color: var(--color-accent-orange);
}
.input-field {
  display: block;
  padding: 0.375rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  margin-top: 0.25rem;
}
.input-field:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}
</style>
