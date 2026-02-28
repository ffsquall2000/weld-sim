<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold mb-1">{{ $t('contactWorkbench.title') }}</h1>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">{{ $t('contactWorkbench.subtitle') }}</p>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left: Parameter Panel (1 col) -->
      <div class="space-y-4">
        <!-- Material Configuration -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('contactWorkbench.materialConfig') }}</h2>

          <div>
            <label class="label-text">{{ $t('contactWorkbench.hornMaterial') }}</label>
            <select v-model="form.horn_material" class="input-field w-full">
              <option v-for="mat in hornMaterials" :key="mat" :value="mat">{{ mat }}</option>
            </select>
          </div>

          <div>
            <label class="label-text">{{ $t('contactWorkbench.workpieceMaterial') }}</label>
            <select v-model="form.workpiece_material" class="input-field w-full">
              <option v-for="mat in workpieceMaterials" :key="mat" :value="mat">{{ mat }}</option>
            </select>
          </div>

          <div>
            <label class="label-text">{{ $t('contactWorkbench.workpieceThickness') }}</label>
            <div class="flex items-center gap-1">
              <input v-model.number="form.workpiece_thickness_mm" type="number" class="input-field flex-1" min="0.1" step="0.1" />
              <span class="text-xs" style="color: var(--color-text-secondary)">mm</span>
            </div>
          </div>
        </div>

        <!-- Anvil Configuration -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('contactWorkbench.anvilConfig') }}</h2>

          <div>
            <label class="label-text">{{ $t('contactWorkbench.anvilType') }}</label>
            <select v-model="form.anvil_type" class="input-field w-full">
              <option value="flat">{{ $t('contactWorkbench.anvilFlat') }}</option>
              <option value="grooved">{{ $t('contactWorkbench.anvilGrooved') }}</option>
              <option value="knurled">{{ $t('contactWorkbench.anvilKnurled') }}</option>
              <option value="contoured">{{ $t('contactWorkbench.anvilContoured') }}</option>
            </select>
          </div>

          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.widthMm') }}</label>
              <input v-model.number="form.anvil_params.width_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.depthMm') }}</label>
              <input v-model.number="form.anvil_params.depth_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.heightMm') }}</label>
              <input v-model.number="form.anvil_params.height_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
          </div>

          <!-- Groove params (shown when grooved) -->
          <div v-if="form.anvil_type === 'grooved'" class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.grooveWidth') }}</label>
              <input v-model.number="form.anvil_params.groove_width_mm" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.grooveDepth') }}</label>
              <input v-model.number="form.anvil_params.groove_depth_mm" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
          </div>

          <!-- Knurl params (shown when knurled) -->
          <div v-if="form.anvil_type === 'knurled'" class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.knurlPitch') }}</label>
              <input v-model.number="form.anvil_params.knurl_pitch_mm" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.knurlDepth') }}</label>
              <input v-model.number="form.anvil_params.knurl_depth_mm" type="number" class="input-field w-full" min="0.05" step="0.05" />
            </div>
          </div>
        </div>

        <!-- Process Parameters -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('contactWorkbench.processParams') }}</h2>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.frequencyHz') }}</label>
              <input v-model.number="form.frequency_hz" type="number" class="input-field w-full" min="1000" step="1000" />
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.amplitudeUm') }}</label>
              <input v-model.number="form.amplitude_um" type="number" class="input-field w-full" min="1" step="1" />
            </div>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.weldTime') }}</label>
              <div class="flex items-center gap-1">
                <input v-model.number="form.weld_time_s" type="number" class="input-field flex-1" min="0.01" step="0.05" />
                <span class="text-xs" style="color: var(--color-text-secondary)">s</span>
              </div>
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.initialTemp') }}</label>
              <div class="flex items-center gap-1">
                <input v-model.number="form.initial_temp_c" type="number" class="input-field flex-1" min="0" step="5" />
                <span class="text-xs" style="color: var(--color-text-secondary)">&deg;C</span>
              </div>
            </div>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('contactWorkbench.contactType') }}</label>
              <select v-model="form.contact_type" class="input-field w-full">
                <option value="penalty">Penalty</option>
                <option value="nitsche">Nitsche</option>
              </select>
            </div>
            <div>
              <label class="label-text">{{ $t('contactWorkbench.frictionCoeff') }}</label>
              <input v-model.number="form.friction_coefficient" type="number" class="input-field w-full" min="0" max="1" step="0.05" />
            </div>
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="card space-y-2">
          <button class="btn-action w-full" :disabled="loadingDocker" @click="doCheckDocker">
            {{ loadingDocker ? $t('contactWorkbench.checking') : $t('contactWorkbench.checkDocker') }}
          </button>
          <button class="btn-action w-full" :disabled="loadingPreview" @click="doAnvilPreview">
            {{ loadingPreview ? $t('contactWorkbench.previewing') : $t('contactWorkbench.previewAnvil') }}
          </button>
          <button class="btn-primary w-full" :disabled="loadingContact" @click="doContactAnalysis">
            {{ loadingContact ? $t('contactWorkbench.analyzing') : $t('contactWorkbench.runContact') }}
          </button>
          <button class="btn-primary w-full" :disabled="loadingThermal" @click="doThermalAnalysis">
            {{ loadingThermal ? $t('contactWorkbench.analyzing') : $t('contactWorkbench.runThermal') }}
          </button>
          <button class="btn-action w-full" :disabled="loadingFull" @click="doFullAnalysis" style="border-color: var(--color-accent-orange); color: var(--color-accent-orange)">
            {{ loadingFull ? $t('contactWorkbench.analyzing') : $t('contactWorkbench.runFull') }}
          </button>

          <!-- Docker Status -->
          <div v-if="dockerStatus !== null" class="text-sm flex items-center gap-2 mt-1 px-1">
            <span :style="{ color: dockerStatus.available ? '#22c55e' : '#ef4444' }">
              {{ dockerStatus.available ? '\u2713' : '\u2717' }}
            </span>
            <span style="color: var(--color-text-secondary)">
              Docker: {{ dockerStatus.available ? $t('contactWorkbench.dockerAvailable') : $t('contactWorkbench.dockerUnavailable') }}
            </span>
          </div>
        </div>

        <!-- Error -->
        <div v-if="error" class="p-3 rounded text-sm" style="background-color: rgba(220,38,38,0.1); color: #dc2626">
          {{ error }}
        </div>
      </div>

      <!-- Right: Preview + Results (2 cols) -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Anvil Preview -->
        <div class="card">
          <h2 class="section-title mb-3">{{ $t('contactWorkbench.anvilPreview') }}</h2>
          <div class="preview-area">
            <canvas ref="anvilCanvas" width="700" height="320" class="w-full" style="border: 1px solid var(--color-border); border-radius: 0.375rem; background-color: var(--color-bg-primary)"></canvas>
            <div v-if="!anvilVertices" class="preview-placeholder">
              {{ $t('contactWorkbench.previewPlaceholder') }}
            </div>
          </div>
          <div v-if="anvilInfo" class="text-sm mt-2" style="color: var(--color-text-secondary)">
            {{ $t('contactWorkbench.anvilTypeLabel') }}: {{ anvilInfo.anvil_type }} |
            {{ $t('contactWorkbench.contactFace') }}: {{ anvilInfo.contact_face }}
          </div>
        </div>

        <!-- Results Tabs -->
        <div v-if="hasResults" class="card">
          <div class="result-tabs mb-4">
            <button
              v-for="tab in availableTabs"
              :key="tab.id"
              class="result-tab"
              :class="{ active: activeTab === tab.id }"
              @click="activeTab = tab.id"
            >
              {{ tab.label }}
            </button>
          </div>

          <!-- Contact Tab -->
          <div v-if="activeTab === 'contact' && contactResult">
            <div class="grid grid-cols-3 gap-3 mb-4">
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.contactPressure') }}</div>
                <div class="metric-value">{{ contactResult.contact_pressure?.toFixed(2) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.slipDistance') }}</div>
                <div class="metric-value">{{ contactResult.slip_distance?.toFixed(3) ?? '--' }} mm</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.deformation') }}</div>
                <div class="metric-value">{{ contactResult.deformation?.toFixed(4) ?? '--' }} mm</div>
              </div>
            </div>
            <div class="grid grid-cols-3 gap-3">
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.stress') }}</div>
                <div class="metric-value">{{ contactResult.stress?.toFixed(2) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.weldQuality') }}</div>
                <div class="metric-value" :style="{ color: qualityColor(contactResult.weld_quality) }">
                  {{ contactResult.weld_quality != null ? (contactResult.weld_quality * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.solveTime') }}</div>
                <div class="metric-value">{{ contactResult.solve_time_s?.toFixed(2) ?? '--' }} s</div>
              </div>
            </div>
          </div>

          <!-- Thermal Tab -->
          <div v-if="activeTab === 'thermal' && thermalResult">
            <div class="grid grid-cols-3 gap-3 mb-4">
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.maxTemperature') }}</div>
                <div class="metric-value" :style="{ color: tempColor(thermalResult.max_temperature_c) }">
                  {{ thermalResult.max_temperature_c?.toFixed(1) ?? '--' }} &deg;C
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.meltZone') }}</div>
                <div class="metric-value">
                  {{ thermalResult.melt_zone_fraction != null ? (thermalResult.melt_zone_fraction * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.weldQuality') }}</div>
                <div class="metric-value" :style="{ color: qualityColor(thermalResult.weld_quality) }">
                  {{ thermalResult.weld_quality != null ? (thermalResult.weld_quality * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
            </div>

            <!-- Thermal History Chart -->
            <div v-if="thermalResult.thermal_history?.length" class="mt-4">
              <h3 class="text-sm font-semibold mb-2">{{ $t('contactWorkbench.thermalHistory') }}</h3>
              <div class="thermal-chart-area">
                <canvas ref="thermalCanvas" width="700" height="200" class="w-full" style="border: 1px solid var(--color-border); border-radius: 0.375rem; background-color: var(--color-bg-primary)"></canvas>
              </div>
            </div>

            <div class="grid grid-cols-1 gap-3 mt-4">
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.solveTime') }}</div>
                <div class="metric-value">{{ thermalResult.solve_time_s?.toFixed(2) ?? '--' }} s</div>
              </div>
            </div>
          </div>

          <!-- Combined Tab -->
          <div v-if="activeTab === 'combined' && fullResult">
            <div class="grid grid-cols-2 gap-4 mb-4">
              <!-- Contact Summary -->
              <div>
                <h3 class="text-sm font-semibold mb-2" style="color: var(--color-accent-orange)">
                  {{ $t('contactWorkbench.tabContact') }}
                </h3>
                <div class="space-y-1 text-sm">
                  <div>{{ $t('contactWorkbench.contactPressure') }}: <span class="font-mono font-bold">{{ fullResult.contact.contact_pressure?.toFixed(2) }} MPa</span></div>
                  <div>{{ $t('contactWorkbench.slipDistance') }}: <span class="font-mono">{{ fullResult.contact.slip_distance?.toFixed(3) }} mm</span></div>
                  <div>{{ $t('contactWorkbench.stress') }}: <span class="font-mono">{{ fullResult.contact.stress?.toFixed(2) }} MPa</span></div>
                  <div>{{ $t('contactWorkbench.weldQuality') }}: <span class="font-mono" :style="{ color: qualityColor(fullResult.contact.weld_quality) }">{{ fullResult.contact.weld_quality != null ? (fullResult.contact.weld_quality * 100).toFixed(1) + '%' : '--' }}</span></div>
                </div>
              </div>
              <!-- Thermal Summary -->
              <div>
                <h3 class="text-sm font-semibold mb-2" style="color: #ef4444">
                  {{ $t('contactWorkbench.tabThermal') }}
                </h3>
                <div class="space-y-1 text-sm">
                  <div>{{ $t('contactWorkbench.maxTemperature') }}: <span class="font-mono font-bold" :style="{ color: tempColor(fullResult.thermal.max_temperature_c) }">{{ fullResult.thermal.max_temperature_c?.toFixed(1) }} &deg;C</span></div>
                  <div>{{ $t('contactWorkbench.meltZone') }}: <span class="font-mono">{{ fullResult.thermal.melt_zone_fraction != null ? (fullResult.thermal.melt_zone_fraction * 100).toFixed(1) + '%' : '--' }}</span></div>
                  <div>{{ $t('contactWorkbench.weldQuality') }}: <span class="font-mono" :style="{ color: qualityColor(fullResult.thermal.weld_quality) }">{{ fullResult.thermal.weld_quality != null ? (fullResult.thermal.weld_quality * 100).toFixed(1) + '%' : '--' }}</span></div>
                </div>
              </div>
            </div>

            <!-- Combined Assessment -->
            <div class="grid grid-cols-2 gap-3">
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.overallQuality') }}</div>
                <div class="metric-value" :style="{ color: qualityColor(fullResult.weld_quality) }">
                  {{ fullResult.weld_quality != null ? (fullResult.weld_quality * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('contactWorkbench.totalSolveTime') }}</div>
                <div class="metric-value">{{ fullResult.total_solve_time_s?.toFixed(2) ?? '--' }} s</div>
              </div>
            </div>
          </div>
        </div>

        <!-- No Results Placeholder -->
        <div
          v-if="!hasResults && !anvilVertices"
          class="card text-sm text-center py-12"
          style="color: var(--color-text-secondary)"
        >
          {{ $t('contactWorkbench.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  analyzeContact,
  checkDocker,
  previewAnvil,
  analyzeThermal,
  fullAnalysis,
  type ContactAnalyzeResponse,
  type ThermalResponse,
  type FullAnalysisResponse,
  type DockerCheckResponse,
  type AnvilPreviewResponse,
} from '@/api/contact'

const { t } = useI18n()

// --- Form State ---
const form = ref({
  horn_material: 'Titanium Ti-6Al-4V',
  workpiece_material: 'Copper C11000',
  workpiece_thickness_mm: 0.5,
  anvil_type: 'flat',
  anvil_params: {
    width_mm: 20,
    depth_mm: 15,
    height_mm: 10,
    groove_width_mm: 2.0,
    groove_depth_mm: 1.0,
    knurl_pitch_mm: 1.0,
    knurl_depth_mm: 0.3,
  },
  contact_type: 'penalty' as 'penalty' | 'nitsche',
  frequency_hz: 20000,
  amplitude_um: 30,
  weld_time_s: 0.5,
  initial_temp_c: 25,
  friction_coefficient: 0.3,
})

const hornMaterials = [
  'Titanium Ti-6Al-4V',
  'Steel D2',
  'Aluminum 7075-T6',
  'M2 High Speed Steel',
  'CPM 10V',
]

const workpieceMaterials = [
  'Copper C11000',
  'Aluminum 1100',
  'Aluminum 3003',
  'Nickel 200',
  'Steel 1018',
  'Brass C26000',
]

// --- Loading States ---
const loadingDocker = ref(false)
const loadingPreview = ref(false)
const loadingContact = ref(false)
const loadingThermal = ref(false)
const loadingFull = ref(false)
const error = ref<string | null>(null)

// --- Results ---
const dockerStatus = ref<DockerCheckResponse | null>(null)
const anvilVertices = ref<number[][] | null>(null)
const anvilFaces = ref<number[][] | null>(null)
const anvilInfo = ref<{ anvil_type: string; contact_face: string } | null>(null)
const contactResult = ref<ContactAnalyzeResponse | null>(null)
const thermalResult = ref<ThermalResponse | null>(null)
const fullResult = ref<FullAnalysisResponse | null>(null)

// --- Canvas refs ---
const anvilCanvas = ref<HTMLCanvasElement | null>(null)
const thermalCanvas = ref<HTMLCanvasElement | null>(null)

// --- Tabs ---
const activeTab = ref<string>('contact')

const availableTabs = computed(() => {
  const tabs: Array<{ id: string; label: string }> = []
  if (contactResult.value) {
    tabs.push({ id: 'contact', label: t('contactWorkbench.tabContact') })
  }
  if (thermalResult.value) {
    tabs.push({ id: 'thermal', label: t('contactWorkbench.tabThermal') })
  }
  if (fullResult.value) {
    tabs.push({ id: 'combined', label: t('contactWorkbench.tabCombined') })
  }
  return tabs
})

const hasResults = computed(() => {
  return !!(contactResult.value || thermalResult.value || fullResult.value)
})

// --- Anvil Rendering ---
function renderAnvil() {
  const canvas = anvilCanvas.value
  if (!canvas || !anvilVertices.value || !anvilFaces.value) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  const verts = anvilVertices.value
  const faces = anvilFaces.value

  ctx.clearRect(0, 0, w, h)

  if (verts.length === 0) return

  // Compute bounding box
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  for (const v of verts) {
    if (v[0] < minX) minX = v[0]
    if (v[0] > maxX) maxX = v[0]
    if (v[1] < minY) minY = v[1]
    if (v[1] > maxY) maxY = v[1]
  }

  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1
  const margin = 40
  const scaleX = (w - margin * 2) / rangeX
  const scaleY = (h - margin * 2) / rangeY
  const scale = Math.min(scaleX, scaleY)
  const offX = (w - rangeX * scale) / 2
  const offY = (h - rangeY * scale) / 2

  function project(v: number[]): [number, number] {
    return [
      offX + (v[0] - minX) * scale,
      offY + (rangeY - (v[1] - minY)) * scale,
    ]
  }

  // Draw faces as wireframe
  ctx.strokeStyle = 'rgba(234, 88, 12, 0.4)'
  ctx.lineWidth = 0.5
  for (const face of faces) {
    if (face.length < 3) continue
    ctx.beginPath()
    const [x0, y0] = project(verts[face[0]])
    ctx.moveTo(x0, y0)
    for (let i = 1; i < face.length; i++) {
      const [x, y] = project(verts[face[i]])
      ctx.lineTo(x, y)
    }
    ctx.closePath()
    ctx.stroke()
  }

  // Draw vertices
  ctx.fillStyle = 'rgba(234, 88, 12, 0.7)'
  for (const v of verts) {
    const [x, y] = project(v)
    ctx.beginPath()
    ctx.arc(x, y, 1.5, 0, Math.PI * 2)
    ctx.fill()
  }
}

// --- Thermal History Rendering ---
function renderThermalHistory() {
  const canvas = thermalCanvas.value
  if (!canvas || !thermalResult.value?.thermal_history?.length) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  const data = thermalResult.value.thermal_history

  ctx.clearRect(0, 0, w, h)

  if (data.length === 0) return

  const margin = { top: 20, right: 20, bottom: 30, left: 50 }
  const plotW = w - margin.left - margin.right
  const plotH = h - margin.top - margin.bottom

  const minT = Math.min(...data)
  const maxT = Math.max(...data)
  const rangeT = maxT - minT || 1

  // Draw axes
  ctx.strokeStyle = 'rgba(128, 128, 128, 0.3)'
  ctx.lineWidth = 1
  ctx.beginPath()
  ctx.moveTo(margin.left, margin.top)
  ctx.lineTo(margin.left, margin.top + plotH)
  ctx.lineTo(margin.left + plotW, margin.top + plotH)
  ctx.stroke()

  // Y-axis labels
  ctx.fillStyle = 'rgba(128, 128, 128, 0.7)'
  ctx.font = '10px monospace'
  ctx.textAlign = 'right'
  for (let i = 0; i <= 4; i++) {
    const val = minT + (rangeT * i) / 4
    const y = margin.top + plotH - (plotH * i) / 4
    ctx.fillText(val.toFixed(0) + '\u00B0C', margin.left - 5, y + 3)
  }

  // X-axis label
  ctx.textAlign = 'center'
  ctx.fillText(t('contactWorkbench.timeSteps'), margin.left + plotW / 2, h - 5)

  // Draw line
  ctx.strokeStyle = '#ef4444'
  ctx.lineWidth = 2
  ctx.beginPath()
  for (let i = 0; i < data.length; i++) {
    const x = margin.left + (plotW * i) / (data.length - 1 || 1)
    const y = margin.top + plotH - (plotH * (data[i] - minT)) / rangeT
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()

  // Fill gradient area under the curve
  ctx.lineTo(margin.left + plotW, margin.top + plotH)
  ctx.lineTo(margin.left, margin.top + plotH)
  ctx.closePath()
  const gradient = ctx.createLinearGradient(0, margin.top, 0, margin.top + plotH)
  gradient.addColorStop(0, 'rgba(239, 68, 68, 0.2)')
  gradient.addColorStop(1, 'rgba(239, 68, 68, 0)')
  ctx.fillStyle = gradient
  ctx.fill()
}

watch([anvilVertices, anvilFaces], () => {
  nextTick(() => renderAnvil())
})

watch(() => thermalResult.value, () => {
  nextTick(() => renderThermalHistory())
})

// --- Actions ---
async function doCheckDocker() {
  loadingDocker.value = true
  error.value = null
  try {
    const res = await checkDocker()
    dockerStatus.value = res.data
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('contactWorkbench.dockerCheckFailed')
  } finally {
    loadingDocker.value = false
  }
}

async function doAnvilPreview() {
  loadingPreview.value = true
  error.value = null
  try {
    const res = await previewAnvil({
      anvil_type: form.value.anvil_type,
      width_mm: form.value.anvil_params.width_mm,
      depth_mm: form.value.anvil_params.depth_mm,
      height_mm: form.value.anvil_params.height_mm,
      groove_width_mm: form.value.anvil_params.groove_width_mm,
      groove_depth_mm: form.value.anvil_params.groove_depth_mm,
      knurl_pitch_mm: form.value.anvil_params.knurl_pitch_mm,
      knurl_depth_mm: form.value.anvil_params.knurl_depth_mm,
    })
    anvilVertices.value = res.data.vertices
    anvilFaces.value = res.data.faces
    anvilInfo.value = { anvil_type: res.data.anvil_type, contact_face: res.data.contact_face }
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('contactWorkbench.previewFailed')
  } finally {
    loadingPreview.value = false
  }
}

async function doContactAnalysis() {
  loadingContact.value = true
  error.value = null
  try {
    const res = await analyzeContact({
      horn_material: form.value.horn_material,
      workpiece_material: form.value.workpiece_material,
      workpiece_thickness_mm: form.value.workpiece_thickness_mm,
      anvil_type: form.value.anvil_type,
      anvil_params: {
        width_mm: form.value.anvil_params.width_mm,
        depth_mm: form.value.anvil_params.depth_mm,
        height_mm: form.value.anvil_params.height_mm,
      },
      contact_type: form.value.contact_type,
      frequency_hz: form.value.frequency_hz,
      amplitude_um: form.value.amplitude_um,
    })
    contactResult.value = res.data
    activeTab.value = 'contact'
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('contactWorkbench.contactFailed')
  } finally {
    loadingContact.value = false
  }
}

async function doThermalAnalysis() {
  loadingThermal.value = true
  error.value = null
  try {
    const contactPressure = contactResult.value?.contact_pressure ?? 10
    const res = await analyzeThermal({
      workpiece_material: form.value.workpiece_material,
      frequency_hz: form.value.frequency_hz,
      amplitude_um: form.value.amplitude_um,
      weld_time_s: form.value.weld_time_s,
      contact_pressure_mpa: contactPressure,
      initial_temp_c: form.value.initial_temp_c,
      friction_coefficient: form.value.friction_coefficient,
    })
    thermalResult.value = res.data
    activeTab.value = 'thermal'
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('contactWorkbench.thermalFailed')
  } finally {
    loadingThermal.value = false
  }
}

async function doFullAnalysis() {
  loadingFull.value = true
  error.value = null
  try {
    const res = await fullAnalysis({
      horn_material: form.value.horn_material,
      workpiece_material: form.value.workpiece_material,
      workpiece_thickness_mm: form.value.workpiece_thickness_mm,
      anvil_type: form.value.anvil_type,
      anvil_params: {
        width_mm: form.value.anvil_params.width_mm,
        depth_mm: form.value.anvil_params.depth_mm,
        height_mm: form.value.anvil_params.height_mm,
      },
      contact_type: form.value.contact_type,
      frequency_hz: form.value.frequency_hz,
      amplitude_um: form.value.amplitude_um,
      weld_time_s: form.value.weld_time_s,
      initial_temp_c: form.value.initial_temp_c,
      friction_coefficient: form.value.friction_coefficient,
    })
    fullResult.value = res.data
    contactResult.value = res.data.contact
    thermalResult.value = res.data.thermal
    activeTab.value = 'combined'
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('contactWorkbench.fullFailed')
  } finally {
    loadingFull.value = false
  }
}

// --- Helpers ---
function qualityColor(val: number | null | undefined): string {
  if (val == null) return ''
  if (val >= 0.8) return '#22c55e'
  if (val >= 0.6) return '#eab308'
  return '#ef4444'
}

function tempColor(val: number | null | undefined): string {
  if (val == null) return ''
  if (val >= 500) return '#ef4444'
  if (val >= 200) return '#eab308'
  return '#22c55e'
}
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
}

.section-title {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-accent-orange);
  margin: 0;
}

.label-text {
  display: block;
  font-size: 0.75rem;
  color: var(--color-text-secondary);
  margin-bottom: 0.125rem;
}

.input-field {
  display: block;
  padding: 0.375rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  margin-top: 0.125rem;
  box-sizing: border-box;
}

.input-field:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}

.btn-primary {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 700;
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

.btn-action {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
  transition: all 0.15s;
}

.btn-action:hover:not(:disabled) {
  border-color: var(--color-accent-orange);
  color: var(--color-accent-orange);
}

.btn-action:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.preview-area {
  position: relative;
}

.preview-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  color: var(--color-text-secondary);
  pointer-events: none;
}

/* Result tabs */
.result-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--color-border);
}

.result-tab {
  padding: 8px 16px;
  font-size: 13px;
  font-weight: 600;
  background: transparent;
  color: var(--color-text-secondary);
  border: none;
  border-bottom: 2px solid transparent;
  cursor: pointer;
  transition: all 0.15s;
}

.result-tab:hover {
  color: var(--color-text-primary);
}

.result-tab.active {
  color: var(--color-accent-orange);
  border-bottom-color: var(--color-accent-orange);
}

/* Metrics */
.metric-card {
  padding: 12px;
  border-radius: 8px;
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
}

.metric-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
}

.font-mono {
  font-family: 'SF Mono', 'Menlo', monospace;
}

.thermal-chart-area {
  position: relative;
}
</style>
