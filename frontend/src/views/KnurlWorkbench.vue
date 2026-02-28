<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold mb-1">{{ $t('knurlWorkbench.title') }}</h1>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">{{ $t('knurlWorkbench.subtitle') }}</p>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left: Parameter Panel (1 col) -->
      <div class="space-y-4">
        <!-- Horn Geometry -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('knurlWorkbench.hornGeometry') }}</h2>

          <div>
            <label class="label-text">{{ $t('knurlWorkbench.hornType') }}</label>
            <select v-model="form.horn_type" class="input-field w-full">
              <option value="cylindrical">{{ $t('knurlWorkbench.typeCylindrical') }}</option>
              <option value="exponential">{{ $t('knurlWorkbench.typeExponential') }}</option>
              <option value="stepped">{{ $t('knurlWorkbench.typeStepped') }}</option>
              <option value="block">{{ $t('knurlWorkbench.typeBlock') }}</option>
            </select>
          </div>

          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.widthMm') }}</label>
              <input v-model.number="form.width_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.heightMm') }}</label>
              <input v-model.number="form.height_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.lengthMm') }}</label>
              <input v-model.number="form.length_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
          </div>
        </div>

        <!-- Knurl Parameters -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('knurlWorkbench.knurlParams') }}</h2>

          <div>
            <label class="label-text">{{ $t('knurlWorkbench.knurlType') }}</label>
            <select v-model="form.knurl_type" class="input-field w-full">
              <option value="none">{{ $t('knurlWorkbench.knurlNone') }}</option>
              <option value="linear">{{ $t('knurlWorkbench.knurlLinear') }}</option>
              <option value="cross_hatch">{{ $t('knurlWorkbench.knurlCrossHatch') }}</option>
              <option value="diamond">{{ $t('knurlWorkbench.knurlDiamond') }}</option>
            </select>
          </div>

          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.pitch') }}</label>
              <input v-model.number="form.knurl_pitch_mm" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.depth') }}</label>
              <input v-model.number="form.knurl_depth_mm" type="number" class="input-field w-full" min="0.05" step="0.05" />
            </div>
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.toothWidth') }}</label>
              <input v-model.number="form.tooth_width_mm" type="number" class="input-field w-full" min="0.05" step="0.05" />
            </div>
          </div>
        </div>

        <!-- Simulation Parameters -->
        <div class="card space-y-3">
          <h2 class="section-title">{{ $t('knurlWorkbench.simParams') }}</h2>

          <div>
            <label class="label-text">{{ $t('knurlWorkbench.material') }}</label>
            <select v-model="form.material" class="input-field w-full">
              <option v-for="mat in materialOptions" :key="mat.name" :value="mat.name">
                {{ mat.name }} ({{ mat.E_gpa }} GPa)
              </option>
            </select>
          </div>

          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.targetFreq') }}</label>
              <div class="flex items-center gap-1">
                <input v-model.number="form.frequency_khz" type="number" class="input-field flex-1" min="1" step="0.5" />
                <span class="text-xs" style="color: var(--color-text-secondary)">kHz</span>
              </div>
            </div>
            <div>
              <label class="label-text">{{ $t('knurlWorkbench.meshDensity') }}</label>
              <select v-model="form.mesh_density" class="input-field w-full">
                <option value="coarse">{{ $t('knurlWorkbench.meshCoarse') }}</option>
                <option value="medium">{{ $t('knurlWorkbench.meshMedium') }}</option>
                <option value="fine">{{ $t('knurlWorkbench.meshFine') }}</option>
              </select>
            </div>
          </div>

          <div>
            <label class="label-text">{{ $t('knurlWorkbench.nModes') }}</label>
            <input v-model.number="form.n_modes" type="number" class="input-field w-full" min="1" max="20" step="1" />
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="card space-y-2">
          <button class="btn-action w-full" :disabled="loadingPreview" @click="doPreview">
            {{ loadingPreview ? $t('knurlWorkbench.previewing') : $t('knurlWorkbench.preview') }}
          </button>
          <button class="btn-primary w-full" :disabled="loadingAnalyze" @click="doAnalyze">
            {{ loadingAnalyze ? $t('knurlWorkbench.analyzing') : $t('knurlWorkbench.analyze') }}
          </button>
          <button class="btn-action w-full" :disabled="loadingCompare" @click="doCompare">
            {{ loadingCompare ? $t('knurlWorkbench.comparing') : $t('knurlWorkbench.compare') }}
          </button>
          <button class="btn-action w-full" :disabled="loadingOptimize" @click="doOptimize">
            {{ loadingOptimize ? $t('knurlWorkbench.optimizing') : $t('knurlWorkbench.optimize') }}
          </button>
          <button class="btn-action w-full" :disabled="loadingExport" @click="doExport">
            {{ loadingExport ? $t('knurlWorkbench.exporting') : $t('knurlWorkbench.exportStep') }}
          </button>
        </div>

        <!-- Error -->
        <div v-if="error" class="p-3 rounded text-sm" style="background-color: rgba(220,38,38,0.1); color: #dc2626">
          {{ error }}
        </div>
      </div>

      <!-- Right: Preview + Results (2 cols) -->
      <div class="lg:col-span-2 space-y-6">
        <!-- Mesh Preview -->
        <div class="card">
          <h2 class="section-title mb-3">{{ $t('knurlWorkbench.meshPreview') }}</h2>
          <div v-if="meshStats" class="text-sm mb-2" style="color: var(--color-text-secondary)">
            {{ $t('knurlWorkbench.nodes') }}: {{ meshStats.nodes.toLocaleString() }} |
            {{ $t('knurlWorkbench.elements') }}: {{ meshStats.elements.toLocaleString() }}
          </div>
          <div class="preview-area">
            <canvas ref="meshCanvas" width="700" height="360" class="w-full" style="border: 1px solid var(--color-border); border-radius: 0.375rem; background-color: var(--color-bg-primary)"></canvas>
            <div v-if="!meshVertices" class="preview-placeholder">
              {{ $t('knurlWorkbench.previewPlaceholder') }}
            </div>
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

          <!-- Modal Analysis Tab -->
          <div v-if="activeTab === 'modal' && analyzeResult">
            <div class="grid grid-cols-3 gap-3 mb-4">
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.closestMode') }}</div>
                <div class="metric-value">{{ analyzeResult.closest_mode?.toLocaleString() ?? '--' }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.freqDeviation') }}</div>
                <div class="metric-value">{{ analyzeResult.frequency_deviation_hz?.toFixed(1) ?? '--' }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.amplitudeUniformity') }}</div>
                <div class="metric-value" :style="{ color: uniformityColor(analyzeResult.amplitude_uniformity) }">
                  {{ analyzeResult.amplitude_uniformity != null ? (analyzeResult.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
            </div>

            <!-- Mode Shapes Table -->
            <div v-if="analyzeResult.mode_shapes?.length" class="max-h-60 overflow-y-auto">
              <table class="w-full text-sm">
                <thead>
                  <tr style="border-bottom: 1px solid var(--color-border)">
                    <th class="text-left py-1 px-2">#</th>
                    <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.freqHz') }}</th>
                    <th class="text-left py-1 px-2">{{ $t('knurlWorkbench.modeType') }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr
                    v-for="(mode, i) in analyzeResult.mode_shapes"
                    :key="i"
                    :style="{
                      backgroundColor: mode.frequency_hz === analyzeResult.closest_mode ? 'rgba(234,88,12,0.1)' : 'transparent',
                      fontWeight: mode.frequency_hz === analyzeResult.closest_mode ? '700' : '400',
                    }"
                  >
                    <td class="py-1 px-2">{{ i + 1 }}</td>
                    <td class="py-1 px-2 text-right font-mono">{{ mode.frequency_hz?.toLocaleString() }}</td>
                    <td class="py-1 px-2">
                      <span class="mode-badge" :style="{ backgroundColor: modeColor(mode.mode_type) }">
                        {{ mode.mode_type }}
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <!-- Stress Tab -->
          <div v-if="activeTab === 'stress' && analyzeResult">
            <div class="grid grid-cols-2 gap-3">
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.maxStress') }}</div>
                <div class="metric-value">{{ analyzeResult.max_stress_mpa?.toFixed(1) ?? '--' }} MPa</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.amplitudeUniformity') }}</div>
                <div class="metric-value" :style="{ color: uniformityColor(analyzeResult.amplitude_uniformity) }">
                  {{ analyzeResult.amplitude_uniformity != null ? (analyzeResult.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}
                </div>
              </div>
            </div>
          </div>

          <!-- Compare Tab -->
          <div v-if="activeTab === 'compare' && compareResult">
            <div class="grid grid-cols-2 gap-3 mb-4">
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.freqShift') }}</div>
                <div class="metric-value">{{ compareResult.frequency_shift_hz?.toFixed(1) ?? '--' }} Hz</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.freqShiftPercent') }}</div>
                <div class="metric-value">{{ compareResult.frequency_shift_percent?.toFixed(2) ?? '--' }}%</div>
              </div>
            </div>

            <div class="grid grid-cols-2 gap-4">
              <!-- With Knurl -->
              <div>
                <h3 class="text-sm font-semibold mb-2" style="color: var(--color-accent-orange)">
                  {{ $t('knurlWorkbench.withKnurl') }}
                </h3>
                <div class="space-y-1 text-sm">
                  <div>{{ $t('knurlWorkbench.closestMode') }}: <span class="font-mono font-bold">{{ compareResult.with_knurl.closest_mode?.toLocaleString() }} Hz</span></div>
                  <div>{{ $t('knurlWorkbench.maxStress') }}: <span class="font-mono">{{ compareResult.with_knurl.max_stress_mpa?.toFixed(1) }} MPa</span></div>
                  <div>{{ $t('knurlWorkbench.amplitudeUniformity') }}: <span class="font-mono">{{ compareResult.with_knurl.amplitude_uniformity != null ? (compareResult.with_knurl.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}</span></div>
                </div>
              </div>
              <!-- Without Knurl -->
              <div>
                <h3 class="text-sm font-semibold mb-2" style="color: var(--color-text-secondary)">
                  {{ $t('knurlWorkbench.withoutKnurl') }}
                </h3>
                <div class="space-y-1 text-sm">
                  <div>{{ $t('knurlWorkbench.closestMode') }}: <span class="font-mono font-bold">{{ compareResult.without_knurl.closest_mode?.toLocaleString() }} Hz</span></div>
                  <div>{{ $t('knurlWorkbench.maxStress') }}: <span class="font-mono">{{ compareResult.without_knurl.max_stress_mpa?.toFixed(1) }} MPa</span></div>
                  <div>{{ $t('knurlWorkbench.amplitudeUniformity') }}: <span class="font-mono">{{ compareResult.without_knurl.amplitude_uniformity != null ? (compareResult.without_knurl.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}</span></div>
                </div>
              </div>
            </div>
          </div>

          <!-- Optimization Tab -->
          <div v-if="activeTab === 'optimize' && optimizeResult">
            <div v-if="optimizeResult.summary" class="text-sm mb-4 p-3 rounded" style="background-color: var(--color-bg-card); border: 1px solid var(--color-border)">
              {{ optimizeResult.summary }}
            </div>

            <!-- Best Candidates -->
            <div class="grid grid-cols-2 gap-3 mb-4">
              <div v-if="optimizeResult.best_frequency_match" class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.bestFreqMatch') }}</div>
                <div class="text-sm mt-1">
                  <div>{{ optimizeResult.best_frequency_match.knurl_type }} | {{ $t('knurlWorkbench.pitch') }}: {{ optimizeResult.best_frequency_match.knurl_pitch_mm }}mm</div>
                  <div class="font-mono">{{ $t('knurlWorkbench.freqDeviation') }}: {{ optimizeResult.best_frequency_match.frequency_deviation_hz?.toFixed(1) }} Hz</div>
                </div>
              </div>
              <div v-if="optimizeResult.best_uniformity" class="metric-card">
                <div class="metric-label">{{ $t('knurlWorkbench.bestUniformity') }}</div>
                <div class="text-sm mt-1">
                  <div>{{ optimizeResult.best_uniformity.knurl_type }} | {{ $t('knurlWorkbench.pitch') }}: {{ optimizeResult.best_uniformity.knurl_pitch_mm }}mm</div>
                  <div class="font-mono">{{ $t('knurlWorkbench.amplitudeUniformity') }}: {{ optimizeResult.best_uniformity.amplitude_uniformity != null ? (optimizeResult.best_uniformity.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}</div>
                </div>
              </div>
            </div>

            <!-- Candidates Table -->
            <div v-if="optimizeResult.candidates?.length" class="space-y-2">
              <h3 class="text-sm font-semibold">{{ $t('knurlWorkbench.candidates') }} ({{ optimizeResult.candidates.length }})</h3>
              <div class="max-h-60 overflow-y-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr style="border-bottom: 1px solid var(--color-border)">
                      <th class="text-left py-1 px-2">{{ $t('knurlWorkbench.knurlType') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.pitch') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.depth') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.freqDevHz') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.uniformityShort') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.stressMpa') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="(c, i) in optimizeResult.candidates" :key="i">
                      <td class="py-1 px-2">{{ c.knurl_type }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ c.knurl_pitch_mm }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ c.knurl_depth_mm }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ c.frequency_deviation_hz?.toFixed(1) }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ c.amplitude_uniformity != null ? (c.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ c.max_stress_mpa?.toFixed(1) }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Pareto Front -->
            <div v-if="optimizeResult.pareto_front?.length" class="mt-4 space-y-2">
              <h3 class="text-sm font-semibold" style="color: var(--color-accent-orange)">{{ $t('knurlWorkbench.paretoFront') }} ({{ optimizeResult.pareto_front.length }})</h3>
              <div class="max-h-48 overflow-y-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr style="border-bottom: 1px solid var(--color-border)">
                      <th class="text-left py-1 px-2">{{ $t('knurlWorkbench.knurlType') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.pitch') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.depth') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.freqDevHz') }}</th>
                      <th class="text-right py-1 px-2">{{ $t('knurlWorkbench.uniformityShort') }}</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="(p, i) in optimizeResult.pareto_front"
                      :key="i"
                      style="background-color: rgba(234,88,12,0.05)"
                    >
                      <td class="py-1 px-2">{{ p.knurl_type }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ p.knurl_pitch_mm }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ p.knurl_depth_mm }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ p.frequency_deviation_hz?.toFixed(1) }}</td>
                      <td class="py-1 px-2 text-right font-mono">{{ p.amplitude_uniformity != null ? (p.amplitude_uniformity * 100).toFixed(1) + '%' : '--' }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>

        <!-- No Results Placeholder -->
        <div
          v-if="!hasResults && !meshVertices"
          class="card text-sm text-center py-12"
          style="color: var(--color-text-secondary)"
        >
          {{ $t('knurlWorkbench.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  generateKnurlMesh,
  analyzeKnurl,
  compareKnurl,
  optimizeKnurl,
  exportKnurlStep,
  getStepDownloadUrl,
  type KnurlAnalyzeResponse,
  type KnurlCompareResponse,
  type KnurlOptimizeResponse,
} from '@/api/knurl-fea'

const { t } = useI18n()

// --- Form State ---
const form = ref({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  knurl_type: 'linear',
  knurl_pitch_mm: 1.0,
  knurl_depth_mm: 0.3,
  tooth_width_mm: 0.5,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium',
  n_modes: 6,
})

const materialOptions = [
  { name: 'Titanium Ti-6Al-4V', E_gpa: 113.8 },
  { name: 'Steel D2', E_gpa: 210 },
  { name: 'Aluminum 7075-T6', E_gpa: 71.7 },
  { name: 'Copper C11000', E_gpa: 117 },
  { name: 'Nickel 200', E_gpa: 204 },
  { name: 'M2 High Speed Steel', E_gpa: 220 },
  { name: 'CPM 10V', E_gpa: 222 },
  { name: 'PM60 Powder Steel', E_gpa: 230 },
  { name: 'HAP40 Powder HSS', E_gpa: 228 },
  { name: 'HAP72 Powder HSS', E_gpa: 235 },
]

// --- Loading States ---
const loadingPreview = ref(false)
const loadingAnalyze = ref(false)
const loadingCompare = ref(false)
const loadingOptimize = ref(false)
const loadingExport = ref(false)
const error = ref<string | null>(null)

// --- Results ---
const meshVertices = ref<number[][] | null>(null)
const meshFaces = ref<number[][] | null>(null)
const meshStats = ref<{ nodes: number; elements: number } | null>(null)
const analyzeResult = ref<KnurlAnalyzeResponse | null>(null)
const compareResult = ref<KnurlCompareResponse | null>(null)
const optimizeResult = ref<KnurlOptimizeResponse | null>(null)

// --- Canvas ref ---
const meshCanvas = ref<HTMLCanvasElement | null>(null)

// --- Tabs ---
const activeTab = ref<string>('modal')

const availableTabs = computed(() => {
  const tabs: Array<{ id: string; label: string }> = []
  if (analyzeResult.value) {
    tabs.push({ id: 'modal', label: t('knurlWorkbench.tabModal') })
    tabs.push({ id: 'stress', label: t('knurlWorkbench.tabStress') })
  }
  if (compareResult.value) {
    tabs.push({ id: 'compare', label: t('knurlWorkbench.tabCompare') })
  }
  if (optimizeResult.value) {
    tabs.push({ id: 'optimize', label: t('knurlWorkbench.tabOptimize') })
  }
  return tabs
})

const hasResults = computed(() => {
  return !!(analyzeResult.value || compareResult.value || optimizeResult.value)
})

// --- Mesh Rendering ---
function renderMesh() {
  const canvas = meshCanvas.value
  if (!canvas || !meshVertices.value || !meshFaces.value) return

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const w = canvas.width
  const h = canvas.height
  const verts = meshVertices.value
  const faces = meshFaces.value

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

watch([meshVertices, meshFaces], () => {
  nextTick(() => renderMesh())
})

// --- Actions ---
async function doPreview() {
  loadingPreview.value = true
  error.value = null
  try {
    const res = await generateKnurlMesh({
      horn_type: form.value.horn_type,
      width_mm: form.value.width_mm,
      height_mm: form.value.height_mm,
      length_mm: form.value.length_mm,
      knurl_type: form.value.knurl_type,
      knurl_pitch_mm: form.value.knurl_pitch_mm,
      knurl_depth_mm: form.value.knurl_depth_mm,
      mesh_density: form.value.mesh_density,
    })
    meshVertices.value = res.data.vertices
    meshFaces.value = res.data.faces
    meshStats.value = res.data.mesh_stats
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlWorkbench.previewFailed')
  } finally {
    loadingPreview.value = false
  }
}

async function doAnalyze() {
  loadingAnalyze.value = true
  error.value = null
  try {
    const res = await analyzeKnurl({
      horn_type: form.value.horn_type,
      width_mm: form.value.width_mm,
      height_mm: form.value.height_mm,
      length_mm: form.value.length_mm,
      knurl_type: form.value.knurl_type,
      knurl_pitch_mm: form.value.knurl_pitch_mm,
      knurl_depth_mm: form.value.knurl_depth_mm,
      material: form.value.material,
      frequency_khz: form.value.frequency_khz,
      n_modes: form.value.n_modes,
      mesh_density: form.value.mesh_density,
    })
    analyzeResult.value = res.data
    activeTab.value = 'modal'
    // Also update mesh if returned
    if (res.data.mesh_preview) {
      meshVertices.value = res.data.mesh_preview.vertices
      meshFaces.value = res.data.mesh_preview.faces
    }
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlWorkbench.analyzeFailed')
  } finally {
    loadingAnalyze.value = false
  }
}

async function doCompare() {
  loadingCompare.value = true
  error.value = null
  try {
    const res = await compareKnurl({
      horn_type: form.value.horn_type,
      width_mm: form.value.width_mm,
      height_mm: form.value.height_mm,
      length_mm: form.value.length_mm,
      knurl_type: form.value.knurl_type,
      knurl_pitch_mm: form.value.knurl_pitch_mm,
      knurl_depth_mm: form.value.knurl_depth_mm,
      material: form.value.material,
      frequency_khz: form.value.frequency_khz,
      n_modes: form.value.n_modes,
      mesh_density: form.value.mesh_density,
    })
    compareResult.value = res.data
    activeTab.value = 'compare'
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlWorkbench.compareFailed')
  } finally {
    loadingCompare.value = false
  }
}

async function doOptimize() {
  loadingOptimize.value = true
  error.value = null
  try {
    const res = await optimizeKnurl({
      horn_type: form.value.horn_type,
      width_mm: form.value.width_mm,
      height_mm: form.value.height_mm,
      length_mm: form.value.length_mm,
      material: form.value.material,
      frequency_khz: form.value.frequency_khz,
      n_candidates: 10,
    })
    optimizeResult.value = res.data
    activeTab.value = 'optimize'
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlWorkbench.optimizeFailed')
  } finally {
    loadingOptimize.value = false
  }
}

async function doExport() {
  loadingExport.value = true
  error.value = null
  try {
    const res = await exportKnurlStep({
      horn_type: form.value.horn_type,
      width_mm: form.value.width_mm,
      height_mm: form.value.height_mm,
      length_mm: form.value.length_mm,
      knurl_type: form.value.knurl_type,
      knurl_pitch_mm: form.value.knurl_pitch_mm,
      knurl_depth_mm: form.value.knurl_depth_mm,
    })
    // Open download URL
    const url = getStepDownloadUrl(res.data.filename)
    window.open(url, '_blank')
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlWorkbench.exportFailed')
  } finally {
    loadingExport.value = false
  }
}

// --- Helpers ---
function modeColor(type: string): string {
  switch (type) {
    case 'longitudinal': return '#22c55e'
    case 'flexural': return '#3b82f6'
    case 'torsional': return '#a855f7'
    default: return '#6b7280'
  }
}

function uniformityColor(val: number | null | undefined): string {
  if (val == null) return ''
  if (val >= 0.9) return '#22c55e'
  if (val >= 0.7) return '#eab308'
  return '#ef4444'
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

.mode-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  color: #fff;
}

.font-mono {
  font-family: 'SF Mono', 'Menlo', monospace;
}
</style>
