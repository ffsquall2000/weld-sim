<template>
  <div class="workbench">
    <!-- Header -->
    <div class="workbench-header">
      <h1 class="workbench-title">Analysis Workbench</h1>
      <span class="workbench-subtitle">Unified FEA simulation pipeline</span>
    </div>

    <!-- Main content: left 3D viewer + right config panel -->
    <div class="workbench-body">
      <!-- Left: 3D Viewer -->
      <div class="viewer-panel">
        <FEAViewer
          :mesh="viewerMesh"
          :scalar-field="viewerScalar"
          scalar-label="Displacement"
          placeholder="Upload a STEP file or configure horn parameters to begin"
          style="height: 100%"
        />
      </div>

      <!-- Right: Configuration Panel -->
      <div class="config-panel">
        <div class="config-scroll">
          <!-- Section: Geometry Input -->
          <div class="config-section">
            <h3 class="section-title">Geometry Input</h3>

            <!-- Source toggle -->
            <div class="source-toggle">
              <button
                class="toggle-btn"
                :class="{ active: inputMode === 'parametric' }"
                @click="inputMode = 'parametric'"
              >
                Parametric
              </button>
              <button
                class="toggle-btn"
                :class="{ active: inputMode === 'file' }"
                @click="inputMode = 'file'"
              >
                STEP File
              </button>
            </div>

            <!-- Parametric inputs -->
            <template v-if="inputMode === 'parametric'">
              <div class="field">
                <label class="field-label">Horn Type</label>
                <select v-model="form.horn_type" class="input-field">
                  <option value="cylindrical">Cylindrical</option>
                  <option value="flat">Flat</option>
                  <option value="blade">Blade</option>
                  <option value="exponential">Exponential</option>
                  <option value="block">Block</option>
                </select>
              </div>
              <div class="field-row">
                <div class="field">
                  <label class="field-label">Width (mm)</label>
                  <input v-model.number="form.width_mm" type="number" class="input-field" min="1" step="0.5" />
                </div>
                <div class="field">
                  <label class="field-label">Height (mm)</label>
                  <input v-model.number="form.height_mm" type="number" class="input-field" min="1" step="0.5" />
                </div>
                <div class="field">
                  <label class="field-label">Length (mm)</label>
                  <input v-model.number="form.length_mm" type="number" class="input-field" min="1" step="0.5" />
                </div>
              </div>
            </template>

            <!-- File upload -->
            <template v-if="inputMode === 'file'">
              <div
                class="drop-zone"
                :class="{ dragging: isDragging }"
                @dragover.prevent="isDragging = true"
                @dragleave="isDragging = false"
                @drop.prevent="handleDrop"
                @click="triggerFileInput"
              >
                <div class="drop-icon">&#x1F4C1;</div>
                <p class="drop-text">
                  {{ uploadedFile ? uploadedFile.name : 'Drop STEP file here or click to browse' }}
                </p>
                <p v-if="!uploadedFile" class="drop-hint">.step, .stp</p>
                <input
                  ref="fileInput"
                  type="file"
                  class="hidden"
                  accept=".step,.stp"
                  @change="handleFileSelect"
                />
              </div>
              <div v-if="uploading" class="upload-status">
                <div class="spinner-small" />
                <span>Analyzing geometry...</span>
              </div>
            </template>
          </div>

          <!-- Section: Simulation Parameters -->
          <div class="config-section">
            <h3 class="section-title">Simulation Parameters</h3>
            <div class="field">
              <label class="field-label">Material</label>
              <select v-model="form.material" class="input-field">
                <option v-for="mat in materials" :key="mat.name" :value="mat.name">
                  {{ mat.name }} ({{ mat.E_gpa }} GPa)
                </option>
              </select>
            </div>
            <div class="field-row">
              <div class="field">
                <label class="field-label">Frequency (kHz)</label>
                <input v-model.number="form.frequency_khz" type="number" class="input-field" min="1" step="0.5" />
              </div>
              <div class="field">
                <label class="field-label">Mesh Density</label>
                <select v-model="form.mesh_density" class="input-field">
                  <option value="coarse">Coarse</option>
                  <option value="medium">Medium</option>
                  <option value="fine">Fine</option>
                  <option value="adaptive">Adaptive</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Section: Analysis Modules -->
          <div class="config-section">
            <h3 class="section-title">Analysis Modules</h3>
            <div class="modules-list">
              <label
                v-for="mod in moduleList"
                :key="mod.id"
                class="module-item"
              >
                <input
                  type="checkbox"
                  :checked="selected.has(mod.id)"
                  @change="toggle(mod.id)"
                  class="module-checkbox"
                />
                <div class="module-info">
                  <span class="module-name">{{ mod.label }}</span>
                  <span v-if="getDependencyLabel(mod.id)" class="module-deps">
                    {{ getDependencyLabel(mod.id) }}
                  </span>
                </div>
              </label>
            </div>
          </div>

          <!-- Run Button -->
          <button
            class="run-btn"
            :disabled="running || orderedModules.length === 0"
            @click="runAnalysis"
          >
            {{ running ? 'Running...' : `Run Selected (${orderedModules.length} modules)` }}
          </button>

          <!-- Progress -->
          <FEAProgress
            ref="progressRef"
            :visible="running"
            :task-id="taskId"
            title="Analysis Chain"
            @cancel="cancelAnalysis"
            @complete="onComplete"
            @error="onError"
          />

          <!-- Error display -->
          <div v-if="errorMsg" class="error-box">
            {{ errorMsg }}
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom: Result Tabs -->
    <div v-if="hasAnyResult" class="results-panel">
      <div class="result-tabs">
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

      <div class="result-content">
        <!-- Modal Results -->
        <div v-if="activeTab === 'modal' && chainResult?.modal">
          <div class="result-meta">
            Solve time: {{ chainResult.modal.solve_time_s?.toFixed(1) ?? '--' }}s |
            Nodes: {{ chainResult.node_count ?? '--' }} |
            Elements: {{ chainResult.element_count ?? '--' }}
          </div>
          <div v-if="chainResult.modal.mode_shapes?.length" class="mode-table-wrap">
            <table class="mode-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Frequency (Hz)</th>
                  <th>Mode Type</th>
                  <th>Deviation</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(mode, i) in chainResult.modal.mode_shapes"
                  :key="i"
                  :class="{ 'closest-mode': mode.frequency_hz === chainResult.modal.closest_mode_hz }"
                >
                  <td>{{ Number(i) + 1 }}</td>
                  <td class="font-mono">{{ mode.frequency_hz?.toLocaleString() }}</td>
                  <td>
                    <span class="mode-badge" :style="{ backgroundColor: modeColor(mode.mode_type) }">
                      {{ mode.mode_type }}
                    </span>
                  </td>
                  <td class="font-mono">
                    {{ chainResult.modal.target_frequency_hz
                      ? (((mode.frequency_hz - chainResult.modal.target_frequency_hz) / chainResult.modal.target_frequency_hz) * 100).toFixed(1)
                      : '--' }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <div v-else class="no-data">No mode shapes available</div>
        </div>

        <!-- Harmonic Results -->
        <div v-if="activeTab === 'harmonic' && chainResult?.harmonic">
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">Gain at Target</div>
              <div class="metric-value">{{ chainResult.harmonic.gain_at_target?.toFixed(2) ?? '--' }}x</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Q Factor</div>
              <div class="metric-value">{{ chainResult.harmonic.Q_factor?.toFixed(0) ?? '--' }}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Resonant Freq</div>
              <div class="metric-value">{{ chainResult.harmonic.resonant_frequency_hz?.toLocaleString() ?? '--' }} Hz</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Uniformity</div>
              <div class="metric-value">{{ chainResult.harmonic.uniformity_percent?.toFixed(1) ?? '--' }}%</div>
            </div>
          </div>
        </div>

        <!-- Stress Results -->
        <div v-if="activeTab === 'stress' && chainResult?.stress">
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">Max Stress</div>
              <div class="metric-value">{{ chainResult.stress.max_von_mises_mpa?.toFixed(1) ?? '--' }} MPa</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Safety Factor</div>
              <div class="metric-value" :class="safetyFactorClass">
                {{ chainResult.stress.safety_factor?.toFixed(2) ?? '--' }}
              </div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Max Displacement</div>
              <div class="metric-value">{{ chainResult.stress.max_displacement_um?.toFixed(2) ?? '--' }} um</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Yield Strength</div>
              <div class="metric-value">{{ chainResult.stress.yield_strength_mpa?.toFixed(0) ?? '--' }} MPa</div>
            </div>
          </div>
        </div>

        <!-- Fatigue Results -->
        <div v-if="activeTab === 'fatigue' && chainResult?.fatigue">
          <div class="metrics-grid">
            <div class="metric-card">
              <div class="metric-label">Min Safety Factor</div>
              <div class="metric-value">{{ chainResult.fatigue.min_safety_factor?.toFixed(2) ?? '--' }}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Life Estimate</div>
              <div class="metric-value">{{ formatLife(chainResult.fatigue.life_cycles) }}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Critical Location</div>
              <div class="metric-value text-sm">{{ chainResult.fatigue.critical_location ?? '--' }}</div>
            </div>
            <div class="metric-card">
              <div class="metric-label">Assessment</div>
              <div class="metric-value" :class="fatigueAssessmentClass">
                {{ chainResult.fatigue.assessment ?? '--' }}
              </div>
            </div>
          </div>
        </div>

        <!-- No data fallback for active tab -->
        <div v-if="activeTab && !getTabData(activeTab)" class="no-data">
          No data -- this module has not been executed yet.
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, defineAsyncComponent } from 'vue'
import { useAnalysisDependencies } from '@/composables/useAnalysisDependencies'
import { runAnalysisChain, type ChainRequest, type ChainResponse } from '@/api/analysis'
import { geometryApi, type FEAMaterial, type GeometryAnalysisResponse } from '@/api/geometry'
import FEAProgress from '@/components/FEAProgress.vue'
import { generateTaskId } from '@/utils/uuid'

const FEAViewer = defineAsyncComponent(() =>
  import('@/components/viewer/FEAViewer.vue'),
)

// --- Dependency system ---
const { selected, toggle, orderedModules, getDependencyLabel } = useAnalysisDependencies()

const moduleList = [
  { id: 'modal', label: 'Modal Analysis' },
  { id: 'harmonic', label: 'Harmonic Response' },
  { id: 'stress', label: 'Stress Analysis' },
  { id: 'fatigue', label: 'Fatigue Assessment' },
  { id: 'static', label: 'Static Analysis' },
]

// --- Form state ---
const inputMode = ref<'parametric' | 'file'>('parametric')
const form = ref({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium' as string,
})

const materials = ref<FEAMaterial[]>([])

// --- File upload state ---
const fileInput = ref<HTMLInputElement | null>(null)
const isDragging = ref(false)
const uploading = ref(false)
const uploadedFile = ref<File | null>(null)
const cadResult = ref<GeometryAnalysisResponse | null>(null)

// --- 3D Viewer state ---
const viewerMesh = ref<{ vertices: number[][]; faces: number[][] } | null>(null)
const viewerScalar = ref<number[] | null>(null)

// --- Analysis state ---
const running = ref(false)
const taskId = ref('')
const errorMsg = ref<string | null>(null)
const chainResult = ref<ChainResponse | null>(null)
const progressRef = ref<InstanceType<typeof FEAProgress> | null>(null)

// --- Result tabs ---
const activeTab = ref<string>('modal')

const availableTabs = computed(() => {
  const tabs = []
  if (chainResult.value?.modal) tabs.push({ id: 'modal', label: 'Modal' })
  if (chainResult.value?.harmonic) tabs.push({ id: 'harmonic', label: 'Harmonic' })
  if (chainResult.value?.stress) tabs.push({ id: 'stress', label: 'Stress' })
  if (chainResult.value?.fatigue) tabs.push({ id: 'fatigue', label: 'Fatigue' })
  return tabs
})

const hasAnyResult = computed(() => {
  const r = chainResult.value
  return r && (r.modal || r.harmonic || r.stress || r.fatigue)
})

const safetyFactorClass = computed(() => {
  const sf = chainResult.value?.stress?.safety_factor
  if (!sf) return ''
  if (sf >= 2) return 'text-safe'
  if (sf >= 1) return 'text-warn'
  return 'text-danger'
})

const fatigueAssessmentClass = computed(() => {
  const a = chainResult.value?.fatigue?.assessment
  if (!a) return ''
  if (a === 'PASS' || a === 'pass') return 'text-safe'
  return 'text-danger'
})

function getTabData(tabId: string): any {
  if (!chainResult.value) return null
  return (chainResult.value as any)[tabId] ?? null
}

// --- File upload ---
function triggerFileInput() {
  fileInput.value?.click()
}

function handleDrop(e: DragEvent) {
  isDragging.value = false
  const files = e.dataTransfer?.files
  if (files?.[0]) processFile(files[0])
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  if (input.files?.[0]) processFile(input.files[0])
}

async function processFile(file: File) {
  uploading.value = true
  uploadedFile.value = file
  errorMsg.value = null
  try {
    const res = await geometryApi.uploadCAD(file)
    cadResult.value = res.data
    if (res.data.mesh) {
      viewerMesh.value = res.data.mesh
    }
    // Auto-fill form from CAD analysis
    form.value.horn_type = res.data.horn_type
    form.value.width_mm = Math.round(res.data.dimensions['width_mm'] ?? 25)
    form.value.height_mm = Math.round(res.data.dimensions['height_mm'] ?? 80)
    form.value.length_mm = Math.round(res.data.dimensions['length_mm'] ?? 25)
  } catch (err: any) {
    errorMsg.value = err.response?.data?.detail || err.message || 'Upload failed'
  } finally {
    uploading.value = false
  }
}

// --- Analysis execution ---
async function runAnalysis() {
  running.value = true
  errorMsg.value = null
  chainResult.value = null

  const tid = generateTaskId()
  taskId.value = tid

  const req: ChainRequest = {
    modules: orderedModules.value,
    material: form.value.material,
    frequency_khz: form.value.frequency_khz,
    mesh_density: form.value.mesh_density,
    horn_type: form.value.horn_type,
    width_mm: form.value.width_mm,
    height_mm: form.value.height_mm,
    length_mm: form.value.length_mm,
    task_id: tid,
  }

  try {
    const result = await runAnalysisChain(req)
    chainResult.value = result
    // Set active tab to first available result
    if (result.modal) activeTab.value = 'modal'
    else if (result.harmonic) activeTab.value = 'harmonic'
    else if (result.stress) activeTab.value = 'stress'
    else if (result.fatigue) activeTab.value = 'fatigue'
  } catch (err: any) {
    errorMsg.value = err.response?.data?.detail || err.message || 'Analysis failed'
  } finally {
    running.value = false
  }
}

function cancelAnalysis() {
  progressRef.value?.requestCancel()
  errorMsg.value = 'Analysis cancelled'
  running.value = false
}

function onComplete(result: any) {
  if (result) {
    chainResult.value = result
    if (result.modal) activeTab.value = 'modal'
  }
  running.value = false
}

function onError(error: string) {
  errorMsg.value = error
  running.value = false
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

function formatLife(cycles: number | undefined | null): string {
  if (!cycles) return '--'
  if (cycles >= 1e9) return `${(cycles / 1e9).toFixed(1)}B`
  if (cycles >= 1e6) return `${(cycles / 1e6).toFixed(1)}M`
  if (cycles >= 1e3) return `${(cycles / 1e3).toFixed(1)}K`
  return cycles.toFixed(0)
}

// --- Lifecycle ---
onMounted(async () => {
  try {
    const res = await geometryApi.getMaterials()
    materials.value = res.data
  } catch {
    materials.value = [
      { name: 'Titanium Ti-6Al-4V', E_gpa: 113.8, density_kg_m3: 4430, poisson_ratio: 0.342 },
      { name: 'Steel D2', E_gpa: 210, density_kg_m3: 7700, poisson_ratio: 0.3 },
      { name: 'Aluminum 7075-T6', E_gpa: 71.7, density_kg_m3: 2810, poisson_ratio: 0.33 },
    ]
  }
})
</script>

<style scoped>
.workbench {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}

.workbench-header {
  display: flex;
  align-items: baseline;
  gap: 12px;
  padding: 16px 24px;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.workbench-title {
  font-size: 20px;
  font-weight: 700;
  margin: 0;
}

.workbench-subtitle {
  font-size: 13px;
  color: var(--color-text-secondary);
}

.workbench-body {
  display: flex;
  flex: 1;
  min-height: 0;
}

/* Left: 3D Viewer */
.viewer-panel {
  flex: 1;
  min-width: 0;
  border-right: 1px solid var(--color-border);
}

/* Right: Config Panel */
.config-panel {
  width: 380px;
  min-width: 320px;
  display: flex;
  flex-direction: column;
}

.config-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}

.config-section {
  margin-bottom: 20px;
}

.section-title {
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--color-accent-orange);
  margin: 0 0 12px 0;
}

/* Source toggle */
.source-toggle {
  display: flex;
  gap: 0;
  margin-bottom: 12px;
  border-radius: 6px;
  overflow: hidden;
  border: 1px solid var(--color-border);
}

.toggle-btn {
  flex: 1;
  padding: 6px 12px;
  font-size: 12px;
  font-weight: 600;
  background: transparent;
  color: var(--color-text-secondary);
  border: none;
  cursor: pointer;
  transition: all 0.15s;
}

.toggle-btn.active {
  background-color: var(--color-accent-orange);
  color: #fff;
}

/* Fields */
.field {
  margin-bottom: 10px;
}

.field-label {
  display: block;
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.field-row {
  display: flex;
  gap: 8px;
  margin-bottom: 10px;
}

.field-row .field {
  flex: 1;
  margin-bottom: 0;
}

.input-field {
  display: block;
  width: 100%;
  padding: 6px 8px;
  border-radius: 6px;
  font-size: 13px;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  box-sizing: border-box;
}

.input-field:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}

/* Drop zone */
.drop-zone {
  border: 2px dashed var(--color-border);
  border-radius: 8px;
  padding: 24px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
}

.drop-zone.dragging {
  border-color: var(--color-accent-orange);
  background-color: rgba(234, 88, 12, 0.05);
}

.drop-icon {
  font-size: 32px;
  margin-bottom: 8px;
}

.drop-text {
  font-size: 13px;
  font-weight: 500;
  margin: 0;
}

.drop-hint {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin: 4px 0 0 0;
}

.hidden {
  display: none;
}

.upload-status {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  font-size: 12px;
  color: var(--color-text-secondary);
}

.spinner-small {
  width: 16px;
  height: 16px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-accent-orange);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Modules */
.modules-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.module-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 6px;
  cursor: pointer;
  transition: background-color 0.15s;
  border: 1px solid var(--color-border);
}

.module-item:hover {
  background-color: var(--color-bg-card);
}

.module-checkbox {
  margin-top: 2px;
  accent-color: var(--color-accent-orange);
}

.module-info {
  display: flex;
  flex-direction: column;
}

.module-name {
  font-size: 13px;
  font-weight: 500;
}

.module-deps {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-style: italic;
}

/* Run button */
.run-btn {
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  font-weight: 700;
  font-size: 14px;
  background-color: var(--color-accent-orange);
  color: #fff;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
  margin-bottom: 12px;
}

.run-btn:hover:not(:disabled) {
  opacity: 0.9;
}

.run-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Error box */
.error-box {
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 13px;
  background-color: rgba(220, 38, 38, 0.1);
  color: #dc2626;
  margin-top: 8px;
}

/* Results Panel (bottom) */
.results-panel {
  flex-shrink: 0;
  border-top: 1px solid var(--color-border);
  max-height: 320px;
  display: flex;
  flex-direction: column;
}

.result-tabs {
  display: flex;
  gap: 0;
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
  padding: 0 16px;
}

.result-tab {
  padding: 10px 20px;
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

.result-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 24px;
}

.result-meta {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-bottom: 12px;
}

/* Mode table */
.mode-table-wrap {
  max-height: 200px;
  overflow-y: auto;
}

.mode-table {
  width: 100%;
  font-size: 13px;
  border-collapse: collapse;
}

.mode-table th {
  text-align: left;
  padding: 6px 10px;
  border-bottom: 1px solid var(--color-border);
  font-size: 11px;
  text-transform: uppercase;
  color: var(--color-text-secondary);
}

.mode-table td {
  padding: 6px 10px;
}

.mode-table tr.closest-mode {
  background-color: rgba(234, 88, 12, 0.1);
  font-weight: 700;
}

.font-mono {
  font-family: 'SF Mono', 'Menlo', monospace;
}

.mode-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 11px;
  color: #fff;
}

/* Metrics grid */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px;
}

.metric-card {
  padding: 14px;
  border-radius: 8px;
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
}

.metric-label {
  font-size: 11px;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  margin-bottom: 6px;
}

.metric-value {
  font-size: 20px;
  font-weight: 700;
}

.metric-value.text-sm {
  font-size: 14px;
}

.text-safe {
  color: #22c55e;
}

.text-warn {
  color: #eab308;
}

.text-danger {
  color: #ef4444;
}

/* No data */
.no-data {
  font-size: 13px;
  color: var(--color-text-secondary);
  text-align: center;
  padding: 24px;
}
</style>
