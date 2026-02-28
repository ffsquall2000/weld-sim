<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('acoustic.title') }}</h1>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Left: Parameter Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('acoustic.paramPanel') }}</h2>

        <!-- Horn Geometry -->
        <div>
          <label class="label-text">{{ $t('acoustic.hornGeometry') }}</label>
          <div class="space-y-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('acoustic.hornType') }}</label>
              <select v-model="form.horn_type" class="input-field w-full">
                <option value="cylindrical">{{ $t('acoustic.typeCylindrical') }}</option>
                <option value="flat">{{ $t('acoustic.typeFlat') }}</option>
                <option value="exponential">{{ $t('acoustic.typeExponential') }}</option>
                <option value="blade">{{ $t('acoustic.typeBlade') }}</option>
                <option value="stepped">{{ $t('acoustic.typeStepped') }}</option>
                <option value="block">{{ $t('acoustic.typeBlock') }}</option>
              </select>
            </div>
            <div class="grid grid-cols-3 gap-2">
              <div>
                <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('acoustic.widthMm') }}</label>
                <input v-model.number="form.width_mm" type="number" class="input-field w-full" min="1" step="0.5" />
              </div>
              <div>
                <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('acoustic.heightMm') }}</label>
                <input v-model.number="form.height_mm" type="number" class="input-field w-full" min="1" step="0.5" />
              </div>
              <div>
                <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('acoustic.lengthMm') }}</label>
                <input v-model.number="form.length_mm" type="number" class="input-field w-full" min="1" step="0.5" />
              </div>
            </div>
          </div>
        </div>

        <!-- Material -->
        <div>
          <label class="label-text">{{ $t('acoustic.material') }}</label>
          <select v-model="form.material" class="input-field w-full">
            <option v-for="mat in materialOptions" :key="mat.name" :value="mat.name">
              {{ mat.name }} ({{ mat.E_gpa }} GPa)
            </option>
          </select>
        </div>

        <!-- Target Frequency -->
        <div>
          <label class="label-text">{{ $t('acoustic.targetFrequency') }}</label>
          <div class="flex items-center gap-2">
            <input v-model.number="form.frequency_khz" type="number" class="input-field flex-1" min="1" step="0.5" />
            <span class="text-sm" style="color: var(--color-text-secondary)">kHz</span>
          </div>
        </div>

        <!-- Mesh Density -->
        <div>
          <label class="label-text">{{ $t('acoustic.meshDensity') }}</label>
          <select v-model="form.mesh_density" class="input-field w-full">
            <option value="coarse">{{ $t('acoustic.meshCoarse') }}</option>
            <option value="medium">{{ $t('acoustic.meshMedium') }}</option>
            <option value="fine">{{ $t('acoustic.meshFine') }}</option>
          </select>
        </div>

        <button class="btn-primary w-full" :disabled="analyzing" @click="runAnalysis">
          {{ analyzing ? $t('acoustic.analyzing') : $t('acoustic.analyze') }}
        </button>
      </div>

      <!-- Right: Results Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('acoustic.resultsPanel') }}</h2>

        <!-- Modal Frequencies Table -->
        <div v-if="result" class="space-y-3">
          <h3 class="text-sm font-semibold" style="color: var(--color-accent-orange)">
            {{ $t('acoustic.modalFrequencies') }}
          </h3>
          <div class="max-h-56 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">#</th>
                  <th class="text-right py-1 px-2">{{ $t('acoustic.freqHz') }}</th>
                  <th class="text-left py-1 px-2">{{ $t('acoustic.modeType') }}</th>
                  <th class="text-center py-1 px-2">{{ $t('acoustic.deviation') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(mode, i) in result.modes"
                  :key="i"
                  :style="{
                    backgroundColor: isClosestMode(mode) ? 'rgba(234,88,12,0.1)' : 'transparent',
                    fontWeight: isClosestMode(mode) ? '700' : '400',
                  }"
                >
                  <td class="py-1 px-2">{{ i + 1 }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ mode.frequency_hz.toLocaleString() }}</td>
                  <td class="py-1 px-2">
                    <span
                      class="px-2 py-0.5 rounded text-xs"
                      :style="{ backgroundColor: modeColor(mode.mode_type), color: '#fff' }"
                    >{{ modeLabel(mode.mode_type) }}</span>
                  </td>
                  <td class="py-1 px-2 text-center font-mono">
                    {{ modeDeviation(mode).toFixed(1) }}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Amplitude Uniformity -->
        <div v-if="result?.amplitude_uniformity != null" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.amplitudeUniformity') }}</h3>
          <div class="p-3 rounded" :style="{ backgroundColor: uniformityBg, border: `1px solid ${uniformityBorder}` }">
            <div class="text-lg font-bold" :style="{ color: uniformityColor }">
              {{ (result.amplitude_uniformity * 100).toFixed(1) }}%
            </div>
            <div class="text-xs mt-1" :style="{ color: uniformityColor }">
              {{ uniformityLabel }}
            </div>
          </div>
        </div>

        <!-- Stress Hotspots -->
        <div v-if="result?.stress_hotspots && result.stress_hotspots.length > 0" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.stressHotspots') }}</h3>
          <div class="max-h-40 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">{{ $t('acoustic.location') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('acoustic.stressMpa') }}</th>
                  <th class="text-center py-1 px-2">{{ $t('acoustic.severity') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(hs, i) in result.stress_hotspots" :key="i">
                  <td class="py-1 px-2">({{ hs.location.map(v => v.toFixed(1)).join(', ') }})</td>
                  <td class="py-1 px-2 text-right font-mono">{{ hs.von_mises_mpa?.toFixed(1) }}</td>
                  <td class="py-1 px-2 text-center">
                    <span
                      class="px-2 py-0.5 rounded text-xs"
                      :style="{ backgroundColor: severityColor(hotspotSeverity(hs)), color: '#fff' }"
                    >{{ hotspotSeverity(hs) }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Modal Bar Chart -->
        <div v-if="result?.modes?.length" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.modeChart') }}</h3>
          <ModalBarChart
            :modes="result.modes.map((m, i) => ({ modeNumber: i + 1, frequency: m.frequency_hz, type: (m.mode_type as 'longitudinal' | 'flexural' | 'torsional' | 'unknown') }))"
            :target-frequency="result.target_frequency_hz"
            style="height: 200px"
          />
        </div>

        <!-- FRF Chart -->
        <div v-if="result?.harmonic_response" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.frfChart') }}</h3>
          <FRFChart
            :frequencies="result.harmonic_response.frequencies_hz"
            :amplitudes="result.harmonic_response.amplitudes"
            :target-frequency="result.target_frequency_hz"
            :peak-frequency="harmonicPeakFreq ?? undefined"
            :peak-amplitude="harmonicPeakAmplitude ?? undefined"
          />
        </div>

        <!-- Harmonic Response -->
        <div v-if="result?.harmonic_response" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.harmonicResponse') }}</h3>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.peakAmplitude') }}</span>
              <div class="font-bold">{{ harmonicPeakAmplitude?.toFixed(4) ?? '--' }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.peakFrequency') }}</span>
              <div class="font-bold">{{ harmonicPeakFreq?.toFixed(0) ?? '--' }} Hz</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.freqDeviation') }}</span>
              <div class="font-bold">{{ result.frequency_deviation_percent?.toFixed(2) }}%</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.maxStress') }}</span>
              <div class="font-bold">{{ result.stress_max_mpa?.toFixed(1) }} MPa</div>
            </div>
          </div>
          <!-- Sweep Data Points -->
          <div class="max-h-40 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-right py-1 px-2">{{ $t('acoustic.freqHz') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('acoustic.amplitude') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(f, i) in result.harmonic_response.frequencies_hz"
                  :key="i"
                  :style="{ fontWeight: (result.harmonic_response?.amplitudes[i] ?? 0) >= 0.95 ? '700' : '400' }"
                >
                  <td class="py-1 px-2 text-right font-mono">{{ f.toLocaleString() }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ result.harmonic_response?.amplitudes[i]?.toFixed(4) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Error -->
        <div
          v-if="error"
          class="p-3 rounded text-sm"
          style="background-color: rgba(220, 38, 38, 0.1); color: #dc2626"
        >
          {{ error }}
        </div>

        <!-- No results placeholder -->
        <div
          v-if="!result && !error"
          class="text-sm text-center py-12"
          style="color: var(--color-text-secondary)"
        >
          {{ $t('acoustic.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import apiClient from '@/api/client'
import FRFChart from '@/components/charts/FRFChart.vue'
import ModalBarChart from '@/components/charts/ModalBarChart.vue'

const { t } = useI18n()

const analyzing = ref(false)
const error = ref<string | null>(null)

interface AcousticForm {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  material: string
  frequency_khz: number
  mesh_density: string
}

const form = ref<AcousticForm>({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  frequency_khz: 20,
  mesh_density: 'medium',
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

interface ModeShape {
  frequency_hz: number
  mode_type: string
  participation_factor: number
  effective_mass_ratio: number
  displacement_max: number
}

interface StressHotspot {
  location: number[]  // [x, y, z]
  von_mises_mpa: number
  node_index: number
}

interface HarmonicResponse {
  frequencies_hz: number[]
  amplitudes: number[]
}

interface AcousticResult {
  modes: ModeShape[]
  closest_mode_hz: number
  target_frequency_hz: number
  frequency_deviation_percent: number
  amplitude_uniformity: number
  stress_hotspots: StressHotspot[]
  stress_max_mpa: number
  harmonic_response: HarmonicResponse | null
  node_count: number
  element_count: number
  solve_time_s: number
}

const result = ref<AcousticResult | null>(null)

const uniformityColor = computed(() => {
  if (!result.value) return ''
  const u = result.value.amplitude_uniformity
  if (u >= 0.9) return '#22c55e'
  if (u >= 0.7) return '#eab308'
  return '#ef4444'
})

const uniformityBg = computed(() => {
  if (!result.value) return ''
  const u = result.value.amplitude_uniformity
  if (u >= 0.9) return 'rgba(34,197,94,0.1)'
  if (u >= 0.7) return 'rgba(234,179,8,0.1)'
  return 'rgba(239,68,68,0.1)'
})

const uniformityBorder = computed(() => {
  if (!result.value) return ''
  const u = result.value.amplitude_uniformity
  if (u >= 0.9) return 'rgba(34,197,94,0.3)'
  if (u >= 0.7) return 'rgba(234,179,8,0.3)'
  return 'rgba(239,68,68,0.3)'
})

const uniformityLabel = computed(() => {
  if (!result.value) return ''
  const u = result.value.amplitude_uniformity
  if (u >= 0.9) return t('acoustic.uniformityExcellent')
  if (u >= 0.7) return t('acoustic.uniformityAcceptable')
  return t('acoustic.uniformityPoor')
})

function modeColor(type: string): string {
  switch (type) {
    case 'longitudinal': return '#22c55e'
    case 'flexural': return '#3b82f6'
    case 'torsional': return '#a855f7'
    default: return '#6b7280'
  }
}

function modeLabel(type: string): string {
  switch (type) {
    case 'longitudinal': return t('acoustic.modeLongitudinal')
    case 'flexural': return t('acoustic.modeFlexural')
    case 'torsional': return t('acoustic.modeTorsional')
    default: return type
  }
}

function severityColor(severity: string): string {
  switch (severity) {
    case 'low': return '#22c55e'
    case 'medium': return '#eab308'
    case 'high': return '#ef4444'
    default: return '#6b7280'
  }
}

function isClosestMode(mode: ModeShape): boolean {
  if (!result.value) return false
  return Math.abs(mode.frequency_hz - result.value.closest_mode_hz) < 1.0
}

function modeDeviation(mode: ModeShape): number {
  if (!result.value || result.value.target_frequency_hz <= 0) return 0
  return Math.abs(mode.frequency_hz - result.value.target_frequency_hz) / result.value.target_frequency_hz * 100
}

function hotspotSeverity(hs: StressHotspot): string {
  if (!result.value) return 'low'
  const maxStress = result.value.stress_max_mpa
  if (maxStress <= 0) return 'low'
  const ratio = hs.von_mises_mpa / maxStress
  if (ratio > 0.8) return 'high'
  if (ratio > 0.5) return 'medium'
  return 'low'
}

const harmonicPeakAmplitude = computed(() => {
  if (!result.value?.harmonic_response) return null
  return Math.max(...result.value.harmonic_response.amplitudes)
})

const harmonicPeakFreq = computed(() => {
  if (!result.value?.harmonic_response) return null
  const amps = result.value.harmonic_response.amplitudes
  const freqs = result.value.harmonic_response.frequencies_hz
  const maxIdx = amps.indexOf(Math.max(...amps))
  return freqs[maxIdx]
})

async function runAnalysis() {
  analyzing.value = true
  error.value = null
  result.value = null

  try {
    const res = await apiClient.post<AcousticResult>('/acoustic/analyze', form.value, { timeout: 120000 })
    result.value = res.data
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('acoustic.analyzeFailed')
  } finally {
    analyzing.value = false
  }
}
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
}
.label-text {
  display: block;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: 0.25rem;
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
.btn-primary:hover:not(:disabled) { opacity: 0.9; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
