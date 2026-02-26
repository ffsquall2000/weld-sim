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
                  v-for="(mode, i) in result.modal_frequencies"
                  :key="i"
                  :style="{
                    backgroundColor: mode.is_closest ? 'rgba(234,88,12,0.1)' : 'transparent',
                    fontWeight: mode.is_closest ? '700' : '400',
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
                    {{ mode.deviation_percent?.toFixed(1) }}%
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
                  <td class="py-1 px-2">{{ hs.location }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ hs.stress_mpa?.toFixed(1) }}</td>
                  <td class="py-1 px-2 text-center">
                    <span
                      class="px-2 py-0.5 rounded text-xs"
                      :style="{ backgroundColor: severityColor(hs.severity), color: '#fff' }"
                    >{{ hs.severity }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Harmonic Response -->
        <div v-if="result?.harmonic_response" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('acoustic.harmonicResponse') }}</h3>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.peakAmplitude') }}</span>
              <div class="font-bold">{{ result.harmonic_response.peak_amplitude_um?.toFixed(2) }} um</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.qualityFactor') }}</span>
              <div class="font-bold">{{ result.harmonic_response.quality_factor?.toFixed(0) }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.bandwidth') }}</span>
              <div class="font-bold">{{ result.harmonic_response.bandwidth_hz?.toFixed(0) }} Hz</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('acoustic.impedanceOhm') }}</span>
              <div class="font-bold">{{ result.harmonic_response.impedance_ohm?.toFixed(1) }}</div>
            </div>
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
  { name: 'Steel H13', E_gpa: 210 },
  { name: 'Copper C110', E_gpa: 117 },
  { name: 'Tungsten Carbide', E_gpa: 620 },
  { name: 'Nickel 200', E_gpa: 204 },
  { name: 'Inconel 718', E_gpa: 200 },
  { name: 'Beryllium Copper', E_gpa: 131 },
  { name: 'Monel 400', E_gpa: 179 },
]

interface ModalFrequency {
  frequency_hz: number
  mode_type: string
  deviation_percent: number
  is_closest: boolean
}

interface StressHotspot {
  location: string
  stress_mpa: number
  severity: string
}

interface HarmonicResponse {
  peak_amplitude_um: number
  quality_factor: number
  bandwidth_hz: number
  impedance_ohm: number
}

interface AcousticResult {
  modal_frequencies: ModalFrequency[]
  amplitude_uniformity: number
  stress_hotspots: StressHotspot[]
  harmonic_response: HarmonicResponse | null
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
