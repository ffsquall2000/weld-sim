<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('fatigue.title') }}</h1>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left: Parameter Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('fatigue.paramPanel') }}</h2>

        <!-- Material Properties -->
        <div>
          <label class="label-text">{{ $t('fatigue.material') }}</label>
          <select v-model="form.material" class="input-field w-full">
            <option v-for="mat in materials" :key="mat.name" :value="mat.name">
              {{ mat.name }}
            </option>
          </select>
        </div>

        <!-- Loading Parameters -->
        <div>
          <label class="label-text">{{ $t('fatigue.loading') }}</label>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">
                {{ $t('fatigue.stressAmplitude') }} (MPa)
              </label>
              <input
                v-model.number="form.stress_amplitude_mpa"
                type="number"
                class="input-field w-full"
                min="1"
                step="1"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">
                {{ $t('fatigue.meanStress') }} (MPa)
              </label>
              <input
                v-model.number="form.mean_stress_mpa"
                type="number"
                class="input-field w-full"
                step="1"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">
                {{ $t('fatigue.frequency') }} (Hz)
              </label>
              <input
                v-model.number="form.frequency_hz"
                type="number"
                class="input-field w-full"
                min="1"
                step="100"
              />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">
                {{ $t('fatigue.targetCycles') }}
              </label>
              <input
                v-model.number="form.target_cycles"
                type="number"
                class="input-field w-full"
                min="1000"
                step="1000000"
              />
            </div>
          </div>
        </div>

        <!-- Surface Finish -->
        <div>
          <label class="label-text">{{ $t('fatigue.surfaceFinish') }}</label>
          <select v-model="form.surface_finish" class="input-field w-full">
            <option value="polished">{{ $t('fatigue.surfacePolished') }}</option>
            <option value="ground">{{ $t('fatigue.surfaceGround') }}</option>
            <option value="machined">{{ $t('fatigue.surfaceMachined') }}</option>
            <option value="as_forged">{{ $t('fatigue.surfaceForged') }}</option>
          </select>
        </div>

        <!-- Stress Concentration -->
        <div>
          <label class="label-text">{{ $t('fatigue.stressConcentration') }}</label>
          <input
            v-model.number="form.stress_concentration_factor"
            type="number"
            class="input-field w-full"
            min="1"
            step="0.1"
          />
        </div>

        <button class="btn-primary w-full" :disabled="analyzing" @click="runAnalysis">
          {{ analyzing ? $t('fatigue.analyzing') : $t('fatigue.analyze') }}
        </button>
      </div>

      <!-- Center: S-N Chart + Goodman Diagram -->
      <div class="card flex flex-col space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('fatigue.snCurve') }}</h2>

        <!-- S-N Curve -->
        <div v-if="result" class="flex-1" style="min-height: 300px">
          <SNChart
            :stress-levels="result.sn_curve.stress_levels"
            :cycles-to-failure="result.sn_curve.cycles_to_failure"
            :operating-point="operatingPoint"
          />
        </div>
        <div
          v-else
          class="flex-1 flex items-center justify-center"
          style="min-height: 300px; background: var(--color-bg-card); border-radius: 8px"
        >
          <span style="color: var(--color-text-secondary)">{{ $t('fatigue.noResults') }}</span>
        </div>

        <!-- Goodman Diagram (simplified display) -->
        <div v-if="result" class="p-4 rounded" style="background: var(--color-bg-card)">
          <h3 class="text-sm font-semibold mb-2">{{ $t('fatigue.goodmanDiagram') }}</h3>
          <div class="grid grid-cols-3 gap-2 text-sm">
            <div>
              <span style="color: var(--color-text-secondary)">Sut:</span>
              <span class="font-bold ml-1">{{ result.goodman.ultimate_strength_mpa }} MPa</span>
            </div>
            <div>
              <span style="color: var(--color-text-secondary)">Se:</span>
              <span class="font-bold ml-1">{{ result.goodman.endurance_limit_mpa.toFixed(1) }} MPa</span>
            </div>
            <div>
              <span style="color: var(--color-text-secondary)">{{ $t('fatigue.margin') }}:</span>
              <span class="font-bold ml-1" :style="{ color: marginColor }">
                {{ result.goodman.safety_margin.toFixed(2) }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Right: Results Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('fatigue.resultsPanel') }}</h2>

        <!-- Safety Factor -->
        <div v-if="result">
          <div
            class="p-4 rounded-lg"
            :style="{ backgroundColor: safetyBg, border: `1px solid ${safetyBorder}` }"
          >
            <div class="text-sm" style="color: var(--color-text-secondary)">
              {{ $t('fatigue.safetyFactor') }}
            </div>
            <div class="text-3xl font-bold" :style="{ color: safetyColor }">
              {{ result.safety_factor.toFixed(2) }}
            </div>
            <div class="text-sm mt-1" :style="{ color: safetyColor }">
              {{ safetyLabel }}
            </div>
          </div>
        </div>

        <!-- Life Prediction -->
        <div v-if="result" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('fatigue.lifePrediction') }}</h3>
          <div class="grid grid-cols-1 gap-2">
            <div class="p-3 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('fatigue.predictedCycles') }}</span>
              <div class="font-bold text-lg">{{ formatCycles(result.predicted_cycles) }}</div>
            </div>
            <div class="p-3 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('fatigue.operatingHours') }}</span>
              <div class="font-bold">{{ result.operating_hours.toFixed(1) }} h</div>
            </div>
            <div class="p-3 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('fatigue.damageAccumulation') }}</span>
              <div class="font-bold">{{ (result.damage_ratio * 100).toFixed(2) }}%</div>
            </div>
          </div>
        </div>

        <!-- Critical Locations -->
        <div v-if="result?.critical_locations?.length" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('fatigue.criticalLocations') }}</h3>
          <div class="max-h-40 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">{{ $t('fatigue.location') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('fatigue.stressMpa') }}</th>
                  <th class="text-center py-1 px-2">{{ $t('fatigue.risk') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(loc, i) in result.critical_locations" :key="i">
                  <td class="py-1 px-2 font-mono text-xs">
                    ({{ loc.x.toFixed(1) }}, {{ loc.y.toFixed(1) }}, {{ loc.z.toFixed(1) }})
                  </td>
                  <td class="py-1 px-2 text-right font-mono">{{ loc.stress_mpa.toFixed(1) }}</td>
                  <td class="py-1 px-2 text-center">
                    <span
                      class="px-2 py-0.5 rounded text-xs"
                      :style="{ backgroundColor: riskColor(loc.risk), color: '#fff' }"
                    >{{ loc.risk }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Recommendations -->
        <div v-if="result?.recommendations?.length" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('fatigue.recommendations') }}</h3>
          <ul class="text-sm space-y-1">
            <li
              v-for="(rec, i) in result.recommendations"
              :key="i"
              style="color: var(--color-text-secondary)"
            >
              &bull; {{ rec }}
            </li>
          </ul>
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
          {{ $t('fatigue.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import SNChart from '@/components/charts/SNChart.vue'
import apiClient from '@/api/client'

const { t } = useI18n()

const analyzing = ref(false)
const error = ref<string | null>(null)

interface FatigueForm {
  material: string
  stress_amplitude_mpa: number
  mean_stress_mpa: number
  frequency_hz: number
  target_cycles: number
  surface_finish: string
  stress_concentration_factor: number
}

const form = ref<FatigueForm>({
  material: 'Titanium Ti-6Al-4V',
  stress_amplitude_mpa: 200,
  mean_stress_mpa: 50,
  frequency_hz: 20000,
  target_cycles: 1e8,
  surface_finish: 'ground',
  stress_concentration_factor: 1.5,
})

const materials = [
  { name: 'Titanium Ti-6Al-4V', Su: 1000, Se: 510 },
  { name: 'Steel D2', Su: 2000, Se: 800 },
  { name: 'Aluminum 7075-T6', Su: 570, Se: 160 },
  { name: 'M2 High Speed Steel', Su: 1800, Se: 700 },
  { name: 'HAP72 Powder HSS', Su: 2200, Se: 900 },
]

interface CriticalLocation {
  x: number
  y: number
  z: number
  stress_mpa: number
  risk: string
}

interface SNData {
  stress_levels: number[]
  cycles_to_failure: number[]
}

interface GoodmanData {
  ultimate_strength_mpa: number
  endurance_limit_mpa: number
  safety_margin: number
}

interface FatigueResult {
  safety_factor: number
  predicted_cycles: number
  operating_hours: number
  damage_ratio: number
  sn_curve: SNData
  goodman: GoodmanData
  critical_locations: CriticalLocation[]
  recommendations: string[]
}

const result = ref<FatigueResult | null>(null)

const operatingPoint = computed(() => {
  if (!result.value) return null
  return {
    stress: form.value.stress_amplitude_mpa,
    cycles: result.value.predicted_cycles,
  }
})

const safetyColor = computed(() => {
  if (!result.value) return ''
  const sf = result.value.safety_factor
  if (sf >= 2.0) return '#22c55e'
  if (sf >= 1.5) return '#eab308'
  return '#ef4444'
})

const safetyBg = computed(() => {
  if (!result.value) return ''
  const sf = result.value.safety_factor
  if (sf >= 2.0) return 'rgba(34,197,94,0.1)'
  if (sf >= 1.5) return 'rgba(234,179,8,0.1)'
  return 'rgba(239,68,68,0.1)'
})

const safetyBorder = computed(() => {
  if (!result.value) return ''
  const sf = result.value.safety_factor
  if (sf >= 2.0) return 'rgba(34,197,94,0.3)'
  if (sf >= 1.5) return 'rgba(234,179,8,0.3)'
  return 'rgba(239,68,68,0.3)'
})

const safetyLabel = computed(() => {
  if (!result.value) return ''
  const sf = result.value.safety_factor
  if (sf >= 2.0) return t('fatigue.safetyExcellent')
  if (sf >= 1.5) return t('fatigue.safetyAcceptable')
  return t('fatigue.safetyCritical')
})

const marginColor = computed(() => {
  if (!result.value) return ''
  const m = result.value.goodman.safety_margin
  if (m >= 0.5) return '#22c55e'
  if (m >= 0.2) return '#eab308'
  return '#ef4444'
})

function riskColor(risk: string): string {
  switch (risk) {
    case 'low': return '#22c55e'
    case 'medium': return '#eab308'
    case 'high': return '#ef4444'
    default: return '#6b7280'
  }
}

function formatCycles(cycles: number): string {
  if (cycles >= 1e9) return `${(cycles / 1e9).toFixed(2)}G`
  if (cycles >= 1e6) return `${(cycles / 1e6).toFixed(2)}M`
  if (cycles >= 1e3) return `${(cycles / 1e3).toFixed(1)}K`
  return cycles.toFixed(0)
}

async function runAnalysis() {
  analyzing.value = true
  error.value = null
  result.value = null

  try {
    // For now, generate mock results since backend endpoint may not exist
    // In production, this would call: const res = await apiClient.post<FatigueResult>('/fatigue/analyze', form.value)

    // Mock S-N curve data (Basquin equation)
    const Su = materials.find(m => m.name === form.value.material)?.Su ?? 1000
    const Se = materials.find(m => m.name === form.value.material)?.Se ?? 400
    const stressLevels: number[] = []
    const cyclesToFailure: number[] = []

    for (let n = 3; n <= 10; n += 0.5) {
      const N = Math.pow(10, n)
      const S = Se + (Su - Se) * Math.pow(1000 / N, 0.1)
      stressLevels.push(Math.max(Se, S))
      cyclesToFailure.push(N)
    }

    // Calculate safety factor using Goodman criterion
    const stressAmp = form.value.stress_amplitude_mpa
    const meanStress = form.value.mean_stress_mpa
    const Kt = form.value.stress_concentration_factor
    const effectiveStress = stressAmp * Kt
    const safetyFactor = 1 / (effectiveStress / Se + meanStress / Su)

    // Predict cycles to failure
    const b = -0.1 // Basquin exponent
    const predictedCycles = Math.pow((effectiveStress - Se) / (Su - Se), 1 / b) * 1e6

    result.value = {
      safety_factor: safetyFactor,
      predicted_cycles: Math.max(1e3, predictedCycles),
      operating_hours: predictedCycles / form.value.frequency_hz / 3600,
      damage_ratio: form.value.target_cycles / predictedCycles,
      sn_curve: {
        stress_levels: stressLevels,
        cycles_to_failure: cyclesToFailure,
      },
      goodman: {
        ultimate_strength_mpa: Su,
        endurance_limit_mpa: Se,
        safety_margin: safetyFactor - 1,
      },
      critical_locations: [
        { x: 0, y: 40, z: 0, stress_mpa: effectiveStress, risk: safetyFactor < 1.5 ? 'high' : 'medium' },
        { x: 12, y: 35, z: 5, stress_mpa: effectiveStress * 0.8, risk: 'low' },
      ],
      recommendations: safetyFactor < 1.5
        ? [t('fatigue.recReduceStress'), t('fatigue.recImproveFinish'), t('fatigue.recAddFillet')]
        : [t('fatigue.recDesignOk')],
    }
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('fatigue.analyzeFailed')
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
