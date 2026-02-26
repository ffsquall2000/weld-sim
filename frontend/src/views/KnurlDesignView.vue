<template>
  <div class="p-6 max-w-6xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('knurlDesign.title') }}</h1>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Left: Input Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('knurlDesign.inputPanel') }}</h2>

        <!-- Material Inputs -->
        <div>
          <label class="label-text">{{ $t('knurlDesign.materialSection') }}</label>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.upperMaterial') }}</label>
              <select v-model="form.upper_material" class="input-field w-full">
                <option v-for="m in foilMaterials" :key="m" :value="m">{{ m }}</option>
              </select>
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.lowerMaterial') }}</label>
              <select v-model="form.lower_material" class="input-field w-full">
                <option v-for="m in substrateMaterials" :key="m" :value="m">{{ m }}</option>
              </select>
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.upperHardness') }}</label>
              <input v-model.number="form.upper_hardness_hv" type="number" class="input-field w-full" min="1" step="1" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.lowerHardness') }}</label>
              <input v-model.number="form.lower_hardness_hv" type="number" class="input-field w-full" min="1" step="1" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.frictionCoeff') }}</label>
              <input v-model.number="form.mu_base" type="number" class="input-field w-full" min="0.01" max="1.0" step="0.05" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.weldArea') }}</label>
              <input v-model.number="form.weld_area_mm2" type="number" class="input-field w-full" min="1" step="5" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.foilLayers') }}</label>
              <input v-model.number="form.n_layers" type="number" class="input-field w-full" min="1" step="1" />
            </div>
          </div>
        </div>

        <!-- Welding Parameters -->
        <div>
          <label class="label-text">{{ $t('knurlDesign.weldParams') }}</label>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.frequency') }}</label>
              <select v-model="form.frequency_khz" class="input-field w-full">
                <option :value="20">20 kHz</option>
                <option :value="35">35 kHz</option>
                <option :value="40">40 kHz</option>
              </select>
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.amplitude') }}</label>
              <input v-model.number="form.amplitude_um" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.pressure') }}</label>
              <input v-model.number="form.pressure_mpa" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
          </div>
        </div>

        <!-- Optimization Settings -->
        <div>
          <label class="label-text">{{ $t('knurlDesign.optimizationSettings') }}</label>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('knurlDesign.maxCandidates') }}</label>
              <input v-model.number="form.max_results" type="number" class="input-field w-full" min="5" max="50" step="5" />
            </div>
          </div>
        </div>

        <button class="btn-primary w-full" :disabled="optimizing" @click="runOptimization">
          {{ optimizing ? $t('knurlDesign.optimizing') : $t('knurlDesign.optimize') }}
        </button>
      </div>

      <!-- Right: Results Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('knurlDesign.resultsPanel') }}</h2>

        <!-- Best Config Details -->
        <div v-if="bestConfig" class="space-y-3">
          <h3 class="text-sm font-semibold" style="color: var(--color-accent-orange)">
            {{ $t('knurlDesign.bestConfig') }}
          </h3>
          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.patternType') }}</span>
              <div class="font-bold">{{ bestConfig.knurl_type }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.pitch') }}</span>
              <div class="font-bold">{{ bestConfig.pitch_mm?.toFixed(2) }} mm</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.toothWidth') }}</span>
              <div class="font-bold">{{ bestConfig.tooth_width_mm?.toFixed(2) }} mm</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.depth') }}</span>
              <div class="font-bold">{{ bestConfig.depth_mm?.toFixed(2) }} mm</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.score') }}</span>
              <div class="font-bold" style="color: var(--color-accent-orange)">{{ bestConfig.overall_score?.toFixed(3) }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('knurlDesign.energyCoupling') }}</span>
              <div class="font-bold">{{ ((bestConfig.energy_coupling_efficiency ?? 0) * 100).toFixed(1) }}%</div>
            </div>
          </div>
        </div>

        <!-- Recommendations Table -->
        <div v-if="recommendations.length > 0" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('knurlDesign.rankedRecommendations') }}</h3>
          <div class="max-h-72 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">#</th>
                  <th class="text-left py-1 px-2">{{ $t('knurlDesign.pattern') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('knurlDesign.pitch') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('knurlDesign.depth') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('knurlDesign.score') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="(rec, i) in recommendations"
                  :key="i"
                  :style="{ backgroundColor: i === 0 ? 'rgba(234,88,12,0.1)' : 'transparent', fontWeight: i === 0 ? '700' : '400' }"
                >
                  <td class="py-1 px-2">{{ i + 1 }}</td>
                  <td class="py-1 px-2">{{ rec.knurl_type }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ rec.pitch_mm?.toFixed(2) }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ rec.depth_mm?.toFixed(2) }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ rec.overall_score?.toFixed(3) }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Pareto Front -->
        <div v-if="paretoFront.length > 0" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('knurlDesign.paretoFront') }}</h3>
          <div class="max-h-48 overflow-y-auto">
            <table class="w-full text-sm">
              <thead>
                <tr style="border-bottom: 1px solid var(--color-border)">
                  <th class="text-left py-1 px-2">{{ $t('knurlDesign.pattern') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('knurlDesign.strengthScore') }}</th>
                  <th class="text-right py-1 px-2">{{ $t('knurlDesign.damageScore') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(p, i) in paretoFront" :key="i">
                  <td class="py-1 px-2">{{ p.knurl_type }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ p.energy_coupling_efficiency?.toFixed(3) }}</td>
                  <td class="py-1 px-2 text-right font-mono">{{ p.material_damage_index?.toFixed(3) }}</td>
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
          v-if="recommendations.length === 0 && !error"
          class="text-sm text-center py-12"
          style="color: var(--color-text-secondary)"
        >
          {{ $t('knurlDesign.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import apiClient from '@/api/client'

const { t } = useI18n()

const optimizing = ref(false)
const error = ref<string | null>(null)

interface KnurlForm {
  upper_material: string
  lower_material: string
  upper_hardness_hv: number
  lower_hardness_hv: number
  mu_base: number
  weld_area_mm2: number
  frequency_khz: number
  amplitude_um: number
  pressure_mpa: number
  n_layers: number
  max_results: number
}

const form = ref<KnurlForm>({
  upper_material: 'Al',
  lower_material: 'Cu',
  upper_hardness_hv: 23,
  lower_hardness_hv: 50,
  mu_base: 0.3,
  weld_area_mm2: 75,
  frequency_khz: 20,
  amplitude_um: 25,
  pressure_mpa: 0.5,
  n_layers: 40,
  max_results: 20,
})

const foilMaterials = ['Al', 'Cu', 'Ni', 'Al 3003', 'Stainless 304']
const substrateMaterials = ['Cu', 'Ni', 'Al', 'Steel SPCC']

interface KnurlRecommendation {
  knurl_type: string
  pitch_mm: number
  tooth_width_mm: number
  depth_mm: number
  direction: string
  effective_friction: number
  energy_coupling_efficiency: number
  material_damage_index: number
  overall_score: number
  rank: number
}

interface ParetoPoint {
  knurl_type: string
  energy_coupling_efficiency: number
  material_damage_index: number
  overall_score: number
}

const recommendations = ref<KnurlRecommendation[]>([])
const paretoFront = ref<ParetoPoint[]>([])
const bestConfig = ref<KnurlRecommendation | null>(null)

async function runOptimization() {
  optimizing.value = true
  error.value = null
  recommendations.value = []
  paretoFront.value = []
  bestConfig.value = null

  try {
    const res = await apiClient.post<{
      recommendations: KnurlRecommendation[]
      pareto_front: ParetoPoint[]
      analysis_summary: Record<string, unknown>
    }>('/knurl/optimize', form.value, { timeout: 60000 })

    recommendations.value = res.data.recommendations ?? []
    paretoFront.value = res.data.pareto_front ?? []
    bestConfig.value = recommendations.value[0] || null
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('knurlDesign.optimizeFailed')
  } finally {
    optimizing.value = false
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
