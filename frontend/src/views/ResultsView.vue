<template>
  <div class="p-6 max-w-5xl mx-auto">
    <!-- Loading state -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <span style="color: var(--color-text-secondary)">{{ $t('common.loading') }}</span>
    </div>

    <!-- No result state -->
    <div v-else-if="!data" class="text-center py-20">
      <p class="text-lg mb-4" style="color: var(--color-text-secondary)">{{ $t('result.noResult') }}</p>
      <router-link to="/calculate" class="btn-primary inline-block">
        {{ $t('result.goCalculate') }}
      </router-link>
    </div>

    <!-- Results display -->
    <template v-else>
      <!-- Header -->
      <div class="flex items-center justify-between mb-6">
        <h1 class="text-2xl font-bold">{{ $t('result.title') }}</h1>
        <div class="flex gap-3">
          <router-link :to="`/reports/${data.recipe_id}`" class="btn-primary">
            {{ $t('result.exportReport') }}
          </router-link>
          <router-link to="/calculate" class="btn-secondary">
            {{ $t('result.newCalculation') }}
          </router-link>
        </div>
      </div>

      <!-- Recipe info bar -->
      <div
        class="flex flex-wrap gap-4 mb-6 p-3 rounded-lg text-sm"
        style="background-color: var(--color-bg-secondary); border: 1px solid var(--color-border); color: var(--color-text-secondary)"
      >
        <span><strong>{{ $t('result.recipeId') }}:</strong> {{ data.recipe_id.slice(0, 8) }}...</span>
        <span><strong>{{ $t('result.application') }}:</strong> {{ data.application }}</span>
        <span><strong>{{ $t('result.createdAt') }}:</strong> {{ formatDate(data.created_at) }}</span>
      </div>

      <!-- Key parameter cards -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <ParameterCard
          v-for="card in primaryCards"
          :key="card.key"
          :label="card.label"
          :value="card.value"
          :unit="card.unit"
          :safe-min="card.safeMin"
          :safe-max="card.safeMax"
        />
      </div>

      <!-- Secondary cards (power) -->
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
        <ParameterCard
          v-for="card in secondaryCards"
          :key="card.key"
          :label="card.label"
          :value="card.value"
          :unit="card.unit"
          :safe-min="card.safeMin"
          :safe-max="card.safeMax"
        />
      </div>

      <!-- Risk Assessment -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('result.risk') }}</h2>
        <div class="flex flex-wrap gap-2">
          <RiskBadge
            v-for="(level, name) in data.risk_assessment"
            :key="name"
            :label="formatParamName(String(name))"
            :level="String(level)"
          />
        </div>
      </section>

      <!-- Parameter Table -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('result.parameterTable') }}</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr style="border-bottom: 1px solid var(--color-border)">
                <th class="text-left py-2 px-3">{{ $t('result.parameters') }}</th>
                <th class="text-right py-2 px-3 font-mono">{{ $t('result.value') }}</th>
                <th class="text-right py-2 px-3 font-mono">{{ $t('result.safeMin') }}</th>
                <th class="text-right py-2 px-3 font-mono">{{ $t('result.safeMax') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(val, key, idx) in data.parameters"
                :key="key"
                :style="{ backgroundColor: idx % 2 === 0 ? 'transparent' : 'var(--color-bg-card)' }"
              >
                <td class="py-2 px-3">{{ formatParamName(String(key)) }}</td>
                <td class="py-2 px-3 text-right font-mono">{{ formatNumber(val) }}</td>
                <td class="py-2 px-3 text-right font-mono" style="color: var(--color-text-secondary)">
                  {{ safeRange(String(key))?.[0] !== undefined ? formatNumber(safeRange(String(key))![0]) : '—' }}
                </td>
                <td class="py-2 px-3 text-right font-mono" style="color: var(--color-text-secondary)">
                  {{ safeRange(String(key))?.[1] !== undefined ? formatNumber(safeRange(String(key))![1]) : '—' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Validation -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('result.validation') }}</h2>
        <div class="flex items-center gap-2">
          <span
            class="text-lg font-bold"
            :style="{ color: data.validation.status === 'pass' ? 'var(--color-success)' : 'var(--color-danger)' }"
          >
            {{ data.validation.status === 'pass' ? '\u2705' : '\u274C' }}
            {{ data.validation.status === 'pass' ? $t('result.pass') : $t('result.fail') }}
          </span>
        </div>
        <ul v-if="data.validation.messages.length > 0" class="mt-2 space-y-1">
          <li
            v-for="(msg, i) in data.validation.messages"
            :key="i"
            class="text-sm"
            style="color: var(--color-text-secondary)"
          >
            {{ msg }}
          </li>
        </ul>
      </section>

      <!-- Recommendations -->
      <section v-if="data.recommendations.length > 0" class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('result.recommendations') }}</h2>
        <ul class="space-y-2">
          <li
            v-for="(rec, i) in data.recommendations"
            :key="i"
            class="flex items-start gap-2 text-sm"
            style="color: var(--color-text-secondary)"
          >
            <span style="color: var(--color-accent-orange)">&#8226;</span>
            {{ rec }}
          </li>
        </ul>
      </section>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { useCalculationStore } from '@/stores/calculation'
import { recipesApi } from '@/api/recipes'
import type { SimulateResponse } from '@/api/simulation'
import ParameterCard from '@/components/charts/ParameterCard.vue'
import RiskBadge from '@/components/charts/RiskBadge.vue'

const route = useRoute()
const calcStore = useCalculationStore()

const data = ref<SimulateResponse | null>(null)
const loading = ref(true)

// Key parameter cards to highlight
const paramConfig: { key: string; label: string; unit: string }[] = [
  { key: 'amplitude_um', label: 'Amplitude', unit: '\u03BCm' },
  { key: 'pressure_mpa', label: 'Pressure', unit: 'MPa' },
  { key: 'energy_j', label: 'Energy', unit: 'J' },
  { key: 'weld_time_ms', label: 'Weld Time', unit: 'ms' },
]

const powerConfig: { key: string; label: string; unit: string }[] = [
  { key: 'interface_power_w', label: 'Interface Power', unit: 'W' },
  { key: 'machine_power_w', label: 'Machine Power', unit: 'W' },
]

function buildCards(config: { key: string; label: string; unit: string }[]) {
  if (!data.value) return []
  return config
    .filter((c) => data.value!.parameters[c.key] !== undefined)
    .map((c) => {
      const sw = data.value!.safety_window[c.key]
      return {
        key: c.key,
        label: c.label,
        value: data.value!.parameters[c.key] as number,
        unit: c.unit,
        safeMin: sw?.[0],
        safeMax: sw?.[1],
      }
    })
}

const primaryCards = ref<ReturnType<typeof buildCards>>([])
const secondaryCards = ref<ReturnType<typeof buildCards>>([])

function safeRange(key: string): [number, number] | undefined {
  return data.value?.safety_window[key] as [number, number] | undefined
}

function formatParamName(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function formatNumber(val: number): string {
  return Number.isInteger(val) ? val.toString() : val.toFixed(2)
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

onMounted(async () => {
  const id = route.params.id as string

  // First try Pinia store (navigating from wizard)
  if (calcStore.result && calcStore.result.recipe_id === id) {
    data.value = calcStore.result
  } else {
    // Otherwise fetch from API
    try {
      const response = await recipesApi.get(id)
      data.value = response.data as SimulateResponse
    } catch {
      data.value = null
    }
  }

  primaryCards.value = buildCards(paramConfig)
  secondaryCards.value = buildCards(powerConfig)
  loading.value = false
})
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1rem 1.25rem;
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
  text-decoration: none;
  transition: opacity 0.2s;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn-secondary {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
  text-decoration: none;
  transition: border-color 0.2s;
}

.btn-secondary:hover {
  border-color: var(--color-accent-orange);
}
</style>
