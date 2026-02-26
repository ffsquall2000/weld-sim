<template>
  <div class="p-6 max-w-4xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('report.title') }}</h1>

    <!-- Loading -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <span style="color: var(--color-text-secondary)">{{ $t('common.loading') }}</span>
    </div>

    <!-- No data -->
    <div v-else-if="!recipe" class="text-center py-20">
      <p style="color: var(--color-text-secondary)">{{ $t('report.noRecipe') }}</p>
      <router-link to="/history" class="btn-secondary inline-block mt-4">
        {{ $t('history.title') }}
      </router-link>
    </div>

    <template v-else>
      <!-- Recipe Info Header -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('report.recipeInfo') }}</h2>
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
          <div>
            <span style="color: var(--color-text-secondary)">{{ $t('result.recipeId') }}</span>
            <p class="font-mono mt-1">{{ recipe.recipe_id.slice(0, 12) }}...</p>
          </div>
          <div>
            <span style="color: var(--color-text-secondary)">{{ $t('result.application') }}</span>
            <p class="mt-1">{{ recipe.application }}</p>
          </div>
          <div>
            <span style="color: var(--color-text-secondary)">{{ $t('result.createdAt') }}</span>
            <p class="mt-1">{{ formatDate(recipe.created_at) }}</p>
          </div>
        </div>
      </section>

      <!-- Parameter Preview Table -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('report.paramPreview') }}</h2>
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr style="border-bottom: 1px solid var(--color-border)">
                <th class="text-left py-2 px-3">{{ $t('result.parameters') }}</th>
                <th class="text-right py-2 px-3 font-mono">{{ $t('result.value') }}</th>
                <th class="text-right py-2 px-3 font-mono">{{ $t('result.safeRange') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(val, key, idx) in recipe.parameters"
                :key="key"
                :style="{ backgroundColor: idx % 2 === 0 ? 'transparent' : 'var(--color-bg-card)' }"
              >
                <td class="py-2 px-3">{{ formatParamName(String(key)) }}</td>
                <td class="py-2 px-3 text-right font-mono">{{ formatNumber(val) }}</td>
                <td class="py-2 px-3 text-right font-mono" style="color: var(--color-text-secondary)">
                  {{ formatSafeRange(String(key)) }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Export Actions -->
      <section class="card mb-6">
        <h2 class="text-lg font-semibold mb-3">{{ $t('report.exportSection') }}</h2>
        <div class="flex flex-wrap gap-3">
          <button
            v-for="fmt in formats"
            :key="fmt.key"
            class="btn-primary"
            :disabled="exportState[fmt.key] === 'loading'"
            @click="handleExport(fmt.key)"
          >
            <span v-if="exportState[fmt.key] === 'loading'">{{ $t('report.downloading') }}</span>
            <span v-else>{{ fmt.label }}</span>
          </button>
        </div>

        <!-- Download links -->
        <div v-if="Object.keys(downloadLinks).length > 0" class="mt-4 space-y-2">
          <div
            v-for="(url, fmt) in downloadLinks"
            :key="fmt"
            class="flex items-center gap-3 text-sm"
          >
            <span style="color: var(--color-success)" class="font-semibold">
              {{ $t('report.exported') }}: {{ String(fmt).toUpperCase() }}
            </span>
            <a
              :href="url"
              target="_blank"
              class="underline"
              style="color: var(--color-accent-blue)"
            >
              {{ $t('report.download') }}
            </a>
          </div>
        </div>
      </section>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, reactive } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'
import { recipesApi } from '@/api/recipes'
import { reportsApi } from '@/api/reports'
import type { SimulateResponse } from '@/api/simulation'

const route = useRoute()
const { t } = useI18n()
const calcStore = useCalculationStore()

const recipe = ref<SimulateResponse | null>(null)
const loading = ref(true)

type ExportStatus = 'idle' | 'loading' | 'done'
const exportState = reactive<Record<string, ExportStatus>>({
  json: 'idle',
  excel: 'idle',
  pdf: 'idle',
})
const downloadLinks = reactive<Record<string, string>>({})

const formats = [
  { key: 'json', label: t('report.exportJson') },
  { key: 'excel', label: t('report.exportExcel') },
  { key: 'pdf', label: t('report.exportPdf') },
]

async function handleExport(format: string) {
  if (!recipe.value) return
  exportState[format] = 'loading'
  try {
    const response = await reportsApi.export(recipe.value.recipe_id, format)
    const filename = (response.data as { filename?: string })?.filename
    if (filename) {
      downloadLinks[format] = reportsApi.downloadUrl(filename)
    }
    exportState[format] = 'done'
  } catch {
    exportState[format] = 'idle'
  }
}

function formatSafeRange(key: string): string {
  if (!recipe.value) return '\u2014'
  const sw = recipe.value.safety_window[key]
  if (!sw) return '\u2014'
  return `${formatNumber(sw[0])} \u2013 ${formatNumber(sw[1])}`
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

  if (calcStore.result && calcStore.result.recipe_id === id) {
    recipe.value = calcStore.result
  } else {
    try {
      const response = await recipesApi.get(id)
      recipe.value = response.data as SimulateResponse
    } catch {
      recipe.value = null
    }
  }

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
  transition: opacity 0.2s;
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
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
