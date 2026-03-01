<template>
  <div class="p-6 max-w-5xl mx-auto">
    <div class="flex items-center justify-between mb-6">
      <h1 class="text-2xl font-bold">{{ $t('history.title') }}</h1>
      <button class="btn-secondary" @click="fetchHistory">
        {{ $t('history.refresh') }}
      </button>
    </div>

    <!-- Loading -->
    <div v-if="loading" class="flex items-center justify-center py-20">
      <span style="color: var(--color-text-secondary)">{{ $t('common.loading') }}</span>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="text-center py-20">
      <p style="color: var(--color-danger)">{{ $t('common.error') }}: {{ error }}</p>
    </div>

    <!-- Empty -->
    <div v-else-if="recipes.length === 0" class="text-center py-20">
      <p class="text-lg mb-4" style="color: var(--color-text-secondary)">{{ $t('history.empty') }}</p>
      <router-link to="/workbench/calculate" class="btn-primary inline-block">
        {{ $t('history.startNew') }}
      </router-link>
    </div>

    <!-- Table -->
    <div v-else class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr style="border-bottom: 2px solid var(--color-border)">
            <th class="text-left py-3 px-3">{{ $t('history.id') }}</th>
            <th class="text-left py-3 px-3">{{ $t('history.application') }}</th>
            <th class="text-left py-3 px-3">{{ $t('history.materials') }}</th>
            <th class="text-left py-3 px-3">{{ $t('history.date') }}</th>
            <th class="text-right py-3 px-3">{{ $t('history.actions') }}</th>
          </tr>
        </thead>
        <tbody>
          <tr
            v-for="(recipe, idx) in recipes"
            :key="recipe.recipe_id"
            class="cursor-pointer transition-colors"
            :style="{
              backgroundColor: idx % 2 === 0 ? 'transparent' : 'var(--color-bg-card)',
              borderBottom: '1px solid var(--color-border)',
            }"
            @click="navigateToResult(recipe.recipe_id)"
          >
            <td class="py-3 px-3 font-mono text-xs">{{ recipe.recipe_id.slice(0, 8) }}...</td>
            <td class="py-3 px-3">{{ recipe.application }}</td>
            <td class="py-3 px-3" style="color: var(--color-text-secondary)">
              {{ extractMaterials(recipe) }}
            </td>
            <td class="py-3 px-3" style="color: var(--color-text-secondary)">
              {{ formatDate(recipe.created_at) }}
            </td>
            <td class="py-3 px-3 text-right">
              <router-link
                :to="`/workbench/results/${recipe.recipe_id}`"
                class="text-sm font-semibold"
                style="color: var(--color-accent-orange)"
                @click.stop
              >
                {{ $t('history.view') }}
              </router-link>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { recipesApi } from '@/api/recipes'
import type { SimulateResponse } from '@/api/simulation'

const router = useRouter()

const recipes = ref<SimulateResponse[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

async function fetchHistory() {
  loading.value = true
  error.value = null
  try {
    const response = await recipesApi.list(50)
    recipes.value = (response.data as SimulateResponse[] | { recipes: SimulateResponse[] })
      ? (Array.isArray(response.data) ? response.data : (response.data as { recipes: SimulateResponse[] }).recipes ?? [])
      : []
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

function navigateToResult(id: string) {
  router.push(`/workbench/results/${id}`)
}

function extractMaterials(recipe: SimulateResponse): string {
  // Try to extract from parameters if available
  const params = recipe.parameters
  if (!params) return '—'
  // Try common keys
  const keys = Object.keys(params)
  if (keys.length === 0) return '—'
  return keys.slice(0, 2).map((k) => k.replace(/_/g, ' ')).join(', ')
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString()
  } catch {
    return iso
  }
}

onMounted(fetchHistory)
</script>

<style scoped>
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
  transition: border-color 0.2s;
}

.btn-secondary:hover {
  border-color: var(--color-accent-orange);
}

tr:hover {
  background-color: var(--color-bg-card) !important;
}
</style>
