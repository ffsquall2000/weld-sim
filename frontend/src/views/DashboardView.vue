<template>
  <div class="p-6 max-w-5xl mx-auto">
    <!-- Welcome header -->
    <div class="mb-8">
      <h1 class="text-3xl font-bold">
        {{ $t('dashboard.welcome') }}
      </h1>
      <p class="text-xl mt-1" style="color: var(--color-accent-orange)">
        {{ $t('app.title') }}
      </p>
    </div>

    <!-- Quick action cards -->
    <section class="mb-8">
      <h2 class="text-lg font-semibold mb-4">{{ $t('dashboard.quickActions') }}</h2>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <router-link to="/workbench/calculate" class="action-card group">
          <div class="text-2xl mb-2">&#9881;</div>
          <div class="font-semibold text-lg">{{ $t('dashboard.newCalculation') }}</div>
          <p class="text-sm mt-1" style="color: var(--color-text-secondary)">
            {{ $t('dashboard.newCalculationDesc') }}
          </p>
        </router-link>

        <router-link to="/" class="action-card group">
          <div class="text-2xl mb-2">&#128203;</div>
          <div class="font-semibold text-lg">{{ $t('dashboard.viewHistory') }}</div>
          <p class="text-sm mt-1" style="color: var(--color-text-secondary)">
            {{ $t('dashboard.viewHistoryDesc') }}
          </p>
        </router-link>
      </div>
    </section>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Recent calculations -->
      <section class="lg:col-span-2">
        <div class="card">
          <h2 class="text-lg font-semibold mb-4">{{ $t('dashboard.recentCalculations') }}</h2>

          <div v-if="loadingRecent" class="py-8 text-center" style="color: var(--color-text-secondary)">
            {{ $t('common.loading') }}
          </div>

          <div v-else-if="recentRecipes.length === 0" class="py-8 text-center" style="color: var(--color-text-secondary)">
            {{ $t('dashboard.noRecent') }}
          </div>

          <div v-else class="space-y-2">
            <router-link
              v-for="recipe in recentRecipes"
              :key="recipe.recipe_id"
              :to="`/workbench/results/${recipe.recipe_id}`"
              class="flex items-center justify-between p-3 rounded-lg transition-colors"
              style="border: 1px solid var(--color-border)"
            >
              <div>
                <span class="text-sm font-semibold">{{ recipe.application }}</span>
                <span class="text-xs font-mono ml-2" style="color: var(--color-text-secondary)">
                  {{ recipe.recipe_id.slice(0, 8) }}
                </span>
              </div>
              <span class="text-xs" style="color: var(--color-text-secondary)">
                {{ formatDate(recipe.created_at) }}
              </span>
            </router-link>
          </div>
        </div>
      </section>

      <!-- System status -->
      <section>
        <div class="card">
          <h2 class="text-lg font-semibold mb-4">{{ $t('dashboard.systemStatus') }}</h2>
          <div class="space-y-4">
            <div>
              <span class="text-sm" style="color: var(--color-text-secondary)">
                {{ $t('dashboard.materialCount') }}
              </span>
              <p class="text-2xl font-bold mt-1" style="color: var(--color-accent-orange)">
                {{ materialCount ?? '—' }}
              </p>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { recipesApi } from '@/api/recipes'
import { materialsApi } from '@/api/materials'
import type { SimulateResponse } from '@/api/simulation'

const recentRecipes = ref<SimulateResponse[]>([])
const loadingRecent = ref(true)
const materialCount = ref<number | null>(null)

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString()
  } catch {
    return iso
  }
}

onMounted(async () => {
  // Fetch recent recipes
  try {
    const response = await recipesApi.list(5)
    const data = response.data
    recentRecipes.value = Array.isArray(data) ? data : (data as { recipes: SimulateResponse[] }).recipes ?? []
  } catch {
    // Silently fail — dashboard still works
  } finally {
    loadingRecent.value = false
  }

  // Fetch material count
  try {
    const response = await materialsApi.list()
    const data = response.data as { materials: string[] }
    materialCount.value = data.materials?.length ?? 0
  } catch {
    // Non-critical
  }
})
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1rem 1.25rem;
}

.action-card {
  display: block;
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
  text-decoration: none;
  color: var(--color-text-primary);
  transition: border-color 0.2s;
}

.action-card:hover {
  border-color: var(--color-accent-orange);
}
</style>
