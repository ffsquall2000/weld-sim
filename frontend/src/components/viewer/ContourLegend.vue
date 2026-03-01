<template>
  <div class="contour-legend flex flex-col items-center p-2 rounded-lg"
       style="background-color: rgba(33, 38, 45, 0.9); border: 1px solid var(--color-border); backdrop-filter: blur(8px);">

    <!-- Field name and unit -->
    <div class="text-xs font-medium mb-1 text-center max-w-[80px] truncate"
         :title="fieldLabel"
         style="color: var(--color-text-primary);">
      {{ fieldLabel }}
    </div>

    <!-- Max value -->
    <div class="text-xs tabular-nums mb-0.5"
         style="color: var(--color-text-secondary);">
      {{ formatValue(rangeMax) }}
    </div>

    <!-- Gradient bar -->
    <div class="color-bar relative rounded-sm overflow-hidden"
         :style="{ background: gradientStyle }">
    </div>

    <!-- Min value -->
    <div class="text-xs tabular-nums mt-0.5"
         style="color: var(--color-text-secondary);">
      {{ formatValue(rangeMin) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useViewer3DStore, type ColorMapPreset } from '@/stores/viewer3d'

const store = useViewer3DStore()

// Color map gradient CSS definitions (bottom to top, i.e. min to max)
const COLOR_MAP_GRADIENTS: Record<ColorMapPreset, string[]> = {
  jet: [
    'rgb(0, 0, 128)',     // 0%
    'rgb(0, 0, 255)',     // 10%
    'rgb(0, 255, 255)',   // 35%
    'rgb(0, 255, 0)',     // 50%
    'rgb(255, 255, 0)',   // 65%
    'rgb(255, 0, 0)',     // 90%
    'rgb(128, 0, 0)',     // 100%
  ],
  rainbow: [
    'rgb(255, 0, 0)',
    'rgb(255, 128, 0)',
    'rgb(255, 255, 0)',
    'rgb(0, 255, 0)',
    'rgb(0, 128, 255)',
    'rgb(0, 0, 255)',
    'rgb(128, 0, 255)',
  ],
  cool_warm: [
    'rgb(59, 76, 192)',
    'rgb(222, 222, 222)',
    'rgb(180, 4, 38)',
  ],
  viridis: [
    'rgb(68, 1, 84)',
    'rgb(72, 36, 117)',
    'rgb(32, 145, 140)',
    'rgb(139, 197, 63)',
    'rgb(253, 231, 37)',
  ],
}

const fieldLabel = computed(() => {
  if (!store.scalarField) return ''
  // Format field name: replace underscores with spaces, title case
  return store.scalarField
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
})

const range = computed(() => store.activeFieldRange)

const rangeMin = computed(() => range.value?.[0] ?? 0)
const rangeMax = computed(() => range.value?.[1] ?? 1)

const gradientStyle = computed(() => {
  const colors = COLOR_MAP_GRADIENTS[store.colorMap] ?? COLOR_MAP_GRADIENTS.cool_warm
  // Build a linear-gradient from bottom (min) to top (max)
  const stops = colors.map((color, i) => {
    const pct = (i / (colors.length - 1)) * 100
    return `${color} ${pct.toFixed(0)}%`
  })
  return `linear-gradient(to top, ${stops.join(', ')})`
})

function formatValue(value: number): string {
  const abs = Math.abs(value)
  if (abs === 0) return '0'
  if (abs >= 1e6) return value.toExponential(2)
  if (abs >= 1000) return value.toFixed(0)
  if (abs >= 1) return value.toFixed(2)
  if (abs >= 0.01) return value.toFixed(4)
  return value.toExponential(2)
}
</script>

<style scoped>
.contour-legend {
  width: 48px;
}

.color-bar {
  width: 16px;
  height: 180px;
  border: 1px solid var(--color-border);
}
</style>
