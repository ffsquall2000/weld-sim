<template>
  <svg :viewBox="'0 0 200 200'" class="w-full max-w-[200px]" xmlns="http://www.w3.org/2000/svg">
    <!-- Title: layer count -->
    <text x="100" y="16" text-anchor="middle" fill="var(--color-text-secondary)" font-size="10">
      {{ upperLayers }} layers
    </text>

    <!-- Upper layers (stacked foils) -->
    <g v-for="(_, i) in displayLayers" :key="'upper-' + i">
      <rect
        :x="30"
        :y="upperStartY + i * upperLayerHeight"
        :width="100"
        :height="Math.max(upperLayerHeight - 1, 2)"
        :fill="materialColor(upperMaterial)"
        stroke="#444"
        stroke-width="0.5"
        rx="1"
      />
    </g>

    <!-- Lower material (substrate) -->
    <rect
      :x="30"
      :y="lowerY"
      :width="100"
      :height="lowerHeight"
      :fill="materialColor(lowerMaterial)"
      stroke="#444"
      stroke-width="0.5"
      rx="1"
    />

    <!-- Labels on the right -->
    <text :x="140" :y="upperMidY" fill="var(--color-text-primary)" font-size="9" dominant-baseline="middle">
      {{ upperMaterial }}
    </text>
    <text :x="140" :y="upperMidY + 12" fill="var(--color-text-secondary)" font-size="8" dominant-baseline="middle">
      {{ upperThickness }} mm
    </text>

    <text :x="140" :y="lowerMidY" fill="var(--color-text-primary)" font-size="9" dominant-baseline="middle">
      {{ lowerMaterial }}
    </text>
    <text :x="140" :y="lowerMidY + 12" fill="var(--color-text-secondary)" font-size="8" dominant-baseline="middle">
      {{ lowerThickness }} mm
    </text>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  upperMaterial: string
  upperThickness: number
  upperLayers: number
  lowerMaterial: string
  lowerThickness: number
}>()

const materialColors: Record<string, string> = {
  Nickel: '#a0a0a0',
  Aluminum: '#c0c0c0',
  Copper: '#b87333',
  Steel: '#606060',
}

function materialColor(name: string): string {
  for (const [key, color] of Object.entries(materialColors)) {
    if (name.toLowerCase().includes(key.toLowerCase())) return color
  }
  return '#888888'
}

// Show at most 8 visual layers to keep the diagram readable
const displayLayers = computed(() => Math.min(props.upperLayers, 8))

const upperStartY = 24
const availableHeight = 140
const upperTotalHeight = computed(() => availableHeight * 0.65)
const upperLayerHeight = computed(() => upperTotalHeight.value / displayLayers.value)
const lowerY = computed(() => upperStartY + upperTotalHeight.value + 4)
const lowerHeight = computed(() => availableHeight * 0.3)

const upperMidY = computed(() => upperStartY + upperTotalHeight.value / 2)
const lowerMidY = computed(() => lowerY.value + lowerHeight.value / 2)
</script>
