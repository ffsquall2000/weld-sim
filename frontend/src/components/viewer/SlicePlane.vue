<template>
  <div class="slice-plane-controls p-3 rounded-lg"
       style="background-color: rgba(33, 38, 45, 0.9); border: 1px solid var(--color-border); backdrop-filter: blur(8px);">

    <!-- Title row with toggle -->
    <div class="flex items-center justify-between mb-2">
      <span class="text-xs font-medium" style="color: var(--color-text-primary);">
        {{ t('viewer.slicePlane') }}
      </span>
      <button
        class="text-xs px-1.5 py-0.5 rounded"
        :style="{
          backgroundColor: store.slicePlane.enabled ? 'var(--color-accent-blue)' : 'var(--color-bg-primary)',
          color: store.slicePlane.enabled ? '#ffffff' : 'var(--color-text-secondary)',
          border: '1px solid var(--color-border)',
        }"
        @click="store.toggleSlicePlane()">
        {{ store.slicePlane.enabled ? 'ON' : 'OFF' }}
      </button>
    </div>

    <!-- Normal direction selector -->
    <div class="mb-2">
      <label class="text-xs mb-1 block" style="color: var(--color-text-secondary);">
        {{ t('viewer.normal') }}
      </label>
      <div class="flex gap-1">
        <button
          v-for="axis in axes"
          :key="axis.label"
          :class="['axis-btn', { active: isActiveNormal(axis.normal) }]"
          @click="store.setSlicePlaneNormal(axis.normal)">
          {{ axis.label }}
        </button>
      </div>
    </div>

    <!-- Origin position slider -->
    <div class="mb-2">
      <label class="text-xs mb-1 block" style="color: var(--color-text-secondary);">
        {{ t('viewer.position') }}: {{ positionPercent }}%
      </label>
      <input
        type="range"
        min="0"
        max="100"
        :value="positionPercent"
        class="w-full h-1.5 accent-blue-500 cursor-pointer"
        style="accent-color: var(--color-accent-blue);"
        @input="onPositionChange" />
    </div>

    <!-- Plane equation display -->
    <div class="text-xs font-mono px-1.5 py-1 rounded"
         style="background-color: var(--color-bg-primary); color: var(--color-text-secondary); border: 1px solid var(--color-border);">
      {{ planeEquation }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useViewer3DStore } from '@/stores/viewer3d'

const { t } = useI18n()
const store = useViewer3DStore()

interface AxisDef {
  label: string
  normal: [number, number, number]
}

const axes: AxisDef[] = [
  { label: 'X', normal: [1, 0, 0] },
  { label: 'Y', normal: [0, 1, 0] },
  { label: 'Z', normal: [0, 0, 1] },
]

function isActiveNormal(normal: [number, number, number]): boolean {
  const n = store.slicePlane.normal
  return n[0] === normal[0] && n[1] === normal[1] && n[2] === normal[2]
}

// Compute position as percentage along the active normal axis
const positionPercent = computed(() => {
  if (!store.meshData) return 50
  const bounds = store.meshData.bounds
  const n = store.slicePlane.normal
  const o = store.slicePlane.origin

  // Find the dominant axis of the normal
  let axisIdx = 0
  if (Math.abs(n[1]!) > Math.abs(n[axisIdx]!)) axisIdx = 1
  if (Math.abs(n[2]!) > Math.abs(n[axisIdx]!)) axisIdx = 2

  const bMin = bounds[axisIdx * 2]!
  const bMax = bounds[axisIdx * 2 + 1]!
  const range = bMax - bMin
  if (range === 0) return 50

  const pos = o[axisIdx]!
  return Math.round(((pos - bMin) / range) * 100)
})

function onPositionChange(event: Event) {
  const input = event.target as HTMLInputElement
  const percent = parseInt(input.value) / 100

  if (!store.meshData) return
  const bounds = store.meshData.bounds
  const n = store.slicePlane.normal
  const origin: [number, number, number] = [...store.slicePlane.origin]

  // Find the dominant axis
  let axisIdx = 0
  if (Math.abs(n[1]!) > Math.abs(n[axisIdx]!)) axisIdx = 1
  if (Math.abs(n[2]!) > Math.abs(n[axisIdx]!)) axisIdx = 2

  const bMin = bounds[axisIdx * 2]!
  const bMax = bounds[axisIdx * 2 + 1]!
  origin[axisIdx] = bMin + percent * (bMax - bMin)

  store.setSlicePlaneOrigin(origin)
}

// Display plane equation: n . (r - o) = 0
const planeEquation = computed(() => {
  const n = store.slicePlane.normal
  const o = store.slicePlane.origin
  const nx = n[0].toFixed(1)
  const ny = n[1].toFixed(1)
  const nz = n[2].toFixed(1)
  const ox = o[0].toFixed(2)
  const oy = o[1].toFixed(2)
  const oz = o[2].toFixed(2)
  return `${nx}(x-${ox}) + ${ny}(y-${oy}) + ${nz}(z-${oz}) = 0`
})
</script>

<style scoped>
.slice-plane-controls {
  width: 220px;
}

.axis-btn {
  flex: 1;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  border-radius: 4px;
  border: 1px solid var(--color-border);
  background-color: var(--color-bg-primary);
  color: var(--color-text-secondary);
  cursor: pointer;
  transition: all 0.15s;
  text-align: center;
}

.axis-btn:hover {
  border-color: var(--color-accent-blue);
  color: var(--color-text-primary);
}

.axis-btn.active {
  background-color: var(--color-accent-blue);
  border-color: var(--color-accent-blue);
  color: #ffffff;
}
</style>
