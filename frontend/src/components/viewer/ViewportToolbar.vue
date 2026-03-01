<template>
  <div class="viewport-toolbar flex flex-col gap-1 p-1.5 rounded-lg"
       style="background-color: rgba(33, 38, 45, 0.9); border: 1px solid var(--color-border); backdrop-filter: blur(8px);">

    <!-- Display mode toggle group -->
    <div class="flex gap-0.5">
      <button
        v-for="mode in displayModes"
        :key="mode.value"
        :title="mode.label"
        :class="['toolbar-btn', { active: store.displayMode === mode.value }]"
        @click="store.setDisplayMode(mode.value)">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path v-if="mode.value === 'solid'" d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
          <g v-else-if="mode.value === 'wireframe'">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" stroke-dasharray="3 2" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </g>
          <g v-else>
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
            <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
            <line x1="12" y1="22.08" x2="12" y2="12" />
          </g>
        </svg>
      </button>
    </div>

    <!-- Divider -->
    <div class="w-full h-px" style="background-color: var(--color-border);" />

    <!-- Scalar field selector -->
    <div v-if="store.fieldNames.length > 0" class="relative">
      <select
        :value="store.scalarField ?? ''"
        class="toolbar-select w-full text-xs"
        :title="t('viewer.scalarField')"
        @change="onFieldChange($event)">
        <option value="">{{ t('viewer.noField') }}</option>
        <option v-for="name in store.fieldNames" :key="name" :value="name">
          {{ name }}
        </option>
      </select>
    </div>

    <!-- Color map selector -->
    <div v-if="store.scalarField" class="relative">
      <select
        :value="store.colorMap"
        class="toolbar-select w-full text-xs"
        :title="t('viewer.colorMap')"
        @change="onColorMapChange($event)">
        <option v-for="cm in colorMaps" :key="cm.value" :value="cm.value">
          {{ cm.label }}
        </option>
      </select>
    </div>

    <!-- Divider -->
    <div class="w-full h-px" style="background-color: var(--color-border);" />

    <!-- Action buttons -->
    <div class="flex flex-col gap-0.5">
      <!-- Slice plane toggle -->
      <button
        :title="t('viewer.slicePlane')"
        :class="['toolbar-btn', { active: store.slicePlane.enabled }]"
        @click="store.toggleSlicePlane()">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <line x1="3" y1="12" x2="21" y2="12" />
        </svg>
      </button>

      <!-- Show/hide axes -->
      <button
        :title="t('viewer.axes')"
        :class="['toolbar-btn', { active: store.showAxes }]"
        @click="store.toggleAxes()">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="4" y1="20" x2="20" y2="20" stroke="#ef4444" />
          <line x1="4" y1="20" x2="4" y2="4" stroke="#22c55e" />
          <line x1="4" y1="20" x2="12" y2="12" stroke="#3b82f6" />
          <text x="20" y="19" font-size="6" fill="#ef4444" stroke="none">X</text>
          <text x="5" y="5" font-size="6" fill="#22c55e" stroke="none">Y</text>
          <text x="12" y="11" font-size="6" fill="#3b82f6" stroke="none">Z</text>
        </svg>
      </button>

      <!-- Show/hide edges -->
      <button
        :title="t('viewer.edges')"
        :class="['toolbar-btn', { active: store.showEdges }]"
        @click="store.toggleEdges()">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M4 20L12 4L20 20H4Z" />
          <line x1="8" y1="12" x2="16" y2="12" stroke-dasharray="2 2" />
        </svg>
      </button>

      <!-- Reset camera -->
      <button
        :title="t('viewer.resetCamera')"
        class="toolbar-btn"
        @click="emit('resetCamera')">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M15 3h6v6" />
          <path d="M9 21H3v-6" />
          <path d="M21 3l-7 7" />
          <path d="M3 21l7-7" />
        </svg>
      </button>

      <!-- Take screenshot -->
      <button
        :title="t('viewer.screenshot')"
        class="toolbar-btn"
        @click="emit('takeScreenshot')">
        <svg class="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
          <circle cx="12" cy="13" r="4" />
        </svg>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useViewer3DStore, type DisplayMode, type ColorMapPreset } from '@/stores/viewer3d'

const { t } = useI18n()
const store = useViewer3DStore()

const emit = defineEmits<{
  resetCamera: []
  takeScreenshot: []
}>()

const displayModes: { value: DisplayMode; label: string }[] = [
  { value: 'solid', label: 'Solid' },
  { value: 'wireframe', label: 'Wireframe' },
  { value: 'solid_wireframe', label: 'Solid + Wireframe' },
]

const colorMaps: { value: ColorMapPreset; label: string }[] = [
  { value: 'jet', label: 'Jet' },
  { value: 'rainbow', label: 'Rainbow' },
  { value: 'cool_warm', label: 'Cool-Warm' },
  { value: 'viridis', label: 'Viridis' },
]

function onFieldChange(event: Event) {
  const select = event.target as HTMLSelectElement
  store.setScalarField(select.value || null)
}

function onColorMapChange(event: Event) {
  const select = event.target as HTMLSelectElement
  store.setColorMap(select.value as ColorMapPreset)
}
</script>

<style scoped>
.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 4px;
  border: none;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
  background-color: transparent;
  color: var(--color-text-secondary);
}

.toolbar-btn:hover {
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
}

.toolbar-btn.active {
  background-color: var(--color-accent-blue);
  color: #ffffff;
}

.toolbar-select {
  appearance: none;
  padding: 3px 6px;
  border-radius: 4px;
  border: 1px solid var(--color-border);
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  cursor: pointer;
  outline: none;
  min-width: 90px;
}

.toolbar-select:hover {
  border-color: var(--color-accent-blue);
}

.toolbar-select:focus {
  border-color: var(--color-accent-blue);
  box-shadow: 0 0 0 1px var(--color-accent-blue);
}
</style>
