<!-- frontend/src/components/viewer/IsosurfaceControls.vue -->
<template>
  <div class="isosurface-controls">
    <div class="iso-header">
      <span class="iso-title">Isosurfaces</span>
      <button class="add-btn" @click="addDefault" :disabled="isComputing" title="Add isosurface">
        +
      </button>
    </div>

    <div v-if="isComputing" class="computing-indicator">
      <span class="spinner-small"></span>
      Computing...
    </div>

    <div v-if="isosurfaces.length === 0" class="empty-hint">
      Click + to add an isosurface
    </div>

    <div
      v-for="iso in isosurfaces"
      :key="iso.id"
      :class="['iso-item', { hidden: !iso.visible }]"
    >
      <div class="iso-row">
        <button
          :class="['visibility-btn', { active: iso.visible }]"
          @click="$emit('update', iso.id, { visible: !iso.visible })"
          title="Toggle visibility"
        >
          {{ iso.visible ? '👁' : '👁‍🗨' }}
        </button>

        <input
          type="color"
          :value="iso.color"
          @input="$emit('update', iso.id, { color: ($event.target as HTMLInputElement).value })"
          class="color-picker"
          title="Color"
        />

        <span class="threshold-value">{{ iso.threshold.toFixed(3) }}</span>

        <button
          class="remove-btn"
          @click="$emit('remove', iso.id)"
          title="Remove"
        >
          ✕
        </button>
      </div>

      <div class="iso-row">
        <span class="slider-label">Value:</span>
        <input
          type="range"
          :min="scalarRange.min"
          :max="scalarRange.max"
          :step="(scalarRange.max - scalarRange.min) / 100"
          :value="iso.threshold"
          @change="$emit('update', iso.id, { threshold: Number(($event.target as HTMLInputElement).value) })"
          class="threshold-slider"
        />
      </div>

      <div class="iso-row">
        <span class="slider-label">Opacity:</span>
        <input
          type="range"
          min="0.1"
          max="1"
          step="0.05"
          :value="iso.opacity"
          @input="$emit('update', iso.id, { opacity: Number(($event.target as HTMLInputElement).value) })"
          class="opacity-slider"
        />
        <span class="opacity-value">{{ (iso.opacity * 100).toFixed(0) }}%</span>
      </div>
    </div>

    <button
      v-if="isosurfaces.length > 0"
      class="clear-btn"
      @click="$emit('clearAll')"
    >
      Clear All
    </button>
  </div>
</template>

<script setup lang="ts">
import type { IsosurfaceConfig } from '@/composables/useIsosurface'

const props = defineProps<{
  isosurfaces: IsosurfaceConfig[]
  scalarRange: { min: number; max: number }
  isComputing: boolean
}>()

const emit = defineEmits<{
  (e: 'add', threshold: number, color: string, opacity: number): void
  (e: 'update', id: number, updates: Partial<Omit<IsosurfaceConfig, 'id'>>): void
  (e: 'remove', id: number): void
  (e: 'clearAll'): void
}>()

function addDefault() {
  const mid = (props.scalarRange.min + props.scalarRange.max) / 2
  const colors = ['#ff9800', '#2196f3', '#4caf50', '#f44336', '#9c27b0']
  const color = colors[props.isosurfaces.length % colors.length]
  emit('add', mid, color, 0.6)
}
</script>

<style scoped>
.isosurface-controls {
  position: absolute;
  right: 8px;
  bottom: 8px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.85);
  border-radius: 8px;
  padding: 12px;
  min-width: 220px;
  max-height: 300px;
  overflow-y: auto;
}

.iso-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.iso-title {
  color: #fff;
  font-size: 13px;
  font-weight: 500;
}

.add-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 4px;
  background: var(--color-accent-orange, #ff9800);
  color: #fff;
  font-size: 16px;
  cursor: pointer;
  line-height: 1;
}
.add-btn:hover { opacity: 0.9; }
.add-btn:disabled { opacity: 0.5; cursor: not-allowed; }

.computing-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #ff9800;
  font-size: 12px;
  margin-bottom: 8px;
}

.spinner-small {
  width: 12px;
  height: 12px;
  border: 2px solid #ff9800;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.empty-hint {
  color: #666;
  font-size: 12px;
  text-align: center;
  padding: 10px 0;
}

.iso-item {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 6px;
  padding: 8px;
  margin-bottom: 8px;
}
.iso-item.hidden {
  opacity: 0.5;
}

.iso-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}
.iso-row:last-child {
  margin-bottom: 0;
}

.visibility-btn {
  width: 24px;
  height: 24px;
  border: none;
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  font-size: 12px;
}
.visibility-btn.active {
  background: rgba(255, 255, 255, 0.1);
}

.color-picker {
  width: 28px;
  height: 24px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  padding: 0;
}

.threshold-value {
  flex: 1;
  color: #ccc;
  font-size: 12px;
  font-family: monospace;
}

.remove-btn {
  width: 20px;
  height: 20px;
  border: none;
  border-radius: 4px;
  background: rgba(244, 67, 54, 0.2);
  color: #f44336;
  cursor: pointer;
  font-size: 10px;
}
.remove-btn:hover {
  background: rgba(244, 67, 54, 0.4);
}

.slider-label {
  color: #888;
  font-size: 11px;
  min-width: 50px;
}

.threshold-slider,
.opacity-slider {
  flex: 1;
  accent-color: var(--color-accent-orange, #ff9800);
}

.opacity-value {
  color: #888;
  font-size: 11px;
  min-width: 35px;
  text-align: right;
}

.clear-btn {
  width: 100%;
  padding: 6px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  border-radius: 4px;
  background: transparent;
  color: #888;
  font-size: 12px;
  cursor: pointer;
  margin-top: 4px;
}
.clear-btn:hover {
  background: rgba(255, 255, 255, 0.05);
  color: #aaa;
}
</style>
