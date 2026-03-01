<!-- frontend/src/components/viewer/ClippingControls.vue -->
<template>
  <div class="clipping-controls">
    <div class="clipping-header">
      <span class="clipping-title">Cross-Section</span>
      <button class="reset-btn" @click="$emit('reset')" title="Reset">Reset</button>
    </div>

    <!-- X Axis -->
    <div :class="['axis-row', { active: state.x.enabled }]">
      <button
        :class="['axis-btn', { active: state.x.enabled }]"
        @click="$emit('toggleAxis', 'x')"
        title="Toggle X clipping"
      >
        X
      </button>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        :value="state.x.position"
        :disabled="!state.x.enabled"
        @input="$emit('setPosition', 'x', Number(($event.target as HTMLInputElement).value))"
        class="position-slider"
      />
      <button
        :class="['invert-btn', { active: state.x.inverted }]"
        :disabled="!state.x.enabled"
        @click="$emit('invertAxis', 'x')"
        title="Invert direction"
      >
        <span class="flip-icon">&#8644;</span>
      </button>
    </div>

    <!-- Y Axis -->
    <div :class="['axis-row', { active: state.y.enabled }]">
      <button
        :class="['axis-btn', { active: state.y.enabled }]"
        @click="$emit('toggleAxis', 'y')"
        title="Toggle Y clipping"
      >
        Y
      </button>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        :value="state.y.position"
        :disabled="!state.y.enabled"
        @input="$emit('setPosition', 'y', Number(($event.target as HTMLInputElement).value))"
        class="position-slider"
      />
      <button
        :class="['invert-btn', { active: state.y.inverted }]"
        :disabled="!state.y.enabled"
        @click="$emit('invertAxis', 'y')"
        title="Invert direction"
      >
        <span class="flip-icon">&#8644;</span>
      </button>
    </div>

    <!-- Z Axis -->
    <div :class="['axis-row', { active: state.z.enabled }]">
      <button
        :class="['axis-btn', { active: state.z.enabled }]"
        @click="$emit('toggleAxis', 'z')"
        title="Toggle Z clipping"
      >
        Z
      </button>
      <input
        type="range"
        min="0"
        max="1"
        step="0.01"
        :value="state.z.position"
        :disabled="!state.z.enabled"
        @input="$emit('setPosition', 'z', Number(($event.target as HTMLInputElement).value))"
        class="position-slider"
      />
      <button
        :class="['invert-btn', { active: state.z.inverted }]"
        :disabled="!state.z.enabled"
        @click="$emit('invertAxis', 'z')"
        title="Invert direction"
      >
        <span class="flip-icon">&#8644;</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { ClippingState } from '@/composables/useClipping'

defineProps<{
  state: ClippingState
}>()

defineEmits<{
  (e: 'toggleAxis', axis: 'x' | 'y' | 'z'): void
  (e: 'setPosition', axis: 'x' | 'y' | 'z', position: number): void
  (e: 'invertAxis', axis: 'x' | 'y' | 'z'): void
  (e: 'reset'): void
}>()
</script>

<style scoped>
.clipping-controls {
  position: absolute;
  left: 8px;
  bottom: 8px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.8);
  border-radius: 8px;
  padding: 12px;
  min-width: 200px;
}

.clipping-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.clipping-title {
  color: #fff;
  font-size: 13px;
  font-weight: 500;
}

.reset-btn {
  background: rgba(255, 255, 255, 0.1);
  border: none;
  border-radius: 4px;
  color: #888;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
}
.reset-btn:hover {
  background: rgba(255, 255, 255, 0.15);
  color: #aaa;
}

.axis-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
  opacity: 0.5;
  transition: opacity 0.15s;
}
.axis-row.active {
  opacity: 1;
}

.axis-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  font-size: 12px;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.15s;
}

.axis-row:nth-child(2) .axis-btn {
  background: rgba(244, 67, 54, 0.2);
  color: #f44336;
}
.axis-row:nth-child(2) .axis-btn.active {
  background: #f44336;
  color: #fff;
}

.axis-row:nth-child(3) .axis-btn {
  background: rgba(76, 175, 80, 0.2);
  color: #4caf50;
}
.axis-row:nth-child(3) .axis-btn.active {
  background: #4caf50;
  color: #fff;
}

.axis-row:nth-child(4) .axis-btn {
  background: rgba(33, 150, 243, 0.2);
  color: #2196f3;
}
.axis-row:nth-child(4) .axis-btn.active {
  background: #2196f3;
  color: #fff;
}

.position-slider {
  flex: 1;
  accent-color: var(--color-accent-orange, #ff9800);
}
.position-slider:disabled {
  opacity: 0.3;
}

.invert-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.1);
  color: #888;
  cursor: pointer;
  font-size: 14px;
}
.invert-btn:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.15);
}
.invert-btn.active {
  background: var(--color-accent-orange, #ff9800);
  color: #fff;
}
.invert-btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
}

.flip-icon {
  display: inline-block;
}
</style>
