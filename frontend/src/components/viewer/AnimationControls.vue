<!-- frontend/src/components/viewer/AnimationControls.vue -->
<template>
  <div class="animation-controls" v-if="modes.length > 0">
    <!-- Mode selector -->
    <div class="mode-selector">
      <label>Mode:</label>
      <select v-model="currentModeIndexLocal" class="mode-select">
        <option v-for="(mode, idx) in modes" :key="idx" :value="idx">
          Mode {{ idx + 1 }}: {{ mode.frequency_hz.toFixed(1) }} Hz ({{ mode.mode_type }})
        </option>
      </select>
    </div>

    <!-- Playback controls -->
    <div class="playback">
      <button class="ctrl-btn" @click="$emit('stepBackward')" title="Step Back">◄◄</button>
      <button class="ctrl-btn play-btn" @click="$emit('togglePlay')" :title="isPlaying ? 'Pause' : 'Play'">
        {{ isPlaying ? '⏸' : '▶' }}
      </button>
      <button class="ctrl-btn" @click="$emit('stepForward')" title="Step Forward">►►</button>

      <!-- Phase slider -->
      <input
        type="range"
        min="0"
        max="360"
        :value="phase"
        @input="$emit('update:phase', Number(($event.target as HTMLInputElement).value))"
        class="phase-slider"
        title="Phase"
      />

      <button
        :class="['ctrl-btn', { active: loop }]"
        @click="$emit('update:loop', !loop)"
        title="Loop"
      >🔄</button>

      <select v-model="speedLocal" class="speed-select" title="Speed">
        <option :value="0.25">0.25x</option>
        <option :value="0.5">0.5x</option>
        <option :value="1">1x</option>
        <option :value="2">2x</option>
        <option :value="4">4x</option>
      </select>
    </div>

    <!-- Amplitude slider -->
    <div class="amplitude-row">
      <label>Amplitude:</label>
      <input
        type="range"
        min="0.1"
        max="10"
        step="0.1"
        :value="amplitudeScale"
        @input="$emit('update:amplitudeScale', Number(($event.target as HTMLInputElement).value))"
        class="amplitude-slider"
      />
      <span class="amplitude-value">{{ amplitudeScale.toFixed(1) }}x</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ModeData } from '@/composables/useAnimation'

const props = defineProps<{
  modes: ModeData[]
  isPlaying: boolean
  phase: number
  amplitudeScale: number
  speed: number
  loop: boolean
  currentModeIndex: number
}>()

const emit = defineEmits<{
  (e: 'update:isPlaying', v: boolean): void
  (e: 'update:phase', v: number): void
  (e: 'update:amplitudeScale', v: number): void
  (e: 'update:speed', v: number): void
  (e: 'update:loop', v: boolean): void
  (e: 'update:currentModeIndex', v: number): void
  (e: 'togglePlay'): void
  (e: 'stepForward'): void
  (e: 'stepBackward'): void
}>()

// Two-way binding helpers
const currentModeIndexLocal = computed({
  get: () => props.currentModeIndex,
  set: (v) => emit('update:currentModeIndex', v),
})

const speedLocal = computed({
  get: () => props.speed,
  set: (v) => emit('update:speed', v),
})
</script>

<style scoped>
.animation-controls {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: rgba(0, 0, 0, 0.8);
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  z-index: 10;
}

.mode-selector, .playback, .amplitude-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #ccc;
}

.mode-select, .speed-select {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 4px;
  color: #ccc;
  padding: 2px 6px;
  font-size: 12px;
}

.ctrl-btn {
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background: rgba(255,255,255,0.1);
  color: #ccc;
  cursor: pointer;
  font-size: 12px;
}
.ctrl-btn:hover { background: rgba(255,255,255,0.2) }
.ctrl-btn.active { background: var(--color-accent-orange, #ff9800); color: #fff }
.play-btn { width: 36px; font-size: 16px }

.phase-slider, .amplitude-slider {
  flex: 1;
  accent-color: var(--color-accent-orange, #ff9800);
}
.amplitude-value { min-width: 40px; text-align: right }
</style>
