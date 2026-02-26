<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step5') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.equipmentDesc') }}
    </p>

    <div class="space-y-6">
      <!-- Frequency -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.frequency') }} (kHz)</h3>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="f in frequencyPresets"
            :key="f"
            class="preset-btn"
            :class="{ 'preset-btn-active': store.frequency === f && !customFrequency }"
            @click="selectFrequency(f)"
          >
            {{ f }} kHz
          </button>
          <button
            class="preset-btn"
            :class="{ 'preset-btn-active': customFrequency }"
            @click="enableCustomFrequency"
          >
            {{ $t('wizard.custom') }}
          </button>
        </div>
        <input
          v-if="customFrequency"
          v-model.number="store.frequency"
          type="number"
          min="10"
          max="100"
          step="0.1"
          class="field-input w-full mt-3"
          :placeholder="$t('wizard.customFreqPlaceholder')"
        />
      </div>

      <!-- Max Power -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.maxPower') }} (W)</h3>
        <input
          v-model.number="store.maxPower"
          type="number"
          min="500"
          max="10000"
          step="100"
          class="field-input w-full"
        />
        <input
          v-model.number="store.maxPower"
          type="range"
          min="500"
          max="10000"
          step="100"
          class="w-full mt-2 accent-[var(--color-accent-orange)]"
        />
        <div class="flex justify-between text-xs mt-1" style="color: var(--color-text-secondary)">
          <span>500 W</span>
          <span>10,000 W</span>
        </div>
      </div>

      <!-- Booster Gain -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.boosterGain') }}</h3>
        <div class="flex flex-wrap gap-2">
          <button
            v-for="g in gainPresets"
            :key="g"
            class="preset-btn"
            :class="{ 'preset-btn-active': store.boosterGain === g && !customGain }"
            @click="selectGain(g)"
          >
            {{ g }}x
          </button>
          <button
            class="preset-btn"
            :class="{ 'preset-btn-active': customGain }"
            @click="enableCustomGain"
          >
            {{ $t('wizard.custom') }}
          </button>
        </div>
        <input
          v-if="customGain"
          v-model.number="store.boosterGain"
          type="number"
          min="0.5"
          max="5.0"
          step="0.1"
          class="field-input w-full mt-3"
        />
      </div>

      <!-- Effective Area (read-only) -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-2">{{ $t('wizard.effectiveArea') }}</h3>
        <p class="text-2xl font-mono font-bold" style="color: var(--color-accent-orange)">
          {{ store.effectiveArea.toFixed(1) }} mm<sup>2</sup>
        </p>
        <p class="text-xs mt-1" style="color: var(--color-text-secondary)">
          {{ store.weldWidth }} mm x {{ store.weldLength }} mm
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()

const frequencyPresets = [20.0, 30.0, 35.0, 40.0]
const gainPresets = [1.0, 1.5, 2.0, 2.5]

const customFrequency = ref(false)
const customGain = ref(false)

function selectFrequency(f: number) {
  customFrequency.value = false
  store.frequency = f
}

function enableCustomFrequency() {
  customFrequency.value = true
}

function selectGain(g: number) {
  customGain.value = false
  store.boosterGain = g
}

function enableCustomGain() {
  customGain.value = true
}
</script>

<style scoped>
.wizard-card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.25rem;
}

.field-input {
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
  color: var(--color-text-primary);
  border-radius: 0.375rem;
  padding: 0.5rem 0.75rem;
  font-size: 0.875rem;
  outline: none;
  transition: border-color 0.2s;
}

.field-input:focus {
  border-color: var(--color-accent-orange);
}

.preset-btn {
  padding: 0.5rem 1rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  font-weight: 500;
  border: 1px solid var(--color-border);
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  cursor: pointer;
  transition: all 0.2s;
}

.preset-btn:hover {
  border-color: var(--color-accent-orange);
}

.preset-btn-active {
  background-color: var(--color-accent-orange);
  border-color: var(--color-accent-orange);
  color: #fff;
}
</style>
