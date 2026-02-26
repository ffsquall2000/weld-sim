<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step2') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.materialDesc') }}
    </p>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Upper Material (Foil) -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-4">{{ $t('wizard.upperMaterial') }}</h3>

        <label class="field-label">{{ $t('wizard.material') }}</label>
        <select v-model="store.upperMaterial" class="field-input w-full mb-3">
          <option v-for="m in materials" :key="m" :value="m">{{ m }}</option>
        </select>

        <label class="field-label">{{ $t('wizard.thickness') }} (mm)</label>
        <input
          v-model.number="store.upperThickness"
          type="number"
          min="0.001"
          max="10"
          step="0.001"
          class="field-input w-full mb-3"
        />

        <label class="field-label">{{ $t('wizard.layers') }}</label>
        <input
          v-model.number="store.upperLayers"
          type="number"
          min="1"
          max="200"
          step="1"
          class="field-input w-full"
        />
      </div>

      <!-- Lower Material (Substrate) -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-4">{{ $t('wizard.lowerMaterial') }}</h3>

        <label class="field-label">{{ $t('wizard.material') }}</label>
        <select v-model="store.lowerMaterial" class="field-input w-full mb-3">
          <option v-for="m in materials" :key="m" :value="m">{{ m }}</option>
        </select>

        <label class="field-label">{{ $t('wizard.thickness') }} (mm)</label>
        <input
          v-model.number="store.lowerThickness"
          type="number"
          min="0.01"
          max="20"
          step="0.01"
          class="field-input w-full"
        />
      </div>
    </div>

    <!-- Material Stack Visual -->
    <div class="mt-6 wizard-card">
      <h3 class="font-semibold mb-3">{{ $t('wizard.stackPreview') }}</h3>
      <div class="stack-visual">
        <div class="stack-layer stack-upper">
          {{ store.upperMaterial }} {{ store.upperThickness }}mm
          <span style="color: var(--color-text-secondary)"> x {{ store.upperLayers }}</span>
        </div>
        <div class="stack-layer stack-lower">
          {{ store.lowerMaterial }} {{ store.lowerThickness }}mm
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()

const materials = [
  'Nickel 201',
  'Nickel 200',
  'Aluminum 1100',
  'Aluminum 1050',
  'Copper C110',
  'Copper C101',
  'Steel 304',
  'Steel 316L',
]
</script>

<style scoped>
.wizard-card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.25rem;
}

.field-label {
  display: block;
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: 0.25rem;
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

.stack-visual {
  border-radius: 0.5rem;
  overflow: hidden;
  border: 1px solid var(--color-border);
}

.stack-layer {
  padding: 0.75rem 1rem;
  font-size: 0.875rem;
  font-weight: 500;
  text-align: center;
}

.stack-upper {
  background-color: rgba(255, 152, 0, 0.15);
  border-bottom: 1px dashed var(--color-border);
  color: var(--color-accent-orange);
}

.stack-lower {
  background-color: rgba(184, 115, 51, 0.15);
  color: #d4956a;
}
</style>
