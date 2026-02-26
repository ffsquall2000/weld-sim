<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step4') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.geometryDesc') }}
    </p>

    <!-- Basic weld dimensions -->
    <div class="wizard-card mb-6">
      <h3 class="font-semibold mb-4">{{ $t('wizard.weldDimensions') }}</h3>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label class="field-label">{{ $t('wizard.weldWidth') }} (mm)</label>
          <input
            v-model.number="store.weldWidth"
            type="number"
            min="0.1"
            max="100"
            step="0.1"
            class="field-input w-full"
          />
        </div>
        <div>
          <label class="field-label">{{ $t('wizard.weldLength') }} (mm)</label>
          <input
            v-model.number="store.weldLength"
            type="number"
            min="0.1"
            max="500"
            step="0.1"
            class="field-input w-full"
          />
        </div>
      </div>
    </div>

    <!-- File import placeholder -->
    <div class="wizard-card opacity-60">
      <h3 class="font-semibold mb-2">{{ $t('wizard.geometryImport') }}</h3>
      <p class="text-sm mb-4" style="color: var(--color-text-secondary)">
        {{ $t('wizard.geometryImportFuture') }}
      </p>
      <div class="upload-zone">
        <svg class="w-10 h-10 mx-auto mb-2" style="color: var(--color-text-secondary)" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
        <p class="text-sm" style="color: var(--color-text-secondary)">
          {{ $t('wizard.geometryUploadPlaceholder') }}
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()
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

.upload-zone {
  border: 2px dashed var(--color-border);
  border-radius: 0.75rem;
  padding: 2rem;
  text-align: center;
  cursor: not-allowed;
}
</style>
