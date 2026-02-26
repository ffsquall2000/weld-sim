<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step3') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.hornAnvilDesc') }}
    </p>

    <div class="space-y-6 max-h-[60vh] overflow-y-auto pr-1">
      <!-- Horn Type -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.hornType') }}</h3>
        <select v-model="store.hornType" class="field-input w-full">
          <option v-for="h in hornTypes" :key="h.value" :value="h.value">{{ h.label }}</option>
        </select>
        <p class="text-xs mt-2" style="color: var(--color-text-secondary)">
          {{ hornDescription }}
        </p>
      </div>

      <!-- Knurl Pattern -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.knurlPattern') }}</h3>
        <select v-model="store.knurlType" class="field-input w-full mb-3">
          <option v-for="k in knurlTypes" :key="k.value" :value="k.value">{{ k.label }}</option>
        </select>

        <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label class="field-label">{{ $t('wizard.pitch') }} (mm)</label>
            <input
              v-model.number="store.knurlPitch"
              type="number"
              min="0.1"
              max="5.0"
              step="0.1"
              class="field-input w-full"
            />
          </div>
          <div>
            <label class="field-label">{{ $t('wizard.toothWidth') }} (mm)</label>
            <input
              v-model.number="store.knurlToothWidth"
              type="number"
              min="0.05"
              max="3.0"
              step="0.05"
              class="field-input w-full"
            />
          </div>
          <div>
            <label class="field-label">{{ $t('wizard.depth') }} (mm)</label>
            <input
              v-model.number="store.knurlDepth"
              type="number"
              min="0.05"
              max="2.0"
              step="0.05"
              class="field-input w-full"
            />
          </div>
        </div>

        <div class="mt-3 text-sm">
          <span style="color: var(--color-text-secondary)">{{ $t('wizard.contactRatio') }}:</span>
          <span class="ml-2 font-mono font-semibold">{{ store.contactRatio.toFixed(3) }}</span>
        </div>
      </div>

      <!-- Anvil Type -->
      <div class="wizard-card">
        <h3 class="font-semibold mb-3">{{ $t('wizard.anvilType') }}</h3>
        <select v-model="store.anvilType" class="field-input w-full">
          <option v-for="a in anvilTypes" :key="a.value" :value="a.value">{{ a.label }}</option>
        </select>

        <div v-if="store.anvilType === 'resonant'" class="mt-3">
          <label class="field-label">{{ $t('wizard.resonantFreq') }} (kHz)</label>
          <input
            v-model.number="store.anvilResonantFreq"
            type="number"
            min="0"
            max="100"
            step="0.1"
            class="field-input w-full"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()
const { t } = useI18n()

const hornTypes = [
  { value: 'flat', label: t('wizard.hornFlat') },
  { value: 'curved', label: t('wizard.hornCurved') },
  { value: 'segmented', label: t('wizard.hornSegmented') },
  { value: 'blade', label: t('wizard.hornBlade') },
  { value: 'heavy', label: t('wizard.hornHeavy') },
  { value: 'branson_dp', label: t('wizard.hornBransonDp') },
  { value: 'custom', label: t('wizard.hornCustom') },
]

const knurlTypes = [
  { value: 'linear', label: t('wizard.knurlLinear') },
  { value: 'cross_hatch', label: t('wizard.knurlCrossHatch') },
  { value: 'diamond', label: t('wizard.knurlDiamond') },
  { value: 'conical', label: t('wizard.knurlConical') },
  { value: 'spherical', label: t('wizard.knurlSpherical') },
  { value: 'custom', label: t('wizard.knurlCustom') },
]

const anvilTypes = [
  { value: 'fixed_flat', label: t('wizard.anvilFixedFlat') },
  { value: 'knurled', label: t('wizard.anvilKnurled') },
  { value: 'contoured', label: t('wizard.anvilContoured') },
  { value: 'rotary', label: t('wizard.anvilRotary') },
  { value: 'multi_station', label: t('wizard.anvilMultiStation') },
  { value: 'resonant', label: t('wizard.anvilResonant') },
  { value: 'custom', label: t('wizard.anvilCustom') },
]

const hornDescriptions: Record<string, string> = {
  flat: 'wizard.hornFlatDesc',
  curved: 'wizard.hornCurvedDesc',
  segmented: 'wizard.hornSegmentedDesc',
  blade: 'wizard.hornBladeDesc',
  heavy: 'wizard.hornHeavyDesc',
  branson_dp: 'wizard.hornBransonDpDesc',
  custom: 'wizard.hornCustomDesc',
}

const hornDescription = computed(() => t(hornDescriptions[store.hornType] ?? 'wizard.hornFlatDesc'))
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
</style>
