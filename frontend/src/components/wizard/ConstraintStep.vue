<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step6') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.constraintDesc') }}
    </p>

    <div class="wizard-card">
      <h3 class="font-semibold mb-4">{{ $t('wizard.summaryTitle') }}</h3>

      <div class="space-y-3">
        <!-- Application -->
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.step1') }}</span>
          <span class="summary-value">{{ applicationLabel }}</span>
        </div>

        <!-- Materials -->
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.upperMaterial') }}</span>
          <span class="summary-value">
            {{ store.upperMaterial }} {{ store.upperThickness }}mm x {{ store.upperLayers }}
          </span>
        </div>
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.lowerMaterial') }}</span>
          <span class="summary-value">{{ store.lowerMaterial }} {{ store.lowerThickness }}mm</span>
        </div>

        <!-- Horn / Knurl -->
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.hornType') }}</span>
          <span class="summary-value">{{ store.hornType }}</span>
        </div>
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.knurlPattern') }}</span>
          <span class="summary-value">
            {{ store.knurlType }} ({{ store.knurlPitch }}mm / {{ store.knurlToothWidth }}mm / {{ store.knurlDepth }}mm)
          </span>
        </div>
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.anvilType') }}</span>
          <span class="summary-value">{{ store.anvilType }}</span>
        </div>

        <!-- Geometry -->
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.weldDimensions') }}</span>
          <span class="summary-value">
            {{ store.weldWidth }} x {{ store.weldLength }} mm ({{ store.effectiveArea.toFixed(1) }} mm<sup>2</sup>)
          </span>
        </div>

        <!-- Equipment -->
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.frequency') }}</span>
          <span class="summary-value">{{ store.frequency }} kHz</span>
        </div>
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.maxPower') }}</span>
          <span class="summary-value">{{ store.maxPower }} W</span>
        </div>
        <div class="summary-row">
          <span class="summary-label">{{ $t('wizard.boosterGain') }}</span>
          <span class="summary-value">{{ store.boosterGain }}x</span>
        </div>
      </div>
    </div>

    <!-- Ready indicator -->
    <div class="mt-6 text-center">
      <div class="ready-badge">
        <svg class="w-5 h-5 inline-block mr-1" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
        </svg>
        {{ $t('wizard.readyToCalculate') }}
      </div>
    </div>

    <!-- Error display -->
    <div v-if="store.error" class="mt-4 p-3 rounded-lg text-sm" style="background-color: rgba(244, 67, 54, 0.1); color: var(--color-danger)">
      {{ store.error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()
const { t } = useI18n()

const appLabels: Record<string, string> = {
  li_battery_tab: 'wizard.appLiBatteryTab',
  li_battery_busbar: 'wizard.appLiBatteryBusbar',
  li_battery_collector: 'wizard.appLiBatteryCollector',
  general_metal: 'wizard.appGeneralMetal',
}

const applicationLabel = computed(() => t(appLabels[store.application] ?? store.application))
</script>

<style scoped>
.wizard-card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.25rem;
}

.summary-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--color-border);
}

.summary-row:last-child {
  border-bottom: none;
}

.summary-label {
  font-size: 0.875rem;
  color: var(--color-text-secondary);
}

.summary-value {
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text-primary);
}

.ready-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 1.25rem;
  border-radius: 9999px;
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
  font-weight: 600;
  font-size: 0.875rem;
}
</style>
