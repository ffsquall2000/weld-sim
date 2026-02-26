<template>
  <div>
    <h2 class="text-xl font-semibold mb-2">{{ $t('wizard.step1') }}</h2>
    <p class="text-sm mb-6" style="color: var(--color-text-secondary)">
      {{ $t('wizard.appDesc') }}
    </p>
    <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
      <button
        v-for="opt in options"
        :key="opt.value"
        class="app-card text-left"
        :class="{ 'app-card-selected': store.application === opt.value }"
        @click="store.application = opt.value"
      >
        <span class="text-2xl mb-2 block">{{ opt.icon }}</span>
        <span class="font-semibold block mb-1">{{ opt.label }}</span>
        <span class="text-xs" style="color: var(--color-text-secondary)">{{ opt.desc }}</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'

const store = useCalculationStore()
const { t } = useI18n()

const options = [
  {
    value: 'li_battery_tab',
    icon: '\u{1F50B}',
    label: t('wizard.appLiBatteryTab'),
    desc: t('wizard.appLiBatteryTabDesc'),
  },
  {
    value: 'li_battery_busbar',
    icon: '\u26A1',
    label: t('wizard.appLiBatteryBusbar'),
    desc: t('wizard.appLiBatteryBusbarDesc'),
  },
  {
    value: 'li_battery_collector',
    icon: '\u{1F4E6}',
    label: t('wizard.appLiBatteryCollector'),
    desc: t('wizard.appLiBatteryCollectorDesc'),
  },
  {
    value: 'general_metal',
    icon: '\u2699\uFE0F',
    label: t('wizard.appGeneralMetal'),
    desc: t('wizard.appGeneralMetalDesc'),
  },
]
</script>

<style scoped>
.app-card {
  background-color: var(--color-bg-secondary);
  border: 2px solid var(--color-border);
  border-radius: 0.75rem;
  padding: 1.25rem;
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.app-card:hover {
  border-color: var(--color-accent-orange);
}

.app-card-selected {
  border-color: var(--color-accent-orange);
  box-shadow: 0 0 0 1px var(--color-accent-orange);
}
</style>
