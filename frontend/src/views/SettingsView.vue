<template>
  <div class="p-6 max-w-3xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('settings.title') }}</h1>

    <!-- Appearance -->
    <section class="card mb-6">
      <h2 class="text-lg font-semibold mb-4">{{ $t('settings.appearance') }}</h2>

      <!-- Theme toggle -->
      <div class="flex items-center justify-between mb-4">
        <label class="text-sm" style="color: var(--color-text-secondary)">
          {{ $t('settings.theme') }}
        </label>
        <div class="flex rounded-lg overflow-hidden" style="border: 1px solid var(--color-border)">
          <button
            class="px-4 py-2 text-sm font-semibold transition-colors"
            :style="{
              backgroundColor: settingsStore.theme === 'dark' ? 'var(--color-accent-orange)' : 'transparent',
              color: settingsStore.theme === 'dark' ? '#fff' : 'var(--color-text-secondary)',
            }"
            @click="settingsStore.theme = 'dark'"
          >
            {{ $t('settings.dark') }}
          </button>
          <button
            class="px-4 py-2 text-sm font-semibold transition-colors"
            :style="{
              backgroundColor: settingsStore.theme === 'light' ? 'var(--color-accent-orange)' : 'transparent',
              color: settingsStore.theme === 'light' ? '#fff' : 'var(--color-text-secondary)',
            }"
            @click="settingsStore.theme = 'light'"
          >
            {{ $t('settings.light') }}
          </button>
        </div>
      </div>

      <!-- Language -->
      <div class="flex items-center justify-between">
        <label class="text-sm" style="color: var(--color-text-secondary)">
          {{ $t('settings.language') }}
        </label>
        <select
          :value="settingsStore.locale"
          class="settings-select"
          @change="handleLocaleChange"
        >
          <option value="zh-CN">中文</option>
          <option value="en">English</option>
        </select>
      </div>
    </section>

    <!-- Default Parameters -->
    <section class="card mb-6">
      <h2 class="text-lg font-semibold mb-4">{{ $t('settings.defaults') }}</h2>

      <!-- Default frequency -->
      <div class="flex items-center justify-between mb-4">
        <label class="text-sm" style="color: var(--color-text-secondary)">
          {{ $t('settings.defaultFrequency') }}
        </label>
        <select
          v-model.number="calcStore.frequency"
          class="settings-select"
        >
          <option :value="15">15 kHz</option>
          <option :value="20">20 kHz</option>
          <option :value="30">30 kHz</option>
          <option :value="35">35 kHz</option>
          <option :value="40">40 kHz</option>
        </select>
      </div>

      <!-- Default max power -->
      <div class="flex items-center justify-between">
        <label class="text-sm" style="color: var(--color-text-secondary)">
          {{ $t('settings.defaultMaxPower') }}
        </label>
        <input
          v-model.number="calcStore.maxPower"
          type="number"
          min="500"
          max="10000"
          step="100"
          class="settings-input"
        />
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useSettingsStore } from '@/stores/settings'
import { useCalculationStore } from '@/stores/calculation'

const { locale } = useI18n()
const settingsStore = useSettingsStore()
const calcStore = useCalculationStore()

function handleLocaleChange(e: Event) {
  const val = (e.target as HTMLSelectElement).value
  settingsStore.locale = val
  locale.value = val
}
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1rem 1.25rem;
}

.settings-select {
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
  min-width: 120px;
}

.settings-select:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}

.settings-input {
  padding: 0.5rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  width: 120px;
  text-align: right;
  font-family: ui-monospace, monospace;
}

.settings-input:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}
</style>
