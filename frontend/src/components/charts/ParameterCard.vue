<template>
  <div
    class="rounded-lg p-4"
    :style="{
      backgroundColor: 'var(--color-bg-secondary)',
      border: '1px solid var(--color-border)',
      borderLeft: `3px solid ${borderColor}`,
    }"
  >
    <div class="text-sm" style="color: var(--color-text-secondary)">
      {{ label }}
    </div>
    <div class="text-2xl font-bold mt-1 font-mono" style="color: var(--color-text-primary)">
      {{ formattedValue }}
      <span class="text-sm font-normal" style="color: var(--color-text-secondary)">{{ unit }}</span>
    </div>
    <div
      v-if="safeMin !== undefined && safeMax !== undefined"
      class="text-xs mt-2"
      style="color: var(--color-text-secondary)"
    >
      {{ $t('result.safeRangeLabel') }}: {{ safeMin }}{{ unit ? ' ' + unit : '' }} &ndash; {{ safeMax }}{{ unit ? ' ' + unit : '' }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  label: string
  value: number | string
  unit: string
  safeMin?: number
  safeMax?: number
  color?: string
}>(), {
  color: '#ff9800',
})

const isOutOfRange = computed(() => {
  if (props.safeMin === undefined || props.safeMax === undefined) return false
  const numVal = typeof props.value === 'number' ? props.value : parseFloat(String(props.value))
  if (isNaN(numVal)) return false
  return numVal < props.safeMin || numVal > props.safeMax
})

const borderColor = computed(() => (isOutOfRange.value ? '#f44336' : props.color))

const formattedValue = computed(() => {
  if (typeof props.value === 'number') {
    return Number.isInteger(props.value) ? props.value.toString() : props.value.toFixed(2)
  }
  return props.value
})
</script>
