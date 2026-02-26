<template>
  <span
    class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase"
    :style="badgeStyle"
  >
    <span class="inline-block w-2 h-2 rounded-full" :style="{ backgroundColor: levelColor }"></span>
    {{ label }}: {{ level }}
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  label: string
  level: string
}>()

const levelColors: Record<string, string> = {
  low: '#4caf50',
  medium: '#ff9800',
  high: '#f44336',
  critical: '#d32f2f',
}

const levelColor = computed(() => levelColors[props.level] ?? '#8b949e')

const badgeStyle = computed(() => ({
  backgroundColor: `${levelColor.value}20`,
  color: levelColor.value,
  border: `1px solid ${levelColor.value}40`,
}))
</script>
