<template>
  <svg viewBox="0 0 200 200" class="w-full max-w-[200px]" xmlns="http://www.w3.org/2000/svg">
    <!-- Horn shape -->
    <g v-if="hornType === 'flat'">
      <rect x="50" y="40" width="100" height="100" rx="6" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else-if="hornType === 'curved'">
      <path d="M50,40 L150,40 L150,120 Q100,160 50,120 Z" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else-if="hornType === 'segmented'">
      <rect x="50" y="40" width="28" height="100" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
      <rect x="86" y="40" width="28" height="100" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
      <rect x="122" y="40" width="28" height="100" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else-if="hornType === 'blade'">
      <polygon points="60,40 140,40 115,140 85,140" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else-if="hornType === 'heavy'">
      <rect x="30" y="40" width="140" height="100" rx="6" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else-if="hornType === 'branson_dp'">
      <!-- Stepped profile: 3 decreasing widths -->
      <rect x="40" y="40" width="120" height="30" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
      <rect x="55" y="70" width="90" height="30" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
      <rect x="70" y="100" width="60" height="40" rx="3" fill="#ff9800" stroke="#e68900" stroke-width="2" />
    </g>

    <g v-else>
      <!-- custom: dashed outline with ? -->
      <rect
        x="50" y="40" width="100" height="100" rx="6"
        fill="none" stroke="#ff9800" stroke-width="2" stroke-dasharray="6 4"
      />
      <text x="100" y="100" text-anchor="middle" fill="#ff9800" font-size="36" font-weight="bold">?</text>
    </g>

    <!-- Label -->
    <text x="100" y="170" text-anchor="middle" fill="var(--color-text-secondary)" font-size="11">
      {{ hornLabel }}
    </text>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  hornType: string
}>()

const hornLabelKeys: Record<string, string> = {
  flat: 'hornLabel.flat',
  curved: 'hornLabel.curved',
  segmented: 'hornLabel.segmented',
  blade: 'hornLabel.blade',
  heavy: 'hornLabel.heavy',
  branson_dp: 'hornLabel.branson_dp',
  custom: 'hornLabel.custom',
}

const hornLabel = computed(() => {
  const key = hornLabelKeys[props.hornType]
  return key ? t(key) : props.hornType
})
</script>
