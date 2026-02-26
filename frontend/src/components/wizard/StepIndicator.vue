<template>
  <div class="flex items-center justify-between w-full">
    <template v-for="(step, index) in steps" :key="index">
      <!-- Step circle + label -->
      <div class="flex flex-col items-center relative z-10">
        <div
          class="step-circle"
          :class="{
            'step-current': index === currentStep,
            'step-completed': index < currentStep,
            'step-pending': index > currentStep,
          }"
        >
          <svg
            v-if="index < currentStep"
            class="w-4 h-4"
            fill="none"
            stroke="currentColor"
            stroke-width="3"
            viewBox="0 0 24 24"
          >
            <path stroke-linecap="round" stroke-linejoin="round" d="M5 13l4 4L19 7" />
          </svg>
          <span v-else>{{ index + 1 }}</span>
        </div>
        <span
          class="mt-2 text-xs whitespace-nowrap hidden sm:block"
          :class="index <= currentStep ? 'text-[var(--color-text-primary)]' : 'text-[var(--color-text-secondary)]'"
        >
          {{ step }}
        </span>
      </div>
      <!-- Connecting line -->
      <div
        v-if="index < steps.length - 1"
        class="flex-1 h-0.5 mx-1 transition-colors"
        :class="index < currentStep ? 'bg-[var(--color-success)]' : 'bg-[var(--color-border)]'"
      />
    </template>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  currentStep: number
  steps: string[]
}>()
</script>

<style scoped>
.step-circle {
  width: 2.25rem;
  height: 2.25rem;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.875rem;
  font-weight: 700;
  border: 2px solid;
  transition: all 0.2s;
}

.step-current {
  background-color: var(--color-accent-orange);
  border-color: var(--color-accent-orange);
  color: #fff;
}

.step-completed {
  background-color: var(--color-success);
  border-color: var(--color-success);
  color: #fff;
}

.step-pending {
  background-color: transparent;
  border-color: var(--color-border);
  color: var(--color-text-secondary);
}
</style>
