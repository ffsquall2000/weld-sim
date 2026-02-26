<template>
  <div class="p-6 max-w-4xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('nav.calculate') }}</h1>

    <StepIndicator :currentStep="store.currentStep" :steps="stepLabels" />

    <div class="mt-8">
      <ApplicationStep v-if="store.currentStep === 0" />
      <MaterialStep v-else-if="store.currentStep === 1" />
      <HornAnvilStep v-else-if="store.currentStep === 2" />
      <GeometryStep v-else-if="store.currentStep === 3" />
      <EquipmentStep v-else-if="store.currentStep === 4" />
      <ConstraintStep v-else-if="store.currentStep === 5" />
    </div>

    <!-- Navigation buttons -->
    <div class="flex justify-between mt-8">
      <button
        v-if="store.currentStep > 0"
        class="btn-secondary"
        @click="store.prevStep()"
      >
        {{ $t('wizard.back') }}
      </button>
      <div v-else />

      <button
        v-if="store.currentStep < 5"
        class="btn-primary"
        @click="store.nextStep()"
      >
        {{ $t('wizard.next') }}
      </button>
      <button
        v-else
        class="btn-primary"
        :disabled="store.isCalculating"
        @click="handleCalculate"
      >
        {{ store.isCalculating ? $t('wizard.calculating') : $t('wizard.calculate') }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useCalculationStore } from '@/stores/calculation'
import StepIndicator from '@/components/wizard/StepIndicator.vue'
import ApplicationStep from '@/components/wizard/ApplicationStep.vue'
import MaterialStep from '@/components/wizard/MaterialStep.vue'
import HornAnvilStep from '@/components/wizard/HornAnvilStep.vue'
import GeometryStep from '@/components/wizard/GeometryStep.vue'
import EquipmentStep from '@/components/wizard/EquipmentStep.vue'
import ConstraintStep from '@/components/wizard/ConstraintStep.vue'

const store = useCalculationStore()
const router = useRouter()
const { t } = useI18n()

const stepLabels = computed(() => [
  t('wizard.step1'),
  t('wizard.step2'),
  t('wizard.step3'),
  t('wizard.step4'),
  t('wizard.step5'),
  t('wizard.step6'),
])

async function handleCalculate() {
  const recipeId = await store.submitCalculation()
  if (recipeId) {
    router.push({ name: 'results', params: { id: recipeId } })
  }
}
</script>

<style scoped>
.btn-primary {
  padding: 0.625rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: var(--color-accent-orange);
  color: #fff;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
}

.btn-primary:hover:not(:disabled) {
  opacity: 0.9;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  padding: 0.625rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
  transition: border-color 0.2s;
}

.btn-secondary:hover {
  border-color: var(--color-accent-orange);
}
</style>
