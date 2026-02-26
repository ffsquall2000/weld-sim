import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { simulationApi, type SimulateRequest, type SimulateResponse } from '@/api/simulation'

export const useCalculationStore = defineStore('calculation', () => {
  // Step navigation
  const currentStep = ref(0)
  const totalSteps = 6

  // Step 1: Application
  const application = ref('li_battery_tab')

  // Step 2: Materials
  const upperMaterial = ref('Nickel 201')
  const upperThickness = ref(0.1)
  const upperLayers = ref(40)
  const lowerMaterial = ref('Copper C110')
  const lowerThickness = ref(0.3)

  // Step 3: Horn / Anvil / Knurl
  const hornType = ref('flat')
  const knurlType = ref('cross_hatch')
  const knurlPitch = ref(1.2)
  const knurlToothWidth = ref(0.6)
  const knurlDepth = ref(0.35)
  const anvilType = ref('fixed_flat')
  const anvilResonantFreq = ref(0.0)

  // Step 4: Geometry (placeholder values)
  const weldWidth = ref(3.0)
  const weldLength = ref(25.0)

  // Step 5: Equipment
  const frequency = ref(20.0)
  const maxPower = ref(3500)
  const boosterGain = ref(1.5)

  // Computed
  const contactRatio = computed(() => {
    if (knurlPitch.value === 0) return 0
    return knurlToothWidth.value / knurlPitch.value
  })

  const effectiveArea = computed(() => weldWidth.value * weldLength.value)

  // Calculation result
  const result = ref<SimulateResponse | null>(null)
  const isCalculating = ref(false)
  const error = ref<string | null>(null)

  // Navigation actions
  function nextStep() {
    if (currentStep.value < totalSteps - 1) {
      currentStep.value++
    }
  }

  function prevStep() {
    if (currentStep.value > 0) {
      currentStep.value--
    }
  }

  function goToStep(n: number) {
    if (n >= 0 && n < totalSteps) {
      currentStep.value = n
    }
  }

  // Build the API request from current state
  function buildRequest(): SimulateRequest {
    return {
      application: application.value,
      upper_material_type: upperMaterial.value,
      upper_thickness_mm: upperThickness.value,
      upper_layers: upperLayers.value,
      lower_material_type: lowerMaterial.value,
      lower_thickness_mm: lowerThickness.value,
      weld_width_mm: weldWidth.value,
      weld_length_mm: weldLength.value,
      frequency_khz: frequency.value,
      max_power_w: maxPower.value,
      horn_type: hornType.value,
      knurl_type: knurlType.value,
      knurl_pitch_mm: knurlPitch.value,
      knurl_tooth_width_mm: knurlToothWidth.value,
      knurl_depth_mm: knurlDepth.value,
      anvil_type: anvilType.value,
      booster_gain: boosterGain.value,
    }
  }

  async function submitCalculation(): Promise<string | null> {
    isCalculating.value = true
    error.value = null
    try {
      const response = await simulationApi.calculate(buildRequest())
      result.value = response.data
      return response.data.recipe_id
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : String(err)
      error.value = message
      return null
    } finally {
      isCalculating.value = false
    }
  }

  function reset() {
    currentStep.value = 0
    application.value = 'li_battery_tab'
    upperMaterial.value = 'Nickel 201'
    upperThickness.value = 0.1
    upperLayers.value = 40
    lowerMaterial.value = 'Copper C110'
    lowerThickness.value = 0.3
    weldWidth.value = 3.0
    weldLength.value = 25.0
    hornType.value = 'flat'
    knurlType.value = 'cross_hatch'
    knurlPitch.value = 1.2
    knurlToothWidth.value = 0.6
    knurlDepth.value = 0.35
    anvilType.value = 'fixed_flat'
    anvilResonantFreq.value = 0.0
    frequency.value = 20.0
    maxPower.value = 3500
    boosterGain.value = 1.5
    result.value = null
    isCalculating.value = false
    error.value = null
  }

  return {
    // State
    currentStep,
    application,
    upperMaterial,
    upperThickness,
    upperLayers,
    lowerMaterial,
    lowerThickness,
    weldWidth,
    weldLength,
    hornType,
    knurlType,
    knurlPitch,
    knurlToothWidth,
    knurlDepth,
    anvilType,
    anvilResonantFreq,
    frequency,
    maxPower,
    boosterGain,
    result,
    isCalculating,
    error,
    // Computed
    contactRatio,
    effectiveArea,
    // Actions
    nextStep,
    prevStep,
    goToStep,
    submitCalculation,
    reset,
  }
})
