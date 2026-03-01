import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { optimizationApi } from '@/api/v2'

export interface DesignVariable {
  name: string
  var_type: string
  min_value: number | null
  max_value: number | null
  step: number | null
}

export interface Objective {
  metric: string
  direction: string
  weight: number
}

export interface Constraint {
  metric: string
  operator: string
  value: number
}

export interface OptimizationStudy {
  id: string
  name: string
  strategy: string
  status: string
  total_iterations: number
  completed_iterations: number
  design_variables: DesignVariable[]
  objectives: Objective[]
  constraints: Constraint[]
  pareto_front_run_ids: string[]
}

export interface Iteration {
  iteration_number: number
  run_id: string
  design_point: Record<string, number>
  objective_values: Record<string, number>
  constraint_values: Record<string, number>
  feasible: boolean
  pareto_optimal: boolean
}

export interface ParetoPoint {
  run_id: string
  objective_values: Record<string, number>
  design_point: Record<string, number>
}

export const useOptimizationStore = defineStore('optimization', () => {
  // State
  const studies = ref<OptimizationStudy[]>([])
  const currentStudy = ref<OptimizationStudy | null>(null)
  const iterations = ref<Iteration[]>([])
  const paretoFront = ref<ParetoPoint[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Getters
  const studyProgress = computed(() => {
    if (!currentStudy.value) return 0
    const { total_iterations, completed_iterations } = currentStudy.value
    if (total_iterations === 0) return 0
    return Math.round((completed_iterations / total_iterations) * 100)
  })

  const isStudyRunning = computed(
    () => currentStudy.value?.status === 'running'
  )

  const feasibleIterations = computed(() =>
    iterations.value.filter((it) => it.feasible)
  )

  const paretoOptimalIterations = computed(() =>
    iterations.value.filter((it) => it.pareto_optimal)
  )

  const bestIteration = computed(() => {
    const feasible = feasibleIterations.value
    if (feasible.length === 0) return null
    // Return the iteration with the best primary objective value
    if (!currentStudy.value?.objectives.length) return feasible[0]
    const primaryObj = currentStudy.value.objectives[0]!
    return feasible.reduce((best, it) => {
      const bestVal = best.objective_values[primaryObj.metric] ?? 0
      const curVal = it.objective_values[primaryObj.metric] ?? 0
      if (primaryObj.direction === 'minimize') {
        return curVal < bestVal ? it : best
      }
      return curVal > bestVal ? it : best
    })
  })

  // Actions
  async function createStudy(data: {
    simulation_id: string
    name: string
    strategy: string
    total_iterations: number
    design_variables: DesignVariable[]
    objectives: Objective[]
    constraints: Constraint[]
  }) {
    loading.value = true
    error.value = null
    try {
      const response = await optimizationApi.create(data.simulation_id, data)
      studies.value.push(response.data)
      currentStudy.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function fetchStudy(studyId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await optimizationApi.get(studyId)
      currentStudy.value = response.data
      const index = studies.value.findIndex((s) => s.id === studyId)
      if (index !== -1) {
        studies.value[index] = response.data
      }
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function fetchIterations(studyId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await optimizationApi.iterations(studyId)
      iterations.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function fetchPareto(studyId: string) {
    try {
      const response = await optimizationApi.pareto(studyId)
      paretoFront.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return []
    }
  }

  async function pauseStudy(studyId: string) {
    try {
      await optimizationApi.pause(studyId)
      if (currentStudy.value?.id === studyId) {
        currentStudy.value = { ...currentStudy.value, status: 'paused' }
      }
      return true
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return false
    }
  }

  async function resumeStudy(studyId: string) {
    try {
      await optimizationApi.resume(studyId)
      if (currentStudy.value?.id === studyId) {
        currentStudy.value = { ...currentStudy.value, status: 'running' }
      }
      return true
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return false
    }
  }

  function setCurrentStudy(study: OptimizationStudy | null) {
    currentStudy.value = study
  }

  function clearIterations() {
    iterations.value = []
    paretoFront.value = []
  }

  return {
    // State
    studies,
    currentStudy,
    iterations,
    paretoFront,
    loading,
    error,
    // Getters
    studyProgress,
    isStudyRunning,
    feasibleIterations,
    paretoOptimalIterations,
    bestIteration,
    // Actions
    createStudy,
    fetchStudy,
    fetchIterations,
    fetchPareto,
    pauseStudy,
    resumeStudy,
    setCurrentStudy,
    clearIterations,
  }
})
