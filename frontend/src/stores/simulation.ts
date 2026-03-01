import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v2',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

export interface SimulationCase {
  id: string
  project_id: string
  name: string
  analysis_type: string
  solver_backend: string
  configuration: Record<string, any> | null
  boundary_conditions: Record<string, any> | null
  material_assignments: Record<string, any> | null
}

export interface Metric {
  metric_name: string
  value: number
  unit: string | null
}

export interface Run {
  id: string
  simulation_case_id: string
  geometry_version_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  compute_time_s: number | null
  metrics: Metric[]
}

export interface LogEntry {
  timestamp: string
  level: 'info' | 'warn' | 'error'
  message: string
  phase?: string
}

export const useSimulationStore = defineStore('simulation', () => {
  // State
  const simulations = ref<SimulationCase[]>([])
  const currentSimulation = ref<SimulationCase | null>(null)
  const runs = ref<Run[]>([])
  const activeRun = ref<Run | null>(null)
  const logs = ref<LogEntry[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Progress tracking
  const runProgress = ref(0)
  const runPhase = ref<string>('')
  const runElapsedTime = ref(0)

  // Getters
  const simulationsByProject = computed(() => {
    return (projectId: string) =>
      simulations.value.filter((s) => s.project_id === projectId)
  })

  const runsBySimulation = computed(() => {
    return (simId: string) =>
      runs.value.filter((r) => r.simulation_case_id === simId)
  })

  const activeRunMetrics = computed(() => activeRun.value?.metrics ?? [])

  const isRunning = computed(
    () => activeRun.value?.status === 'running' || activeRun.value?.status === 'queued'
  )

  // Actions
  async function fetchSimulations(projectId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await api.get<SimulationCase[]>(
        `/projects/${projectId}/simulations`
      )
      simulations.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function createSimulation(data: {
    project_id: string
    name: string
    analysis_type: string
    solver_backend: string
    configuration?: Record<string, any>
    boundary_conditions?: Record<string, any>
    material_assignments?: Record<string, any>
  }) {
    loading.value = true
    error.value = null
    try {
      const response = await api.post<SimulationCase>(
        `/projects/${data.project_id}/simulations`,
        data
      )
      simulations.value.push(response.data)
      currentSimulation.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function submitRun(data: {
    simulation_case_id: string
    geometry_version_id: string
  }) {
    loading.value = true
    error.value = null
    runProgress.value = 0
    runPhase.value = 'queued'
    runElapsedTime.value = 0
    try {
      const response = await api.post<Run>(
        `/simulations/${data.simulation_case_id}/runs`,
        data
      )
      const run = response.data
      runs.value.push(run)
      activeRun.value = run
      addLog('info', `Run ${run.id} submitted`, 'queued')
      return run
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      addLog('error', `Run submission failed: ${error.value}`)
      return null
    } finally {
      loading.value = false
    }
  }

  async function fetchRuns(simulationCaseId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await api.get<Run[]>(
        `/simulations/${simulationCaseId}/runs`
      )
      runs.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function fetchRunMetrics(runId: string) {
    try {
      const response = await api.get<Metric[]>(`/runs/${runId}/metrics`)
      const run = runs.value.find((r) => r.id === runId)
      if (run) {
        run.metrics = response.data
      }
      if (activeRun.value?.id === runId) {
        activeRun.value = { ...activeRun.value, metrics: response.data }
      }
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return []
    }
  }

  async function cancelRun(runId: string) {
    try {
      await api.post(`/runs/${runId}/cancel`)
      const run = runs.value.find((r) => r.id === runId)
      if (run) {
        run.status = 'cancelled'
      }
      if (activeRun.value?.id === runId) {
        activeRun.value = { ...activeRun.value, status: 'cancelled' }
        runPhase.value = 'cancelled'
      }
      addLog('warn', `Run ${runId} cancelled`)
      return true
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return false
    }
  }

  function setActiveRun(run: Run | null) {
    activeRun.value = run
  }

  function setCurrentSimulation(sim: SimulationCase | null) {
    currentSimulation.value = sim
  }

  // Log management
  function addLog(level: LogEntry['level'], message: string, phase?: string) {
    logs.value.push({
      timestamp: new Date().toISOString(),
      level,
      message,
      phase,
    })
    if (phase) {
      runPhase.value = phase
    }
  }

  function clearLogs() {
    logs.value = []
  }

  function updateProgress(progress: number, phase?: string) {
    runProgress.value = Math.min(100, Math.max(0, progress))
    if (phase) {
      runPhase.value = phase
    }
  }

  return {
    // State
    simulations,
    currentSimulation,
    runs,
    activeRun,
    logs,
    loading,
    error,
    runProgress,
    runPhase,
    runElapsedTime,
    // Getters
    simulationsByProject,
    runsBySimulation,
    activeRunMetrics,
    isRunning,
    // Actions
    fetchSimulations,
    createSimulation,
    submitRun,
    fetchRuns,
    fetchRunMetrics,
    cancelRun,
    setActiveRun,
    setCurrentSimulation,
    addLog,
    clearLogs,
    updateProgress,
  }
})
