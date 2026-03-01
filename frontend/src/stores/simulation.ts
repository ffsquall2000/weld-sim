import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { simulationApi, runApi } from '@/api/v2'
import { useWebSocket, type WSMessage } from '@/composables/useWebSocket'

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

export interface StandardMetricsResult {
  metrics: Record<string, number>
  quality_score: number
  metric_info: Record<string, { unit: string; description: string; range?: number[] }>
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
  const standardMetricsCache = ref<Record<string, StandardMetricsResult>>({})

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
      const response = await simulationApi.list(projectId)
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
      const response = await simulationApi.create(data.project_id, data)
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

  // WebSocket instance for run progress
  const wsInstance = useWebSocket()

  async function submitRun(data: {
    simulation_case_id: string
    geometry_version_id: string
    parameters_override?: any
  }) {
    loading.value = true
    error.value = null
    runProgress.value = 0
    runPhase.value = 'queued'
    runElapsedTime.value = 0
    try {
      const response = await runApi.create(data.simulation_case_id, {
        geometry_version_id: data.geometry_version_id,
        parameters_override: data.parameters_override,
      })
      const run = response.data
      runs.value.push(run)
      activeRun.value = run
      addLog('info', `Run ${run.id} submitted`, 'queued')
      // Connect WebSocket for real-time progress
      watchRunProgress(run.id)
      return run
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      addLog('error', `Run submission failed: ${error.value}`)
      return null
    } finally {
      loading.value = false
    }
  }

  function watchRunProgress(runId: string) {
    wsInstance.clearMessages()
    wsInstance.connectToRun(runId)

    // Watch for incoming WebSocket messages
    watch(
      () => wsInstance.lastMessage.value,
      (msg: WSMessage | null) => {
        if (!msg) return

        switch (msg.type) {
          case 'progress':
            if (msg.percent !== undefined) {
              updateProgress(msg.percent, msg.phase)
            }
            if (msg.elapsed_s !== undefined) {
              runElapsedTime.value = msg.elapsed_s
            }
            if (msg.message) {
              addLog('info', msg.message, msg.phase)
            }
            break

          case 'metric':
            if (msg.metric_name && msg.value !== undefined) {
              addLog('info', `Metric: ${msg.metric_name} = ${msg.value}${msg.unit ? ' ' + msg.unit : ''}`)
            }
            break

          case 'completed':
            runProgress.value = 100
            runPhase.value = 'completed'
            if (activeRun.value?.id === runId) {
              activeRun.value = {
                ...activeRun.value,
                status: 'completed',
                compute_time_s: msg.compute_time_s ?? null,
              }
            }
            addLog('info', 'Run completed', 'completed')
            wsInstance.disconnect()
            break

          case 'failed':
            runPhase.value = 'failed'
            if (activeRun.value?.id === runId) {
              activeRun.value = { ...activeRun.value, status: 'failed' }
            }
            addLog('error', msg.error ?? 'Run failed', 'failed')
            wsInstance.disconnect()
            break

          case 'cancelled':
            runPhase.value = 'cancelled'
            if (activeRun.value?.id === runId) {
              activeRun.value = { ...activeRun.value, status: 'cancelled' }
            }
            addLog('warn', 'Run cancelled', 'cancelled')
            wsInstance.disconnect()
            break
        }
      }
    )
  }

  async function fetchRuns(simulationCaseId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await runApi.list(simulationCaseId)
      runs.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function fetchRunMetrics(runId: string) {
    try {
      const response = await runApi.metrics(runId)
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

  async function fetchStandardMetrics(runId: string): Promise<StandardMetricsResult | null> {
    // Return cached result if available
    if (standardMetricsCache.value[runId]) {
      return standardMetricsCache.value[runId]
    }
    try {
      const response = await runApi.standardMetrics(runId)
      const result: StandardMetricsResult = response.data
      standardMetricsCache.value[runId] = result
      return result
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    }
  }

  async function cancelRun(runId: string) {
    try {
      await runApi.cancel(runId)
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
    // WebSocket
    wsConnected: wsInstance.isConnected,
    standardMetricsCache,
    // Actions
    fetchSimulations,
    createSimulation,
    submitRun,
    watchRunProgress,
    fetchRuns,
    fetchRunMetrics,
    fetchStandardMetrics,
    cancelRun,
    setActiveRun,
    setCurrentSimulation,
    addLog,
    clearLogs,
    updateProgress,
  }
})
