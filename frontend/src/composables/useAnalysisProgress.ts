import { ref, onUnmounted } from 'vue'

export interface ProgressState {
  taskId: string
  taskType: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'
  step: number
  stepName: string
  totalSteps: number
  progress: number
  message: string
  error?: string
}

export function useAnalysisProgress(taskId: string) {
  const state = ref<ProgressState | null>(null)
  const isConnected = ref(false)
  const error = ref<string | null>(null)

  let ws: WebSocket | null = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 3

  function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/analysis/${taskId}`

    ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      isConnected.value = true
      error.value = null
      reconnectAttempts = 0
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)

      if (data.type === 'ping') return

      if (data.type === 'error') {
        error.value = data.message
        return
      }

      if (data.type === 'connected' || data.type === 'progress') {
        state.value = {
          taskId: data.task_id,
          taskType: data.task_type,
          status: data.status,
          step: data.current_step ?? data.step ?? 0,
          stepName: data.step_name ?? '',
          totalSteps: data.total_steps,
          progress: data.progress,
          message: data.message ?? '',
        }
      }

      if (data.type === 'completed') {
        state.value = {
          ...state.value!,
          status: 'completed',
          progress: 1,
        }
      }

      if (data.type === 'failed') {
        state.value = {
          ...state.value!,
          status: 'failed',
          error: data.error,
        }
        error.value = data.error
      }

      if (data.type === 'cancelled') {
        state.value = {
          ...state.value!,
          status: 'cancelled',
        }
      }
    }

    ws.onclose = () => {
      isConnected.value = false

      // Try to reconnect if not completed
      if (state.value?.status === 'running' && reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++
        setTimeout(connect, 1000 * reconnectAttempts)
      }
    }

    ws.onerror = () => {
      error.value = 'WebSocket connection error'
    }
  }

  function disconnect() {
    if (ws) {
      ws.close()
      ws = null
    }
  }

  // Auto-connect
  connect()

  onUnmounted(() => {
    disconnect()
  })

  return {
    state,
    isConnected,
    error,
    disconnect,
    reconnect: connect,
  }
}
