/**
 * Composable for WebSocket connection to backend run/optimization progress.
 */
import { ref, onUnmounted } from 'vue'

export interface WSMessage {
  type: string
  run_id?: string
  optimization_id?: string
  percent?: number
  phase?: string
  message?: string
  elapsed_s?: number
  metric_name?: string
  value?: number
  unit?: string
  status?: string
  compute_time_s?: number
  metrics_summary?: Record<string, number>
  error?: string
  timestamp?: string
}

export function useWebSocket() {
  const ws = ref<WebSocket | null>(null)
  const isConnected = ref(false)
  const lastMessage = ref<WSMessage | null>(null)
  const messages = ref<WSMessage[]>([])
  let pingInterval: ReturnType<typeof setInterval> | null = null

  function connect(path: string) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${protocol}//${window.location.host}/api/v2/ws/${path}`

    disconnect() // Close any existing connection

    ws.value = new WebSocket(url)

    ws.value.onopen = () => {
      isConnected.value = true
      // Start ping interval
      pingInterval = setInterval(() => {
        if (ws.value?.readyState === WebSocket.OPEN) {
          ws.value.send(JSON.stringify({ type: 'ping' }))
        }
      }, 30000)
    }

    ws.value.onmessage = (event) => {
      try {
        const data: WSMessage = JSON.parse(event.data)
        if (data.type === 'pong' || data.type === 'heartbeat') return
        lastMessage.value = data
        messages.value.push(data)
      } catch (e) {
        console.warn('[WS] Failed to parse message:', event.data)
      }
    }

    ws.value.onclose = () => {
      isConnected.value = false
      if (pingInterval) {
        clearInterval(pingInterval)
        pingInterval = null
      }
    }

    ws.value.onerror = (error) => {
      console.error('[WS] Error:', error)
      isConnected.value = false
    }
  }

  function connectToRun(runId: string) {
    connect(`runs/${runId}`)
  }

  function connectToOptimization(optimizationId: string) {
    connect(`optimizations/${optimizationId}`)
  }

  function disconnect() {
    if (pingInterval) {
      clearInterval(pingInterval)
      pingInterval = null
    }
    if (ws.value) {
      ws.value.close()
      ws.value = null
    }
    isConnected.value = false
  }

  function clearMessages() {
    messages.value = []
    lastMessage.value = null
  }

  onUnmounted(() => {
    disconnect()
  })

  return {
    ws,
    isConnected,
    lastMessage,
    messages,
    connect,
    connectToRun,
    connectToOptimization,
    disconnect,
    clearMessages,
  }
}
