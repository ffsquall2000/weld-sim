<template>
  <div v-if="visible" class="fea-progress">
    <div class="fea-progress__header">
      <span class="fea-progress__icon">⚙️</span>
      <span class="fea-progress__title">{{ title }}</span>
      <button
        v-if="canCancel"
        class="fea-progress__cancel"
        @click="$emit('cancel')"
        :disabled="cancelling"
      >
        {{ cancelling ? t('progress.cancelling') : t('progress.cancel') }}
      </button>
    </div>

    <div class="fea-progress__bar-container">
      <div class="fea-progress__bar" :style="{ width: displayPercent + '%' }">
        <span v-if="displayPercent > 15" class="fea-progress__bar-text">
          {{ displayPercent }}%
        </span>
      </div>
      <span v-if="displayPercent <= 15" class="fea-progress__percent-outside">
        {{ displayPercent }}%
      </span>
    </div>

    <div class="fea-progress__details">
      <span class="fea-progress__phase">
        {{ phaseLabel }}
        <template v-if="stepIndex > 0 && totalSteps > 0">
          ({{ stepIndex }}/{{ totalSteps }})
        </template>
      </span>
      <span class="fea-progress__time">
        {{ elapsedStr }}
        <template v-if="estimatedStr"> / {{ t('progress.estimated') }} {{ estimatedStr }}</template>
      </span>
    </div>

    <div v-if="message" class="fea-progress__message">{{ message }}</div>

    <div v-if="error" class="fea-progress__error">
      <span>❌</span> {{ error }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  visible: boolean
  taskId: string
  title?: string
  wsUrl?: string
}>()

const emit = defineEmits<{
  (e: 'cancel'): void
  (e: 'complete', result: any): void
  (e: 'error', error: string): void
  (e: 'progress', data: { phase: string; progress: number; message: string }): void
}>()

const progress = ref(0)
const phase = ref('')
const message = ref('')
const error = ref('')
const stepIndex = ref(0)
const totalSteps = ref(0)
const cancelling = ref(false)
const startTime = ref(0)
const elapsed = ref(0)
const estimatedTotal = ref(0)

let ws: WebSocket | null = null
let timerHandle: ReturnType<typeof setInterval> | null = null

// Phase label mapping
const phaseLabels: Record<string, string> = {
  init: 'progress.phase_init',
  import_step: 'progress.phase_import',
  meshing: 'progress.phase_meshing',
  assembly: 'progress.phase_assembly',
  solving: 'progress.phase_solving',
  classifying: 'progress.phase_classifying',
  packaging: 'progress.phase_packaging',
  harmonic: 'progress.phase_harmonic',
  component_analysis: 'progress.phase_component',
  aggregation: 'progress.phase_aggregation',
}

const displayPercent = computed(() => Math.round(progress.value * 100))

const canCancel = computed(() => progress.value < 1.0 && !error.value)

const phaseLabel = computed(() => {
  const key = phaseLabels[phase.value]
  return key ? t(key) : phase.value || t('progress.preparing')
})

const elapsedStr = computed(() => formatDuration(elapsed.value))
const estimatedStr = computed(() => {
  if (estimatedTotal.value > 0 && progress.value > 0.05 && progress.value < 0.95) {
    return formatDuration(estimatedTotal.value)
  }
  return ''
})

function formatDuration(seconds: number): string {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return m > 0 ? `${m}:${s.toString().padStart(2, '0')}` : `0:${s.toString().padStart(2, '0')}`
}

function connectWs() {
  if (!props.taskId || !props.visible) return

  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const base = props.wsUrl || `${protocol}//${window.location.host}`
  const url = `${base}/api/v1/ws/analysis/${props.taskId}`

  try {
    ws = new WebSocket(url)

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleMessage(data)
      } catch {
        // ignore parse errors
      }
    }

    ws.onclose = () => {
      ws = null
    }

    ws.onerror = () => {
      ws = null
    }
  } catch {
    // WebSocket creation failed
  }
}

function handleMessage(data: any) {
  switch (data.type) {
    case 'progress':
      progress.value = data.progress ?? 0
      phase.value = data.phase ?? data.step_name ?? ''
      message.value = data.message ?? ''
      if (data.step !== undefined) stepIndex.value = data.step
      if (data.total_steps !== undefined) totalSteps.value = data.total_steps

      // Estimate remaining time
      if (progress.value > 0.05) {
        estimatedTotal.value = elapsed.value / progress.value
      }

      emit('progress', {
        phase: phase.value,
        progress: progress.value,
        message: message.value,
      })
      break

    case 'completed':
      progress.value = 1.0
      phase.value = 'complete'
      message.value = t('progress.complete')
      emit('complete', data.result)
      cleanup()
      break

    case 'failed':
      error.value = data.error || t('progress.unknown_error')
      emit('error', error.value)
      cleanup()
      break

    case 'cancelled':
      error.value = t('progress.cancelled')
      emit('error', error.value)
      cleanup()
      break
  }
}

// Allow direct progress updates from parent (for non-WebSocket mode)
function updateProgress(p: number, ph: string, msg: string) {
  progress.value = p
  phase.value = ph
  message.value = msg
  if (p > 0.05) {
    estimatedTotal.value = elapsed.value / p
  }
}

function requestCancel() {
  cancelling.value = true
  // Send cancel via WebSocket if connected
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'cancel', task_id: props.taskId }))
  }
  emit('cancel')
}

function cleanup() {
  if (timerHandle) {
    clearInterval(timerHandle)
    timerHandle = null
  }
  if (ws) {
    ws.close()
    ws = null
  }
}

function reset() {
  progress.value = 0
  phase.value = ''
  message.value = ''
  error.value = ''
  stepIndex.value = 0
  totalSteps.value = 0
  cancelling.value = false
  elapsed.value = 0
  estimatedTotal.value = 0
  startTime.value = Date.now()
}

watch(() => props.visible, (v) => {
  if (v) {
    reset()
    startTime.value = Date.now()
    timerHandle = setInterval(() => {
      elapsed.value = (Date.now() - startTime.value) / 1000
    }, 500)
    connectWs()
  } else {
    cleanup()
  }
})

watch(() => props.taskId, (id) => {
  if (id && props.visible) {
    cleanup()
    connectWs()
  }
})

onMounted(() => {
  if (props.visible) {
    reset()
    startTime.value = Date.now()
    timerHandle = setInterval(() => {
      elapsed.value = (Date.now() - startTime.value) / 1000
    }, 500)
    connectWs()
  }
})

onUnmounted(cleanup)

defineExpose({ updateProgress, reset, requestCancel })
</script>

<style scoped>
.fea-progress {
  background: var(--bg-secondary, #f8f9fa);
  border: 1px solid var(--border-color, #e2e8f0);
  border-radius: 8px;
  padding: 16px;
  margin: 12px 0;
}

.fea-progress__header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.fea-progress__icon {
  font-size: 18px;
  animation: spin 2s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.fea-progress__title {
  flex: 1;
  font-weight: 600;
  font-size: 14px;
  color: var(--text-primary, #1a202c);
}

.fea-progress__cancel {
  padding: 4px 12px;
  font-size: 12px;
  border: 1px solid #e53e3e;
  color: #e53e3e;
  background: transparent;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.fea-progress__cancel:hover:not(:disabled) {
  background: #e53e3e;
  color: white;
}

.fea-progress__cancel:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.fea-progress__bar-container {
  position: relative;
  height: 24px;
  background: var(--bg-tertiary, #e2e8f0);
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 8px;
}

.fea-progress__bar {
  height: 100%;
  background: linear-gradient(90deg, #4299e1, #3182ce);
  border-radius: 12px;
  transition: width 0.5s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
}

.fea-progress__bar-text {
  color: white;
  font-size: 12px;
  font-weight: 600;
}

.fea-progress__percent-outside {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary, #4a5568);
}

.fea-progress__details {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-secondary, #718096);
}

.fea-progress__phase {
  font-weight: 500;
}

.fea-progress__time {
  font-variant-numeric: tabular-nums;
}

.fea-progress__message {
  margin-top: 6px;
  font-size: 12px;
  color: var(--text-tertiary, #a0aec0);
  font-style: italic;
}

.fea-progress__error {
  margin-top: 8px;
  padding: 8px 12px;
  background: #fff5f5;
  border: 1px solid #fed7d7;
  border-radius: 4px;
  color: #c53030;
  font-size: 13px;
}
</style>
