<template>
  <div class="solver-console">
    <!-- Toolbar -->
    <div class="sc-toolbar">
      <div class="sc-toolbar-left">
        <span class="sc-toolbar-title">{{ t('solverConsole.title') }}</span>
        <span v-if="simulationStore.isRunning" class="sc-phase-badge">
          {{ phaseLabel }}
        </span>
      </div>
      <div class="sc-toolbar-right">
        <span v-if="simulationStore.isRunning" class="sc-elapsed">
          {{ formattedElapsedTime }}
        </span>
        <button class="sc-toolbar-btn" @click="toggleAutoScroll" :title="t('solverConsole.autoScroll')">
          <span :class="{ 'sc-icon-active': autoScroll }">&#8645;</span>
        </button>
        <button class="sc-toolbar-btn" @click="clearLogs" :title="t('solverConsole.clear')">
          &#10005;
        </button>
      </div>
    </div>

    <!-- Progress Bar -->
    <div v-if="simulationStore.isRunning || simulationStore.runProgress > 0" class="sc-progress">
      <div
        class="sc-progress-bar"
        :style="{ width: simulationStore.runProgress + '%' }"
        :class="{
          'sc-progress-bar--active': simulationStore.isRunning,
          'sc-progress-bar--complete': simulationStore.runProgress >= 100,
        }"
      />
      <span class="sc-progress-text">{{ simulationStore.runProgress }}%</span>
    </div>

    <!-- Log Output -->
    <div ref="logContainer" class="sc-log-area" @scroll="handleScroll">
      <div v-if="simulationStore.logs.length === 0" class="sc-log-empty">
        {{ t('solverConsole.noLogs') }}
      </div>
      <div
        v-for="(entry, index) in simulationStore.logs"
        :key="index"
        class="sc-log-line"
        :class="logLineClass(entry.level)"
      >
        <span class="sc-log-time">{{ formatTime(entry.timestamp) }}</span>
        <span class="sc-log-level" :class="logLevelClass(entry.level)">
          {{ entry.level.toUpperCase().padEnd(5) }}
        </span>
        <span v-if="entry.phase" class="sc-log-phase">[{{ entry.phase }}]</span>
        <span class="sc-log-message">{{ entry.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useSimulationStore } from '@/stores/simulation'

const { t } = useI18n()
const simulationStore = useSimulationStore()

const logContainer = ref<HTMLElement | null>(null)
const autoScroll = ref(true)
const elapsedTimer = ref<ReturnType<typeof setInterval> | null>(null)

const phaseLabel = computed(() => {
  const phase = simulationStore.runPhase
  switch (phase) {
    case 'queued': return t('solverConsole.phaseQueued')
    case 'meshing': return t('solverConsole.phaseMeshing')
    case 'solving': return t('solverConsole.phaseSolving')
    case 'postprocessing': return t('solverConsole.phasePostprocessing')
    case 'cancelled': return t('solverConsole.phaseCancelled')
    default: return phase
  }
})

const formattedElapsedTime = computed(() => {
  const seconds = simulationStore.runElapsedTime
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
})

function formatTime(iso: string): string {
  const d = new Date(iso)
  const h = d.getHours().toString().padStart(2, '0')
  const m = d.getMinutes().toString().padStart(2, '0')
  const s = d.getSeconds().toString().padStart(2, '0')
  return `${h}:${m}:${s}`
}

function logLineClass(level: string) {
  return {
    'sc-log-line--warn': level === 'warn',
    'sc-log-line--error': level === 'error',
  }
}

function logLevelClass(level: string) {
  return {
    'sc-level--info': level === 'info',
    'sc-level--warn': level === 'warn',
    'sc-level--error': level === 'error',
  }
}

function toggleAutoScroll() {
  autoScroll.value = !autoScroll.value
}

function clearLogs() {
  simulationStore.clearLogs()
}

function handleScroll() {
  if (!logContainer.value) return
  const el = logContainer.value
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30
  if (!atBottom) {
    autoScroll.value = false
  }
}

function scrollToBottom() {
  if (!autoScroll.value || !logContainer.value) return
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}

// Watch for new log entries and auto-scroll
watch(
  () => simulationStore.logs.length,
  () => {
    scrollToBottom()
  }
)

// Elapsed time counter
watch(
  () => simulationStore.isRunning,
  (running) => {
    if (running) {
      simulationStore.runElapsedTime = 0
      elapsedTimer.value = setInterval(() => {
        simulationStore.runElapsedTime++
      }, 1000)
    } else if (elapsedTimer.value) {
      clearInterval(elapsedTimer.value)
      elapsedTimer.value = null
    }
  },
  { immediate: true }
)

onUnmounted(() => {
  if (elapsedTimer.value) {
    clearInterval(elapsedTimer.value)
  }
})
</script>

<style scoped>
.solver-console {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 12px;
}

.sc-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 10px;
  background-color: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
  min-height: 28px;
}

.sc-toolbar-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sc-toolbar-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.sc-phase-badge {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  background-color: rgba(88, 166, 255, 0.15);
  color: var(--color-accent-blue);
  font-weight: 500;
}

.sc-toolbar-right {
  display: flex;
  align-items: center;
  gap: 6px;
}

.sc-elapsed {
  font-size: 11px;
  font-family: ui-monospace, monospace;
  color: var(--color-text-secondary);
}

.sc-toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
  border-radius: 3px;
  transition: background-color 0.15s, color 0.15s;
}

.sc-toolbar-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.sc-icon-active {
  color: var(--color-accent-blue);
}

/* Progress bar */
.sc-progress {
  position: relative;
  height: 18px;
  background-color: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
}

.sc-progress-bar {
  height: 100%;
  background-color: var(--color-accent-blue);
  transition: width 0.3s ease;
  opacity: 0.3;
}

.sc-progress-bar--active {
  opacity: 0.5;
  background-image: linear-gradient(
    45deg,
    rgba(255, 255, 255, 0.1) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.1) 50%,
    rgba(255, 255, 255, 0.1) 75%,
    transparent 75%
  );
  background-size: 20px 20px;
  animation: progress-stripes 1s linear infinite;
}

.sc-progress-bar--complete {
  background-color: var(--color-success);
  opacity: 0.4;
}

.sc-progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 10px;
  font-weight: 600;
  color: var(--color-text-primary);
}

@keyframes progress-stripes {
  from { background-position: 20px 0; }
  to { background-position: 0 0; }
}

/* Log area */
.sc-log-area {
  flex: 1;
  overflow-y: auto;
  overflow-x: auto;
  background-color: #0a0e14;
  font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', ui-monospace, monospace;
  font-size: 11px;
  line-height: 1.6;
  padding: 4px 0;
}

.sc-log-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #4a5568;
  font-style: italic;
}

.sc-log-line {
  display: flex;
  gap: 8px;
  padding: 0 10px;
  white-space: nowrap;
}

.sc-log-line:hover {
  background-color: rgba(255, 255, 255, 0.03);
}

.sc-log-line--warn {
  background-color: rgba(255, 152, 0, 0.05);
}

.sc-log-line--error {
  background-color: rgba(244, 67, 54, 0.08);
}

.sc-log-time {
  color: #4a5568;
  flex-shrink: 0;
}

.sc-log-level {
  flex-shrink: 0;
  width: 40px;
  font-weight: 600;
}

.sc-level--info {
  color: #6b7280;
}

.sc-level--warn {
  color: #f59e0b;
}

.sc-level--error {
  color: #ef4444;
}

.sc-log-phase {
  color: #60a5fa;
  flex-shrink: 0;
}

.sc-log-message {
  color: #c9d1d9;
}
</style>
