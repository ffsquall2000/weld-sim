<template>
  <div class="solver-console">
    <div class="console-toolbar">
      <span class="console-title">{{ $t('layout.panels.solverConsole') }}</span>
      <div class="console-actions">
        <button class="console-btn" @click="clearLogs">
          {{ $t('layout.console.clear') }}
        </button>
      </div>
    </div>
    <div class="console-output" ref="outputEl">
      <div v-if="logs.length === 0" class="console-empty">
        {{ $t('layout.placeholders.solverConsole') }}
      </div>
      <div
        v-for="(log, i) in logs"
        :key="i"
        class="console-line"
        :class="`console-line--${log.level}`"
      >
        <span class="console-line__time">{{ log.time }}</span>
        <span class="console-line__level">[{{ log.level.toUpperCase() }}]</span>
        <span class="console-line__msg">{{ log.message }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

interface LogEntry {
  time: string
  level: 'info' | 'warn' | 'error' | 'debug'
  message: string
}

const logs = ref<LogEntry[]>([
  { time: new Date().toLocaleTimeString(), level: 'info', message: 'Solver console initialized. Ready for simulation.' },
])

const outputEl = ref<HTMLElement>()

function clearLogs() {
  logs.value = []
}

defineExpose({ outputEl })
</script>

<style scoped>
.solver-console {
  display: flex;
  flex-direction: column;
  height: 100%;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  background-color: var(--color-bg-primary);
}

.console-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 8px;
  background-color: var(--color-bg-card);
  border-bottom: 1px solid var(--color-border);
}

.console-title {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-weight: 600;
}

.console-actions {
  display: flex;
  gap: 4px;
}

.console-btn {
  padding: 2px 8px;
  font-size: 10px;
  color: var(--color-text-secondary);
  background: none;
  border: 1px solid var(--color-border);
  border-radius: 3px;
  cursor: pointer;
  transition: color 0.15s, border-color 0.15s;
}

.console-btn:hover {
  color: var(--color-text-primary);
  border-color: var(--color-accent-blue);
}

.console-output {
  flex: 1;
  overflow-y: auto;
  padding: 4px 8px;
  font-size: 11px;
  line-height: 1.6;
}

.console-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-secondary);
  font-size: 11px;
}

.console-line {
  display: flex;
  gap: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}

.console-line__time {
  color: var(--color-text-secondary);
  flex-shrink: 0;
}

.console-line__level {
  flex-shrink: 0;
  font-weight: 600;
}

.console-line--info .console-line__level {
  color: var(--color-accent-blue);
}

.console-line--warn .console-line__level {
  color: var(--color-warning);
}

.console-line--error .console-line__level {
  color: var(--color-danger);
}

.console-line--debug .console-line__level {
  color: var(--color-text-secondary);
}

.console-line__msg {
  color: var(--color-text-primary);
}
</style>
