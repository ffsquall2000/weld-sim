<template>
  <footer class="status-bar">
    <!-- Solver status indicator -->
    <div class="status-bar__section">
      <span
        class="status-indicator"
        :class="`status-indicator--${solverStatus}`"
      />
      <span class="status-bar__label">
        {{ $t(`layout.status.solver.${solverStatus}`) }}
      </span>
    </div>

    <!-- Progress bar (when solver is running) -->
    <div v-if="solverStatus === 'running'" class="status-bar__progress">
      <div class="progress-track">
        <div
          class="progress-fill"
          :style="{ width: `${progress}%` }"
        />
      </div>
      <span class="status-bar__label">{{ progress }}%</span>
    </div>

    <!-- Spacer -->
    <div class="status-bar__spacer" />

    <!-- Project name -->
    <div class="status-bar__section">
      <span class="status-bar__label status-bar__label--muted">
        {{ $t('layout.status.project') }}:
      </span>
      <span class="status-bar__label">{{ projectName }}</span>
    </div>

    <!-- Divider -->
    <div class="status-bar__divider" />

    <!-- Backend name -->
    <div class="status-bar__section">
      <span class="status-bar__label status-bar__label--muted">
        {{ $t('layout.status.backend') }}:
      </span>
      <span class="status-bar__label">{{ backendName }}</span>
    </div>

    <!-- Divider -->
    <div class="status-bar__divider" />

    <!-- Connection status -->
    <div class="status-bar__section">
      <span
        class="status-indicator"
        :class="wsConnected ? 'status-indicator--connected' : 'status-indicator--disconnected'"
      />
      <span class="status-bar__label">
        {{ wsConnected ? $t('layout.status.connected') : $t('layout.status.disconnected') }}
      </span>
    </div>
  </footer>
</template>

<script setup lang="ts">
import { ref } from 'vue'

// These would typically come from Pinia stores.
// For now we use local reactive state as defaults.
const solverStatus = ref<'idle' | 'running' | 'completed' | 'failed'>('idle')
const progress = ref(0)
const projectName = ref('Untitled Project')
const backendName = ref('WeldSim Core v1.0')
const wsConnected = ref(false)

// Expose for external control
defineExpose({
  solverStatus,
  progress,
  projectName,
  backendName,
  wsConnected,
})
</script>

<style scoped>
.status-bar {
  display: flex;
  align-items: center;
  height: 24px;
  min-height: 24px;
  padding: 0 8px;
  background-color: var(--color-bg-secondary);
  border-top: 1px solid var(--color-border);
  font-size: 11px;
  gap: 6px;
  user-select: none;
}

.status-bar__section {
  display: flex;
  align-items: center;
  gap: 4px;
  flex-shrink: 0;
}

.status-bar__spacer {
  flex: 1;
}

.status-bar__label {
  color: var(--color-text-secondary);
  white-space: nowrap;
}

.status-bar__label--muted {
  opacity: 0.7;
}

.status-bar__divider {
  width: 1px;
  height: 12px;
  background-color: var(--color-border);
  flex-shrink: 0;
}

.status-bar__progress {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 120px;
}

.progress-track {
  flex: 1;
  height: 4px;
  background-color: var(--color-bg-card);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background-color: var(--color-accent-blue);
  border-radius: 2px;
  transition: width 0.3s ease;
}

/* Status indicator dot */
.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-indicator--idle {
  background-color: var(--color-text-secondary);
}

.status-indicator--running {
  background-color: var(--color-accent-blue);
  animation: pulse 1.5s ease-in-out infinite;
}

.status-indicator--completed {
  background-color: var(--color-success);
}

.status-indicator--failed {
  background-color: var(--color-danger);
}

.status-indicator--connected {
  background-color: var(--color-success);
}

.status-indicator--disconnected {
  background-color: var(--color-danger);
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
