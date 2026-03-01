<template>
  <div class="workflow-toolbar">
    <!-- Run -->
    <button
      class="toolbar-btn run"
      :disabled="executionStatus === 'running'"
      @click="onRun"
      title="Run Workflow"
    >
      <span class="btn-icon">\u25B6</span>
    </button>

    <!-- Stop -->
    <button
      class="toolbar-btn stop"
      :disabled="executionStatus !== 'running'"
      @click="onStop"
      title="Stop Execution"
    >
      <span class="btn-icon">\u25A0</span>
    </button>

    <!-- Validate -->
    <button
      class="toolbar-btn validate"
      :disabled="executionStatus === 'running'"
      @click="onValidate"
      title="Validate Workflow"
    >
      <span class="btn-icon">\u2713</span>
    </button>

    <div class="toolbar-divider" />

    <!-- Template dropdown -->
    <div class="template-dropdown" ref="dropdownRef">
      <button
        class="toolbar-btn template"
        @click="templateOpen = !templateOpen"
        title="Load Template"
      >
        <span class="btn-icon">\u2630</span>
      </button>
      <div v-if="templateOpen" class="dropdown-menu">
        <button
          v-for="(name, key) in templateNames"
          :key="key"
          class="dropdown-item"
          @click="onLoadTemplate(key)"
        >
          {{ name }}
        </button>
      </div>
    </div>

    <!-- Clear -->
    <button
      class="toolbar-btn clear"
      :disabled="executionStatus === 'running'"
      @click="onClear"
      title="Clear Workflow"
    >
      <span class="btn-icon">\u2715</span>
    </button>

    <!-- Validation errors -->
    <div v-if="validationErrors.length > 0" class="validation-panel">
      <div
        v-for="(err, i) in validationErrors"
        :key="i"
        class="validation-error"
      >
        <span class="error-icon">\u26A0</span>
        {{ err.message }}
      </div>
    </div>

    <!-- Status -->
    <div v-if="executionStatus === 'running'" class="status-badge running-badge">
      Running...
    </div>
    <div v-else-if="executionStatus === 'completed'" class="status-badge completed-badge">
      Completed
    </div>
    <div v-else-if="executionStatus === 'error'" class="status-badge error-badge">
      Error
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useWorkflowStore, templateNames } from '@/stores/workflow'
import { storeToRefs } from 'pinia'

const store = useWorkflowStore()
const { executionStatus, validationErrors } = storeToRefs(store)

const templateOpen = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)

function onRun() {
  store.executeWorkflow()
}

function onStop() {
  store.stopExecution()
}

function onValidate() {
  store.validateWorkflow()
}

function onLoadTemplate(key: string) {
  store.loadTemplate(key)
  templateOpen.value = false
}

function onClear() {
  store.clearWorkflow()
}

// Close dropdown on outside click
function handleClickOutside(e: MouseEvent) {
  if (dropdownRef.value && !dropdownRef.value.contains(e.target as HTMLElement)) {
    templateOpen.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

<style scoped>
.workflow-toolbar {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  background: rgba(22, 27, 34, 0.95);
  border: 1px solid #30363d;
  border-radius: 10px;
  padding: 8px 10px;
  backdrop-filter: blur(8px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  z-index: 10;
  flex-wrap: wrap;
  max-width: 300px;
}

/* --- Buttons --- */
.toolbar-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border: 1px solid #30363d;
  border-radius: 7px;
  background: #21262d;
  color: #e6edf3;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s, color 0.15s;
  flex-shrink: 0;
}
.toolbar-btn:hover:not(:disabled) {
  background: #30363d;
  border-color: #484f58;
}
.toolbar-btn:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 14px;
  line-height: 1;
}

.toolbar-btn.run .btn-icon {
  color: #4ade80;
}
.toolbar-btn.stop .btn-icon {
  color: #f87171;
}
.toolbar-btn.validate .btn-icon {
  color: #58a6ff;
  font-weight: bold;
}
.toolbar-btn.clear .btn-icon {
  color: #8b949e;
}

/* --- Divider --- */
.toolbar-divider {
  width: 1px;
  height: 24px;
  background: #30363d;
  margin: 5px 2px;
  flex-shrink: 0;
}

/* --- Template dropdown --- */
.template-dropdown {
  position: relative;
}

.dropdown-menu {
  position: absolute;
  top: 40px;
  right: 0;
  min-width: 200px;
  background: #21262d;
  border: 1px solid #30363d;
  border-radius: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
  overflow: hidden;
  z-index: 20;
}

.dropdown-item {
  display: block;
  width: 100%;
  padding: 10px 14px;
  border: none;
  background: transparent;
  color: #e6edf3;
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: background 0.12s;
}
.dropdown-item:hover {
  background: rgba(88, 166, 255, 0.1);
}

/* --- Validation errors --- */
.validation-panel {
  width: 100%;
  margin-top: 6px;
  max-height: 120px;
  overflow-y: auto;
}

.validation-error {
  display: flex;
  align-items: flex-start;
  gap: 6px;
  padding: 5px 8px;
  font-size: 11px;
  color: #f87171;
  line-height: 1.4;
}

.error-icon {
  flex-shrink: 0;
  font-size: 12px;
}

/* --- Status badges --- */
.status-badge {
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
  white-space: nowrap;
}

.running-badge {
  background: rgba(245, 158, 11, 0.15);
  color: #f59e0b;
  animation: pulse-badge 1.5s ease-in-out infinite;
}

.completed-badge {
  background: rgba(74, 222, 128, 0.15);
  color: #4ade80;
}

.error-badge {
  background: rgba(248, 113, 113, 0.15);
  color: #f87171;
}

@keyframes pulse-badge {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
</style>
