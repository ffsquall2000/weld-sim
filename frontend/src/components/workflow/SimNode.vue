<template>
  <div :class="['sim-node', `status-${data.status}`, `type-${data.type}`]">
    <div class="node-header">
      <span class="node-icon">{{ icon }}</span>
      <span class="node-title">{{ data.label }}</span>
    </div>
    <div class="node-body">
      <div class="node-status-indicator">
        <span class="status-dot" />
        <span class="status-text">{{ statusText }}</span>
      </div>
    </div>
    <Handle type="target" :position="Position.Left" class="handle-target" />
    <Handle type="source" :position="Position.Right" class="handle-source" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Handle, Position } from '@vue-flow/core'
import type { SimNodeData } from '@/composables/useWorkflowGraph'

const props = defineProps<{
  data: SimNodeData
}>()

const iconMap: Record<string, string> = {
  geometry: '\u25B3',       // triangle
  mesh: '\u25A6',           // square with lines
  material: '\u25C6',       // diamond
  boundary_condition: '\u2B21', // hexagon
  solver: '\u2699',         // gear
  post_process: '\u25CE',   // bullseye
  compare: '\u21C6',        // left-right arrows
  optimize: '\u27F3',       // clockwise arrow
}

const statusTextMap: Record<string, string> = {
  idle: 'Idle',
  configured: 'Configured',
  running: 'Running...',
  completed: 'Completed',
  error: 'Error',
}

const icon = computed(() => iconMap[props.data.type] || '\u25CF')
const statusText = computed(() => statusTextMap[props.data.status] || props.data.status)
</script>

<style scoped>
/* --- Base node --- */
.sim-node {
  min-width: 160px;
  border-radius: 8px;
  border: 2px solid var(--node-accent, #555);
  background: #1c2028;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-size: 13px;
  color: #e6edf3;
  transition: border-color 0.2s, box-shadow 0.2s;
  cursor: grab;
}

.sim-node:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
}

/* --- Header --- */
.node-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px 6px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  font-weight: 600;
}

.node-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
  flex-shrink: 0;
}

.node-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* --- Body --- */
.node-body {
  padding: 6px 12px 8px;
}

.node-status-indicator {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #8b949e;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--status-color, #484f58);
  flex-shrink: 0;
}

.status-text {
  text-transform: capitalize;
}

/* --- Handles --- */
.handle-target,
.handle-source {
  width: 10px !important;
  height: 10px !important;
  background: #484f58 !important;
  border: 2px solid #1c2028 !important;
  border-radius: 50% !important;
  transition: background 0.15s;
}

.handle-target:hover,
.handle-source:hover {
  background: var(--node-accent, #58a6ff) !important;
}

/* --- Type accent colors --- */
.type-geometry {
  --node-accent: #58a6ff;
}
.type-mesh {
  --node-accent: #2dd4bf;
}
.type-material {
  --node-accent: #f59e0b;
}
.type-boundary_condition {
  --node-accent: #f97316;
}
.type-solver {
  --node-accent: #a78bfa;
}
.type-post_process {
  --node-accent: #4ade80;
}
.type-compare {
  --node-accent: #818cf8;
}
.type-optimize {
  --node-accent: #fb7185;
}

/* --- Status colors --- */
.status-idle {
  --status-color: #484f58;
}
.status-configured {
  --status-color: #58a6ff;
  border-color: var(--node-accent);
}
.status-running {
  --status-color: #f59e0b;
  border-color: #f59e0b;
  box-shadow: 0 0 12px rgba(245, 158, 11, 0.3);
}
.status-running .status-dot {
  animation: pulse-dot 1s ease-in-out infinite;
}
.status-completed {
  --status-color: #4ade80;
  border-color: #4ade80;
}
.status-error {
  --status-color: #f87171;
  border-color: #f87171;
  box-shadow: 0 0 12px rgba(248, 113, 113, 0.3);
}

@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}
</style>
