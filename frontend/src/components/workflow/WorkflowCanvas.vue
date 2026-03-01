<template>
  <div
    class="workflow-canvas"
    @drop="onDrop"
    @dragover.prevent
    @dragenter.prevent
  >
    <VueFlow
      v-model:nodes="nodes"
      v-model:edges="edges"
      :node-types="nodeTypes"
      :connection-mode="ConnectionMode.Loose"
      :snap-to-grid="true"
      :snap-grid="[15, 15]"
      :default-edge-options="defaultEdgeOptions"
      @node-click="onNodeClick"
      @connect="onConnect"
      @edges-change="onEdgesChange"
      @pane-click="onPaneClick"
      fit-view-on-init
      class="vue-flow-dark"
    >
      <Background :gap="20" :size="1" pattern-color="#2a2f38" />
      <Controls position="bottom-left" />
      <MiniMap
        :node-color="minimapNodeColor"
        :mask-color="'rgba(13, 17, 23, 0.7)'"
        position="bottom-right"
      />
    </VueFlow>

    <NodePalette class="palette-overlay" />
    <WorkflowToolbar class="toolbar-overlay" />
  </div>
</template>

<script setup lang="ts">
import { markRaw, computed } from 'vue'
import { VueFlow, ConnectionMode, useVueFlow } from '@vue-flow/core'
import type { Node, Edge, Connection, EdgeChange } from '@vue-flow/core'
import { Background } from '@vue-flow/background'
import { Controls } from '@vue-flow/controls'
import { MiniMap } from '@vue-flow/minimap'
import { storeToRefs } from 'pinia'

import SimNode from './SimNode.vue'
import NodePalette from './NodePalette.vue'
import WorkflowToolbar from './WorkflowToolbar.vue'
import { useWorkflowStore } from '@/stores/workflow'
import { isValidConnection as checkConnection } from '@/composables/useWorkflowGraph'
import type { SimNodeData, SimNodeType } from '@/composables/useWorkflowGraph'

const store = useWorkflowStore()
const { nodes, edges } = storeToRefs(store)

const { project } = useVueFlow()

const nodeTypes = {
  simNode: markRaw(SimNode),
}

const defaultEdgeOptions = {
  type: 'smoothstep',
  animated: false,
  style: { stroke: '#484f58', strokeWidth: 2 },
}

// Color map for minimap nodes
const typeColorMap: Record<string, string> = {
  geometry: '#58a6ff',
  mesh: '#2dd4bf',
  material: '#f59e0b',
  boundary_condition: '#f97316',
  solver: '#a78bfa',
  post_process: '#4ade80',
  compare: '#818cf8',
  optimize: '#fb7185',
}

function minimapNodeColor(node: Node): string {
  const data = node.data as SimNodeData | undefined
  if (data?.type) {
    return typeColorMap[data.type] || '#484f58'
  }
  return '#484f58'
}

function onNodeClick(_event: MouseEvent, node: Node) {
  store.setSelectedNode(node.id)
}

function onPaneClick() {
  store.setSelectedNode(null)
}

function onConnect(connection: Connection) {
  if (!connection.source || !connection.target) return

  // Find source and target node types for validation
  const sourceNode = nodes.value.find((n) => n.id === connection.source)
  const targetNode = nodes.value.find((n) => n.id === connection.target)
  if (!sourceNode || !targetNode) return

  const sourceType = (sourceNode.data as SimNodeData).type
  const targetType = (targetNode.data as SimNodeData).type

  if (!checkConnection(sourceType, targetType)) {
    // Invalid connection - could add a notification here
    return
  }

  store.connectNodes(connection.source, connection.target)
}

function onEdgesChange(changes: EdgeChange[]) {
  for (const change of changes) {
    if (change.type === 'remove') {
      store.disconnectNodes(change.id)
    }
  }
}

function onDrop(event: DragEvent) {
  if (!event.dataTransfer) return

  const nodeType = event.dataTransfer.getData('application/workflow-node-type') as SimNodeType
  const nodeLabel = event.dataTransfer.getData('application/workflow-node-label')

  if (!nodeType || !nodeLabel) return

  // Convert screen coordinates to flow coordinates
  const position = project({
    x: event.clientX,
    y: event.clientY,
  })

  store.addNode(nodeType, nodeLabel, { x: position.x, y: position.y })
}
</script>

<style>
/* Global vue-flow styles - must not be scoped */
@import '@vue-flow/core/dist/style.css';
@import '@vue-flow/core/dist/theme-default.css';
@import '@vue-flow/controls/dist/style.css';
@import '@vue-flow/minimap/dist/style.css';
</style>

<style scoped>
.workflow-canvas {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 600px;
  background: #0d1117;
}

.palette-overlay {
  position: absolute;
  top: 12px;
  left: 12px;
}

.toolbar-overlay {
  position: absolute;
  top: 12px;
  right: 12px;
}

/* --- Dark theme overrides for vue-flow --- */
.vue-flow-dark :deep(.vue-flow__pane) {
  background: #0d1117;
}

.vue-flow-dark :deep(.vue-flow__edge-path) {
  stroke: #484f58;
  stroke-width: 2;
}

.vue-flow-dark :deep(.vue-flow__edge.selected .vue-flow__edge-path) {
  stroke: #58a6ff;
}

.vue-flow-dark :deep(.vue-flow__edge:hover .vue-flow__edge-path) {
  stroke: #6e7681;
}

.vue-flow-dark :deep(.vue-flow__connection-path) {
  stroke: #58a6ff;
  stroke-width: 2;
  stroke-dasharray: 5;
}

.vue-flow-dark :deep(.vue-flow__controls) {
  background: rgba(22, 27, 34, 0.95);
  border: 1px solid #30363d;
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  overflow: hidden;
}

.vue-flow-dark :deep(.vue-flow__controls-button) {
  background: transparent;
  border: none;
  border-bottom: 1px solid #30363d;
  color: #8b949e;
  fill: #8b949e;
  width: 32px;
  height: 32px;
}

.vue-flow-dark :deep(.vue-flow__controls-button:hover) {
  background: rgba(88, 166, 255, 0.1);
  color: #e6edf3;
  fill: #e6edf3;
}

.vue-flow-dark :deep(.vue-flow__controls-button svg) {
  fill: currentColor;
}

.vue-flow-dark :deep(.vue-flow__minimap) {
  background: rgba(22, 27, 34, 0.95);
  border: 1px solid #30363d;
  border-radius: 8px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

.vue-flow-dark :deep(.vue-flow__minimap-mask) {
  fill: rgba(13, 17, 23, 0.7);
}

/* Selection highlight */
.vue-flow-dark :deep(.vue-flow__node.selected .sim-node) {
  box-shadow: 0 0 0 2px #58a6ff, 0 4px 16px rgba(88, 166, 255, 0.25);
}
</style>
