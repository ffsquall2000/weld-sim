<template>
  <div class="node-palette" :class="{ collapsed }">
    <button class="palette-toggle" @click="collapsed = !collapsed" :title="collapsed ? 'Expand palette' : 'Collapse palette'">
      <span class="toggle-icon">{{ collapsed ? '\u25B6' : '\u25C0' }}</span>
      <span v-if="!collapsed" class="toggle-label">Nodes</span>
    </button>

    <div v-if="!collapsed" class="palette-body">
      <div v-for="category in categories" :key="category.name" class="palette-category">
        <div class="category-header" @click="category.open = !category.open">
          <span class="category-chevron">{{ category.open ? '\u25BE' : '\u25B8' }}</span>
          <span class="category-name">{{ category.name }}</span>
        </div>
        <div v-if="category.open" class="category-items">
          <div
            v-for="item in category.items"
            :key="item.type"
            class="palette-item"
            draggable="true"
            @dragstart="onDragStart($event, item)"
          >
            <span class="item-icon" :style="{ color: item.color }">{{ item.icon }}</span>
            <span class="item-label">{{ item.label }}</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import type { SimNodeType } from '@/composables/useWorkflowGraph'

interface PaletteItem {
  type: SimNodeType
  label: string
  icon: string
  color: string
}

interface Category {
  name: string
  open: boolean
  items: PaletteItem[]
}

const collapsed = ref(false)

const categories = reactive<Category[]>([
  {
    name: 'Input',
    open: true,
    items: [
      { type: 'geometry', label: 'Geometry', icon: '\u25B3', color: '#58a6ff' },
      { type: 'material', label: 'Material', icon: '\u25C6', color: '#f59e0b' },
    ],
  },
  {
    name: 'Processing',
    open: true,
    items: [
      { type: 'mesh', label: 'Mesh', icon: '\u25A6', color: '#2dd4bf' },
      { type: 'boundary_condition', label: 'Boundary Condition', icon: '\u2B21', color: '#f97316' },
    ],
  },
  {
    name: 'Solving',
    open: true,
    items: [
      { type: 'solver', label: 'Solver', icon: '\u2699', color: '#a78bfa' },
    ],
  },
  {
    name: 'Analysis',
    open: true,
    items: [
      { type: 'post_process', label: 'Post Process', icon: '\u25CE', color: '#4ade80' },
      { type: 'compare', label: 'Compare', icon: '\u21C6', color: '#818cf8' },
      { type: 'optimize', label: 'Optimize', icon: '\u27F3', color: '#fb7185' },
    ],
  },
])

function onDragStart(event: DragEvent, item: PaletteItem) {
  if (!event.dataTransfer) return
  event.dataTransfer.setData('application/workflow-node-type', item.type)
  event.dataTransfer.setData('application/workflow-node-label', item.label)
  event.dataTransfer.effectAllowed = 'move'
}
</script>

<style scoped>
.node-palette {
  background: rgba(22, 27, 34, 0.95);
  border: 1px solid #30363d;
  border-radius: 10px;
  backdrop-filter: blur(8px);
  min-width: 180px;
  max-height: calc(100vh - 120px);
  overflow-y: auto;
  z-index: 10;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  font-size: 13px;
  color: #e6edf3;
  transition: min-width 0.2s;
}

.node-palette.collapsed {
  min-width: 40px;
}

.node-palette::-webkit-scrollbar {
  width: 4px;
}
.node-palette::-webkit-scrollbar-thumb {
  background: #484f58;
  border-radius: 2px;
}

/* --- Toggle button --- */
.palette-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 10px 12px;
  border: none;
  background: transparent;
  color: #8b949e;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border-bottom: 1px solid #30363d;
  transition: color 0.15s;
}
.palette-toggle:hover {
  color: #e6edf3;
}

.toggle-icon {
  font-size: 10px;
}

/* --- Categories --- */
.palette-body {
  padding: 4px 0;
}

.palette-category {
  margin-bottom: 2px;
}

.category-header {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 6px 12px;
  cursor: pointer;
  color: #8b949e;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  user-select: none;
  transition: color 0.15s;
}
.category-header:hover {
  color: #e6edf3;
}

.category-chevron {
  font-size: 10px;
  width: 12px;
  text-align: center;
}

/* --- Palette items --- */
.category-items {
  padding: 0 6px 4px;
}

.palette-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 6px;
  cursor: grab;
  user-select: none;
  transition: background 0.15s;
}
.palette-item:hover {
  background: rgba(88, 166, 255, 0.08);
}
.palette-item:active {
  cursor: grabbing;
  background: rgba(88, 166, 255, 0.15);
}

.item-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
  flex-shrink: 0;
}

.item-label {
  font-size: 12px;
  white-space: nowrap;
}
</style>
