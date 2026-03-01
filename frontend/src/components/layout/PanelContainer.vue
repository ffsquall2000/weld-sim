<template>
  <div class="panel-container" :class="{ 'panel-container--collapsed': isCollapsed }">
    <!-- Tab bar -->
    <div class="panel-tabs">
      <div class="panel-tabs__list">
        <button
          v-for="panel in visiblePanels"
          :key="panel.id"
          class="panel-tab"
          :class="{ 'panel-tab--active': activePanel?.id === panel.id }"
          @click="layoutStore.setActiveTab(position, panel.id)"
        >
          <span class="panel-tab__label">{{ $t(panel.title) }}</span>
        </button>
      </div>
      <div class="panel-tabs__actions">
        <button
          class="panel-collapse-btn"
          :title="isCollapsed ? $t('layout.expand') : $t('layout.collapse')"
          @click="layoutStore.toggleCollapse(position)"
        >
          <span class="panel-collapse-icon">{{ collapseIcon }}</span>
        </button>
      </div>
    </div>

    <!-- Panel content -->
    <div v-show="!isCollapsed" class="panel-content">
      <component
        :is="resolveComponent(activePanel?.component)"
        v-if="activePanel"
      />
      <div v-else class="panel-empty">
        {{ $t('layout.noPanel') }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, defineAsyncComponent, type Component } from 'vue'
import { useLayoutStore } from '@/stores/layout'

const props = defineProps<{
  position: 'left' | 'center' | 'right' | 'bottom'
}>()

const layoutStore = useLayoutStore()

const visiblePanels = computed(() => layoutStore.panelsByPosition[props.position] ?? [])

const activePanel = computed(() => layoutStore.getActivePanel(props.position))

const isCollapsed = computed(() => {
  if (props.position === 'left') return layoutStore.leftCollapsed
  if (props.position === 'right') return layoutStore.rightCollapsed
  if (props.position === 'bottom') return layoutStore.bottomCollapsed
  return false
})

const collapseIcon = computed(() => {
  if (isCollapsed.value) {
    if (props.position === 'left') return '\u25B6'
    if (props.position === 'right') return '\u25C0'
    if (props.position === 'bottom') return '\u25B2'
    return '\u25B6'
  }
  if (props.position === 'left') return '\u25C0'
  if (props.position === 'right') return '\u25B6'
  if (props.position === 'bottom') return '\u25BC'
  return '\u25C0'
})

// Lazy-loaded components for each panel type
const panelComponents: Record<string, Component> = {
  ProjectExplorer: defineAsyncComponent(() => import('@/components/panels/ProjectExplorer.vue')),
  GeometryTree: defineAsyncComponent(() => import('@/components/panels/GeometryTree.vue')),
  VtkViewport: defineAsyncComponent(() => import('@/components/panels/VtkViewportPanel.vue')),
  WorkflowCanvas: defineAsyncComponent(() => import('@/components/panels/WorkflowCanvasPanel.vue')),
  PropertyEditor: defineAsyncComponent(() => import('@/components/panels/PropertyEditor.vue')),
  MetricsPanel: defineAsyncComponent(() => import('@/components/panels/MetricsPanel.vue')),
  SolverConsole: defineAsyncComponent(() => import('@/components/panels/SolverConsole.vue')),
  MaterialSelector: defineAsyncComponent(() => import('@/components/panels/MaterialSelector.vue')),
}

function resolveComponent(name?: string): Component | undefined {
  if (!name) return undefined
  return panelComponents[name]
}
</script>

<style scoped>
.panel-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background-color: var(--color-bg-secondary);
}

.panel-container--collapsed {
  /* When collapsed, only show the tab bar */
}

.panel-tabs {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 30px;
  min-height: 30px;
  background-color: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
}

.panel-tabs__list {
  display: flex;
  overflow-x: auto;
  flex: 1;
  min-width: 0;
}

.panel-tabs__list::-webkit-scrollbar {
  display: none;
}

.panel-tabs__actions {
  display: flex;
  align-items: center;
  padding-right: 2px;
}

.panel-tab {
  display: flex;
  align-items: center;
  padding: 0 12px;
  height: 30px;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 11px;
  font-weight: 500;
  white-space: nowrap;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s, background-color 0.15s;
}

.panel-tab:hover {
  color: var(--color-text-primary);
  background-color: var(--color-bg-card);
}

.panel-tab--active {
  color: var(--color-text-primary);
  border-bottom-color: var(--color-accent-blue);
  background-color: var(--color-bg-secondary);
}

.panel-tab__label {
  line-height: 1;
}

.panel-collapse-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 8px;
  cursor: pointer;
  border-radius: 3px;
  transition: background-color 0.15s, color 0.15s;
}

.panel-collapse-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.panel-collapse-icon {
  line-height: 1;
}

.panel-content {
  flex: 1;
  overflow: auto;
  min-height: 0;
}

.panel-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-secondary);
  font-size: 12px;
}
</style>
