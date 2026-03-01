<template>
  <div class="app-shell">
    <!-- Top Menu Bar -->
    <TopMenuBar />

    <!-- Main content area -->
    <div class="app-shell__body">
      <!-- Left Panel -->
      <div
        v-if="hasLeft"
        class="app-shell__panel app-shell__panel--left"
        :style="leftStyle"
      >
        <PanelContainer position="left" />
      </div>

      <!-- Left-Center Divider -->
      <ResizeDivider
        v-if="hasLeft && !layoutStore.leftCollapsed"
        direction="vertical"
        @resize="onResizeLeft"
      />

      <!-- Center Panel -->
      <div class="app-shell__panel app-shell__panel--center">
        <!-- Center has two regions: main viewport and bottom panel -->
        <div class="app-shell__center-main">
          <PanelContainer position="center" />
        </div>

        <!-- Center-Bottom Divider -->
        <ResizeDivider
          v-if="hasBottom && !layoutStore.bottomCollapsed"
          direction="horizontal"
          @resize="onResizeBottom"
        />

        <!-- Bottom Panel -->
        <div
          v-if="hasBottom"
          class="app-shell__panel app-shell__panel--bottom"
          :style="bottomStyle"
        >
          <PanelContainer position="bottom" />
        </div>
      </div>

      <!-- Center-Right Divider -->
      <ResizeDivider
        v-if="hasRight && !layoutStore.rightCollapsed"
        direction="vertical"
        @resize="onResizeRight"
      />

      <!-- Right Panel -->
      <div
        v-if="hasRight"
        class="app-shell__panel app-shell__panel--right"
        :style="rightStyle"
      >
        <PanelContainer position="right" />
      </div>
    </div>

    <!-- Status Bar -->
    <StatusBar />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted } from 'vue'
import { useLayoutStore } from '@/stores/layout'
import TopMenuBar from './TopMenuBar.vue'
import StatusBar from './StatusBar.vue'
import PanelContainer from './PanelContainer.vue'
import ResizeDivider from './ResizeDivider.vue'

const layoutStore = useLayoutStore()

// Computed: whether each position has visible panels
const hasLeft = computed(() => layoutStore.hasVisiblePanels('left'))
const hasRight = computed(() => layoutStore.hasVisiblePanels('right'))
const hasBottom = computed(() => layoutStore.hasVisiblePanels('bottom'))

// Styles for panel sizing
const leftStyle = computed(() => {
  if (layoutStore.leftCollapsed) {
    return { width: '30px', minWidth: '30px' }
  }
  return { width: `${layoutStore.leftWidth}%` }
})

const rightStyle = computed(() => {
  if (layoutStore.rightCollapsed) {
    return { width: '30px', minWidth: '30px' }
  }
  return { width: `${layoutStore.rightWidth}%` }
})

const bottomStyle = computed(() => {
  if (layoutStore.bottomCollapsed) {
    return { height: '30px', minHeight: '30px' }
  }
  return { height: `${layoutStore.bottomHeight}%` }
})

// Resize handlers: convert pixel delta to percentage delta
function onResizeLeft(delta: number) {
  const bodyWidth = document.querySelector('.app-shell__body')?.clientWidth ?? window.innerWidth
  const percentDelta = (delta / bodyWidth) * 100
  layoutStore.updateLeftWidth(percentDelta)
}

function onResizeRight(delta: number) {
  const bodyWidth = document.querySelector('.app-shell__body')?.clientWidth ?? window.innerWidth
  // Moving right divider to the left = smaller right panel (negative delta)
  const percentDelta = -(delta / bodyWidth) * 100
  layoutStore.updateRightWidth(percentDelta)
}

function onResizeBottom(delta: number) {
  const bodyHeight = document.querySelector('.app-shell__center-main')?.parentElement?.clientHeight ?? window.innerHeight
  // Moving down = smaller main, larger bottom (positive delta = bottom grows)
  const percentDelta = -(delta / bodyHeight) * 100
  layoutStore.updateBottomHeight(percentDelta)
}

// Load persisted layout on mount
onMounted(() => {
  layoutStore.loadLayout()
})

// Save layout before unload
function handleBeforeUnload() {
  layoutStore.saveLayout()
}

onMounted(() => {
  window.addEventListener('beforeunload', handleBeforeUnload)
})

onUnmounted(() => {
  window.removeEventListener('beforeunload', handleBeforeUnload)
})
</script>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background-color: var(--color-bg-primary);
}

.app-shell__body {
  display: flex;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.app-shell__panel {
  overflow: hidden;
  flex-shrink: 0;
}

.app-shell__panel--left {
  border-right: 1px solid var(--color-border);
}

.app-shell__panel--right {
  border-left: 1px solid var(--color-border);
}

.app-shell__panel--center {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  min-height: 0;
}

.app-shell__center-main {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}

.app-shell__panel--bottom {
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}
</style>
