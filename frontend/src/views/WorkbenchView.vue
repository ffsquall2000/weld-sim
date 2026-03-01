<template>
  <div class="workbench-view">
    <!-- Loading -->
    <div v-if="loading" class="wb-loading">
      <span class="wb-spinner" />
      <span>{{ t('common.loading') }}</span>
    </div>

    <!-- Error -->
    <div v-else-if="error" class="wb-error">
      <span>{{ error }}</span>
    </div>

    <!-- Workbench Layout -->
    <div v-else class="wb-layout">
      <!-- Left Panel -->
      <div
        class="wb-panel wb-panel--left"
        :style="{ width: leftCollapsed ? '32px' : leftWidth + '%' }"
      >
        <PanelContainer position="left" />
      </div>

      <!-- Left Resize Divider -->
      <ResizeDivider
        v-if="!leftCollapsed"
        direction="vertical"
        @resize="onLeftResize"
      />

      <!-- Center Panel -->
      <div class="wb-panel wb-panel--center">
        <!-- Center content area: Viewer + Workflow tabs -->
        <div class="wb-center-top">
          <PanelContainer position="center" />
        </div>

        <!-- Bottom Resize Divider -->
        <ResizeDivider
          v-if="!bottomCollapsed"
          direction="horizontal"
          @resize="onBottomResize"
        />

        <!-- Bottom Panel -->
        <div
          class="wb-panel wb-panel--bottom"
          :style="{ height: bottomCollapsed ? '32px' : bottomHeight + '%' }"
        >
          <PanelContainer position="bottom" />
        </div>
      </div>

      <!-- Right Resize Divider -->
      <ResizeDivider
        v-if="!rightCollapsed"
        direction="vertical"
        @resize="onRightResize"
      />

      <!-- Right Panel -->
      <div
        class="wb-panel wb-panel--right"
        :style="{ width: rightCollapsed ? '32px' : rightWidth + '%' }"
      >
        <PanelContainer position="right" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useProjectStore } from '@/stores/project'
import { useGeometryStore } from '@/stores/geometry'
import { useSimulationStore } from '@/stores/simulation'
import { useLayoutStore } from '@/stores/layout'
import PanelContainer from '@/components/layout/PanelContainer.vue'
import ResizeDivider from '@/components/layout/ResizeDivider.vue'

const { t } = useI18n()
const route = useRoute()
const projectStore = useProjectStore()
const geometryStore = useGeometryStore()
const simulationStore = useSimulationStore()
const layoutStore = useLayoutStore()

const loading = ref(false)
const error = ref<string | null>(null)

const leftWidth = computed(() => layoutStore.leftWidth)
const rightWidth = computed(() => layoutStore.rightWidth)
const bottomHeight = computed(() => layoutStore.bottomHeight)
const leftCollapsed = computed(() => layoutStore.leftCollapsed)
const rightCollapsed = computed(() => layoutStore.rightCollapsed)
const bottomCollapsed = computed(() => layoutStore.bottomCollapsed)

function onLeftResize(delta: number) {
  layoutStore.updateLeftWidth(delta * 0.1)
}

function onRightResize(delta: number) {
  layoutStore.updateRightWidth(-delta * 0.1)
}

function onBottomResize(delta: number) {
  layoutStore.updateBottomHeight(-delta * 0.1)
}

async function loadProjectData(projectId: string) {
  loading.value = true
  error.value = null
  try {
    await projectStore.fetchProject(projectId)
    if (!projectStore.currentProject) {
      error.value = 'Project not found'
      return
    }
    // Load project-related data in parallel
    await Promise.all([
      geometryStore.fetchGeometries(projectId),
      simulationStore.fetchSimulations(projectId),
    ])
  } catch (err: unknown) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

// Load layout
onMounted(() => {
  layoutStore.loadLayout()
})

// Watch route params for project ID changes
watch(
  () => route.params.projectId as string,
  (newId) => {
    if (newId) {
      loadProjectData(newId)
    }
  },
  { immediate: true }
)
</script>

<style scoped>
.workbench-view {
  height: calc(100vh - 0px);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.wb-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  height: 100%;
  color: var(--color-text-secondary);
  font-size: 14px;
}

.wb-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-accent-orange);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.wb-error {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-danger);
  font-size: 14px;
}

/* Layout */
.wb-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
  height: 100%;
}

.wb-panel {
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.wb-panel--left {
  min-width: 32px;
  border-right: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}

.wb-panel--right {
  min-width: 32px;
  border-left: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}

.wb-panel--center {
  flex: 1;
  min-width: 200px;
  display: flex;
  flex-direction: column;
}

.wb-center-top {
  flex: 1;
  overflow: hidden;
  min-height: 100px;
}

.wb-panel--bottom {
  min-height: 32px;
  border-top: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}
</style>
