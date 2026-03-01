<template>
  <div class="project-explorer">
    <!-- Project Header -->
    <div class="pe-header" v-if="projectStore.currentProject">
      <div class="pe-project-info">
        <span class="pe-project-icon">&#9641;</span>
        <div class="pe-project-text">
          <div class="pe-project-name">{{ projectStore.currentProject.name }}</div>
          <div class="pe-project-type">{{ projectStore.currentProject.application_type }}</div>
        </div>
      </div>
    </div>
    <div class="pe-header pe-header--empty" v-else>
      <span class="pe-empty-text">{{ t('projectExplorer.noProject') }}</span>
    </div>

    <!-- Tree Sections -->
    <div class="pe-tree">
      <!-- Geometries Section -->
      <div class="pe-section">
        <button
          class="pe-section-header"
          @click="toggleSection('geometries')"
          @contextmenu.prevent="showContextMenu($event, 'geometries')"
        >
          <span class="pe-expand-icon">{{ expandedSections.geometries ? '&#9660;' : '&#9654;' }}</span>
          <span class="pe-section-icon">&#9651;</span>
          <span class="pe-section-title">{{ t('projectExplorer.geometries') }}</span>
          <span class="pe-section-count">{{ geometryStore.geometries.length }}</span>
        </button>
        <div v-if="expandedSections.geometries" class="pe-section-content">
          <div
            v-for="geo in geometryStore.geometries"
            :key="geo.id"
            class="pe-tree-item"
            :class="{ 'pe-tree-item--selected': geometryStore.currentGeometry?.id === geo.id }"
            @click="selectGeometry(geo)"
            @contextmenu.prevent="showContextMenu($event, 'geometry', geo)"
          >
            <span class="pe-item-icon">&#9632;</span>
            <span class="pe-item-label">{{ geo.label || `v${geo.version_number}` }}</span>
            <span class="pe-item-badge pe-badge--source">{{ geo.source_type }}</span>
          </div>
          <div v-if="geometryStore.geometries.length === 0" class="pe-empty-item">
            {{ t('projectExplorer.noGeometries') }}
          </div>
        </div>
      </div>

      <!-- Simulations Section -->
      <div class="pe-section">
        <button
          class="pe-section-header"
          @click="toggleSection('simulations')"
          @contextmenu.prevent="showContextMenu($event, 'simulations')"
        >
          <span class="pe-expand-icon">{{ expandedSections.simulations ? '&#9660;' : '&#9654;' }}</span>
          <span class="pe-section-icon">&#9881;</span>
          <span class="pe-section-title">{{ t('projectExplorer.simulations') }}</span>
          <span class="pe-section-count">{{ simulationStore.simulations.length }}</span>
        </button>
        <div v-if="expandedSections.simulations" class="pe-section-content">
          <div
            v-for="sim in simulationStore.simulations"
            :key="sim.id"
            class="pe-tree-item"
            :class="{ 'pe-tree-item--selected': simulationStore.currentSimulation?.id === sim.id }"
            @click="selectSimulation(sim)"
            @contextmenu.prevent="showContextMenu($event, 'simulation', sim)"
          >
            <span class="pe-item-icon">&#9881;</span>
            <span class="pe-item-label">{{ sim.name }}</span>
            <span class="pe-item-badge pe-badge--type">{{ sim.analysis_type }}</span>
          </div>
          <div v-if="simulationStore.simulations.length === 0" class="pe-empty-item">
            {{ t('projectExplorer.noSimulations') }}
          </div>
        </div>
      </div>

      <!-- Runs Section -->
      <div class="pe-section">
        <button
          class="pe-section-header"
          @click="toggleSection('runs')"
        >
          <span class="pe-expand-icon">{{ expandedSections.runs ? '&#9660;' : '&#9654;' }}</span>
          <span class="pe-section-icon">&#9654;</span>
          <span class="pe-section-title">{{ t('projectExplorer.runs') }}</span>
          <span class="pe-section-count">{{ simulationStore.runs.length }}</span>
        </button>
        <div v-if="expandedSections.runs" class="pe-section-content">
          <div
            v-for="run in simulationStore.runs"
            :key="run.id"
            class="pe-tree-item"
            :class="{ 'pe-tree-item--selected': simulationStore.activeRun?.id === run.id }"
            @click="selectRun(run)"
          >
            <span class="pe-item-icon">&#9656;</span>
            <span class="pe-item-label">{{ run.id.slice(0, 8) }}</span>
            <span
              class="pe-item-badge"
              :class="statusBadgeClass(run.status)"
            >{{ run.status }}</span>
          </div>
          <div v-if="simulationStore.runs.length === 0" class="pe-empty-item">
            {{ t('projectExplorer.noRuns') }}
          </div>
        </div>
      </div>

      <!-- Optimizations Section -->
      <div class="pe-section">
        <button
          class="pe-section-header"
          @click="toggleSection('optimizations')"
          @contextmenu.prevent="showContextMenu($event, 'optimizations')"
        >
          <span class="pe-expand-icon">{{ expandedSections.optimizations ? '&#9660;' : '&#9654;' }}</span>
          <span class="pe-section-icon">&#10004;</span>
          <span class="pe-section-title">{{ t('projectExplorer.optimizations') }}</span>
          <span class="pe-section-count">{{ optimizationStore.studies.length }}</span>
        </button>
        <div v-if="expandedSections.optimizations" class="pe-section-content">
          <div
            v-for="study in optimizationStore.studies"
            :key="study.id"
            class="pe-tree-item"
            @click="selectOptimization(study)"
          >
            <span class="pe-item-icon">&#10038;</span>
            <span class="pe-item-label">{{ study.name }}</span>
            <span
              class="pe-item-badge"
              :class="statusBadgeClass(study.status)"
            >{{ study.status }}</span>
          </div>
          <div v-if="optimizationStore.studies.length === 0" class="pe-empty-item">
            {{ t('projectExplorer.noOptimizations') }}
          </div>
        </div>
      </div>
    </div>

    <!-- Context Menu -->
    <Teleport to="body">
      <div
        v-if="contextMenu.visible"
        class="pe-context-menu"
        :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }"
        @click.stop
      >
        <button class="pe-context-item" @click="handleContextAction('new')">
          {{ t('projectExplorer.contextNew') }}
        </button>
        <button
          v-if="contextMenu.item"
          class="pe-context-item"
          @click="handleContextAction('rename')"
        >
          {{ t('projectExplorer.contextRename') }}
        </button>
        <button
          v-if="contextMenu.item"
          class="pe-context-item pe-context-item--danger"
          @click="handleContextAction('delete')"
        >
          {{ t('projectExplorer.contextDelete') }}
        </button>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { reactive, onMounted, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useProjectStore } from '@/stores/project'
import { useSimulationStore, type Run } from '@/stores/simulation'
import { useGeometryStore, type GeometryVersion } from '@/stores/geometry'
import { useOptimizationStore, type OptimizationStudy } from '@/stores/optimization'

const { t } = useI18n()
const router = useRouter()
const projectStore = useProjectStore()
const simulationStore = useSimulationStore()
const geometryStore = useGeometryStore()
const optimizationStore = useOptimizationStore()

const expandedSections = reactive({
  geometries: true,
  simulations: true,
  runs: false,
  optimizations: false,
})

const contextMenu = reactive({
  visible: false,
  x: 0,
  y: 0,
  type: '' as string,
  item: null as any,
})

function toggleSection(section: keyof typeof expandedSections) {
  expandedSections[section] = !expandedSections[section]
}

function selectGeometry(geo: GeometryVersion) {
  geometryStore.setCurrentGeometry(geo)
}

function selectSimulation(sim: any) {
  simulationStore.setCurrentSimulation(sim)
}

function selectRun(run: Run) {
  simulationStore.setActiveRun(run)
}

function selectOptimization(study: OptimizationStudy) {
  optimizationStore.setCurrentStudy(study)
  router.push({ name: 'optimization', params: { studyId: study.id } })
}

function statusBadgeClass(status: string) {
  switch (status) {
    case 'completed': return 'pe-badge--success'
    case 'running': return 'pe-badge--running'
    case 'queued': return 'pe-badge--queued'
    case 'failed': return 'pe-badge--error'
    case 'cancelled': return 'pe-badge--warn'
    case 'paused': return 'pe-badge--warn'
    default: return 'pe-badge--default'
  }
}

function showContextMenu(event: MouseEvent, type: string, item?: any) {
  contextMenu.visible = true
  contextMenu.x = event.clientX
  contextMenu.y = event.clientY
  contextMenu.type = type
  contextMenu.item = item ?? null
}

function handleContextAction(action: string) {
  contextMenu.visible = false
  // Placeholder for context menu actions
  // In a real implementation, these would call store actions
  if (action === 'new') {
    // Create new item based on contextMenu.type
  } else if (action === 'rename') {
    // Rename the item
  } else if (action === 'delete') {
    // Delete the item
  }
}

function closeContextMenu() {
  contextMenu.visible = false
}

onMounted(() => {
  document.addEventListener('click', closeContextMenu)
})

onUnmounted(() => {
  document.removeEventListener('click', closeContextMenu)
})
</script>

<style scoped>
.project-explorer {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 12px;
}

.pe-header {
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-bg-primary);
}

.pe-header--empty {
  display: flex;
  align-items: center;
  justify-content: center;
}

.pe-empty-text {
  color: var(--color-text-secondary);
  font-style: italic;
}

.pe-project-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pe-project-icon {
  font-size: 18px;
  color: var(--color-accent-orange);
}

.pe-project-text {
  min-width: 0;
}

.pe-project-name {
  font-weight: 600;
  color: var(--color-text-primary);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pe-project-type {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.pe-tree {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
}

.pe-section {
  border-bottom: 1px solid var(--color-border);
}

.pe-section-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 10px;
  border: none;
  background: none;
  color: var(--color-text-primary);
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  cursor: pointer;
  transition: background-color 0.15s;
}

.pe-section-header:hover {
  background-color: var(--color-bg-card);
}

.pe-expand-icon {
  font-size: 8px;
  width: 10px;
  color: var(--color-text-secondary);
}

.pe-section-icon {
  font-size: 12px;
  color: var(--color-accent-blue);
}

.pe-section-title {
  flex: 1;
  text-align: left;
}

.pe-section-count {
  font-size: 10px;
  color: var(--color-text-secondary);
  background-color: var(--color-bg-card);
  padding: 1px 6px;
  border-radius: 8px;
  font-weight: 500;
}

.pe-section-content {
  padding: 2px 0;
}

.pe-tree-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px 4px 28px;
  cursor: pointer;
  color: var(--color-text-secondary);
  transition: background-color 0.1s, color 0.1s;
}

.pe-tree-item:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.pe-tree-item--selected {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  border-left: 2px solid var(--color-accent-blue);
  padding-left: 26px;
}

.pe-item-icon {
  font-size: 10px;
  width: 14px;
  text-align: center;
  flex-shrink: 0;
}

.pe-item-label {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pe-item-badge {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 500;
  text-transform: uppercase;
  flex-shrink: 0;
}

.pe-badge--source {
  background-color: var(--color-bg-primary);
  color: var(--color-text-secondary);
}

.pe-badge--type {
  background-color: var(--color-bg-primary);
  color: var(--color-accent-blue);
}

.pe-badge--success {
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
}

.pe-badge--running {
  background-color: rgba(88, 166, 255, 0.15);
  color: var(--color-accent-blue);
}

.pe-badge--queued {
  background-color: rgba(139, 148, 158, 0.15);
  color: var(--color-text-secondary);
}

.pe-badge--error {
  background-color: rgba(244, 67, 54, 0.15);
  color: var(--color-danger);
}

.pe-badge--warn {
  background-color: rgba(255, 152, 0, 0.15);
  color: var(--color-warning);
}

.pe-badge--default {
  background-color: var(--color-bg-primary);
  color: var(--color-text-secondary);
}

.pe-empty-item {
  padding: 8px 28px;
  color: var(--color-text-secondary);
  font-style: italic;
  font-size: 11px;
}

/* Context Menu */
.pe-context-menu {
  position: fixed;
  z-index: 9999;
  min-width: 140px;
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}

.pe-context-item {
  display: block;
  width: 100%;
  padding: 6px 12px;
  border: none;
  background: none;
  color: var(--color-text-primary);
  font-size: 12px;
  text-align: left;
  cursor: pointer;
  border-radius: 4px;
  transition: background-color 0.1s;
}

.pe-context-item:hover {
  background-color: var(--color-bg-secondary);
}

.pe-context-item--danger {
  color: var(--color-danger);
}

.pe-context-item--danger:hover {
  background-color: rgba(244, 67, 54, 0.1);
}
</style>
