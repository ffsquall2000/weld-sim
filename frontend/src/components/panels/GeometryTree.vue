<template>
  <div class="geometry-tree">
    <!-- Assembly Root -->
    <div class="gt-root">
      <button class="gt-root-header" @click="rootExpanded = !rootExpanded">
        <span class="gt-expand-icon">{{ rootExpanded ? '&#9660;' : '&#9654;' }}</span>
        <span class="gt-root-icon">&#9635;</span>
        <span class="gt-root-label">{{ assemblyName }}</span>
      </button>
    </div>

    <!-- Body List -->
    <div v-if="rootExpanded" class="gt-bodies">
      <div
        v-for="(body, index) in geometryStore.assemblyBodies"
        :key="body.name"
        class="gt-body"
        :class="{ 'gt-body--selected': geometryStore.selectedBody === body.name }"
        draggable="true"
        @click="selectBody(body.name)"
        @dragstart="onDragStart(index, $event)"
        @dragover.prevent="onDragOver(index)"
        @drop="onDrop(index)"
        @dragend="onDragEnd"
      >
        <span class="gt-body-indent" />
        <span class="gt-body-icon" :class="bodyIconClass(body.bodyType)">
          {{ bodyIcon(body.bodyType) }}
        </span>
        <div class="gt-body-info">
          <span class="gt-body-name">{{ body.name }}</span>
          <span class="gt-body-material">{{ body.material }}</span>
        </div>
        <button
          class="gt-visibility-btn"
          :class="{ 'gt-visibility-btn--hidden': !body.visible }"
          :title="body.visible ? t('geometryTree.hide') : t('geometryTree.show')"
          @click.stop="toggleVisibility(body.name)"
        >
          {{ body.visible ? '&#128065;' : '&#128064;' }}
        </button>
      </div>
    </div>

    <!-- Geometry Version Info -->
    <div v-if="geometryStore.currentGeometry" class="gt-version-info">
      <div class="gt-version-header">{{ t('geometryTree.versionInfo') }}</div>
      <div class="gt-version-row">
        <span class="gt-version-label">{{ t('geometryTree.version') }}</span>
        <span class="gt-version-value">v{{ geometryStore.currentGeometry.version_number }}</span>
      </div>
      <div class="gt-version-row">
        <span class="gt-version-label">{{ t('geometryTree.source') }}</span>
        <span class="gt-version-value">{{ geometryStore.currentGeometry.source_type }}</span>
      </div>
      <div v-if="geometryStore.currentGeometry.label" class="gt-version-row">
        <span class="gt-version-label">{{ t('geometryTree.label') }}</span>
        <span class="gt-version-value">{{ geometryStore.currentGeometry.label }}</span>
      </div>
      <div class="gt-version-row">
        <span class="gt-version-label">{{ t('geometryTree.mesh') }}</span>
        <span
          class="gt-version-value"
          :class="geometryStore.currentGeometry.mesh_file_path ? 'gt-value--ok' : 'gt-value--missing'"
        >
          {{ geometryStore.currentGeometry.mesh_file_path ? t('geometryTree.meshReady') : t('geometryTree.meshNotGenerated') }}
        </span>
      </div>
    </div>

    <!-- Empty State -->
    <div v-if="geometryStore.assemblyBodies.length === 0 && rootExpanded" class="gt-empty">
      {{ t('geometryTree.noComponents') }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useGeometryStore } from '@/stores/geometry'

const { t } = useI18n()
const geometryStore = useGeometryStore()

const rootExpanded = ref(true)
const assemblyName = ref('Welding Assembly')
const dragIndex = ref<number | null>(null)
const dropTargetIndex = ref<number | null>(null)

function selectBody(name: string) {
  geometryStore.selectBody(name)
}

function toggleVisibility(name: string) {
  geometryStore.toggleBodyVisibility(name)
}

function bodyIcon(type: string): string {
  switch (type) {
    case 'horn': return '\u2B22'
    case 'anvil': return '\u25A0'
    case 'workpiece_upper': return '\u25AD'
    case 'workpiece_lower': return '\u25AD'
    default: return '\u25CF'
  }
}

function bodyIconClass(type: string) {
  return {
    'gt-icon--horn': type === 'horn',
    'gt-icon--anvil': type === 'anvil',
    'gt-icon--workpiece': type === 'workpiece_upper' || type === 'workpiece_lower',
    'gt-icon--other': type === 'other',
  }
}

// Drag and drop
function onDragStart(index: number, event: DragEvent) {
  dragIndex.value = index
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', String(index))
  }
}

function onDragOver(index: number) {
  dropTargetIndex.value = index
}

function onDrop(index: number) {
  if (dragIndex.value !== null && dragIndex.value !== index) {
    geometryStore.reorderBodies(dragIndex.value, index)
  }
  dragIndex.value = null
  dropTargetIndex.value = null
}

function onDragEnd() {
  dragIndex.value = null
  dropTargetIndex.value = null
}
</script>

<style scoped>
.geometry-tree {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 12px;
}

.gt-root {
  border-bottom: 1px solid var(--color-border);
}

.gt-root-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 8px 10px;
  border: none;
  background: none;
  color: var(--color-text-primary);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.15s;
}

.gt-root-header:hover {
  background-color: var(--color-bg-card);
}

.gt-expand-icon {
  font-size: 8px;
  width: 10px;
  color: var(--color-text-secondary);
}

.gt-root-icon {
  font-size: 14px;
  color: var(--color-accent-orange);
}

.gt-root-label {
  flex: 1;
  text-align: left;
}

/* Body Items */
.gt-bodies {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.gt-body {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px 5px 20px;
  cursor: pointer;
  transition: background-color 0.1s;
  border-bottom: 1px solid rgba(48, 54, 61, 0.3);
}

.gt-body:hover {
  background-color: var(--color-bg-card);
}

.gt-body--selected {
  background-color: var(--color-bg-card);
  border-left: 2px solid var(--color-accent-blue);
  padding-left: 18px;
}

.gt-body-indent {
  width: 8px;
  height: 1px;
  background-color: var(--color-border);
  flex-shrink: 0;
}

.gt-body-icon {
  font-size: 12px;
  width: 16px;
  text-align: center;
  flex-shrink: 0;
}

.gt-icon--horn {
  color: var(--color-accent-orange);
}

.gt-icon--anvil {
  color: var(--color-text-secondary);
}

.gt-icon--workpiece {
  color: var(--color-accent-blue);
}

.gt-icon--other {
  color: var(--color-text-secondary);
}

.gt-body-info {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}

.gt-body-name {
  font-size: 12px;
  color: var(--color-text-primary);
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.gt-body-material {
  font-size: 10px;
  color: var(--color-text-secondary);
}

.gt-visibility-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border: none;
  background: none;
  font-size: 13px;
  cursor: pointer;
  border-radius: 3px;
  flex-shrink: 0;
  transition: opacity 0.15s;
}

.gt-visibility-btn--hidden {
  opacity: 0.3;
}

/* Version Info */
.gt-version-info {
  border-top: 1px solid var(--color-border);
  padding: 8px 12px;
  background-color: var(--color-bg-primary);
}

.gt-version-header {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 6px;
}

.gt-version-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 2px 0;
}

.gt-version-label {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.gt-version-value {
  font-size: 11px;
  color: var(--color-text-primary);
  font-family: ui-monospace, monospace;
}

.gt-value--ok {
  color: var(--color-success);
}

.gt-value--missing {
  color: var(--color-text-secondary);
  font-style: italic;
}

/* Empty */
.gt-empty {
  padding: 20px;
  text-align: center;
  color: var(--color-text-secondary);
  font-style: italic;
}
</style>
