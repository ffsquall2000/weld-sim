<template>
  <div class="vtk-viewport relative w-full h-full min-h-[300px] overflow-hidden rounded-lg"
       @keydown="onKeyDown"
       tabindex="0">
    <!-- VTK.js render container -->
    <div ref="vtkContainer" class="absolute inset-0" />

    <!-- Floating toolbar -->
    <ViewportToolbar class="absolute top-2 left-2 z-10" />

    <!-- Contour / color bar legend -->
    <ContourLegend v-if="hasScalarField" class="absolute right-2 top-2 z-10" />

    <!-- Slice plane controls -->
    <SlicePlane v-if="store.slicePlane.enabled" class="absolute bottom-2 left-2 z-10" />

    <!-- Loading overlay -->
    <div v-if="store.isLoading"
         class="absolute inset-0 flex items-center justify-center z-20"
         style="background-color: rgba(13, 17, 23, 0.75);">
      <div class="flex flex-col items-center gap-3">
        <div class="w-10 h-10 border-3 border-t-transparent rounded-full animate-spin"
             style="border-color: var(--color-accent-blue); border-top-color: transparent;" />
        <span style="color: var(--color-text-secondary);">{{ t('common.loading') }}</span>
      </div>
    </div>

    <!-- Empty state placeholder -->
    <div v-if="!store.hasMesh && !store.isLoading"
         class="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
      <div class="flex flex-col items-center gap-2 text-center px-4">
        <svg class="w-16 h-16 opacity-30" style="color: var(--color-text-secondary);"
             viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
          <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
          <line x1="12" y1="22.08" x2="12" y2="12" />
        </svg>
        <span class="text-sm opacity-50" style="color: var(--color-text-secondary);">
          {{ t('geometry.viewerPlaceholder') }}
        </span>
      </div>
    </div>

    <!-- WebGL not supported warning -->
    <div v-if="!viewer.webGLSupported.value"
         class="absolute inset-0 flex items-center justify-center z-20"
         style="background-color: var(--color-bg-primary);">
      <div class="flex flex-col items-center gap-3 text-center px-6">
        <svg class="w-12 h-12" style="color: var(--color-danger);"
             viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
        <span style="color: var(--color-text-primary);">WebGL is not available</span>
        <span class="text-sm" style="color: var(--color-text-secondary);">
          3D rendering requires WebGL support in your browser.
        </span>
      </div>
    </div>

    <!-- Keyboard shortcut hints (shown briefly on focus) -->
    <div v-if="showShortcutHints"
         class="absolute bottom-2 right-2 z-10 text-xs px-2 py-1 rounded opacity-70"
         style="background-color: var(--color-bg-card); color: var(--color-text-secondary); border: 1px solid var(--color-border);">
      F: Fit | W: Wireframe | R: Reset
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { useI18n } from 'vue-i18n'
import { useViewer3DStore } from '@/stores/viewer3d'
import { useVtkViewer } from '@/composables/useVtkViewer'
import ViewportToolbar from './ViewportToolbar.vue'
import ContourLegend from './ContourLegend.vue'
import SlicePlane from './SlicePlane.vue'

const { t } = useI18n()
const store = useViewer3DStore()

const vtkContainer = ref<HTMLElement | null>(null)
const showShortcutHints = ref(false)
let shortcutHintTimer: ReturnType<typeof setTimeout> | null = null

// Initialize VTK viewer composable
const viewer = useVtkViewer(vtkContainer)

// Whether a scalar field is active
const hasScalarField = computed(() => store.scalarField !== null && store.hasMesh)

// Watch store meshData for changes and load into VTK
watch(() => store.meshData, (newMesh) => {
  if (newMesh) {
    viewer.loadMeshFromData(newMesh)
  }
}, { deep: false })

// Watch display mode
watch(() => store.displayMode, (mode) => {
  viewer.setDisplayMode(mode)
})

// Watch scalar field
watch(() => store.scalarField, (fieldName) => {
  const range = store.colorMapRange ?? undefined
  viewer.setScalarField(fieldName, range)
})

// Watch color map range
watch(() => store.colorMapRange, (range) => {
  if (store.scalarField) {
    viewer.setScalarField(store.scalarField, range ?? undefined)
  }
})

// Watch color map preset
watch(() => store.colorMap, (preset) => {
  viewer.setColorMap(preset)
})

// Watch edge visibility
watch(() => store.showEdges, (visible) => {
  viewer.setEdgeVisibility(visible)
})

// Watch axes visibility
watch(() => store.showAxes, (visible) => {
  viewer.setAxesVisibility(visible)
})

// Keyboard shortcuts
function onKeyDown(event: KeyboardEvent) {
  // Ignore if modifier keys are pressed or if user is typing in an input
  if (event.ctrlKey || event.metaKey || event.altKey) return
  const target = event.target as HTMLElement
  if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT') return

  switch (event.key.toLowerCase()) {
    case 'f':
      event.preventDefault()
      viewer.resetCamera()
      break
    case 'w':
      event.preventDefault()
      if (store.displayMode === 'solid') {
        store.setDisplayMode('wireframe')
      } else if (store.displayMode === 'wireframe') {
        store.setDisplayMode('solid_wireframe')
      } else {
        store.setDisplayMode('solid')
      }
      break
    case 'r':
      event.preventDefault()
      viewer.resetCamera()
      break
  }
}

// Show shortcut hints briefly on focus
function onFocus() {
  showShortcutHints.value = true
  if (shortcutHintTimer) clearTimeout(shortcutHintTimer)
  shortcutHintTimer = setTimeout(() => {
    showShortcutHints.value = false
  }, 3000)
}

onMounted(() => {
  const container = vtkContainer.value?.parentElement
  if (container) {
    container.addEventListener('focus', onFocus)
  }
})

onBeforeUnmount(() => {
  if (shortcutHintTimer) clearTimeout(shortcutHintTimer)
  const container = vtkContainer.value?.parentElement
  if (container) {
    container.removeEventListener('focus', onFocus)
  }
})

// Expose viewer methods for parent components
defineExpose({
  resetCamera: () => viewer.resetCamera(),
  takeScreenshot: () => viewer.takeScreenshot(),
  render: () => viewer.render(),
})
</script>

<style scoped>
.vtk-viewport {
  background-color: #1a1a2e;
  outline: none;
}

.vtk-viewport:focus {
  box-shadow: 0 0 0 2px var(--color-accent-blue);
}
</style>
