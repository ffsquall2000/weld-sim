import { ref, onMounted, onUnmounted, type Ref } from 'vue'
import { useLayoutStore, type PanelConfig } from '@/stores/layout'

/**
 * Composable for resizable panel drag behavior.
 *
 * @param direction - 'horizontal' for height changes, 'vertical' for width changes
 * @param minSize - minimum size in pixels
 * @param maxSize - maximum size in pixels
 *
 * Returns a reactive `size` ref and a `startDrag` handler to attach to a divider mousedown.
 */
export function useResizable(
  direction: 'horizontal' | 'vertical',
  minSize: number = 100,
  maxSize: number = 800
) {
  const size = ref(300)
  const isDragging = ref(false)
  let startPos = 0
  let startSize = 0

  function startDrag(e: MouseEvent) {
    isDragging.value = true
    startPos = direction === 'vertical' ? e.clientX : e.clientY
    startSize = size.value
    document.addEventListener('mousemove', onDrag)
    document.addEventListener('mouseup', stopDrag)
    document.body.style.cursor = direction === 'vertical' ? 'col-resize' : 'row-resize'
    document.body.style.userSelect = 'none'
  }

  function onDrag(e: MouseEvent) {
    if (!isDragging.value) return
    const currentPos = direction === 'vertical' ? e.clientX : e.clientY
    const delta = currentPos - startPos
    let newSize = startSize + delta
    newSize = Math.max(minSize, Math.min(maxSize, newSize))
    size.value = newSize
  }

  function stopDrag() {
    isDragging.value = false
    document.removeEventListener('mousemove', onDrag)
    document.removeEventListener('mouseup', stopDrag)
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
  }

  onUnmounted(() => {
    document.removeEventListener('mousemove', onDrag)
    document.removeEventListener('mouseup', stopDrag)
  })

  return {
    size,
    isDragging,
    startDrag,
  }
}

/**
 * Composable for dynamic panel registration.
 *
 * Allows components to register/unregister themselves as panels
 * in the layout system at runtime.
 */
export function usePanelRegistry() {
  const layoutStore = useLayoutStore()
  const registeredPanels: Ref<string[]> = ref([])

  /**
   * Register a new panel in the layout system.
   * If a panel with the same id already exists, it will not be duplicated.
   */
  function registerPanel(config: PanelConfig) {
    const existing = layoutStore.panels.find((p) => p.id === config.id)
    if (!existing) {
      layoutStore.panels.push(config)
      registeredPanels.value.push(config.id)
    }
  }

  /**
   * Unregister a panel from the layout system.
   * Only removes panels that were registered by this instance.
   */
  function unregisterPanel(panelId: string) {
    const idx = layoutStore.panels.findIndex((p) => p.id === panelId)
    if (idx !== -1) {
      layoutStore.panels.splice(idx, 1)
    }
    const regIdx = registeredPanels.value.indexOf(panelId)
    if (regIdx !== -1) {
      registeredPanels.value.splice(regIdx, 1)
    }
  }

  /**
   * Clean up: unregister all panels that were registered by this composable instance.
   */
  function unregisterAll() {
    for (const id of [...registeredPanels.value]) {
      unregisterPanel(id)
    }
  }

  // Automatically clean up on unmount
  onUnmounted(() => {
    unregisterAll()
  })

  return {
    registeredPanels,
    registerPanel,
    unregisterPanel,
    unregisterAll,
  }
}
