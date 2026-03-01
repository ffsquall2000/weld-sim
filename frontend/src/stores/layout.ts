import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface PanelConfig {
  id: string
  title: string
  component: string
  visible: boolean
  position: 'left' | 'center' | 'right' | 'bottom'
  width?: number
  height?: number
  order: number
}

export interface LayoutState {
  panels: PanelConfig[]
  leftWidth: number
  rightWidth: number
  bottomHeight: number
  activeTab: Record<string, string>
}

const STORAGE_KEY = 'weldsim-layout'

function getDefaultPanels(): PanelConfig[] {
  return [
    {
      id: 'project-explorer',
      title: 'layout.panels.projectExplorer',
      component: 'ProjectExplorer',
      visible: true,
      position: 'left',
      order: 0,
    },
    {
      id: 'geometry-tree',
      title: 'layout.panels.geometryTree',
      component: 'GeometryTree',
      visible: true,
      position: 'left',
      order: 1,
    },
    {
      id: 'vtk-viewport',
      title: 'layout.panels.vtkViewport',
      component: 'VtkViewport',
      visible: true,
      position: 'center',
      order: 0,
    },
    {
      id: 'workflow-canvas',
      title: 'layout.panels.workflowCanvas',
      component: 'WorkflowCanvas',
      visible: true,
      position: 'center',
      order: 1,
    },
    {
      id: 'property-editor',
      title: 'layout.panels.propertyEditor',
      component: 'PropertyEditor',
      visible: true,
      position: 'right',
      order: 0,
    },
    {
      id: 'metrics-panel',
      title: 'layout.panels.metricsPanel',
      component: 'MetricsPanel',
      visible: true,
      position: 'right',
      order: 1,
    },
    {
      id: 'solver-console',
      title: 'layout.panels.solverConsole',
      component: 'SolverConsole',
      visible: true,
      position: 'bottom',
      order: 0,
    },
  ]
}

function getDefaultActiveTab(): Record<string, string> {
  return {
    left: 'project-explorer',
    center: 'vtk-viewport',
    right: 'property-editor',
    bottom: 'solver-console',
  }
}

export const useLayoutStore = defineStore('layout', () => {
  // State
  const panels = ref<PanelConfig[]>(getDefaultPanels())
  const leftWidth = ref(20)
  const rightWidth = ref(25)
  const bottomHeight = ref(25)
  const activeTab = ref<Record<string, string>>(getDefaultActiveTab())
  const leftCollapsed = ref(false)
  const rightCollapsed = ref(false)
  const bottomCollapsed = ref(false)

  // Getters
  const panelsByPosition = computed(() => {
    const grouped: Record<string, PanelConfig[]> = {
      left: [],
      center: [],
      right: [],
      bottom: [],
    }
    for (const panel of panels.value) {
      if (panel.visible && grouped[panel.position]) {
        grouped[panel.position]!.push(panel)
      }
    }
    // Sort each group by order
    for (const key of Object.keys(grouped)) {
      grouped[key]!.sort((a, b) => a.order - b.order)
    }
    return grouped
  })

  const hasVisiblePanels = computed(() => {
    return (position: string) => (panelsByPosition.value[position]?.length ?? 0) > 0
  })

  const getActivePanel = computed(() => {
    return (position: string): PanelConfig | undefined => {
      const positionPanels = panelsByPosition.value[position]
      if (!positionPanels || positionPanels.length === 0) return undefined
      const activeId = activeTab.value[position]
      return positionPanels.find((p) => p.id === activeId) ?? positionPanels[0]
    }
  })

  // Actions
  function togglePanel(panelId: string) {
    const panel = panels.value.find((p) => p.id === panelId)
    if (panel) {
      panel.visible = !panel.visible
      // If we just hid the active tab, switch to another visible panel
      if (!panel.visible && activeTab.value[panel.position] === panelId) {
        const remaining = panels.value.filter(
          (p) => p.position === panel.position && p.visible
        )
        if (remaining.length > 0) {
          activeTab.value[panel.position] = remaining[0]!.id
        }
      }
    }
  }

  function setActiveTab(position: string, panelId: string) {
    activeTab.value[position] = panelId
  }

  function updateLeftWidth(delta: number) {
    const newWidth = leftWidth.value + delta
    if (newWidth >= 10 && newWidth <= 40) {
      leftWidth.value = newWidth
    }
  }

  function updateRightWidth(delta: number) {
    const newWidth = rightWidth.value + delta
    if (newWidth >= 10 && newWidth <= 40) {
      rightWidth.value = newWidth
    }
  }

  function updateBottomHeight(delta: number) {
    const newHeight = bottomHeight.value + delta
    if (newHeight >= 10 && newHeight <= 50) {
      bottomHeight.value = newHeight
    }
  }

  function toggleCollapse(position: 'left' | 'right' | 'bottom') {
    if (position === 'left') leftCollapsed.value = !leftCollapsed.value
    else if (position === 'right') rightCollapsed.value = !rightCollapsed.value
    else if (position === 'bottom') bottomCollapsed.value = !bottomCollapsed.value
  }

  function resetLayout() {
    panels.value = getDefaultPanels()
    leftWidth.value = 20
    rightWidth.value = 25
    bottomHeight.value = 25
    activeTab.value = getDefaultActiveTab()
    leftCollapsed.value = false
    rightCollapsed.value = false
    bottomCollapsed.value = false
    localStorage.removeItem(STORAGE_KEY)
  }

  function saveLayout() {
    const state: LayoutState & {
      leftCollapsed: boolean
      rightCollapsed: boolean
      bottomCollapsed: boolean
    } = {
      panels: panels.value,
      leftWidth: leftWidth.value,
      rightWidth: rightWidth.value,
      bottomHeight: bottomHeight.value,
      activeTab: activeTab.value,
      leftCollapsed: leftCollapsed.value,
      rightCollapsed: rightCollapsed.value,
      bottomCollapsed: bottomCollapsed.value,
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  }

  function loadLayout() {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return
    try {
      const state = JSON.parse(raw) as LayoutState & {
        leftCollapsed?: boolean
        rightCollapsed?: boolean
        bottomCollapsed?: boolean
      }
      if (state.panels && Array.isArray(state.panels)) {
        panels.value = state.panels
      }
      if (typeof state.leftWidth === 'number') leftWidth.value = state.leftWidth
      if (typeof state.rightWidth === 'number') rightWidth.value = state.rightWidth
      if (typeof state.bottomHeight === 'number') bottomHeight.value = state.bottomHeight
      if (state.activeTab) activeTab.value = state.activeTab
      if (typeof state.leftCollapsed === 'boolean') leftCollapsed.value = state.leftCollapsed
      if (typeof state.rightCollapsed === 'boolean') rightCollapsed.value = state.rightCollapsed
      if (typeof state.bottomCollapsed === 'boolean') bottomCollapsed.value = state.bottomCollapsed
    } catch {
      // Corrupted layout data, ignore
    }
  }

  return {
    // State
    panels,
    leftWidth,
    rightWidth,
    bottomHeight,
    activeTab,
    leftCollapsed,
    rightCollapsed,
    bottomCollapsed,
    // Getters
    panelsByPosition,
    hasVisiblePanels,
    getActivePanel,
    // Actions
    togglePanel,
    setActiveTab,
    updateLeftWidth,
    updateRightWidth,
    updateBottomHeight,
    toggleCollapse,
    resetLayout,
    saveLayout,
    loadLayout,
  }
})
