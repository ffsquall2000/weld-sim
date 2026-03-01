<template>
  <header class="top-menu-bar">
    <!-- Logo / App name -->
    <div class="top-menu-bar__brand">
      <span class="brand-icon">&#9881;</span>
      <span class="brand-name">{{ $t('app.subtitle') }}</span>
    </div>

    <!-- Menu items -->
    <nav class="top-menu-bar__menus">
      <div
        v-for="menu in menus"
        :key="menu.id"
        class="menu-item"
        :class="{ 'menu-item--open': openMenu === menu.id }"
        @mouseenter="onMenuEnter(menu.id)"
        @mouseleave="onMenuLeave"
      >
        <button
          class="menu-item__trigger"
          @click="toggleMenu(menu.id)"
        >
          {{ $t(menu.label) }}
        </button>

        <!-- Dropdown -->
        <div v-if="openMenu === menu.id" class="menu-dropdown">
          <template v-for="item in menu.items" :key="item.id">
            <div
              v-if="item.type === 'separator'"
              class="menu-dropdown__separator"
            />
            <button
              v-else-if="item.type === 'toggle'"
              class="menu-dropdown__item"
              @click="handleAction(item)"
            >
              <span class="menu-dropdown__check">
                {{ item.checked?.() ? '\u2713' : '' }}
              </span>
              <span class="menu-dropdown__label">{{ $t(item.label) }}</span>
              <span v-if="item.shortcut" class="menu-dropdown__shortcut">
                {{ item.shortcut }}
              </span>
            </button>
            <button
              v-else
              class="menu-dropdown__item"
              :disabled="item.disabled?.()"
              @click="handleAction(item)"
            >
              <span class="menu-dropdown__check" />
              <span class="menu-dropdown__label">{{ $t(item.label) }}</span>
              <span v-if="item.shortcut" class="menu-dropdown__shortcut">
                {{ item.shortcut }}
              </span>
            </button>
          </template>
        </div>
      </div>
    </nav>

    <!-- Spacer -->
    <div class="top-menu-bar__spacer" />

    <!-- Right side actions -->
    <div class="top-menu-bar__actions">
      <button class="action-btn" @click="settingsStore.toggleTheme()">
        {{ settingsStore.theme === 'dark' ? '\u263E' : '\u2600' }}
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useLayoutStore } from '@/stores/layout'
import { useSettingsStore } from '@/stores/settings'

const router = useRouter()
const layoutStore = useLayoutStore()
const settingsStore = useSettingsStore()

const openMenu = ref<string | null>(null)
let closeTimeout: ReturnType<typeof setTimeout> | null = null

interface MenuItem {
  id: string
  label: string
  type?: 'action' | 'separator' | 'toggle'
  shortcut?: string
  action?: () => void
  checked?: () => boolean
  disabled?: () => boolean
}

interface MenuGroup {
  id: string
  label: string
  items: MenuItem[]
}

const menus: MenuGroup[] = [
  {
    id: 'file',
    label: 'layout.menu.file',
    items: [
      { id: 'new', label: 'layout.menu.newProject', shortcut: 'Ctrl+N', action: () => router.push('/') },
      { id: 'open', label: 'layout.menu.open', shortcut: 'Ctrl+O', action: () => router.push('/') },
      { id: 'sep1', label: '', type: 'separator' },
      { id: 'save', label: 'layout.menu.save', shortcut: 'Ctrl+S', action: () => layoutStore.saveLayout() },
      { id: 'sep2', label: '', type: 'separator' },
      { id: 'export', label: 'layout.menu.export' },
    ],
  },
  {
    id: 'view',
    label: 'layout.menu.view',
    items: [
      {
        id: 'toggle-project-explorer',
        label: 'layout.panels.projectExplorer',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'project-explorer')?.visible ?? false,
        action: () => layoutStore.togglePanel('project-explorer'),
      },
      {
        id: 'toggle-geometry-tree',
        label: 'layout.panels.geometryTree',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'geometry-tree')?.visible ?? false,
        action: () => layoutStore.togglePanel('geometry-tree'),
      },
      {
        id: 'toggle-vtk-viewport',
        label: 'layout.panels.vtkViewport',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'vtk-viewport')?.visible ?? false,
        action: () => layoutStore.togglePanel('vtk-viewport'),
      },
      {
        id: 'toggle-workflow-canvas',
        label: 'layout.panels.workflowCanvas',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'workflow-canvas')?.visible ?? false,
        action: () => layoutStore.togglePanel('workflow-canvas'),
      },
      {
        id: 'toggle-property-editor',
        label: 'layout.panels.propertyEditor',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'property-editor')?.visible ?? false,
        action: () => layoutStore.togglePanel('property-editor'),
      },
      {
        id: 'toggle-metrics-panel',
        label: 'layout.panels.metricsPanel',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'metrics-panel')?.visible ?? false,
        action: () => layoutStore.togglePanel('metrics-panel'),
      },
      {
        id: 'toggle-solver-console',
        label: 'layout.panels.solverConsole',
        type: 'toggle',
        checked: () => layoutStore.panels.find((p) => p.id === 'solver-console')?.visible ?? false,
        action: () => layoutStore.togglePanel('solver-console'),
      },
      { id: 'sep-view', label: '', type: 'separator' },
      {
        id: 'reset-layout',
        label: 'layout.menu.resetLayout',
        action: () => layoutStore.resetLayout(),
      },
    ],
  },
  {
    id: 'simulation',
    label: 'layout.menu.simulation',
    items: [
      { id: 'run', label: 'layout.menu.run', shortcut: 'F5', action: () => router.push('/workbench/calculate') },
      { id: 'stop', label: 'layout.menu.stop', shortcut: 'Shift+F5' },
    ],
  },
  {
    id: 'tools',
    label: 'layout.menu.tools',
    items: [
      { id: 'calculate', label: 'nav.calculate', action: () => router.push('/workbench/calculate') },
      { id: 'horn-design', label: 'nav.hornDesign', action: () => router.push('/workbench/horn-design') },
      { id: 'knurl-design', label: 'nav.knurlDesign', action: () => router.push('/workbench/knurl-design') },
      { id: 'acoustic', label: 'nav.acoustic', action: () => router.push('/workbench/acoustic') },
      { id: 'sep-tools', label: '', type: 'separator' },
      { id: 'geometry', label: 'nav.geometry', action: () => router.push('/workbench/geometry') },
      { id: 'fatigue', label: 'nav.fatigue', action: () => router.push('/workbench/fatigue') },
      { id: 'sep-tools-2', label: '', type: 'separator' },
      { id: 'settings', label: 'nav.settings', action: () => router.push('/settings') },
    ],
  },
]

function toggleMenu(menuId: string) {
  if (openMenu.value === menuId) {
    openMenu.value = null
  } else {
    openMenu.value = menuId
  }
}

function onMenuEnter(menuId: string) {
  if (closeTimeout) {
    clearTimeout(closeTimeout)
    closeTimeout = null
  }
  // Only switch menus if one is already open
  if (openMenu.value !== null) {
    openMenu.value = menuId
  }
}

function onMenuLeave() {
  closeTimeout = setTimeout(() => {
    openMenu.value = null
  }, 200)
}

function handleAction(item: MenuItem) {
  if (item.action) {
    item.action()
  }
  // Keep toggle menus open; close for regular actions
  if (item.type !== 'toggle') {
    openMenu.value = null
  }
}
</script>

<style scoped>
.top-menu-bar {
  display: flex;
  align-items: center;
  height: 32px;
  min-height: 32px;
  padding: 0 8px;
  background-color: var(--color-bg-secondary);
  border-bottom: 1px solid var(--color-border);
  user-select: none;
  gap: 0;
}

.top-menu-bar__brand {
  display: flex;
  align-items: center;
  gap: 6px;
  padding-right: 16px;
  margin-right: 4px;
  border-right: 1px solid var(--color-border);
}

.brand-icon {
  font-size: 14px;
  color: var(--color-accent-orange);
}

.brand-name {
  font-size: 12px;
  font-weight: 700;
  color: var(--color-accent-orange);
  white-space: nowrap;
  letter-spacing: 0.5px;
}

.top-menu-bar__menus {
  display: flex;
  align-items: center;
}

.top-menu-bar__spacer {
  flex: 1;
}

.top-menu-bar__actions {
  display: flex;
  align-items: center;
  gap: 4px;
}

.action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 24px;
  background: none;
  border: none;
  color: var(--color-text-secondary);
  font-size: 14px;
  cursor: pointer;
  border-radius: 3px;
  transition: background-color 0.15s, color 0.15s;
}

.action-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

/* Menu trigger */
.menu-item {
  position: relative;
}

.menu-item__trigger {
  display: flex;
  align-items: center;
  height: 32px;
  padding: 0 10px;
  background: none;
  border: none;
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.menu-item__trigger:hover,
.menu-item--open .menu-item__trigger {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

/* Dropdown */
.menu-dropdown {
  position: absolute;
  top: 100%;
  left: 0;
  min-width: 200px;
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  z-index: 1000;
  padding: 4px 0;
}

.menu-dropdown__item {
  display: flex;
  align-items: center;
  width: 100%;
  padding: 6px 12px;
  background: none;
  border: none;
  color: var(--color-text-primary);
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.1s;
  text-align: left;
  gap: 4px;
}

.menu-dropdown__item:hover {
  background-color: var(--color-accent-blue);
  color: #fff;
}

.menu-dropdown__item:disabled {
  color: var(--color-text-secondary);
  opacity: 0.5;
  cursor: not-allowed;
}

.menu-dropdown__item:disabled:hover {
  background: none;
  color: var(--color-text-secondary);
}

.menu-dropdown__check {
  width: 16px;
  text-align: center;
  flex-shrink: 0;
  font-size: 11px;
}

.menu-dropdown__label {
  flex: 1;
}

.menu-dropdown__shortcut {
  color: var(--color-text-secondary);
  font-size: 11px;
  margin-left: 16px;
  flex-shrink: 0;
}

.menu-dropdown__item:hover .menu-dropdown__shortcut {
  color: rgba(255, 255, 255, 0.7);
}

.menu-dropdown__separator {
  height: 1px;
  margin: 4px 8px;
  background-color: var(--color-border);
}
</style>
