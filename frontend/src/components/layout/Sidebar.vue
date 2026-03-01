<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <span class="sidebar-logo">{{ $t('app.subtitle') }}</span>
    </div>

    <nav class="sidebar-nav">
      <!-- Projects / Home -->
      <router-link
        to="/"
        class="nav-item"
        :class="{ active: isActive('projects') }"
      >
        <span class="nav-icon">&#9638;</span>
        <span class="nav-label">{{ $t('nav.projects') }}</span>
      </router-link>

      <!-- Workbench Tools Section -->
      <div class="nav-section">
        <span class="nav-section-label">{{ $t('nav.workbench') }}</span>
      </div>

      <router-link
        v-for="item in toolItems"
        :key="item.name"
        :to="item.path"
        class="nav-item nav-item--tool"
        :class="{ active: isActive(item.name) }"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-label">{{ $t(item.label) }}</span>
      </router-link>

      <!-- Spacer -->
      <div class="nav-spacer" />

      <!-- Settings -->
      <router-link
        to="/settings"
        class="nav-item"
        :class="{ active: isActive('settings') }"
      >
        <span class="nav-icon">&#9881;</span>
        <span class="nav-label">{{ $t('nav.settings') }}</span>
      </router-link>
    </nav>

    <div class="sidebar-footer">
      <button class="theme-toggle" @click="settingsStore.toggleTheme()">
        {{ settingsStore.theme === 'dark' ? $t('settings.switchToLight') : $t('settings.switchToDark') }}
      </button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'

const route = useRoute()
const settingsStore = useSettingsStore()

const toolItems = [
  { name: 'workbench-calculate', path: '/workbench/calculate', icon: '\u2699', label: 'nav.calculate' },
  { name: 'workbench-geometry', path: '/workbench/geometry', icon: '\u25B3', label: 'nav.geometry' },
  { name: 'workbench-horn-design', path: '/workbench/horn-design', icon: '\u2B22', label: 'nav.hornDesign' },
  { name: 'workbench-knurl-design', path: '/workbench/knurl-design', icon: '\u2592', label: 'nav.knurlDesign' },
  { name: 'workbench-acoustic', path: '/workbench/acoustic', icon: '\u223F', label: 'nav.acoustic' },
  { name: 'workbench-fatigue', path: '/workbench/fatigue', icon: '\u26A0', label: 'nav.fatigue' },
]

function isActive(name: string): boolean {
  return route.name === name
}
</script>

<style scoped>
.sidebar {
  width: 200px;
  min-width: 200px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: var(--color-bg-secondary);
  border-right: 1px solid var(--color-border);
}

.sidebar-header {
  padding: 20px 16px;
  border-bottom: 1px solid var(--color-border);
}

.sidebar-logo {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-accent-orange);
}

.sidebar-nav {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 8px 0;
  overflow-y: auto;
}

.nav-section {
  padding: 16px 16px 4px;
}

.nav-section-label {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--color-text-secondary);
  opacity: 0.7;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  text-decoration: none;
  color: var(--color-text-secondary);
  font-size: 14px;
  border-left: 3px solid transparent;
  transition: background-color 0.15s, color 0.15s, border-color 0.15s;
}

.nav-item--tool {
  padding-left: 24px;
  font-size: 13px;
}

.nav-item:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.nav-item.active {
  border-left-color: var(--color-accent-orange);
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.nav-icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
}

.nav-label {
  white-space: nowrap;
}

.nav-spacer {
  flex: 1;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid var(--color-border);
}

.theme-toggle {
  width: 100%;
  padding: 6px 12px;
  background-color: var(--color-bg-card);
  color: var(--color-text-secondary);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.theme-toggle:hover {
  color: var(--color-text-primary);
  border-color: var(--color-accent-orange);
}
</style>
