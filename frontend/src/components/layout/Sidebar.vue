<template>
  <aside class="sidebar">
    <div class="sidebar-header">
      <span class="sidebar-logo">{{ $t('app.subtitle') }}</span>
    </div>

    <nav class="sidebar-nav">
      <router-link
        v-for="item in navItems"
        :key="item.name"
        :to="item.path"
        class="nav-item"
        :class="{ active: isActive(item.name) }"
      >
        <span class="nav-icon">{{ item.icon }}</span>
        <span class="nav-label">{{ $t(item.label) }}</span>
      </router-link>
    </nav>

    <div class="sidebar-footer">
      <button class="theme-toggle" @click="settingsStore.toggleTheme()">
        {{ settingsStore.theme === 'dark' ? 'Light' : 'Dark' }}
      </button>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import { useSettingsStore } from '@/stores/settings'

const route = useRoute()
const settingsStore = useSettingsStore()

const navItems = [
  { name: 'dashboard', path: '/', icon: '\u25A6', label: 'nav.dashboard' },
  { name: 'calculate', path: '/calculate', icon: '\u2699', label: 'nav.calculate' },
  { name: 'history', path: '/history', icon: '\u23F3', label: 'nav.history' },
  { name: 'settings', path: '/settings', icon: '\u2638', label: 'nav.settings' },
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
