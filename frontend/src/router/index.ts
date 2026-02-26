import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/calculate', name: 'calculate', component: () => import('@/views/CalculateView.vue') },
  { path: '/results/:id', name: 'results', component: () => import('@/views/ResultsView.vue') },
  { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue') },
  { path: '/reports/:id', name: 'reports', component: () => import('@/views/ReportsView.vue') },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
]

export const router = createRouter({ history: createWebHistory(), routes })
