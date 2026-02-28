import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/workbench', name: 'workbench', component: () => import('@/views/AnalysisWorkbench.vue') },
  { path: '/calculate', name: 'calculate', component: () => import('@/views/CalculateView.vue') },
  { path: '/geometry', name: 'geometry', component: () => import('@/views/GeometryView.vue') },
  { path: '/horn-design', name: 'horn-design', component: () => import('@/views/HornDesignView.vue') },
  { path: '/knurl-design', name: 'knurl-design', component: () => import('@/views/KnurlDesignView.vue') },
  { path: '/knurl-workbench', name: 'knurl-workbench', component: () => import('@/views/KnurlWorkbench.vue') },
  { path: '/acoustic', name: 'acoustic', component: () => import('@/views/AcousticView.vue') },
  { path: '/simulation', name: 'simulation', component: () => import('@/views/SimulationView.vue') },
  { path: '/fatigue', name: 'fatigue', component: () => import('@/views/FatigueView.vue') },
  { path: '/results/:id', name: 'results', component: () => import('@/views/ResultsView.vue') },
  { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue') },
  { path: '/reports/:id', name: 'reports', component: () => import('@/views/ReportsView.vue') },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
]

export const router = createRouter({ history: createWebHistory(), routes })
