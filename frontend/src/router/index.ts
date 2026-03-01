import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'projects', component: () => import('@/views/ProjectView.vue') },
  { path: '/workbench/:projectId', name: 'workbench', component: () => import('@/views/WorkbenchView.vue'), meta: { useShell: true } },
  { path: '/optimize/:studyId', name: 'optimization', component: () => import('@/views/OptimizationView.vue') },
  { path: '/compare/:comparisonId', name: 'comparison', component: () => import('@/views/ComparisonView.vue') },
  { path: '/dashboard', name: 'dashboard', component: () => import('@/views/DashboardView.vue') },
  { path: '/calculate', name: 'calculate', component: () => import('@/views/CalculateView.vue') },
  { path: '/geometry', name: 'geometry', component: () => import('@/views/GeometryView.vue') },
  { path: '/horn-design', name: 'horn-design', component: () => import('@/views/HornDesignView.vue') },
  { path: '/knurl-design', name: 'knurl-design', component: () => import('@/views/KnurlDesignView.vue') },
  { path: '/acoustic', name: 'acoustic', component: () => import('@/views/AcousticView.vue') },
  { path: '/fatigue', name: 'fatigue', component: () => import('@/views/FatigueView.vue') },
  { path: '/results/:id', name: 'results', component: () => import('@/views/ResultsView.vue') },
  { path: '/history', name: 'history', component: () => import('@/views/HistoryView.vue') },
  { path: '/reports/:id', name: 'reports', component: () => import('@/views/ReportsView.vue') },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
]

export const router = createRouter({ history: createWebHistory(), routes })
