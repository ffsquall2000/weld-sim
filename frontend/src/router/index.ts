import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  // Main routes
  { path: '/', name: 'projects', component: () => import('@/views/ProjectView.vue') },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue') },
  { path: '/optimize/:studyId', name: 'optimization', component: () => import('@/views/OptimizationView.vue') },
  { path: '/compare/:comparisonId', name: 'comparison', component: () => import('@/views/ComparisonView.vue') },

  // Workbench tool sub-routes (use AppShell layout) -- static paths first
  { path: '/workbench/calculate', name: 'workbench-calculate', component: () => import('@/views/CalculateView.vue'), meta: { useShell: true } },
  { path: '/workbench/geometry', name: 'workbench-geometry', component: () => import('@/views/GeometryView.vue'), meta: { useShell: true } },
  { path: '/workbench/horn-design', name: 'workbench-horn-design', component: () => import('@/views/HornDesignView.vue'), meta: { useShell: true } },
  { path: '/workbench/knurl-design', name: 'workbench-knurl-design', component: () => import('@/views/KnurlDesignView.vue'), meta: { useShell: true } },
  { path: '/workbench/acoustic', name: 'workbench-acoustic', component: () => import('@/views/AcousticView.vue'), meta: { useShell: true } },
  { path: '/workbench/fatigue', name: 'workbench-fatigue', component: () => import('@/views/FatigueView.vue'), meta: { useShell: true } },
  { path: '/workbench/results/:id', name: 'workbench-results', component: () => import('@/views/ResultsView.vue'), meta: { useShell: true } },
  { path: '/workbench/reports/:id', name: 'workbench-reports', component: () => import('@/views/ReportsView.vue'), meta: { useShell: true } },

  // Workbench project view (dynamic :projectId param) -- after static paths
  { path: '/workbench/:projectId', name: 'workbench', component: () => import('@/views/WorkbenchView.vue'), meta: { useShell: true } },

  // Legacy redirects for backward compatibility
  { path: '/dashboard', redirect: '/' },
  { path: '/history', redirect: '/' },
  { path: '/calculate', redirect: '/workbench/calculate' },
  { path: '/geometry', redirect: '/workbench/geometry' },
  { path: '/horn-design', redirect: '/workbench/horn-design' },
  { path: '/knurl-design', redirect: '/workbench/knurl-design' },
  { path: '/acoustic', redirect: '/workbench/acoustic' },
  { path: '/fatigue', redirect: '/workbench/fatigue' },
  { path: '/results/:id', redirect: '/workbench/results/' },
  { path: '/reports/:id', redirect: '/workbench/reports/' },
]

export const router = createRouter({ history: createWebHistory(), routes })
