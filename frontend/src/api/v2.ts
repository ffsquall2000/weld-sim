/**
 * Axios client for the v2 simulation platform API.
 */
import axios from 'axios'

const apiV2 = axios.create({
  baseURL: '/api/v2',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// Request interceptor for error logging
apiV2.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('[API v2]', error.response?.status, error.config?.url, error.response?.data)
    return Promise.reject(error)
  }
)

export default apiV2

// ----- Projects -----
export const projectApi = {
  list: (params?: { skip?: number; limit?: number; application_type?: string; search?: string }) =>
    apiV2.get('/projects', { params }),
  get: (id: string) => apiV2.get(`/projects/${id}`),
  create: (data: { name: string; description?: string; application_type: string; settings?: any; tags?: string[] }) =>
    apiV2.post('/projects', data),
  update: (id: string, data: any) => apiV2.patch(`/projects/${id}`, data),
  delete: (id: string) => apiV2.delete(`/projects/${id}`),
}

// ----- Geometries -----
export const geometryApi = {
  list: (projectId: string) => apiV2.get(`/projects/${projectId}/geometries`),
  get: (id: string) => apiV2.get(`/geometries/${id}`),
  create: (projectId: string, data: any) => apiV2.post(`/projects/${projectId}/geometries`, data),
  generate: (id: string) => apiV2.post(`/geometries/${id}/generate`),
  mesh: (id: string) => apiV2.post(`/geometries/${id}/mesh`),
  preview: (id: string) => apiV2.get(`/geometries/${id}/preview`),
  upload: (projectId: string, file: File, label?: string) => {
    const formData = new FormData()
    formData.append('file', file)
    if (label) formData.append('label', label)
    return apiV2.post(`/geometries/upload?project_id=${projectId}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },
}

// ----- Simulations -----
export const simulationApi = {
  list: (projectId: string) => apiV2.get(`/projects/${projectId}/simulations`),
  get: (id: string) => apiV2.get(`/simulations/${id}`),
  create: (projectId: string, data: any) => apiV2.post(`/projects/${projectId}/simulations`, data),
  update: (id: string, data: any) => apiV2.patch(`/simulations/${id}`, data),
  validate: (id: string) => apiV2.post(`/simulations/${id}/validate`),
}

// ----- Runs -----
export const runApi = {
  list: (simulationId: string) => apiV2.get(`/simulations/${simulationId}/runs`),
  get: (id: string) => apiV2.get(`/runs/${id}`),
  create: (simulationId: string, data: { geometry_version_id: string; parameters_override?: any }) =>
    apiV2.post(`/simulations/${simulationId}/runs`, data),
  cancel: (id: string) => apiV2.post(`/runs/${id}/cancel`),
  metrics: (id: string) => apiV2.get(`/runs/${id}/metrics`),
  artifacts: (id: string) => apiV2.get(`/runs/${id}/artifacts`),
  downloadArtifact: (runId: string, artifactId: string) =>
    apiV2.get(`/runs/${runId}/artifacts/${artifactId}/download`, { responseType: 'blob' }),
  fieldResults: (runId: string, fieldName: string) =>
    apiV2.get(`/runs/${runId}/results/field/${fieldName}`),
  standardMetrics: (runId: string) => apiV2.get(`/runs/${runId}/metrics/standard`),
  metricsCatalog: () => apiV2.get('/metrics/catalog'),
}

// ----- Materials -----
export const materialApiV2 = {
  list: (params?: { category?: string; search?: string }) =>
    apiV2.get('/materials', { params }),
  fea: () => apiV2.get('/materials/fea'),
  get: (id: string) => apiV2.get(`/materials/${id}`),
  create: (data: any) => apiV2.post('/materials', data),
}

// ----- Comparisons -----
export const comparisonApi = {
  create: (projectId: string, data: any) => apiV2.post(`/projects/${projectId}/comparisons`, data),
  get: (id: string) => apiV2.get(`/comparisons/${id}`),
  refresh: (id: string) => apiV2.post(`/comparisons/${id}/refresh`),
}

// ----- Optimizations -----
export const optimizationApi = {
  create: (simulationId: string, data: any) => apiV2.post(`/simulations/${simulationId}/optimize`, data),
  get: (id: string) => apiV2.get(`/optimizations/${id}`),
  iterations: (id: string) => apiV2.get(`/optimizations/${id}/iterations`),
  pareto: (id: string) => apiV2.get(`/optimizations/${id}/pareto`),
  pause: (id: string) => apiV2.post(`/optimizations/${id}/pause`),
  resume: (id: string) => apiV2.post(`/optimizations/${id}/resume`),
}

// ----- Workflows -----
export const workflowApi = {
  validate: (data: any) => apiV2.post('/workflows/validate', data),
  execute: (data: any) => apiV2.post('/workflows/execute', data),
  status: (id: string) => apiV2.get(`/workflows/${id}/status`),
}
