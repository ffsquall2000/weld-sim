import apiClient from './client'

export const materialsApi = {
  list: () => apiClient.get<{ materials: string[] }>('/materials'),
  get: (type: string) => apiClient.get(`/materials/${encodeURIComponent(type)}`),
  getCombination: (a: string, b: string) =>
    apiClient.get(`/materials/combination/${encodeURIComponent(a)}/${encodeURIComponent(b)}`),
}
