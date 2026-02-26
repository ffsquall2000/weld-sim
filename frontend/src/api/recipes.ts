import apiClient from './client'

export const recipesApi = {
  list: (limit = 50) => apiClient.get('/recipes', { params: { limit } }),
  get: (id: string) => apiClient.get(`/recipes/${id}`),
}
