import apiClient from './client'

export const reportsApi = {
  export: (recipeId: string, format: string) =>
    apiClient.post('/reports/generate', { recipe_id: recipeId, format }),
  downloadUrl: (filename: string) => `/api/v2/reports/download/${filename}`,
}
