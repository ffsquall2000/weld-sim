import apiClient from './client'

export const reportsApi = {
  export: (recipeId: string, format: string) =>
    apiClient.post('/reports/export', { recipe_id: recipeId, format }),
  downloadUrl: (filename: string) => `/api/v1/reports/download/${filename}`,
}
