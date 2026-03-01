import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api/v2',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

export default apiClient
