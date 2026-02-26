import apiClient from './client'

export interface SimulateRequest {
  application: string
  upper_material_type: string
  upper_thickness_mm: number
  upper_layers: number
  lower_material_type: string
  lower_thickness_mm: number
  weld_width_mm: number
  weld_length_mm: number
  frequency_khz: number
  max_power_w: number
  horn_type?: string
  knurl_type?: string
  knurl_pitch_mm?: number
  knurl_tooth_width_mm?: number
  knurl_depth_mm?: number
  anvil_type?: string
  booster_gain?: number
}

export interface SimulateResponse {
  recipe_id: string
  application: string
  parameters: Record<string, number>
  safety_window: Record<string, [number, number]>
  risk_assessment: Record<string, string>
  quality_estimate: Record<string, number>
  recommendations: string[]
  validation: { status: string; messages: string[] }
  created_at: string
}

export const simulationApi = {
  calculate: (data: SimulateRequest) => apiClient.post<SimulateResponse>('/simulate', data),
  getSchema: (app: string) => apiClient.get(`/simulate/schema/${app}`),
}
