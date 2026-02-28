import apiClient from './client'

// --- Request Types ---

export interface ContactAnalyzeRequest {
  horn_material: string
  workpiece_material: string
  workpiece_thickness_mm: number
  anvil_type: string
  anvil_params: Record<string, number>
  contact_type: 'penalty' | 'nitsche'
  frequency_hz: number
  amplitude_um: number
}

export interface DockerCheckResponse {
  available: boolean
  image: string
}

export interface AnvilPreviewRequest {
  anvil_type: string
  width_mm: number
  depth_mm: number
  height_mm: number
  groove_width_mm?: number
  groove_depth_mm?: number
  knurl_pitch_mm?: number
  knurl_depth_mm?: number
}

export interface AnvilPreviewResponse {
  vertices: number[][]
  faces: number[][]
  anvil_type: string
  params: Record<string, number>
  contact_face: string
}

export interface ThermalRequest {
  workpiece_material: string
  frequency_hz: number
  amplitude_um: number
  weld_time_s: number
  contact_pressure_mpa: number
  initial_temp_c: number
  friction_coefficient: number
}

export interface FullAnalysisRequest {
  horn_material: string
  workpiece_material: string
  workpiece_thickness_mm: number
  anvil_type: string
  anvil_params: Record<string, number>
  contact_type: 'penalty' | 'nitsche'
  frequency_hz: number
  amplitude_um: number
  weld_time_s: number
  initial_temp_c: number
  friction_coefficient: number
}

// --- Response Types ---

export interface ContactAnalyzeResponse {
  contact_pressure: number
  slip_distance: number
  deformation: number
  stress: number
  weld_quality: number
  solve_time_s: number
}

export interface ThermalResponse {
  max_temperature_c: number
  melt_zone_fraction: number
  weld_quality: number
  thermal_history: number[]
  solve_time_s: number
}

export interface FullAnalysisResponse {
  contact: ContactAnalyzeResponse
  thermal: ThermalResponse
  weld_quality: number
  total_solve_time_s: number
}

// --- API Functions ---

export function analyzeContact(req: ContactAnalyzeRequest) {
  return apiClient.post<ContactAnalyzeResponse>('/contact/analyze', req, { timeout: 300000 })
}

export function checkDocker() {
  return apiClient.post<DockerCheckResponse>('/contact/check-docker', {}, { timeout: 15000 })
}

export function previewAnvil(req: AnvilPreviewRequest) {
  return apiClient.post<AnvilPreviewResponse>('/contact/anvil-preview', req, { timeout: 60000 })
}

export function analyzeThermal(req: ThermalRequest) {
  return apiClient.post<ThermalResponse>('/contact/thermal', req, { timeout: 300000 })
}

export function fullAnalysis(req: FullAnalysisRequest) {
  return apiClient.post<FullAnalysisResponse>('/contact/full-analysis', req, { timeout: 600000 })
}
