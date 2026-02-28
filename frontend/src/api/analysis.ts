import apiClient from './client'

export interface ChainRequest {
  modules: string[]
  material: string
  frequency_khz: number
  mesh_density: string
  horn_type?: string
  width_mm?: number
  height_mm?: number
  length_mm?: number
  damping_model?: string
  damping_ratio?: number
  freq_range_percent?: number
  n_freq_points?: number
  surface_finish?: string
  Kt_global?: number
  reliability_pct?: number
  task_id?: string
}

export interface HarmonicResult {
  frequencies_hz: number[]
  gain: number
  q_factor: number
  contact_face_uniformity: number
  resonance_hz: number
  solve_time_s: number
}

export interface StressResult {
  max_stress_mpa: number
  safety_factor: number
  max_displacement_mm: number
  contact_face_uniformity: number
  resonance_hz: number
  solve_time_s: number
}

export interface FatigueResult {
  min_safety_factor: number
  estimated_life_cycles: number
  estimated_hours_at_20khz: number
  critical_locations: Array<{ element_id: number; safety_factor: number; x: number; y: number; z: number }>
  sn_curve_name: string
  corrected_endurance_mpa: number
  max_stress_mpa: number
}

export interface ChainResponse {
  task_id: string
  modules_executed: string[]
  modal?: any
  harmonic?: HarmonicResult
  stress?: StressResult
  fatigue?: FatigueResult
  total_solve_time_s: number
  node_count: number
  element_count: number
}

export async function runAnalysisChain(req: ChainRequest): Promise<ChainResponse> {
  const resp = await apiClient.post('/geometry/fea/run-chain', req, { timeout: 1800000 })
  return resp.data
}

export async function runModalAnalysis(params: any) {
  const resp = await apiClient.post('/geometry/fea/run', params, { timeout: 660000 })
  return resp.data
}

export async function runHarmonicAnalysis(params: any) {
  const resp = await apiClient.post('/geometry/fea/run-harmonic', params, { timeout: 660000 })
  return resp.data
}

export async function runStressAnalysis(params: any) {
  const resp = await apiClient.post('/geometry/fea/run-stress', params, { timeout: 660000 })
  return resp.data
}

export async function runFatigueAnalysis(params: any) {
  const resp = await apiClient.post('/geometry/fea/run-fatigue', params, { timeout: 1800000 })
  return resp.data
}
