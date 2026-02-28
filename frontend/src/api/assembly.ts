import apiClient from './client'

export interface ComponentRequest {
  name: string
  horn_type: string
  dimensions: Record<string, number>
  material_name: string
  mesh_size: number
}

export interface AssemblyAnalysisRequest {
  components: ComponentRequest[]
  coupling_method: string
  penalty_factor: number
  analyses: string[]
  frequency_hz: number
  n_modes: number
  damping_ratio: number
  use_gmsh: boolean
  task_id?: string
}

export interface AssemblyAnalysisResponse {
  success: boolean
  message: string
  n_total_dof: number
  n_components: number
  frequencies_hz: number[]
  mode_types: string[]
  resonance_frequency_hz: number
  gain: number
  q_factor: number
  uniformity: number
  gain_chain: Record<string, number>
  impedance: Record<string, number>
  transmission_coefficients: Record<string, number>
  solve_time_s: number
}

export interface AssemblyMaterial {
  name: string
  E_gpa: number
  density_kg_m3: number
  poisson_ratio: number
  acoustic_velocity_m_s: number | null
}

export interface BoosterProfile {
  profile: string
  description: string
}

export const assemblyApi = {
  analyze: (request: AssemblyAnalysisRequest) =>
    apiClient.post<AssemblyAnalysisResponse>('/assembly/analyze', request, { timeout: 660000 }),

  modal: (request: AssemblyAnalysisRequest) =>
    apiClient.post<AssemblyAnalysisResponse>('/assembly/modal', request, { timeout: 660000 }),

  getMaterials: () =>
    apiClient.get<AssemblyMaterial[]>('/assembly/materials'),

  getProfiles: () =>
    apiClient.get<BoosterProfile[]>('/assembly/profiles'),
}
