import apiClient from './client'

export interface GeometryAnalysisResponse {
  horn_type: string
  dimensions: Record<string, number>
  contact_dimensions: { width_mm: number; length_mm: number } | null
  gain_estimate: number
  confidence: number
  knurl: Record<string, any> | null
  bounding_box: number[]
  volume_mm3: number
  surface_area_mm2: number
  mesh: { vertices: number[][]; faces: number[][] } | null
}

export interface FEARequest {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  material: string
  frequency_khz: number
  mesh_density: string
}

export interface ModeShape {
  frequency_hz: number
  mode_type: string
  participation_factor: number
  effective_mass_ratio: number
  displacement_max: number
}

export interface FEAResponse {
  mode_shapes: ModeShape[]
  closest_mode_hz: number
  target_frequency_hz: number
  frequency_deviation_percent: number
  node_count: number
  element_count: number
  solve_time_s: number
  mesh: { vertices: number[][]; faces: number[][] } | null
  stress_max_mpa: number | null
  temperature_max_c: number | null
  task_id?: string
}

export interface PDFAnalysisResponse {
  detected_dimensions: Array<{
    label: string
    value_mm: number
    tolerance_mm: number
    type: string
    confidence: number
    page: number
  }>
  tolerances: Array<{ label: string; tolerance_mm: number }>
  notes: string[]
  confidence: number
  page_count: number
}

export interface FEAMaterial {
  name: string
  E_gpa: number
  density_kg_m3: number
  poisson_ratio: number
}

export const geometryApi = {
  uploadCAD: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return apiClient.post<GeometryAnalysisResponse>('/geometry/upload/cad', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    })
  },

  uploadPDF: (file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    return apiClient.post<PDFAnalysisResponse>('/geometry/upload/pdf', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 60000,
    })
  },

  runFEA: (request: FEARequest) =>
    apiClient.post<FEAResponse>('/geometry/fea/run', request, { timeout: 120000 }),

  runFEAOnStep: (file: File, material: string, frequencyKhz: number, meshDensity: string) => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('material', material)
    formData.append('frequency_khz', frequencyKhz.toString())
    formData.append('mesh_density', meshDensity)
    return apiClient.post<FEAResponse>('/geometry/fea/run-step', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 180000,
    })
  },

  getMaterials: () => apiClient.get<FEAMaterial[]>('/geometry/fea/materials'),
}
