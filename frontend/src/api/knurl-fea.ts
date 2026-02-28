import apiClient from './client'

// --- Request Types ---

export interface KnurlGenerateRequest {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  knurl_type: string
  knurl_pitch_mm: number
  knurl_depth_mm: number
  mesh_density: string
}

export interface KnurlAnalyzeRequest {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  knurl_type: string
  knurl_pitch_mm: number
  knurl_depth_mm: number
  material: string
  frequency_khz: number
  n_modes: number
  mesh_density: string
}

export interface KnurlOptimizeRequest {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  material: string
  frequency_khz: number
  n_candidates: number
}

export interface KnurlExportRequest {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  knurl_type: string
  knurl_pitch_mm: number
  knurl_depth_mm: number
}

// --- Response Types ---

export interface KnurlGenerateResponse {
  vertices: number[][]
  faces: number[][]
  knurl_info: Record<string, any>
  mesh_stats: { nodes: number; elements: number }
}

export interface ModeShapeEntry {
  frequency_hz: number
  mode_type: string
  participation_factor?: number
  effective_mass_ratio?: number
  displacement_max?: number
}

export interface KnurlAnalyzeResponse {
  mode_shapes: ModeShapeEntry[]
  closest_mode: number
  frequency_deviation_hz: number
  max_stress_mpa: number
  amplitude_uniformity: number
  mesh_preview: { vertices: number[][]; faces: number[][] } | null
}

export interface KnurlCompareResponse {
  with_knurl: KnurlAnalyzeResponse
  without_knurl: KnurlAnalyzeResponse
  frequency_shift_hz: number
  frequency_shift_percent: number
}

export interface OptimizationCandidate {
  knurl_type: string
  knurl_pitch_mm: number
  knurl_depth_mm: number
  frequency_deviation_hz: number
  amplitude_uniformity: number
  max_stress_mpa: number
  score?: number
}

export interface KnurlOptimizeResponse {
  candidates: OptimizationCandidate[]
  pareto_front: OptimizationCandidate[]
  best_frequency_match: OptimizationCandidate | null
  best_uniformity: OptimizationCandidate | null
  summary: string
}

export interface KnurlExportResponse {
  download_url: string
  filename: string
  file_size_bytes: number
}

// --- API Functions ---

export function generateKnurlMesh(req: KnurlGenerateRequest) {
  return apiClient.post<KnurlGenerateResponse>('/knurl-fea/generate', req, { timeout: 120000 })
}

export function analyzeKnurl(req: KnurlAnalyzeRequest) {
  return apiClient.post<KnurlAnalyzeResponse>('/knurl-fea/analyze', req, { timeout: 300000 })
}

export function compareKnurl(req: KnurlAnalyzeRequest) {
  return apiClient.post<KnurlCompareResponse>('/knurl-fea/compare', req, { timeout: 300000 })
}

export function optimizeKnurl(req: KnurlOptimizeRequest) {
  return apiClient.post<KnurlOptimizeResponse>('/knurl-fea/optimize', req, { timeout: 600000 })
}

export function exportKnurlStep(req: KnurlExportRequest) {
  return apiClient.post<KnurlExportResponse>('/knurl-fea/export-step', req, { timeout: 120000 })
}

export function getStepDownloadUrl(filename: string): string {
  return `/api/v1/knurl-fea/download-step/${encodeURIComponent(filename)}`
}
