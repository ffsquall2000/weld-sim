import axios from 'axios'

const API_BASE = '/api/v1/mesh'

export interface MeshInfo {
  task_id: string
  node_count: number
  face_count: number
  has_normals: boolean
  available_scalars: string[]
  mode_count: number
  modes: Array<{
    index: number
    frequency_hz: number
    mode_type: string
  }>
}

export interface MeshGeometry {
  positions: Float32Array
  indices: Uint32Array
  normals?: Float32Array
  nodeCount: number
  faceCount: number
}

export interface MeshScalars {
  values: Float32Array
  min: number
  max: number
  nodeCount: number
}

export interface ModeShape {
  frequency: number
  modeType: string
  shape: Float32Array
  nodeCount: number
}

/**
 * Fetch mesh metadata as JSON
 */
export async function fetchMeshInfo(taskId: string): Promise<MeshInfo> {
  const { data } = await axios.get<MeshInfo>(`${API_BASE}/${taskId}/info`)
  return data
}

/**
 * Fetch mesh geometry as binary data
 */
export async function fetchMeshGeometry(taskId: string): Promise<MeshGeometry> {
  const response = await axios.get(`${API_BASE}/${taskId}/geometry`, {
    responseType: 'arraybuffer',
  })

  const buffer = response.data as ArrayBuffer
  const view = new DataView(buffer)

  // Parse header (12 bytes)
  const nodeCount = view.getUint32(0, true)
  const faceCount = view.getUint32(4, true)
  const hasNormals = view.getUint32(8, true) === 1

  // Calculate offsets
  const positionsOffset = 12
  const positionsSize = nodeCount * 3 * 4 // float32 = 4 bytes
  const indicesOffset = positionsOffset + positionsSize
  const indicesSize = faceCount * 3 * 4 // uint32 = 4 bytes
  const normalsOffset = indicesOffset + indicesSize

  // Extract typed arrays
  const positions = new Float32Array(buffer, positionsOffset, nodeCount * 3)
  const indices = new Uint32Array(buffer, indicesOffset, faceCount * 3)

  let normals: Float32Array | undefined
  if (hasNormals) {
    normals = new Float32Array(buffer, normalsOffset, nodeCount * 3)
  }

  return { positions, indices, normals, nodeCount, faceCount }
}

/**
 * Fetch scalar field as binary data
 */
export async function fetchMeshScalars(taskId: string, field: string): Promise<MeshScalars> {
  const response = await axios.get(`${API_BASE}/${taskId}/scalars`, {
    params: { field },
    responseType: 'arraybuffer',
  })

  const buffer = response.data as ArrayBuffer
  const values = new Float32Array(buffer)

  // Get min/max from headers
  const min = parseFloat(response.headers['x-scalar-min'] || '0')
  const max = parseFloat(response.headers['x-scalar-max'] || '1')
  const nodeCount = parseInt(response.headers['x-node-count'] || '0', 10)

  return { values, min, max, nodeCount }
}

/**
 * Fetch mode shape as binary data
 */
export async function fetchModeShape(taskId: string, modeIndex: number): Promise<ModeShape> {
  const response = await axios.get(`${API_BASE}/${taskId}/modes/${modeIndex}`, {
    responseType: 'arraybuffer',
  })

  const buffer = response.data as ArrayBuffer
  const view = new DataView(buffer)

  // Parse header (12 bytes)
  const frequency = view.getFloat32(0, true)
  const modeTypeLen = view.getUint32(4, true)
  // reserved = view.getUint32(8, true)

  // Parse mode type string
  const modeTypeBytes = new Uint8Array(buffer, 12, modeTypeLen)
  const modeType = new TextDecoder().decode(modeTypeBytes)

  // Parse shape data
  const shapeOffset = 12 + modeTypeLen
  const nodeCount = (buffer.byteLength - shapeOffset) / 12 // 3 floats per node
  const shape = new Float32Array(buffer, shapeOffset, nodeCount * 3)

  return { frequency, modeType, shape, nodeCount }
}
