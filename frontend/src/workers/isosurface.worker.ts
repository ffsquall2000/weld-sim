// frontend/src/workers/isosurface.worker.ts
// Marching Tetrahedra isosurface extraction WebWorker

/**
 * Marching Tetrahedra lookup table
 * For a tetrahedron with 4 vertices, each can be inside (1) or outside (0) the isosurface.
 * 16 possible configurations (2^4), each mapping to 0-2 triangles.
 */

// Edge table: maps edge index to vertex pair indices
const EDGES: [number, number][] = [
  [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
]

// Triangle table: for each configuration (0-15), lists edges that form triangles
// Each entry is an array of edge indices; every 3 edges form one triangle
const TRI_TABLE: number[][] = [
  [],                    // 0000 - all outside
  [0, 2, 1],             // 0001 - v0 inside
  [0, 3, 4],             // 0010 - v1 inside
  [1, 2, 3, 3, 4, 1],    // 0011 - v0, v1 inside
  [1, 5, 3],             // 0100 - v2 inside
  [0, 2, 5, 5, 3, 0],    // 0101 - v0, v2 inside
  [0, 5, 4, 0, 1, 5],    // 0110 - v1, v2 inside
  [2, 5, 4],             // 0111 - v0, v1, v2 inside
  [2, 4, 5],             // 1000 - v3 inside
  [0, 4, 5, 0, 5, 1],    // 1001 - v0, v3 inside
  [0, 3, 5, 0, 5, 2],    // 1010 - v1, v3 inside
  [1, 3, 5],             // 1011 - v0, v1, v3 inside
  [1, 4, 2, 4, 5, 2],    // 1100 - v2, v3 inside
  [0, 4, 3],             // 1101 - v0, v2, v3 inside
  [0, 1, 2, 1, 5, 2],    // 1110 - v1, v2, v3 inside
  []                     // 1111 - all inside
]

interface IsosurfaceRequest {
  type: 'extract'
  positions: Float32Array  // (N*3) vertex positions
  tetrahedra: Uint32Array  // (T*4) tetrahedron connectivity
  scalars: Float32Array    // (N) scalar values at each vertex
  threshold: number
}

interface IsosurfaceResponse {
  type: 'result'
  vertices: Float32Array   // (M*3) triangle vertices
  normals: Float32Array    // (M*3) vertex normals
  triangleCount: number
}

/**
 * Linear interpolation of vertex position along edge
 */
function interpolateVertex(
  positions: Float32Array,
  scalars: Float32Array,
  v1: number,
  v2: number,
  threshold: number,
  out: Float32Array,
  outIndex: number
) {
  const s1 = scalars[v1]
  const s2 = scalars[v2]
  const t = (threshold - s1) / (s2 - s1 + 1e-10)

  const p1x = positions[v1 * 3]
  const p1y = positions[v1 * 3 + 1]
  const p1z = positions[v1 * 3 + 2]
  const p2x = positions[v2 * 3]
  const p2y = positions[v2 * 3 + 1]
  const p2z = positions[v2 * 3 + 2]

  out[outIndex] = p1x + t * (p2x - p1x)
  out[outIndex + 1] = p1y + t * (p2y - p1y)
  out[outIndex + 2] = p1z + t * (p2z - p1z)
}

/**
 * Compute normal from triangle vertices
 */
function computeNormal(
  v0x: number, v0y: number, v0z: number,
  v1x: number, v1y: number, v1z: number,
  v2x: number, v2y: number, v2z: number
): [number, number, number] {
  const ax = v1x - v0x
  const ay = v1y - v0y
  const az = v1z - v0z
  const bx = v2x - v0x
  const by = v2y - v0y
  const bz = v2z - v0z

  let nx = ay * bz - az * by
  let ny = az * bx - ax * bz
  let nz = ax * by - ay * bx

  const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1
  return [nx / len, ny / len, nz / len]
}

/**
 * Extract isosurface using Marching Tetrahedra
 */
function extractIsosurface(
  positions: Float32Array,
  tetrahedra: Uint32Array,
  scalars: Float32Array,
  threshold: number
): { vertices: Float32Array; normals: Float32Array; triangleCount: number } {
  const numTets = tetrahedra.length / 4

  // Pre-allocate (worst case: 2 triangles per tet = 6 vertices per tet)
  const maxVertices = numTets * 6 * 3
  const vertexBuffer = new Float32Array(maxVertices)
  const normalBuffer = new Float32Array(maxVertices)
  let vertexCount = 0

  // Process each tetrahedron
  for (let t = 0; t < numTets; t++) {
    const i0 = tetrahedra[t * 4]
    const i1 = tetrahedra[t * 4 + 1]
    const i2 = tetrahedra[t * 4 + 2]
    const i3 = tetrahedra[t * 4 + 3]

    const tetIndices = [i0, i1, i2, i3]

    // Determine configuration
    let config = 0
    if (scalars[i0] >= threshold) config |= 1
    if (scalars[i1] >= threshold) config |= 2
    if (scalars[i2] >= threshold) config |= 4
    if (scalars[i3] >= threshold) config |= 8

    const triEdges = TRI_TABLE[config]
    if (triEdges.length === 0) continue

    // Generate triangles
    for (let e = 0; e < triEdges.length; e += 3) {
      const edge0 = EDGES[triEdges[e]]
      const edge1 = EDGES[triEdges[e + 1]]
      const edge2 = EDGES[triEdges[e + 2]]

      const vIdx = vertexCount * 3

      // Interpolate vertices along edges
      interpolateVertex(positions, scalars, tetIndices[edge0[0]], tetIndices[edge0[1]], threshold, vertexBuffer, vIdx)
      interpolateVertex(positions, scalars, tetIndices[edge1[0]], tetIndices[edge1[1]], threshold, vertexBuffer, vIdx + 3)
      interpolateVertex(positions, scalars, tetIndices[edge2[0]], tetIndices[edge2[1]], threshold, vertexBuffer, vIdx + 6)

      // Compute normal for this triangle
      const [nx, ny, nz] = computeNormal(
        vertexBuffer[vIdx], vertexBuffer[vIdx + 1], vertexBuffer[vIdx + 2],
        vertexBuffer[vIdx + 3], vertexBuffer[vIdx + 4], vertexBuffer[vIdx + 5],
        vertexBuffer[vIdx + 6], vertexBuffer[vIdx + 7], vertexBuffer[vIdx + 8]
      )

      // Same normal for all three vertices (flat shading)
      normalBuffer[vIdx] = nx; normalBuffer[vIdx + 1] = ny; normalBuffer[vIdx + 2] = nz
      normalBuffer[vIdx + 3] = nx; normalBuffer[vIdx + 4] = ny; normalBuffer[vIdx + 5] = nz
      normalBuffer[vIdx + 6] = nx; normalBuffer[vIdx + 7] = ny; normalBuffer[vIdx + 8] = nz

      vertexCount += 3
    }
  }

  // Trim buffers to actual size
  const finalVertices = new Float32Array(vertexBuffer.buffer, 0, vertexCount * 3)
  const finalNormals = new Float32Array(normalBuffer.buffer, 0, vertexCount * 3)

  return {
    vertices: finalVertices,
    normals: finalNormals,
    triangleCount: vertexCount / 3
  }
}

// WebWorker message handler
self.onmessage = (event: MessageEvent<IsosurfaceRequest>) => {
  const { type, positions, tetrahedra, scalars, threshold } = event.data

  if (type === 'extract') {
    const result = extractIsosurface(positions, tetrahedra, scalars, threshold)

    const response: IsosurfaceResponse = {
      type: 'result',
      vertices: result.vertices,
      normals: result.normals,
      triangleCount: result.triangleCount
    }

    // Transfer buffers back (no copy)
    self.postMessage(response, [result.vertices.buffer, result.normals.buffer])
  }
}

export {} // Make this a module
