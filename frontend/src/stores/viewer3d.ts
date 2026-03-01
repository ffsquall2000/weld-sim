import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

export interface MeshData {
  points: Float32Array       // flattened [x0,y0,z0, x1,y1,z1, ...]
  cells: Uint32Array         // connectivity
  cellTypes: number[]        // VTK cell type IDs
  pointData: Record<string, Float32Array>  // scalar/vector fields keyed by name
  bounds: [number, number, number, number, number, number]  // xmin,xmax,ymin,ymax,zmin,zmax
}

export type DisplayMode = 'solid' | 'wireframe' | 'solid_wireframe'
export type ColorMapPreset = 'jet' | 'rainbow' | 'cool_warm' | 'viridis'

export interface SlicePlaneState {
  enabled: boolean
  normal: [number, number, number]
  origin: [number, number, number]
}

export const useViewer3DStore = defineStore('viewer3d', () => {
  // Active mesh identifier
  const activeMeshId = ref<string | null>(null)

  // Scalar field currently being visualized
  const scalarField = ref<string | null>(null)

  // Display mode
  const displayMode = ref<DisplayMode>('solid')

  // Edge visibility
  const showEdges = ref(false)

  // Axes widget visibility
  const showAxes = ref(true)

  // Color map range: null = auto range
  const colorMapRange = ref<[number, number] | null>(null)

  // Color map preset name
  const colorMap = ref<ColorMapPreset>('cool_warm')

  // Slice plane configuration
  const slicePlane = ref<SlicePlaneState>({
    enabled: false,
    normal: [1, 0, 0],
    origin: [0, 0, 0],
  })

  // Currently loaded mesh data
  const meshData = ref<MeshData | null>(null)

  // Available scalar field names extracted from meshData.pointData
  const fieldNames = computed<string[]>(() => {
    if (!meshData.value?.pointData) return []
    return Object.keys(meshData.value.pointData)
  })

  // Whether mesh data is currently being loaded
  const isLoading = ref(false)

  // Whether a mesh is loaded
  const hasMesh = computed(() => meshData.value !== null)

  // Current scalar field data array
  const activeFieldData = computed<Float32Array | null>(() => {
    if (!scalarField.value || !meshData.value?.pointData) return null
    return meshData.value.pointData[scalarField.value] ?? null
  })

  // Current scalar field data range
  const activeFieldRange = computed<[number, number] | null>(() => {
    if (colorMapRange.value) return colorMapRange.value
    const data = activeFieldData.value
    if (!data || data.length === 0) return null
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < data.length; i++) {
      const v = data[i]!
      if (v < min) min = v
      if (v > max) max = v
    }
    if (min === max) {
      // Avoid zero-range
      return [min - 0.5, max + 0.5]
    }
    return [min, max]
  })

  // --- Actions ---

  function loadMesh(id: string, data: MeshData) {
    activeMeshId.value = id
    meshData.value = data
    scalarField.value = null
    colorMapRange.value = null

    // Auto-select first available scalar field
    const keys = Object.keys(data.pointData)
    if (keys.length > 0) {
      scalarField.value = keys[0]!
    }

    // Center slice plane origin at mesh center
    const b = data.bounds
    slicePlane.value.origin = [
      (b[0] + b[1]) / 2,
      (b[2] + b[3]) / 2,
      (b[4] + b[5]) / 2,
    ]
  }

  function clearMesh() {
    activeMeshId.value = null
    meshData.value = null
    scalarField.value = null
    colorMapRange.value = null
  }

  function setScalarField(fieldName: string | null) {
    scalarField.value = fieldName
    colorMapRange.value = null  // reset to auto range
  }

  function setDisplayMode(mode: DisplayMode) {
    displayMode.value = mode
  }

  function setColorMap(preset: ColorMapPreset) {
    colorMap.value = preset
  }

  function setColorMapRange(range: [number, number] | null) {
    colorMapRange.value = range
  }

  function toggleEdges() {
    showEdges.value = !showEdges.value
  }

  function toggleAxes() {
    showAxes.value = !showAxes.value
  }

  function toggleSlicePlane() {
    slicePlane.value = {
      ...slicePlane.value,
      enabled: !slicePlane.value.enabled,
    }
  }

  function setSlicePlaneNormal(normal: [number, number, number]) {
    slicePlane.value = {
      ...slicePlane.value,
      normal,
    }
  }

  function setSlicePlaneOrigin(origin: [number, number, number]) {
    slicePlane.value = {
      ...slicePlane.value,
      origin,
    }
  }

  function setLoading(loading: boolean) {
    isLoading.value = loading
  }

  return {
    // State
    activeMeshId,
    scalarField,
    displayMode,
    showEdges,
    showAxes,
    colorMapRange,
    colorMap,
    slicePlane,
    meshData,
    isLoading,

    // Computed
    fieldNames,
    hasMesh,
    activeFieldData,
    activeFieldRange,

    // Actions
    loadMesh,
    clearMesh,
    setScalarField,
    setDisplayMode,
    setColorMap,
    setColorMapRange,
    toggleEdges,
    toggleAxes,
    toggleSlicePlane,
    setSlicePlaneNormal,
    setSlicePlaneOrigin,
    setLoading,
  }
})
