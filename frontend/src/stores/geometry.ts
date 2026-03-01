import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { geometryApi } from '@/api/v2'

export interface GeometryVersion {
  id: string
  project_id: string
  version_number: number
  label: string | null
  source_type: string
  parametric_params: Record<string, any> | null
  file_path: string | null
  mesh_file_path: string | null
  metadata_json: Record<string, any> | null
}

export interface AssemblyBody {
  name: string
  material: string
  visible: boolean
  bodyType: 'horn' | 'anvil' | 'workpiece_upper' | 'workpiece_lower' | 'other'
}

export const useGeometryStore = defineStore('geometry', () => {
  // State
  const geometries = ref<GeometryVersion[]>([])
  const currentGeometry = ref<GeometryVersion | null>(null)
  const assemblyBodies = ref<AssemblyBody[]>([
    { name: 'Horn', material: 'Ti-6Al-4V', visible: true, bodyType: 'horn' },
    { name: 'Anvil', material: 'Tool Steel D2', visible: true, bodyType: 'anvil' },
    { name: 'Workpiece Upper', material: 'Nickel 201', visible: true, bodyType: 'workpiece_upper' },
    { name: 'Workpiece Lower', material: 'Copper C110', visible: true, bodyType: 'workpiece_lower' },
  ])
  const selectedBody = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  const meshProgress = ref(0)

  // Getters
  const geometriesByProject = computed(() => {
    return (projectId: string) =>
      geometries.value
        .filter((g) => g.project_id === projectId)
        .sort((a, b) => b.version_number - a.version_number)
  })

  const latestGeometry = computed(() => {
    if (geometries.value.length === 0) return null
    return geometries.value.reduce((latest, g) =>
      g.version_number > latest.version_number ? g : latest
    )
  })

  const visibleBodies = computed(() =>
    assemblyBodies.value.filter((b) => b.visible)
  )

  // Actions
  async function fetchGeometries(projectId: string) {
    loading.value = true
    error.value = null
    try {
      const response = await geometryApi.list(projectId)
      geometries.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function createGeometry(data: {
    project_id: string
    label?: string
    source_type: string
    parametric_params?: Record<string, any>
  }) {
    loading.value = true
    error.value = null
    try {
      const response = await geometryApi.create(data.project_id, data)
      geometries.value.push(response.data)
      currentGeometry.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function generateGeometry(id: string) {
    loading.value = true
    error.value = null
    try {
      const response = await geometryApi.generate(id)
      const index = geometries.value.findIndex((g) => g.id === id)
      if (index !== -1) {
        geometries.value[index] = response.data
      }
      if (currentGeometry.value?.id === id) {
        currentGeometry.value = response.data
      }
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function generateMesh(id: string) {
    loading.value = true
    error.value = null
    meshProgress.value = 0
    try {
      const response = await geometryApi.mesh(id)
      const index = geometries.value.findIndex((g) => g.id === id)
      if (index !== -1) {
        geometries.value[index] = response.data
      }
      if (currentGeometry.value?.id === id) {
        currentGeometry.value = response.data
      }
      meshProgress.value = 100
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function uploadGeometry(projectId: string, file: File) {
    loading.value = true
    error.value = null
    try {
      const response = await geometryApi.upload(projectId, file)
      geometries.value.push(response.data)
      currentGeometry.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  function setCurrentGeometry(geometry: GeometryVersion | null) {
    currentGeometry.value = geometry
  }

  function selectBody(name: string | null) {
    selectedBody.value = name
  }

  function toggleBodyVisibility(name: string) {
    const body = assemblyBodies.value.find((b) => b.name === name)
    if (body) {
      body.visible = !body.visible
    }
  }

  function updateBodyMaterial(name: string, material: string) {
    const body = assemblyBodies.value.find((b) => b.name === name)
    if (body) {
      body.material = material
    }
  }

  function reorderBodies(fromIndex: number, toIndex: number) {
    const item = assemblyBodies.value.splice(fromIndex, 1)[0]
    if (item) {
      assemblyBodies.value.splice(toIndex, 0, item)
    }
  }

  return {
    // State
    geometries,
    currentGeometry,
    assemblyBodies,
    selectedBody,
    loading,
    error,
    meshProgress,
    // Getters
    geometriesByProject,
    latestGeometry,
    visibleBodies,
    // Actions
    fetchGeometries,
    createGeometry,
    generateGeometry,
    generateMesh,
    uploadGeometry,
    setCurrentGeometry,
    selectBody,
    toggleBodyVisibility,
    updateBodyMaterial,
    reorderBodies,
  }
})
