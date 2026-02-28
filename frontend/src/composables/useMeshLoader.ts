import { ref, shallowRef, computed } from 'vue'
import {
  fetchMeshInfo,
  fetchMeshGeometry,
  fetchMeshScalars,
  fetchModeShape,
  type MeshInfo,
  type MeshGeometry,
  type MeshScalars,
  type ModeShape,
} from '@/api/mesh'

export function useMeshLoader() {
  const isLoading = ref(false)
  const error = ref<string | null>(null)

  // Use shallowRef for large typed arrays to avoid Vue's deep reactivity overhead
  const meshInfo = ref<MeshInfo | null>(null)
  const geometry = shallowRef<MeshGeometry | null>(null)
  const currentScalars = shallowRef<MeshScalars | null>(null)
  const currentMode = shallowRef<ModeShape | null>(null)

  // Cache for loaded data
  const scalarCache = new Map<string, MeshScalars>()
  const modeCache = new Map<number, ModeShape>()

  const hasGeometry = computed(() => geometry.value !== null)
  const nodeCount = computed(() => geometry.value?.nodeCount ?? 0)
  const faceCount = computed(() => geometry.value?.faceCount ?? 0)

  async function loadMesh(taskId: string) {
    isLoading.value = true
    error.value = null

    // Clear caches
    scalarCache.clear()
    modeCache.clear()
    currentScalars.value = null
    currentMode.value = null

    try {
      // Load info and geometry in parallel
      const [info, geo] = await Promise.all([
        fetchMeshInfo(taskId),
        fetchMeshGeometry(taskId),
      ])

      meshInfo.value = info
      geometry.value = geo

      // Auto-load first scalar field if available
      if (info.available_scalars.length > 0) {
        await loadScalars(taskId, info.available_scalars[0]!)
      }
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Failed to load mesh'
      throw e
    } finally {
      isLoading.value = false
    }
  }

  async function loadScalars(taskId: string, field: string) {
    // Check cache first
    const cacheKey = `${taskId}:${field}`
    if (scalarCache.has(cacheKey)) {
      currentScalars.value = scalarCache.get(cacheKey)!
      return currentScalars.value
    }

    try {
      const scalars = await fetchMeshScalars(taskId, field)
      scalarCache.set(cacheKey, scalars)
      currentScalars.value = scalars
      return scalars
    } catch (e) {
      error.value = `Failed to load scalar field '${field}'`
      throw e
    }
  }

  async function loadMode(taskId: string, modeIndex: number) {
    // Check cache first
    if (modeCache.has(modeIndex)) {
      currentMode.value = modeCache.get(modeIndex)!
      return currentMode.value
    }

    try {
      const mode = await fetchModeShape(taskId, modeIndex)
      modeCache.set(modeIndex, mode)
      currentMode.value = mode
      return mode
    } catch (e) {
      error.value = `Failed to load mode ${modeIndex}`
      throw e
    }
  }

  function clearCache() {
    scalarCache.clear()
    modeCache.clear()
    meshInfo.value = null
    geometry.value = null
    currentScalars.value = null
    currentMode.value = null
  }

  return {
    // State
    isLoading,
    error,
    meshInfo,
    geometry,
    currentScalars,
    currentMode,

    // Computed
    hasGeometry,
    nodeCount,
    faceCount,

    // Methods
    loadMesh,
    loadScalars,
    loadMode,
    clearCache,
  }
}
