// frontend/src/composables/useIsosurface.ts
import { ref, shallowRef, onBeforeUnmount } from 'vue'
import * as THREE from 'three'

export interface IsosurfaceConfig {
  id: number
  threshold: number
  color: string
  opacity: number
  visible: boolean
}

export function useIsosurface(scene: THREE.Scene) {
  const isComputing = ref(false)
  const error = ref<string | null>(null)
  const isosurfaces = ref<IsosurfaceConfig[]>([])

  // Store mesh references
  const meshMap = new Map<number, THREE.Mesh>()
  let nextId = 1
  let worker: Worker | null = null

  // Source data (set by caller)
  const positions = shallowRef<Float32Array | null>(null)
  const tetrahedra = shallowRef<Uint32Array | null>(null)
  const scalars = shallowRef<Float32Array | null>(null)
  const scalarRange = ref<{ min: number; max: number }>({ min: 0, max: 1 })

  function setSourceData(
    pos: Float32Array,
    tets: Uint32Array,
    scal: Float32Array
  ) {
    positions.value = pos
    tetrahedra.value = tets
    scalars.value = scal

    // Compute scalar range
    let min = Infinity
    let max = -Infinity
    for (let i = 0; i < scal.length; i++) {
      if (scal[i] < min) min = scal[i]
      if (scal[i] > max) max = scal[i]
    }
    scalarRange.value = { min, max }
  }

  function getWorker(): Worker {
    if (!worker) {
      worker = new Worker(
        new URL('../workers/isosurface.worker.ts', import.meta.url),
        { type: 'module' }
      )
    }
    return worker
  }

  async function addIsosurface(threshold: number, color = '#ff9800', opacity = 0.6): Promise<number> {
    if (!positions.value || !tetrahedra.value || !scalars.value) {
      throw new Error('Source data not set')
    }

    const id = nextId++
    const config: IsosurfaceConfig = {
      id,
      threshold,
      color,
      opacity,
      visible: true
    }
    isosurfaces.value.push(config)

    await computeIsosurface(id, threshold, color, opacity)
    return id
  }

  function computeIsosurface(
    id: number,
    threshold: number,
    color: string,
    opacity: number
  ): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!positions.value || !tetrahedra.value || !scalars.value) {
        reject(new Error('Source data not set'))
        return
      }

      isComputing.value = true
      error.value = null

      const w = getWorker()

      const handler = (event: MessageEvent) => {
        const { type, vertices, normals, triangleCount } = event.data

        if (type === 'result') {
          w.removeEventListener('message', handler)
          isComputing.value = false

          if (triangleCount === 0) {
            // No isosurface at this threshold
            resolve()
            return
          }

          // Create mesh
          const geometry = new THREE.BufferGeometry()
          geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3))
          geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3))

          const material = new THREE.MeshPhongMaterial({
            color: new THREE.Color(color),
            opacity,
            transparent: opacity < 1,
            side: THREE.DoubleSide,
            shininess: 30,
          })

          const mesh = new THREE.Mesh(geometry, material)
          mesh.name = `isosurface-${id}`

          // Remove old mesh if updating
          const oldMesh = meshMap.get(id)
          if (oldMesh) {
            scene.remove(oldMesh)
            oldMesh.geometry.dispose()
            ;(oldMesh.material as THREE.Material).dispose()
          }

          meshMap.set(id, mesh)
          scene.add(mesh)
          resolve()
        }
      }

      w.addEventListener('message', handler)

      // Copy arrays (they'll be transferred)
      const posCopy = new Float32Array(positions.value)
      const tetCopy = new Uint32Array(tetrahedra.value)
      const scalCopy = new Float32Array(scalars.value)

      w.postMessage(
        {
          type: 'extract',
          positions: posCopy,
          tetrahedra: tetCopy,
          scalars: scalCopy,
          threshold
        },
        [posCopy.buffer, tetCopy.buffer, scalCopy.buffer]
      )
    })
  }

  function updateIsosurface(id: number, updates: Partial<Omit<IsosurfaceConfig, 'id'>>) {
    const idx = isosurfaces.value.findIndex(iso => iso.id === id)
    if (idx === -1) return

    const config = isosurfaces.value[idx]
    const mesh = meshMap.get(id)

    if (updates.threshold !== undefined && updates.threshold !== config.threshold) {
      config.threshold = updates.threshold
      computeIsosurface(id, updates.threshold, config.color, config.opacity)
    }

    if (updates.color !== undefined && mesh) {
      config.color = updates.color
      ;(mesh.material as THREE.MeshPhongMaterial).color.set(updates.color)
    }

    if (updates.opacity !== undefined && mesh) {
      config.opacity = updates.opacity
      const mat = mesh.material as THREE.MeshPhongMaterial
      mat.opacity = updates.opacity
      mat.transparent = updates.opacity < 1
    }

    if (updates.visible !== undefined && mesh) {
      config.visible = updates.visible
      mesh.visible = updates.visible
    }

    isosurfaces.value[idx] = { ...config }
  }

  function removeIsosurface(id: number) {
    const idx = isosurfaces.value.findIndex(iso => iso.id === id)
    if (idx === -1) return

    isosurfaces.value.splice(idx, 1)

    const mesh = meshMap.get(id)
    if (mesh) {
      scene.remove(mesh)
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
      meshMap.delete(id)
    }
  }

  function clearAll() {
    for (const id of meshMap.keys()) {
      removeIsosurface(id)
    }
    isosurfaces.value = []
  }

  function dispose() {
    clearAll()
    if (worker) {
      worker.terminate()
      worker = null
    }
  }

  onBeforeUnmount(dispose)

  return {
    isComputing,
    error,
    isosurfaces,
    scalarRange,
    setSourceData,
    addIsosurface,
    updateIsosurface,
    removeIsosurface,
    clearAll,
    dispose,
  }
}
