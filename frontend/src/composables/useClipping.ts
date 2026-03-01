// frontend/src/composables/useClipping.ts
import { ref, reactive, watch, onBeforeUnmount } from 'vue'
import * as THREE from 'three'

export interface ClippingAxis {
  enabled: boolean
  position: number // normalized 0-1
  inverted: boolean
}

export interface ClippingState {
  x: ClippingAxis
  y: ClippingAxis
  z: ClippingAxis
}

export function useClipping(renderer: THREE.WebGLRenderer | null) {
  const state = reactive<ClippingState>({
    x: { enabled: false, position: 0.5, inverted: false },
    y: { enabled: false, position: 0.5, inverted: false },
    z: { enabled: false, position: 0.5, inverted: false },
  })

  // Bounding box for the mesh (set by caller)
  const bounds = ref<THREE.Box3>(new THREE.Box3(
    new THREE.Vector3(-1, -1, -1),
    new THREE.Vector3(1, 1, 1)
  ))

  // THREE.Plane objects
  const planeX = new THREE.Plane(new THREE.Vector3(-1, 0, 0), 0)
  const planeY = new THREE.Plane(new THREE.Vector3(0, -1, 0), 0)
  const planeZ = new THREE.Plane(new THREE.Vector3(0, 0, -1), 0)

  const planes: THREE.Plane[] = []

  function updatePlanes() {
    planes.length = 0
    const box = bounds.value

    if (state.x.enabled) {
      const pos = THREE.MathUtils.lerp(box.min.x, box.max.x, state.x.position)
      const dir = state.x.inverted ? 1 : -1
      planeX.set(new THREE.Vector3(dir, 0, 0), -dir * pos)
      planes.push(planeX)
    }

    if (state.y.enabled) {
      const pos = THREE.MathUtils.lerp(box.min.y, box.max.y, state.y.position)
      const dir = state.y.inverted ? 1 : -1
      planeY.set(new THREE.Vector3(0, dir, 0), -dir * pos)
      planes.push(planeY)
    }

    if (state.z.enabled) {
      const pos = THREE.MathUtils.lerp(box.min.z, box.max.z, state.z.position)
      const dir = state.z.inverted ? 1 : -1
      planeZ.set(new THREE.Vector3(0, 0, dir), -dir * pos)
      planes.push(planeZ)
    }

    // Apply to renderer
    if (renderer) {
      renderer.clippingPlanes = planes
      renderer.localClippingEnabled = planes.length > 0
    }
  }

  // Watch state changes
  watch(
    () => [
      state.x.enabled, state.x.position, state.x.inverted,
      state.y.enabled, state.y.position, state.y.inverted,
      state.z.enabled, state.z.position, state.z.inverted,
    ],
    updatePlanes,
    { immediate: true }
  )

  function setBounds(box: THREE.Box3) {
    bounds.value = box.clone()
    updatePlanes()
  }

  function toggleAxis(axis: 'x' | 'y' | 'z') {
    state[axis].enabled = !state[axis].enabled
  }

  function setPosition(axis: 'x' | 'y' | 'z', position: number) {
    state[axis].position = Math.max(0, Math.min(1, position))
  }

  function invertAxis(axis: 'x' | 'y' | 'z') {
    state[axis].inverted = !state[axis].inverted
  }

  function reset() {
    state.x = { enabled: false, position: 0.5, inverted: false }
    state.y = { enabled: false, position: 0.5, inverted: false }
    state.z = { enabled: false, position: 0.5, inverted: false }
  }

  function dispose() {
    if (renderer) {
      renderer.clippingPlanes = []
      renderer.localClippingEnabled = false
    }
  }

  onBeforeUnmount(dispose)

  return {
    state,
    planes,
    bounds,
    setBounds,
    toggleAxis,
    setPosition,
    invertAxis,
    reset,
    dispose,
    updatePlanes,
  }
}
