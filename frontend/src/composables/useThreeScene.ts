// frontend/src/composables/useThreeScene.ts
import { ref, onMounted, onBeforeUnmount, type Ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

export interface ThreeSceneOptions {
  antialias?: boolean
  background?: string
  enableClipping?: boolean
}

export function useThreeScene(
  containerRef: Ref<HTMLDivElement | null>,
  options: ThreeSceneOptions = {},
) {
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000)
  let renderer: THREE.WebGLRenderer | null = null
  let controls: OrbitControls | null = null
  let animationId: number | null = null
  const isReady = ref(false)

  // Clipping planes (for cross-section feature)
  const clippingPlanes: THREE.Plane[] = []

  function init() {
    const container = containerRef.value
    if (!container) return

    const { antialias = true, background = '#1a1a2e', enableClipping = false } = options

    renderer = new THREE.WebGLRenderer({
      antialias,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.setClearColor(new THREE.Color(background))
    if (enableClipping) {
      renderer.localClippingEnabled = true
    }
    container.appendChild(renderer.domElement)

    // Camera
    camera.aspect = container.clientWidth / container.clientHeight
    camera.position.set(0, 0.15, 0.3)
    camera.updateProjectionMatrix()

    // Controls
    controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.05

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(1, 2, 3)
    scene.add(dirLight)
    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x0f0e0d, 0.3)
    scene.add(hemiLight)

    isReady.value = true
  }

  function animate(customUpdate?: () => void) {
    function loop() {
      animationId = requestAnimationFrame(loop)
      controls?.update()
      if (customUpdate) customUpdate()
      if (renderer) renderer.render(scene, camera)
    }
    loop()
  }

  function resize() {
    const container = containerRef.value
    if (!container || !renderer) return
    const w = container.clientWidth
    const h = container.clientHeight
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
  }

  function fitToObject(object: THREE.Object3D) {
    const box = new THREE.Box3().setFromObject(object)
    const center = box.getCenter(new THREE.Vector3())
    const size = box.getSize(new THREE.Vector3())
    const maxDim = Math.max(size.x, size.y, size.z)
    const dist = maxDim / (2 * Math.tan((camera.fov * Math.PI) / 360))
    camera.position.copy(center).add(new THREE.Vector3(0, dist * 0.3, dist * 1.2))
    controls?.target.copy(center)
    controls?.update()
  }

  function dispose() {
    if (animationId !== null) cancelAnimationFrame(animationId)
    controls?.dispose()
    renderer?.dispose()
    scene.traverse((obj) => {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose()
        if (Array.isArray(obj.material)) obj.material.forEach((m) => m.dispose())
        else obj.material.dispose()
      }
    })
    if (renderer?.domElement.parentElement) {
      renderer.domElement.parentElement.removeChild(renderer.domElement)
    }
    isReady.value = false
  }

  let resizeObserver: ResizeObserver | null = null

  onMounted(() => {
    init()
    resizeObserver = new ResizeObserver(() => resize())
    if (containerRef.value) resizeObserver.observe(containerRef.value)
  })

  onBeforeUnmount(() => {
    resizeObserver?.disconnect()
    dispose()
  })

  return {
    scene,
    camera,
    renderer: renderer as THREE.WebGLRenderer,
    controls: controls as OrbitControls,
    clippingPlanes,
    isReady,
    animate,
    resize,
    fitToObject,
    dispose,
  }
}
