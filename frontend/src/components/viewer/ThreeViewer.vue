<template>
  <div ref="containerRef" class="three-viewer-container">
    <canvas ref="canvasRef" class="three-viewer-canvas" />
    <div v-if="!hasMesh" class="three-viewer-placeholder">
      <span>{{ placeholder }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted, onUnmounted, computed } from 'vue'

export interface MeshData {
  vertices: number[][]
  faces: number[][]
}

const props = withDefaults(
  defineProps<{
    mesh?: MeshData | null
    scalarField?: number[] | null
    wireframe?: boolean
    showAxes?: boolean
    placeholder?: string
  }>(),
  {
    mesh: null,
    scalarField: null,
    wireframe: false,
    showAxes: true,
    placeholder: '',
  },
)

const containerRef = ref<HTMLDivElement | null>(null)
const canvasRef = ref<HTMLCanvasElement | null>(null)

const hasMesh = computed(() => {
  return props.mesh && props.mesh.vertices.length > 0 && props.mesh.faces.length > 0
})

let isDragging = false
let prevMouse = { x: 0, y: 0 }
let rotation = { x: -0.5, y: 0.5 }
let zoom = 1.0

// Jet colormap: maps value in [0,1] to [r, g, b]
function jetColormap(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  let r: number, g: number, b: number
  if (t < 0.125) {
    r = 0; g = 0; b = 0.5 + t * 4
  } else if (t < 0.375) {
    r = 0; g = (t - 0.125) * 4; b = 1
  } else if (t < 0.625) {
    r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4
  } else if (t < 0.875) {
    r = 1; g = 1 - (t - 0.625) * 4; b = 0
  } else {
    r = 1 - (t - 0.875) * 4; g = 0; b = 0
  }
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}

function onMouseDown(e: MouseEvent) {
  isDragging = true
  prevMouse = { x: e.clientX, y: e.clientY }
}

function onMouseMove(e: MouseEvent) {
  if (!isDragging) return
  const dx = e.clientX - prevMouse.x
  const dy = e.clientY - prevMouse.y
  rotation.x += dy * 0.01
  rotation.y += dx * 0.01
  prevMouse = { x: e.clientX, y: e.clientY }
  renderFrame()
}

function onMouseUp() {
  isDragging = false
}

function onWheel(e: WheelEvent) {
  e.preventDefault()
  zoom *= e.deltaY > 0 ? 0.9 : 1.1
  zoom = Math.max(0.2, Math.min(5, zoom))
  renderFrame()
}

function resizeCanvas() {
  const canvas = canvasRef.value
  const container = containerRef.value
  if (!canvas || !container) return
  canvas.width = container.clientWidth
  canvas.height = container.clientHeight
  renderFrame()
}

function renderFrame() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = canvas.width
  const H = canvas.height

  // Clear
  ctx.fillStyle = '#1a1a2e'
  ctx.fillRect(0, 0, W, H)

  if (!hasMesh.value || !props.mesh) return

  const verts = props.mesh.vertices
  const faces = props.mesh.faces

  // Bounds
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  let minZ = Infinity, maxZ = -Infinity
  for (const v of verts) {
    const vx = v[0] ?? 0, vy = v[1] ?? 0, vz = v[2] ?? 0
    if (vx < minX) minX = vx; if (vx > maxX) maxX = vx
    if (vy < minY) minY = vy; if (vy > maxY) maxY = vy
    if (vz < minZ) minZ = vz; if (vz > maxZ) maxZ = vz
  }
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const cz = (minZ + maxZ) / 2
  const maxDim = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1)
  const scale = Math.min(W, H) * 0.35 * zoom / maxDim

  const cosX = Math.cos(rotation.x), sinX = Math.sin(rotation.x)
  const cosY = Math.cos(rotation.y), sinY = Math.sin(rotation.y)

  function project(x: number, y: number, z: number): [number, number, number] {
    x -= cx; y -= cy; z -= cz
    const x1 = x * cosY - z * sinY
    const z1 = x * sinY + z * cosY
    const y1 = y * cosX - z1 * sinX
    const z2 = y * sinX + z1 * cosX
    const d = 500
    const factor = d / (d + z2)
    return [W / 2 + x1 * scale * factor, H / 2 - y1 * scale * factor, z2]
  }

  // Project vertices
  const projected = verts.map(v => project(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0))

  // Compute per-vertex colors from scalarField
  let vertexColors: [number, number, number][] | null = null
  if (props.scalarField && props.scalarField.length === verts.length) {
    const vals = props.scalarField
    let mn = Infinity, mx = -Infinity
    for (const v of vals) { if (v < mn) mn = v; if (v > mx) mx = v }
    const range = mx - mn || 1
    vertexColors = vals.map(v => jetColormap((v - mn) / range))
  }

  // Sort faces by depth
  const faceDepths = faces.map((f, idx) => {
    const avgZ = f.reduce((sum, vi) => sum + (projected[vi]?.[2] ?? 0), 0) / f.length
    return { idx, avgZ }
  })
  faceDepths.sort((a, b) => b.avgZ - a.avgZ)

  // Draw faces
  for (const { idx } of faceDepths) {
    const face = faces[idx]
    if (!face || face.length < 3) continue
    const pts = face.map(vi => projected[vi] ?? [0, 0, 0] as [number, number, number])

    const p0 = pts[0]!, p1 = pts[1]!, p2 = pts[2]!
    const ax = p1[0] - p0[0], ay = p1[1] - p0[1]
    const bx = p2[0] - p0[0], by = p2[1] - p0[1]
    const cross = ax * by - ay * bx

    ctx.beginPath()
    ctx.moveTo(p0[0], p0[1])
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i]![0], pts[i]![1])
    }
    ctx.closePath()

    if (!props.wireframe) {
      if (vertexColors) {
        // Average face vertex colors
        let rSum = 0, gSum = 0, bSum = 0
        for (const vi of face) {
          const c = vertexColors[vi]
          if (c) { rSum += c[0]; gSum += c[1]; bSum += c[2] }
        }
        const n = face.length
        const brightness = Math.min(1.2, Math.max(0.4, 0.7 + cross * 0.0001))
        ctx.fillStyle = `rgb(${Math.round(rSum / n * brightness)}, ${Math.round(gSum / n * brightness)}, ${Math.round(bSum / n * brightness)})`
      } else {
        const brightness = Math.min(200, Math.max(50, 100 + cross * 0.01))
        ctx.fillStyle = `rgb(${Math.round(brightness * 0.5)}, ${Math.round(brightness * 0.7)}, ${Math.round(brightness)})`
      }
      ctx.fill()
    }
    ctx.strokeStyle = props.wireframe ? 'rgba(100,180,255,0.6)' : 'rgba(100,180,255,0.15)'
    ctx.lineWidth = props.wireframe ? 1 : 0.5
    ctx.stroke()
  }

  // Axes
  if (props.showAxes) {
    const axisLen = maxDim * 0.3
    const ox = cx - maxDim * 0.45, oy = cy - maxDim * 0.45, oz = cz - maxDim * 0.45
    const originPt = project(ox, oy, oz)
    const xEnd = project(ox + axisLen, oy, oz)
    const yEnd = project(ox, oy + axisLen, oz)
    const zEnd = project(ox, oy, oz + axisLen)

    ctx.lineWidth = 2
    ctx.strokeStyle = '#ef4444'
    ctx.beginPath(); ctx.moveTo(originPt[0], originPt[1]); ctx.lineTo(xEnd[0], xEnd[1]); ctx.stroke()
    ctx.fillStyle = '#ef4444'; ctx.font = '12px sans-serif'; ctx.fillText('X', xEnd[0] + 5, xEnd[1])

    ctx.strokeStyle = '#22c55e'
    ctx.beginPath(); ctx.moveTo(originPt[0], originPt[1]); ctx.lineTo(yEnd[0], yEnd[1]); ctx.stroke()
    ctx.fillStyle = '#22c55e'; ctx.fillText('Y', yEnd[0] + 5, yEnd[1])

    ctx.strokeStyle = '#3b82f6'
    ctx.beginPath(); ctx.moveTo(originPt[0], originPt[1]); ctx.lineTo(zEnd[0], zEnd[1]); ctx.stroke()
    ctx.fillStyle = '#3b82f6'; ctx.fillText('Z', zEnd[0] + 5, zEnd[1])
  }
}

// Public method for parent to reset camera
function resetCamera() {
  rotation = { x: -0.5, y: 0.5 }
  zoom = 1.0
  renderFrame()
}

defineExpose({ resetCamera, renderFrame })

// Watch for prop changes
watch(() => props.mesh, () => renderFrame(), { deep: true })
watch(() => props.scalarField, () => renderFrame(), { deep: true })
watch(() => props.wireframe, () => renderFrame())
watch(() => props.showAxes, () => renderFrame())

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
  const canvas = canvasRef.value
  if (canvas) {
    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('mouseleave', onMouseUp)
    canvas.addEventListener('wheel', onWheel, { passive: false })
  }
  resizeCanvas()
  if (containerRef.value) {
    resizeObserver = new ResizeObserver(() => resizeCanvas())
    resizeObserver.observe(containerRef.value)
  }
})

onUnmounted(() => {
  const canvas = canvasRef.value
  if (canvas) {
    canvas.removeEventListener('mousedown', onMouseDown)
    canvas.removeEventListener('mousemove', onMouseMove)
    canvas.removeEventListener('mouseup', onMouseUp)
    canvas.removeEventListener('mouseleave', onMouseUp)
    canvas.removeEventListener('wheel', onWheel)
  }
  resizeObserver?.disconnect()
})
</script>

<style scoped>
.three-viewer-container {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 300px;
  background-color: #1a1a2e;
  border-radius: 0.5rem;
  overflow: hidden;
}

.three-viewer-canvas {
  display: block;
  width: 100%;
  height: 100%;
}

.three-viewer-placeholder {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #555;
  font-size: 14px;
  pointer-events: none;
}
</style>
