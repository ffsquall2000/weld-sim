<!-- frontend/src/components/viewer/ColorBar.vue -->
<template>
  <div class="colorbar">
    <div class="colorbar-label">{{ label }}</div>
    <div class="colorbar-gradient" ref="gradientRef">
      <canvas ref="canvasRef" width="20" height="200" />
    </div>
    <div class="colorbar-ticks">
      <span>{{ max.toExponential(2) }}</span>
      <span>{{ ((max + min) / 2).toExponential(2) }}</span>
      <span>{{ min.toExponential(2) }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'
import { useColormap, type ColormapName } from '@/composables/useColormap'

const props = withDefaults(defineProps<{
  colormap?: ColormapName
  min?: number
  max?: number
  label?: string
}>(), {
  colormap: 'jet',
  min: 0,
  max: 1,
  label: 'Value',
})

const canvasRef = ref<HTMLCanvasElement | null>(null)
const { sampleColor } = useColormap()

function drawGradient() {
  const canvas = canvasRef.value
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const h = canvas.height
  for (let y = 0; y < h; y++) {
    const t = 1 - y / h // top = max, bottom = min
    const [r, g, b] = sampleColor(props.colormap, t)
    ctx.fillStyle = `rgb(${r * 255}, ${g * 255}, ${b * 255})`
    ctx.fillRect(0, y, canvas.width, 1)
  }
}

watch(() => props.colormap, drawGradient)
onMounted(drawGradient)
</script>

<style scoped>
.colorbar {
  position: absolute;
  right: 12px;
  top: 50%;
  transform: translateY(-50%);
  display: flex;
  gap: 4px;
  z-index: 10;
  background: rgba(0, 0, 0, 0.7);
  border-radius: 6px;
  padding: 8px;
}

.colorbar-label {
  writing-mode: vertical-rl;
  text-orientation: mixed;
  font-size: 11px;
  color: #aaa;
  transform: rotate(180deg);
}

.colorbar-gradient canvas {
  border-radius: 3px;
}

.colorbar-ticks {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  font-size: 10px;
  color: #aaa;
  min-width: 60px;
}
</style>
