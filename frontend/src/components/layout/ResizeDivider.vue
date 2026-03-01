<template>
  <div
    :class="[
      'resize-divider',
      direction === 'vertical' ? 'resize-divider--vertical' : 'resize-divider--horizontal',
      { 'resize-divider--dragging': isDragging },
    ]"
    @mousedown.prevent="onMouseDown"
  >
    <div class="resize-divider__handle" />
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'

const props = defineProps<{
  direction: 'horizontal' | 'vertical'
}>()

const emit = defineEmits<{
  resize: [delta: number]
}>()

const isDragging = ref(false)
let startPos = 0

function onMouseDown(e: MouseEvent) {
  isDragging.value = true
  startPos = props.direction === 'vertical' ? e.clientX : e.clientY
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
  document.body.style.cursor = props.direction === 'vertical' ? 'col-resize' : 'row-resize'
  document.body.style.userSelect = 'none'
}

function onMouseMove(e: MouseEvent) {
  if (!isDragging.value) return
  const currentPos = props.direction === 'vertical' ? e.clientX : e.clientY
  const delta = currentPos - startPos
  if (delta !== 0) {
    emit('resize', delta)
    startPos = currentPos
  }
}

function onMouseUp() {
  isDragging.value = false
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
  document.body.style.cursor = ''
  document.body.style.userSelect = ''
}

onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
})
</script>

<style scoped>
.resize-divider {
  position: relative;
  z-index: 10;
  flex-shrink: 0;
}

.resize-divider--vertical {
  width: 4px;
  cursor: col-resize;
}

.resize-divider--horizontal {
  height: 4px;
  cursor: row-resize;
}

.resize-divider__handle {
  position: absolute;
  background-color: var(--color-border);
  transition: background-color 0.15s;
}

.resize-divider--vertical .resize-divider__handle {
  top: 0;
  bottom: 0;
  left: 1px;
  width: 2px;
}

.resize-divider--horizontal .resize-divider__handle {
  left: 0;
  right: 0;
  top: 1px;
  height: 2px;
}

.resize-divider:hover .resize-divider__handle,
.resize-divider--dragging .resize-divider__handle {
  background-color: var(--color-accent-blue);
}

.resize-divider--dragging {
  /* Expand the hit area while dragging to prevent losing the mouse */
}

.resize-divider--vertical:hover,
.resize-divider--vertical.resize-divider--dragging {
  width: 4px;
}

.resize-divider--horizontal:hover,
.resize-divider--horizontal.resize-divider--dragging {
  height: 4px;
}
</style>
