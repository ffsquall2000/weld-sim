<template>
  <svg viewBox="0 0 200 150" class="w-full max-w-[200px]" xmlns="http://www.w3.org/2000/svg">
    <!-- Pattern area -->
    <rect x="10" y="5" width="180" height="90" fill="var(--color-bg-secondary)" stroke="var(--color-border)" stroke-width="1" rx="2" />

    <!-- Linear: horizontal parallel lines -->
    <g v-if="knurlType === 'linear'">
      <line
        v-for="i in lineCount" :key="'l-' + i"
        :x1="15" :y1="10 + i * lineSpacing" :x2="185" :y2="10 + i * lineSpacing"
        stroke="#ff9800" stroke-width="1.5"
      />
    </g>

    <!-- Cross hatch: grid -->
    <g v-else-if="knurlType === 'cross_hatch'">
      <line
        v-for="i in lineCount" :key="'h-' + i"
        :x1="15" :y1="10 + i * lineSpacing" :x2="185" :y2="10 + i * lineSpacing"
        stroke="#ff9800" stroke-width="1"
      />
      <line
        v-for="i in vLineCount" :key="'v-' + i"
        :x1="15 + i * vLineSpacing" :y1="10" :x2="15 + i * vLineSpacing" :y2="90"
        stroke="#ff9800" stroke-width="1"
      />
    </g>

    <!-- Diamond: 45-degree crossing lines -->
    <g v-else-if="knurlType === 'diamond'">
      <line
        v-for="i in diagCount" :key="'d1-' + i"
        :x1="15 + (i - 1) * diagSpacing - 80" :y1="10"
        :x2="15 + (i - 1) * diagSpacing" :y2="90"
        stroke="#ff9800" stroke-width="1"
      />
      <line
        v-for="i in diagCount" :key="'d2-' + i"
        :x1="15 + (i - 1) * diagSpacing" :y1="10"
        :x2="15 + (i - 1) * diagSpacing - 80" :y2="90"
        stroke="#ff9800" stroke-width="1"
      />
    </g>

    <!-- Conical: grid of circles -->
    <g v-else-if="knurlType === 'conical'">
      <circle
        v-for="(pos, i) in gridPositions" :key="'c-' + i"
        :cx="pos.x" :cy="pos.y" :r="dotRadius"
        fill="none" stroke="#ff9800" stroke-width="1.5"
      />
    </g>

    <!-- Spherical: grid of filled dots -->
    <g v-else>
      <circle
        v-for="(pos, i) in gridPositions" :key="'s-' + i"
        :cx="pos.x" :cy="pos.y" :r="dotRadius"
        fill="#ff9800"
      />
    </g>

    <!-- Cross-section below: trapezoidal teeth -->
    <g transform="translate(10, 100)">
      <line x1="0" y1="0" x2="180" y2="0" stroke="var(--color-border)" stroke-width="1" />

      <!-- Teeth -->
      <g v-for="i in toothCount" :key="'t-' + i">
        <polygon
          :points="toothPoints(i - 1)"
          fill="#ff980040"
          stroke="#ff9800"
          stroke-width="1"
        />
      </g>

      <!-- Annotations -->
      <!-- Pitch arrow -->
      <line x1="20" y1="40" x2="60" y2="40" stroke="var(--color-text-secondary)" stroke-width="0.8" marker-end="url(#arrowhead)" marker-start="url(#arrowhead-rev)" />
      <text x="40" y="48" text-anchor="middle" fill="var(--color-text-secondary)" font-size="7">
        p={{ pitch }}
      </text>

      <!-- Depth label -->
      <text x="150" y="18" fill="var(--color-text-secondary)" font-size="7">
        d={{ depth }}
      </text>
    </g>

    <!-- Arrowhead markers -->
    <defs>
      <marker id="arrowhead" markerWidth="6" markerHeight="4" refX="5" refY="2" orient="auto">
        <polygon points="0 0, 6 2, 0 4" fill="var(--color-text-secondary)" />
      </marker>
      <marker id="arrowhead-rev" markerWidth="6" markerHeight="4" refX="1" refY="2" orient="auto-start-reverse">
        <polygon points="0 0, 6 2, 0 4" fill="var(--color-text-secondary)" />
      </marker>
    </defs>
  </svg>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  knurlType: string
  pitch: number
  toothWidth: number
  depth: number
}>()

// Line patterns
const lineCount = 7
const lineSpacing = computed(() => 80 / (lineCount + 1))
const vLineCount = 9
const vLineSpacing = computed(() => 170 / (vLineCount + 1))

// Diamond
const diagCount = 14
const diagSpacing = computed(() => 260 / (diagCount + 1))

// Grid positions for conical/spherical
const dotRadius = 4
const gridPositions = computed(() => {
  const positions: { x: number; y: number }[] = []
  const cols = 8
  const rows = 4
  const xSpacing = 170 / (cols + 1)
  const ySpacing = 80 / (rows + 1)
  for (let r = 1; r <= rows; r++) {
    for (let c = 1; c <= cols; c++) {
      positions.push({ x: 15 + c * xSpacing, y: 10 + r * ySpacing })
    }
  }
  return positions
})

// Tooth cross section
const toothCount = 4
const toothSpacing = 40
const toothDepth = 25

function toothPoints(index: number): string {
  const x = 20 + index * toothSpacing
  const topWidth = 12
  const bottomWidth = 20
  // Trapezoid: narrow top, wide base
  return `${x + (bottomWidth - topWidth) / 2},0 ${x + (bottomWidth + topWidth) / 2},0 ${x + bottomWidth},${toothDepth} ${x},${toothDepth}`
}
</script>
