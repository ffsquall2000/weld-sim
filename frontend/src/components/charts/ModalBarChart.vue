<!-- frontend/src/components/charts/ModalBarChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="modal-bar-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([BarChart, GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, CanvasRenderer])

export interface ModalData {
  modeNumber: number
  frequency: number
  type: 'longitudinal' | 'flexural' | 'torsional' | 'unknown'
}

const props = defineProps<{
  modes: ModalData[]
  targetFrequency?: number
  parasiticRange?: number // Range around target to highlight parasitic modes (default 500 Hz)
}>()

const parasiticThreshold = computed(() => props.parasiticRange ?? 500)

const modeColors: Record<string, string> = {
  longitudinal: '#58a6ff', // blue
  flexural: '#ff9800',     // orange
  torsional: '#3fb950',    // green
  unknown: '#8b949e',      // gray
}

const isParasitic = (mode: ModalData) => {
  if (!props.targetFrequency) return false
  if (mode.type === 'longitudinal') return false
  const delta = Math.abs(mode.frequency - props.targetFrequency)
  return delta <= parasiticThreshold.value
}

const getBarColor = (mode: ModalData) => {
  if (isParasitic(mode)) return '#f44336' // red for parasitic
  return modeColors[mode.type] || modeColors.unknown
}

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    axisPointer: { type: 'shadow' },
    formatter: (params: any) => {
      const p = params[0]
      const mode = props.modes[p.dataIndex]
      if (!mode) return ''
      const parasitic = isParasitic(mode) ? ' (Parasitic!)' : ''
      return `Mode ${mode.modeNumber}<br/>Frequency: ${mode.frequency.toFixed(1)} Hz<br/>Type: ${mode.type}${parasitic}`
    },
  },
  legend: {
    data: ['Longitudinal', 'Flexural', 'Torsional', 'Parasitic'],
    top: 5,
    textStyle: { color: '#8b949e' },
  },
  grid: { left: 80, right: 30, top: 50, bottom: 30 },
  xAxis: {
    type: 'value',
    name: 'Frequency (Hz)',
    nameLocation: 'center',
    nameGap: 25,
    axisLabel: { color: '#8b949e' },
    axisLine: { lineStyle: { color: '#30363d' } },
    nameTextStyle: { color: '#8b949e' },
  },
  yAxis: {
    type: 'category',
    name: 'Mode',
    data: props.modes.map(m => `Mode ${m.modeNumber}`),
    axisLabel: { color: '#8b949e' },
    axisLine: { lineStyle: { color: '#30363d' } },
    nameTextStyle: { color: '#8b949e' },
  },
  series: [{
    type: 'bar',
    data: props.modes.map(mode => ({
      value: mode.frequency,
      itemStyle: { color: getBarColor(mode) },
    })),
    barWidth: '60%',
    markLine: props.targetFrequency ? {
      silent: true,
      symbol: 'none',
      data: [{
        xAxis: props.targetFrequency,
        label: {
          formatter: `Target: ${props.targetFrequency} Hz`,
          color: '#58a6ff',
          position: 'end',
        },
        lineStyle: { color: '#58a6ff', type: 'dashed', width: 2 },
      }],
    } : undefined,
  }],
}))
</script>

<style scoped>
.modal-bar-chart { width: 100%; height: 300px; }
</style>
