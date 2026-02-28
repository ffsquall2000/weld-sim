<!-- frontend/src/components/charts/GainChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="gain-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent, CanvasRenderer])

export interface GainDataPoint {
  frequency: number
  gain: number        // Gain ratio (dimensionless)
  phase?: number      // Phase angle (degrees), optional
}

const props = withDefaults(defineProps<{
  dataPoints: GainDataPoint[]
  targetFrequency?: number
  targetGain?: number
  minAcceptableGain?: number
  maxAcceptableGain?: number
}>(), {
  dataPoints: () => [],
})

const peakGain = computed(() => {
  if (props.dataPoints.length === 0) return null
  const maxPoint = props.dataPoints.reduce((max, p) => p.gain > max.gain ? p : max, props.dataPoints[0])
  return maxPoint
})

const chartOption = computed(() => {
  const markLines: any[] = []

  // Target frequency line
  if (props.targetFrequency !== undefined) {
    markLines.push({
      xAxis: props.targetFrequency,
      label: { formatter: `Target: ${props.targetFrequency} Hz`, color: '#58a6ff', position: 'start' },
      lineStyle: { color: '#58a6ff', type: 'dashed', width: 2 },
    })
  }

  // Acceptable gain range lines
  if (props.minAcceptableGain !== undefined) {
    markLines.push({
      yAxis: props.minAcceptableGain,
      label: { formatter: `Min: ${props.minAcceptableGain}`, color: '#d29922', position: 'end' },
      lineStyle: { color: '#d29922', type: 'dotted', width: 1 },
    })
  }

  if (props.maxAcceptableGain !== undefined) {
    markLines.push({
      yAxis: props.maxAcceptableGain,
      label: { formatter: `Max: ${props.maxAcceptableGain}`, color: '#d29922', position: 'end' },
      lineStyle: { color: '#d29922', type: 'dotted', width: 1 },
    })
  }

  const markPoints: any[] = []

  // Peak gain point
  if (peakGain.value) {
    markPoints.push({
      coord: [peakGain.value.frequency, peakGain.value.gain],
      symbol: 'circle',
      symbolSize: 10,
      itemStyle: { color: '#f44336' },
      label: {
        formatter: `Peak: ${peakGain.value.gain.toFixed(2)}`,
        position: 'top',
        color: '#f44336',
      },
    })
  }

  // Target gain point
  if (props.targetFrequency !== undefined && props.targetGain !== undefined) {
    markPoints.push({
      coord: [props.targetFrequency, props.targetGain],
      symbol: 'diamond',
      symbolSize: 12,
      itemStyle: { color: '#3fb950' },
      label: {
        formatter: `Target: ${props.targetGain.toFixed(2)}`,
        position: 'right',
        color: '#3fb950',
      },
    })
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        const p = params[0]
        const point = props.dataPoints.find(d => d.frequency === p.value[0])
        const phaseStr = point?.phase !== undefined ? `<br/>Phase: ${point.phase.toFixed(1)}deg` : ''
        return `Frequency: ${p.value[0].toFixed(1)} Hz<br/>Gain: ${p.value[1].toFixed(3)}${phaseStr}`
      },
    },
    grid: { left: 70, right: 30, top: 40, bottom: 60 },
    xAxis: {
      type: 'value',
      name: 'Frequency (Hz)',
      nameLocation: 'center',
      nameGap: 35,
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    yAxis: {
      type: 'value',
      name: 'Gain (ratio)',
      nameLocation: 'center',
      nameGap: 50,
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    dataZoom: [
      { type: 'inside', xAxisIndex: 0 },
      { type: 'slider', xAxisIndex: 0, bottom: 5, height: 20 },
    ],
    series: [{
      type: 'line',
      data: props.dataPoints.map(p => [p.frequency, p.gain]),
      smooth: true,
      lineStyle: { color: '#ff9800', width: 2 },
      itemStyle: { color: '#ff9800' },
      symbol: 'none',
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(255, 152, 0, 0.3)' },
            { offset: 1, color: 'rgba(255, 152, 0, 0.05)' },
          ],
        },
      },
      markLine: markLines.length > 0 ? {
        silent: true,
        symbol: 'none',
        data: markLines,
      } : undefined,
      markPoint: markPoints.length > 0 ? {
        data: markPoints,
      } : undefined,
    }],
  }
})
</script>

<style scoped>
.gain-chart { width: 100%; height: 350px; }
</style>
