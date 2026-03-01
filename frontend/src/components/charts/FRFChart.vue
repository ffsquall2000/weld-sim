<!-- frontend/src/components/charts/FRFChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="frf-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, GridComponent, TooltipComponent, MarkLineComponent, MarkPointComponent, DataZoomComponent, CanvasRenderer])

const props = defineProps<{
  frequencies: number[]
  amplitudes: number[]
  targetFrequency?: number
  peakFrequency?: number
  peakAmplitude?: number
}>()

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    trigger: 'axis',
    formatter: (params: any) => {
      const p = params[0]
      return `Frequency: ${p.value[0].toFixed(1)} Hz<br/>Amplitude: ${p.value[1].toExponential(3)} m`
    },
  },
  grid: { left: 80, right: 30, top: 40, bottom: 60 },
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
    type: 'log',
    name: 'Amplitude (m)',
    nameLocation: 'center',
    nameGap: 55,
    axisLabel: { color: '#8b949e', formatter: (v: number) => v.toExponential(1) },
    axisLine: { lineStyle: { color: '#30363d' } },
    nameTextStyle: { color: '#8b949e' },
  },
  dataZoom: [
    { type: 'inside', xAxisIndex: 0 },
    { type: 'slider', xAxisIndex: 0, bottom: 5, height: 20 },
  ],
  series: [{
    type: 'line',
    data: props.frequencies.map((f, i) => [f, props.amplitudes[i]]),
    smooth: true,
    lineStyle: { color: '#ff9800', width: 2 },
    itemStyle: { color: '#ff9800' },
    symbol: 'none',
    markLine: props.targetFrequency ? {
      silent: true,
      data: [{ xAxis: props.targetFrequency, label: { formatter: 'Target', color: '#58a6ff' }, lineStyle: { color: '#58a6ff', type: 'dashed' } }],
    } : undefined,
    markPoint: props.peakFrequency ? {
      data: [{
        coord: [props.peakFrequency, props.peakAmplitude],
        symbol: 'circle',
        symbolSize: 10,
        itemStyle: { color: '#f44336' },
        label: { formatter: `Peak: ${props.peakFrequency?.toFixed(0)} Hz`, position: 'top', color: '#f44336' },
      }],
    } : undefined,
  }],
}))
</script>

<style scoped>
.frf-chart { width: 100%; height: 350px; }
</style>
