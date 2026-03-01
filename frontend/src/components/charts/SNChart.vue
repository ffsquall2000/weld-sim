<!-- frontend/src/components/charts/SNChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="sn-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, CanvasRenderer])

export interface SNDataPoint {
  stress: number    // S - Stress amplitude (MPa)
  cycles: number    // N - Number of cycles to failure
}

const props = withDefaults(defineProps<{
  dataPoints?: SNDataPoint[]
  curveData?: SNDataPoint[]       // Pre-calculated S-N curve points
  fatigueLimit?: number           // Endurance limit stress (MPa)
  designCycles?: number           // Design life cycles
  designStress?: number           // Design stress
}>(), {
  dataPoints: () => [],
  curveData: () => [],
})

const chartOption = computed(() => {
  const series: any[] = []

  // S-N curve line
  if (props.curveData.length > 0) {
    series.push({
      name: 'S-N Curve',
      type: 'line',
      data: props.curveData.map(p => [p.cycles, p.stress]),
      smooth: true,
      lineStyle: { color: '#ff9800', width: 2 },
      itemStyle: { color: '#ff9800' },
      symbol: 'none',
    })
  }

  // Experimental data points
  if (props.dataPoints.length > 0) {
    series.push({
      name: 'Test Data',
      type: 'scatter',
      data: props.dataPoints.map(p => [p.cycles, p.stress]),
      symbolSize: 8,
      itemStyle: { color: '#58a6ff' },
    })
  }

  // Design point
  if (props.designCycles && props.designStress) {
    series.push({
      name: 'Design Point',
      type: 'scatter',
      data: [[props.designCycles, props.designStress]],
      symbolSize: 12,
      symbol: 'diamond',
      itemStyle: { color: '#3fb950' },
    })
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const cycles = params.value[0]
        const stress = params.value[1]
        return `${params.seriesName}<br/>Cycles: ${cycles.toExponential(2)}<br/>Stress: ${stress.toFixed(1)} MPa`
      },
    },
    legend: {
      data: series.map(s => s.name),
      top: 5,
      textStyle: { color: '#8b949e' },
    },
    grid: { left: 70, right: 30, top: 50, bottom: 50 },
    xAxis: {
      type: 'log',
      name: 'Cycles to Failure (N)',
      nameLocation: 'center',
      nameGap: 35,
      axisLabel: {
        color: '#8b949e',
        formatter: (v: number) => v.toExponential(0),
      },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    yAxis: {
      type: 'log',
      name: 'Stress Amplitude (MPa)',
      nameLocation: 'center',
      nameGap: 50,
      axisLabel: {
        color: '#8b949e',
        formatter: (v: number) => v.toFixed(0),
      },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    series: series.length > 0 ? series.map(s => ({
      ...s,
      markLine: s.name === 'S-N Curve' && props.fatigueLimit ? {
        silent: true,
        symbol: 'none',
        data: [{
          yAxis: props.fatigueLimit,
          label: {
            formatter: `Fatigue Limit: ${props.fatigueLimit} MPa`,
            color: '#d29922',
            position: 'end',
          },
          lineStyle: { color: '#d29922', type: 'dashed', width: 2 },
        }],
      } : undefined,
    })) : [],
  }
})
</script>

<style scoped>
.sn-chart { width: 100%; height: 350px; }
</style>
