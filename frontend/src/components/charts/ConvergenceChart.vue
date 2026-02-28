<!-- frontend/src/components/charts/ConvergenceChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="convergence-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, MarkLineComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([LineChart, ScatterChart, GridComponent, TooltipComponent, MarkLineComponent, LegendComponent, CanvasRenderer])

export interface ConvergencePoint {
  meshSize: number      // Characteristic element size (mm)
  dof: number           // Degrees of freedom
  value: number         // Result value (e.g., stress, frequency)
}

const props = withDefaults(defineProps<{
  dataPoints: ConvergencePoint[]
  richardsonExtrapolation?: number   // Extrapolated value at h=0
  convergenceOrder?: number           // Observed convergence order
  targetError?: number                // Target relative error (%)
  valueName?: string                  // Name of the value being plotted
  valueUnit?: string                  // Unit of the value
}>(), {
  dataPoints: () => [],
  valueName: 'Value',
  valueUnit: '',
})

const chartOption = computed(() => {
  const series: any[] = []

  // Main convergence data
  if (props.dataPoints.length > 0) {
    series.push({
      name: props.valueName,
      type: 'line',
      data: props.dataPoints.map(p => [p.meshSize, p.value]),
      lineStyle: { color: '#ff9800', width: 2 },
      itemStyle: { color: '#ff9800' },
      symbolSize: 8,
    })
  }

  // Richardson extrapolation point
  if (props.richardsonExtrapolation !== undefined && props.dataPoints.length > 0) {
    series.push({
      name: 'Richardson Extrapolation',
      type: 'scatter',
      data: [[0, props.richardsonExtrapolation]],
      symbolSize: 12,
      symbol: 'diamond',
      itemStyle: { color: '#3fb950' },
    })
  }

  const markLines: any[] = []

  // Richardson extrapolation horizontal line
  if (props.richardsonExtrapolation !== undefined) {
    markLines.push({
      yAxis: props.richardsonExtrapolation,
      label: {
        formatter: `Extrapolated: ${props.richardsonExtrapolation.toFixed(2)} ${props.valueUnit}`,
        color: '#3fb950',
        position: 'end',
      },
      lineStyle: { color: '#3fb950', type: 'dashed', width: 2 },
    })
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (params.seriesName === 'Richardson Extrapolation') {
          return `Richardson Extrapolation<br/>${props.valueName}: ${params.value[1].toFixed(2)} ${props.valueUnit}`
        }
        const point = props.dataPoints.find(p => p.meshSize === params.value[0])
        if (!point) return ''
        const error = props.richardsonExtrapolation
          ? ((Math.abs(point.value - props.richardsonExtrapolation) / props.richardsonExtrapolation) * 100).toFixed(2)
          : 'N/A'
        return `Mesh Size: ${point.meshSize.toFixed(2)} mm<br/>DOF: ${point.dof.toLocaleString()}<br/>${props.valueName}: ${point.value.toFixed(2)} ${props.valueUnit}<br/>Error: ${error}%`
      },
    },
    legend: {
      data: series.map(s => s.name),
      top: 5,
      textStyle: { color: '#8b949e' },
    },
    grid: { left: 70, right: 30, top: 50, bottom: 50 },
    xAxis: {
      type: 'value',
      name: 'Mesh Size (mm)',
      nameLocation: 'center',
      nameGap: 35,
      min: 0,
      inverse: true,  // Smaller mesh (finer) on the right
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    yAxis: {
      type: 'value',
      name: `${props.valueName} (${props.valueUnit})`,
      nameLocation: 'center',
      nameGap: 50,
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    series: series.map((s, i) => ({
      ...s,
      markLine: i === 0 && markLines.length > 0 ? {
        silent: true,
        symbol: 'none',
        data: markLines,
      } : undefined,
    })),
  }
})
</script>

<style scoped>
.convergence-chart { width: 100%; height: 350px; }
</style>
