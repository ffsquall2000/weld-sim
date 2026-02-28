<!-- frontend/src/components/charts/ParetoChart.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="pareto-chart" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { ScatterChart, LineChart } from 'echarts/charts'
import { GridComponent, TooltipComponent, LegendComponent, MarkPointComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([ScatterChart, LineChart, GridComponent, TooltipComponent, LegendComponent, MarkPointComponent, CanvasRenderer])

export interface ParetoPoint {
  x: number           // Objective 1 value
  y: number           // Objective 2 value
  label?: string      // Optional point label
  isOptimal?: boolean // Part of Pareto front
}

const props = withDefaults(defineProps<{
  points: ParetoPoint[]
  xAxisName?: string
  yAxisName?: string
  xUnit?: string
  yUnit?: string
  selectedIndex?: number  // Index of selected/highlighted point
}>(), {
  points: () => [],
  xAxisName: 'Objective 1',
  yAxisName: 'Objective 2',
  xUnit: '',
  yUnit: '',
})

defineEmits<{
  (e: 'pointSelect', index: number): void
}>()

const paretoFront = computed(() => {
  return props.points.filter(p => p.isOptimal !== false)
})

const dominatedPoints = computed(() => {
  return props.points.filter(p => p.isOptimal === false)
})

const chartOption = computed(() => {
  const series: any[] = []

  // Pareto front line (connecting optimal points, sorted by x)
  const sortedFront = [...paretoFront.value].sort((a, b) => a.x - b.x)
  if (sortedFront.length > 0) {
    series.push({
      name: 'Pareto Front',
      type: 'line',
      data: sortedFront.map(p => [p.x, p.y]),
      lineStyle: { color: '#ff9800', width: 2, type: 'dashed' },
      itemStyle: { color: '#ff9800' },
      symbol: 'circle',
      symbolSize: 10,
    })
  }

  // Pareto optimal points (scatter for interaction)
  if (paretoFront.value.length > 0) {
    series.push({
      name: 'Optimal Solutions',
      type: 'scatter',
      data: paretoFront.value.map((p) => ({
        value: [p.x, p.y],
        itemStyle: {
          color: props.selectedIndex === props.points.indexOf(p) ? '#3fb950' : '#ff9800',
          borderColor: '#fff',
          borderWidth: 2,
        },
      })),
      symbolSize: 14,
      emphasis: {
        itemStyle: {
          color: '#3fb950',
          borderColor: '#fff',
          borderWidth: 3,
        },
      },
    })
  }

  // Dominated points
  if (dominatedPoints.value.length > 0) {
    series.push({
      name: 'Dominated Solutions',
      type: 'scatter',
      data: dominatedPoints.value.map(p => [p.x, p.y]),
      symbolSize: 8,
      itemStyle: { color: '#8b949e', opacity: 0.6 },
    })
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        const x = params.value[0]
        const y = params.value[1]
        const point = props.points.find(p => p.x === x && p.y === y)
        const labelStr = point?.label ? `<br/>${point.label}` : ''
        return `${params.seriesName}${labelStr}<br/>${props.xAxisName}: ${x.toFixed(2)} ${props.xUnit}<br/>${props.yAxisName}: ${y.toFixed(2)} ${props.yUnit}`
      },
    },
    legend: {
      data: ['Pareto Front', 'Optimal Solutions', 'Dominated Solutions'],
      top: 5,
      textStyle: { color: '#8b949e' },
    },
    grid: { left: 70, right: 30, top: 50, bottom: 50 },
    xAxis: {
      type: 'value',
      name: `${props.xAxisName} (${props.xUnit})`,
      nameLocation: 'center',
      nameGap: 35,
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    yAxis: {
      type: 'value',
      name: `${props.yAxisName} (${props.yUnit})`,
      nameLocation: 'center',
      nameGap: 50,
      axisLabel: { color: '#8b949e' },
      axisLine: { lineStyle: { color: '#30363d' } },
      nameTextStyle: { color: '#8b949e' },
    },
    series,
  }
})
</script>

<style scoped>
.pareto-chart { width: 100%; height: 350px; }
</style>
