<!-- frontend/src/components/charts/SafetyGauge.vue -->
<template>
  <v-chart :option="chartOption" autoresize class="safety-gauge" />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { GaugeChart } from 'echarts/charts'
import { TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([GaugeChart, TooltipComponent, CanvasRenderer])

const props = withDefaults(defineProps<{
  value: number
  min?: number
  max?: number
  title?: string
  unit?: string
  warningThreshold?: number  // percentage (0-100) where yellow starts
  dangerThreshold?: number   // percentage (0-100) where red starts
}>(), {
  min: 0,
  max: 100,
  title: '',
  unit: '',
  warningThreshold: 60,
  dangerThreshold: 80,
})

const normalizedValue = computed(() => {
  const range = props.max - props.min
  if (range === 0) return 0
  return ((props.value - props.min) / range) * 100
})

const chartOption = computed(() => ({
  backgroundColor: 'transparent',
  tooltip: {
    formatter: `{b}: ${props.value.toFixed(2)} ${props.unit}`,
  },
  series: [{
    type: 'gauge',
    startAngle: 200,
    endAngle: -20,
    min: 0,
    max: 100,
    splitNumber: 10,
    radius: '90%',
    center: ['50%', '55%'],
    axisLine: {
      lineStyle: {
        width: 20,
        color: [
          [props.warningThreshold / 100, '#3fb950'],  // green
          [props.dangerThreshold / 100, '#d29922'],   // yellow
          [1, '#f44336'],                              // red
        ],
      },
    },
    pointer: {
      itemStyle: {
        color: 'auto',
      },
      width: 5,
      length: '60%',
    },
    axisTick: {
      distance: -25,
      length: 8,
      lineStyle: { color: '#8b949e', width: 1 },
    },
    splitLine: {
      distance: -30,
      length: 14,
      lineStyle: { color: '#8b949e', width: 2 },
    },
    axisLabel: {
      color: '#8b949e',
      distance: 35,
      fontSize: 10,
      formatter: (value: number) => {
        // Map 0-100 back to actual min-max range
        const actualValue = props.min + (value / 100) * (props.max - props.min)
        return actualValue.toFixed(0)
      },
    },
    anchor: {
      show: true,
      showAbove: true,
      size: 15,
      itemStyle: {
        borderWidth: 3,
        borderColor: '#30363d',
      },
    },
    title: {
      show: true,
      offsetCenter: [0, '70%'],
      fontSize: 14,
      color: '#8b949e',
    },
    detail: {
      valueAnimation: true,
      fontSize: 20,
      fontWeight: 'bold',
      offsetCenter: [0, '45%'],
      formatter: `${props.value.toFixed(1)} ${props.unit}`,
      color: '#c9d1d9',
    },
    data: [{
      value: normalizedValue.value,
      name: props.title,
    }],
  }],
}))
</script>

<style scoped>
.safety-gauge { width: 100%; height: 250px; }
</style>
