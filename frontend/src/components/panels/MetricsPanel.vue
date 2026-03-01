<template>
  <div class="metrics-panel">
    <!-- No Active Run -->
    <div v-if="!simulationStore.activeRun" class="mp-empty">
      <span class="mp-empty-icon">&#9776;</span>
      <span class="mp-empty-text">{{ t('metricsPanel.noActiveRun') }}</span>
    </div>

    <template v-else>
      <!-- Metric Cards Grid -->
      <div class="mp-cards">
        <div
          v-for="metric in displayMetrics"
          :key="metric.name"
          class="mp-card"
          :class="statusClass(metric.status)"
        >
          <div class="mp-card-header">
            <span class="mp-card-name">{{ t(`metricsPanel.metrics.${metric.name}`, metric.name) }}</span>
            <span class="mp-card-status-icon" :class="statusIconClass(metric.status)">
              {{ statusIcon(metric.status) }}
            </span>
          </div>
          <div class="mp-card-value">
            <span class="mp-card-number">{{ formatValue(metric.value) }}</span>
            <span v-if="metric.unit" class="mp-card-unit">{{ metric.unit }}</span>
          </div>
        </div>
      </div>

      <!-- Radar Chart -->
      <div class="mp-chart-section">
        <h4 class="mp-chart-title">{{ t('metricsPanel.radarChart') }}</h4>
        <div class="mp-chart-container">
          <v-chart :option="radarOption" autoresize class="mp-chart" />
        </div>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useSimulationStore } from '@/stores/simulation'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { RadarChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([RadarChart, TooltipComponent, LegendComponent, CanvasRenderer])

const { t } = useI18n()
const simulationStore = useSimulationStore()

// Status thresholds
const thresholds: Record<string, { green: (v: number) => boolean; yellow: (v: number) => boolean }> = {
  frequency_deviation_pct: {
    green: (v) => Math.abs(v) < 1,
    yellow: (v) => Math.abs(v) < 2,
  },
  amplitude_uniformity: {
    green: (v) => v > 0.85,
    yellow: (v) => v > 0.7,
  },
  stress_safety_factor: {
    green: (v) => v > 2,
    yellow: (v) => v > 1.5,
  },
  max_von_mises_stress: {
    green: (v) => v < 200,
    yellow: (v) => v < 350,
  },
  modal_frequency_hz: {
    green: () => true,
    yellow: () => true,
  },
  gain_ratio: {
    green: (v) => v > 1.0 && v < 3.0,
    yellow: (v) => v > 0.5 && v < 4.0,
  },
  contact_pressure_uniformity: {
    green: (v) => v > 0.85,
    yellow: (v) => v > 0.7,
  },
  thermal_rise_c: {
    green: (v) => v < 80,
    yellow: (v) => v < 150,
  },
  fatigue_cycles: {
    green: (v) => v > 1e7,
    yellow: (v) => v > 1e6,
  },
  energy_efficiency: {
    green: (v) => v > 0.8,
    yellow: (v) => v > 0.6,
  },
  weld_strength_estimate: {
    green: (v) => v > 100,
    yellow: (v) => v > 60,
  },
  coupling_loss_db: {
    green: (v) => v < 1.5,
    yellow: (v) => v < 3.0,
  },
}

type MetricStatus = 'pass' | 'warn' | 'fail'

interface DisplayMetric {
  name: string
  value: number
  unit: string | null
  status: MetricStatus
}

function getStatus(name: string, value: number): MetricStatus {
  const t = thresholds[name]
  if (!t) return 'pass'
  if (t.green(value)) return 'pass'
  if (t.yellow(value)) return 'warn'
  return 'fail'
}

const displayMetrics = computed<DisplayMetric[]>(() => {
  const metrics = simulationStore.activeRunMetrics
  if (metrics.length === 0) {
    // Show placeholder metrics
    return defaultMetricNames.map((name) => ({
      name,
      value: 0,
      unit: null,
      status: 'pass' as MetricStatus,
    }))
  }
  return metrics.map((m) => ({
    name: m.metric_name,
    value: m.value,
    unit: m.unit,
    status: getStatus(m.metric_name, m.value),
  }))
})

const defaultMetricNames = [
  'frequency_deviation_pct',
  'amplitude_uniformity',
  'stress_safety_factor',
  'max_von_mises_stress',
  'modal_frequency_hz',
  'gain_ratio',
  'contact_pressure_uniformity',
  'thermal_rise_c',
  'fatigue_cycles',
  'energy_efficiency',
  'weld_strength_estimate',
  'coupling_loss_db',
]

function formatValue(value: number): string {
  if (value === 0) return '-'
  if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(1) + 'M'
  if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(1) + 'k'
  if (Math.abs(value) < 0.01) return value.toExponential(2)
  return value.toFixed(value < 10 ? 3 : 1)
}

function statusIcon(status: MetricStatus): string {
  switch (status) {
    case 'pass': return '\u2713'
    case 'warn': return '\u26A0'
    case 'fail': return '\u2717'
  }
}

function statusClass(status: MetricStatus) {
  return {
    'mp-card--pass': status === 'pass',
    'mp-card--warn': status === 'warn',
    'mp-card--fail': status === 'fail',
  }
}

function statusIconClass(status: MetricStatus) {
  return {
    'mp-status--pass': status === 'pass',
    'mp-status--warn': status === 'warn',
    'mp-status--fail': status === 'fail',
  }
}

// Radar chart
function normalizeMetric(name: string, value: number): number {
  // Normalize to 0-1 range using reasonable reference values
  const ranges: Record<string, [number, number]> = {
    frequency_deviation_pct: [0, 5],
    amplitude_uniformity: [0, 1],
    stress_safety_factor: [0, 5],
    max_von_mises_stress: [0, 500],
    modal_frequency_hz: [15000, 45000],
    gain_ratio: [0, 4],
    contact_pressure_uniformity: [0, 1],
    thermal_rise_c: [0, 200],
    fatigue_cycles: [0, 1e8],
    energy_efficiency: [0, 1],
    weld_strength_estimate: [0, 200],
    coupling_loss_db: [0, 5],
  }
  const range = ranges[name]
  if (!range) return 0.5
  const [min, max] = range
  // For metrics where lower is better, invert
  const invertMetrics = ['frequency_deviation_pct', 'max_von_mises_stress', 'thermal_rise_c', 'coupling_loss_db']
  let normalized = (value - min) / (max - min)
  if (invertMetrics.includes(name)) {
    normalized = 1 - normalized
  }
  return Math.max(0, Math.min(1, normalized))
}

const radarOption = computed(() => {
  const indicators = displayMetrics.value.map((m) => ({
    name: m.name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()).slice(0, 16),
    max: 1,
  }))

  const data = displayMetrics.value.map((m) => normalizeMetric(m.name, m.value))

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
    },
    radar: {
      indicator: indicators,
      shape: 'polygon',
      splitNumber: 4,
      axisName: {
        color: '#8b949e',
        fontSize: 9,
      },
      splitLine: {
        lineStyle: { color: 'rgba(139, 148, 158, 0.2)' },
      },
      splitArea: {
        areaStyle: {
          color: ['rgba(88, 166, 255, 0.02)', 'rgba(88, 166, 255, 0.04)'],
        },
      },
      axisLine: {
        lineStyle: { color: 'rgba(139, 148, 158, 0.15)' },
      },
    },
    series: [
      {
        type: 'radar',
        data: [
          {
            value: data,
            name: t('metricsPanel.currentRun'),
            areaStyle: {
              color: 'rgba(88, 166, 255, 0.15)',
            },
            lineStyle: {
              color: '#58a6ff',
              width: 1.5,
            },
            itemStyle: {
              color: '#58a6ff',
            },
          },
        ],
      },
    ],
  }
})
</script>

<style scoped>
.metrics-panel {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  font-size: 12px;
}

.mp-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 8px;
  color: var(--color-text-secondary);
}

.mp-empty-icon {
  font-size: 24px;
  opacity: 0.5;
}

.mp-empty-text {
  font-size: 12px;
  font-style: italic;
}

/* Metric Cards */
.mp-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
  padding: 8px;
}

.mp-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px 10px;
  transition: border-color 0.2s;
}

.mp-card--pass {
  border-left: 3px solid var(--color-success);
}

.mp-card--warn {
  border-left: 3px solid var(--color-warning);
}

.mp-card--fail {
  border-left: 3px solid var(--color-danger);
}

.mp-card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 4px;
  margin-bottom: 4px;
}

.mp-card-name {
  font-size: 10px;
  color: var(--color-text-secondary);
  line-height: 1.2;
  word-break: break-word;
}

.mp-card-status-icon {
  font-size: 12px;
  flex-shrink: 0;
}

.mp-status--pass {
  color: var(--color-success);
}

.mp-status--warn {
  color: var(--color-warning);
}

.mp-status--fail {
  color: var(--color-danger);
}

.mp-card-value {
  display: flex;
  align-items: baseline;
  gap: 3px;
}

.mp-card-number {
  font-size: 16px;
  font-weight: 700;
  font-family: ui-monospace, monospace;
  color: var(--color-text-primary);
}

.mp-card-unit {
  font-size: 10px;
  color: var(--color-text-secondary);
}

/* Chart */
.mp-chart-section {
  padding: 8px;
  border-top: 1px solid var(--color-border);
}

.mp-chart-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 8px;
}

.mp-chart-container {
  width: 100%;
  height: 280px;
}

.mp-chart {
  width: 100%;
  height: 100%;
}
</style>
