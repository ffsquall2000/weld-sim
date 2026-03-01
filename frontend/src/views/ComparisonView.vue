<template>
  <div class="comparison-view">
    <!-- Header -->
    <div class="cv-header">
      <div class="cv-header-left">
        <button class="cv-back-btn" @click="router.back()">&#8592;</button>
        <h1 class="cv-title">{{ t('comparison.title') }}</h1>
      </div>
      <div class="cv-header-right">
        <button class="cv-export-btn" @click="handleExport('pdf')">
          &#128196; {{ t('comparison.exportPdf') }}
        </button>
        <button class="cv-export-btn" @click="handleExport('excel')">
          &#128202; {{ t('comparison.exportExcel') }}
        </button>
      </div>
    </div>

    <!-- Run Selector -->
    <div class="cv-selector">
      <label class="cv-selector-label">{{ t('comparison.selectRuns') }}</label>
      <div class="cv-selector-chips">
        <div
          v-for="run in simulationStore.runs"
          :key="run.id"
          class="cv-chip"
          :class="{ 'cv-chip--selected': selectedRunIds.has(run.id) }"
          @click="toggleRunSelection(run.id)"
        >
          <span class="cv-chip-label">{{ run.id.slice(0, 8) }}</span>
          <span class="cv-chip-status" :class="chipStatusClass(run.status)">
            {{ run.status }}
          </span>
        </div>
        <div v-if="simulationStore.runs.length === 0" class="cv-no-runs">
          {{ t('comparison.noRunsAvailable') }}
        </div>
      </div>
    </div>

    <!-- Content -->
    <div v-if="selectedRuns.length >= 2" class="cv-content">
      <!-- Quality Score Cards -->
      <section class="cv-section">
        <h3 class="cv-section-title">{{ t('metrics.qualityScore') }}</h3>
        <div class="cv-quality-cards">
          <div
            v-for="run in selectedRuns"
            :key="'qs-' + run.id"
            class="cv-quality-card"
          >
            <div class="cv-quality-card-header">
              <span class="cv-quality-run-id">{{ run.id.slice(0, 8) }}</span>
            </div>
            <div class="cv-quality-score-ring">
              <svg viewBox="0 0 80 80" class="cv-score-svg">
                <circle cx="40" cy="40" r="34" fill="none" stroke="rgba(139,148,158,0.15)" stroke-width="6" />
                <circle
                  cx="40" cy="40" r="34"
                  fill="none"
                  :stroke="qualityScoreColor(getQualityScore(run.id))"
                  stroke-width="6"
                  stroke-linecap="round"
                  :stroke-dasharray="qualityScoreDash(getQualityScore(run.id))"
                  transform="rotate(-90 40 40)"
                />
              </svg>
              <span class="cv-score-value">{{ getQualityScore(run.id).toFixed(1) }}</span>
            </div>
          </div>
        </div>
      </section>

      <!-- Metric Comparison Table -->
      <section class="cv-section">
        <h3 class="cv-section-title">{{ t('comparison.metricComparison') }}</h3>
        <div class="cv-table-wrapper">
          <table class="cv-table">
            <thead>
              <tr>
                <th class="cv-th-metric">{{ t('comparison.metric') }}</th>
                <th v-for="run in selectedRuns" :key="run.id" class="cv-th-run">
                  {{ run.id.slice(0, 8) }}
                </th>
                <th v-if="selectedRuns.length === 2" class="cv-th-delta">
                  {{ t('comparison.delta') }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="metric in comparisonMetrics" :key="metric.name">
                <td class="cv-td-metric" :title="getMetricDescription(metric.name)">
                  {{ getMetricLabel(metric.name) }}
                  <span v-if="getMetricUnit(metric.name)" class="cv-metric-unit-label">
                    ({{ getMetricUnit(metric.name) }})
                  </span>
                </td>
                <td
                  v-for="(value, idx) in metric.values"
                  :key="idx"
                  class="cv-td-value"
                >
                  {{ formatValue(value) }}
                  <span v-if="metric.units[idx]" class="cv-unit">{{ metric.units[idx] }}</span>
                </td>
                <td v-if="selectedRuns.length === 2 && metric.delta !== null" class="cv-td-delta">
                  <span :class="deltaClass(metric.delta, metric.name)">
                    {{ metric.delta > 0 ? '+' : '' }}{{ metric.delta.toFixed(3) }}
                    ({{ metric.deltaPct.toFixed(1) }}%)
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <!-- Standardized Metrics Radar Chart -->
      <section class="cv-section">
        <h3 class="cv-section-title">{{ t('metrics.standardTitle') }}</h3>
        <div class="cv-chart-container">
          <v-chart :option="standardRadarOption" autoresize class="cv-chart" />
        </div>
      </section>

      <!-- Radar Chart Overlay -->
      <section class="cv-section">
        <h3 class="cv-section-title">{{ t('comparison.radarOverlay') }}</h3>
        <div class="cv-chart-container">
          <v-chart :option="radarOption" autoresize class="cv-chart" />
        </div>
      </section>
    </div>

    <!-- Not Enough Runs Selected -->
    <div v-else class="cv-empty">
      <div class="cv-empty-icon">&#9776;</div>
      <h3 class="cv-empty-title">{{ t('comparison.selectAtLeast') }}</h3>
      <p class="cv-empty-desc">{{ t('comparison.selectAtLeastDesc') }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useSimulationStore, type Metric } from '@/stores/simulation'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { RadarChart } from 'echarts/charts'
import { TooltipComponent, LegendComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([RadarChart, TooltipComponent, LegendComponent, CanvasRenderer])

const { t, te } = useI18n()
const router = useRouter()
const simulationStore = useSimulationStore()

const selectedRunIds = ref(new Set<string>())

function toggleRunSelection(id: string) {
  const newSet = new Set(selectedRunIds.value)
  if (newSet.has(id)) {
    newSet.delete(id)
  } else {
    newSet.add(id)
  }
  selectedRunIds.value = newSet
}

const selectedRuns = computed(() =>
  simulationStore.runs.filter((r) => selectedRunIds.value.has(r.id))
)

// Fetch standard metrics when runs are selected
watch(selectedRunIds, async (ids) => {
  for (const id of ids) {
    if (!simulationStore.standardMetricsCache[id]) {
      await simulationStore.fetchStandardMetrics(id)
    }
  }
}, { deep: true })

// Quality score helpers
function getQualityScore(runId: string): number {
  const cached = simulationStore.standardMetricsCache[runId]
  return cached?.quality_score ?? 0
}

function qualityScoreColor(score: number): string {
  if (score >= 80) return '#4caf50'
  if (score >= 60) return '#ff9800'
  if (score >= 40) return '#ffc107'
  return '#f44336'
}

function qualityScoreDash(score: number): string {
  const circumference = 2 * Math.PI * 34
  const filled = (score / 100) * circumference
  return `${filled} ${circumference - filled}`
}

// Metric info helpers
function getMetricLabel(metricName: string): string {
  const key = `metrics.${metricName}`
  if (te(key)) return t(key)
  return metricName.replace(/_/g, ' ')
}

function getMetricDescription(metricName: string): string {
  // Check standardMetricsCache for metric_info
  for (const runId of selectedRunIds.value) {
    const cached = simulationStore.standardMetricsCache[runId]
    if (cached?.metric_info?.[metricName]?.description) {
      return cached.metric_info[metricName].description
    }
  }
  return metricName.replace(/_/g, ' ')
}

function getMetricUnit(metricName: string): string {
  for (const runId of selectedRunIds.value) {
    const cached = simulationStore.standardMetricsCache[runId]
    if (cached?.metric_info?.[metricName]?.unit) {
      return cached.metric_info[metricName].unit
    }
  }
  return ''
}

// Build comparison table data
interface ComparisonMetric {
  name: string
  values: number[]
  units: (string | null)[]
  delta: number | null
  deltaPct: number
}

const comparisonMetrics = computed<ComparisonMetric[]>(() => {
  if (selectedRuns.value.length < 2) return []

  // Collect all unique metric names
  const allNames = new Set<string>()
  for (const run of selectedRuns.value) {
    for (const m of run.metrics || []) {
      allNames.add(m.metric_name)
    }
  }

  return Array.from(allNames).map((name) => {
    const values: number[] = []
    const units: (string | null)[] = []

    for (const run of selectedRuns.value) {
      const metric = (run.metrics || []).find((m: Metric) => m.metric_name === name)
      values.push(metric?.value ?? 0)
      units.push(metric?.unit ?? null)
    }

    let delta: number | null = null
    let deltaPct = 0
    if (selectedRuns.value.length === 2) {
      delta = (values[1] ?? 0) - (values[0] ?? 0)
      deltaPct = (values[0] ?? 0) !== 0 ? (delta / Math.abs(values[0] ?? 0)) * 100 : 0
    }

    return { name, values, units, delta, deltaPct }
  })
})

// Metrics where higher is better
const higherBetterMetrics = new Set([
  'amplitude_uniformity',
  'stress_safety_factor',
  'energy_efficiency',
  'energy_coupling_efficiency',
  'weld_strength_estimate',
  'fatigue_cycles',
  'fatigue_cycle_estimate',
  'gain_ratio',
  'horn_gain',
  'contact_pressure_uniformity',
  'effective_contact_area_mm2',
  'modal_separation_hz',
  'natural_frequency_hz',
])

function deltaClass(delta: number, metricName: string) {
  const higherIsBetter = higherBetterMetrics.has(metricName)
  const isImproved = higherIsBetter ? delta > 0 : delta < 0
  const isRegressed = higherIsBetter ? delta < 0 : delta > 0

  return {
    'cv-delta--improved': isImproved,
    'cv-delta--regressed': isRegressed,
    'cv-delta--neutral': delta === 0,
  }
}

function chipStatusClass(status: string) {
  return {
    'cv-chip-status--completed': status === 'completed',
    'cv-chip-status--running': status === 'running',
    'cv-chip-status--failed': status === 'failed',
  }
}

function formatValue(value: number): string {
  if (value === 0) return '-'
  if (Math.abs(value) >= 1e6) return (value / 1e6).toFixed(2) + 'M'
  if (Math.abs(value) >= 1e3) return (value / 1e3).toFixed(2) + 'k'
  if (Math.abs(value) < 0.01) return value.toExponential(2)
  return value.toFixed(value < 10 ? 4 : 2)
}

// Normalization for radar
const metricRanges: Record<string, [number, number]> = {
  frequency_deviation_pct: [0, 5],
  amplitude_uniformity: [0, 1],
  stress_safety_factor: [0, 5],
  max_von_mises_stress: [0, 500],
  max_von_mises_stress_mpa: [0, 500],
  gain_ratio: [0, 4],
  horn_gain: [0, 4],
  contact_pressure_uniformity: [0, 1],
  thermal_rise_c: [0, 200],
  max_temperature_rise_c: [0, 200],
  energy_efficiency: [0, 1],
  energy_coupling_efficiency: [0, 1],
  weld_strength_estimate: [0, 200],
  coupling_loss_db: [0, 5],
  effective_contact_area_mm2: [0, 3200],
  modal_separation_hz: [0, 5000],
}

const invertMetrics = new Set([
  'frequency_deviation_pct',
  'max_von_mises_stress',
  'max_von_mises_stress_mpa',
  'thermal_rise_c',
  'max_temperature_rise_c',
  'coupling_loss_db',
])

function normalize(name: string, value: number): number {
  const range = metricRanges[name]
  if (!range) return 0.5
  let n = (value - range[0]) / (range[1] - range[0])
  if (invertMetrics.has(name)) n = 1 - n
  return Math.max(0, Math.min(1, n))
}

const colors = ['#58a6ff', '#ff9800', '#4caf50', '#f44336', '#ab47bc', '#26c6da']

// Standardized metrics radar chart (from the standard metrics endpoint)
const standardMetricKeys = [
  'amplitude_uniformity',
  'stress_safety_factor',
  'energy_coupling_efficiency',
  'frequency_deviation_pct',
  'contact_pressure_uniformity',
  'horn_gain',
  'max_von_mises_stress_mpa',
  'max_temperature_rise_c',
]

const standardRadarOption = computed(() => {
  if (selectedRuns.value.length < 2) return {}

  const indicators = standardMetricKeys.map((key) => ({
    name: getMetricLabel(key),
    max: 1,
  }))

  const series = selectedRuns.value.map((run, idx) => {
    const cached = simulationStore.standardMetricsCache[run.id]
    const metricsData = cached?.metrics || {}
    const data = standardMetricKeys.map((key) => {
      const val = metricsData[key] ?? 0
      return normalize(key, val)
    })
    return {
      value: data,
      name: run.id.slice(0, 8),
      areaStyle: {
        color: colors[idx % colors.length]!.replace(')', ', 0.1)').replace('rgb', 'rgba'),
      },
      lineStyle: {
        color: colors[idx % colors.length]!,
        width: 1.5,
      },
      itemStyle: {
        color: colors[idx % colors.length]!,
      },
    }
  })

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (!params.value) return ''
        const firstRunId = selectedRuns.value[0]?.id
        const lines = standardMetricKeys.map((key, i) => {
          const cached = firstRunId ? simulationStore.standardMetricsCache[firstRunId] : undefined
          const unit = cached?.metric_info?.[key]?.unit || ''
          return `${getMetricLabel(key)}: ${(params.value[i] * 100).toFixed(1)}%${unit ? ' (' + unit + ')' : ''}`
        })
        return `<strong>${params.name}</strong><br/>` + lines.join('<br/>')
      },
    },
    legend: {
      data: selectedRuns.value.map((r) => r.id.slice(0, 8)),
      textStyle: { color: '#8b949e', fontSize: 11 },
      bottom: 0,
    },
    radar: {
      indicator: indicators,
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: '#8b949e', fontSize: 9 },
      splitLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.2)' } },
      splitArea: {
        areaStyle: { color: ['rgba(88, 166, 255, 0.02)', 'rgba(88, 166, 255, 0.04)'] },
      },
      axisLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.15)' } },
    },
    series: [{ type: 'radar', data: series }],
  }
})

const radarOption = computed(() => {
  if (comparisonMetrics.value.length === 0) return {}

  const indicators = comparisonMetrics.value.map((m) => ({
    name: m.name.replace(/_/g, ' ').slice(0, 16),
    max: 1,
  }))

  const series = selectedRuns.value.map((run, idx) => {
    const data = comparisonMetrics.value.map((m) => normalize(m.name, m.values[idx] ?? 0))
    return {
      value: data,
      name: run.id.slice(0, 8),
      areaStyle: {
        color: colors[idx % colors.length]!.replace(')', ', 0.1)').replace('rgb', 'rgba'),
      },
      lineStyle: {
        color: colors[idx % colors.length]!,
        width: 1.5,
      },
      itemStyle: {
        color: colors[idx % colors.length]!,
      },
    }
  })

  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'item' },
    legend: {
      data: selectedRuns.value.map((r) => r.id.slice(0, 8)),
      textStyle: { color: '#8b949e', fontSize: 11 },
      bottom: 0,
    },
    radar: {
      indicator: indicators,
      shape: 'polygon',
      splitNumber: 4,
      axisName: { color: '#8b949e', fontSize: 9 },
      splitLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.2)' } },
      splitArea: {
        areaStyle: { color: ['rgba(88, 166, 255, 0.02)', 'rgba(88, 166, 255, 0.04)'] },
      },
      axisLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.15)' } },
    },
    series: [{ type: 'radar', data: series }],
  }
})

function handleExport(format: string) {
  // Placeholder - export implementation would depend on backend
  console.log(`Export comparison as ${format}`)
}
</script>

<style scoped>
.comparison-view {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* Header */
.cv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}

.cv-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.cv-back-btn {
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 18px;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
}

.cv-back-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.cv-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
}

.cv-header-right {
  display: flex;
  gap: 8px;
}

.cv-export-btn {
  padding: 6px 14px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: none;
  color: var(--color-text-secondary);
  font-size: 12px;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.cv-export-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

/* Run Selector */
.cv-selector {
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-bg-primary);
}

.cv-selector-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
  display: block;
}

.cv-selector-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.cv-chip {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border: 1px solid var(--color-border);
  border-radius: 16px;
  background-color: var(--color-bg-card);
  cursor: pointer;
  transition: border-color 0.15s, background-color 0.15s;
}

.cv-chip:hover {
  border-color: var(--color-accent-blue);
}

.cv-chip--selected {
  border-color: var(--color-accent-blue);
  background-color: rgba(88, 166, 255, 0.1);
}

.cv-chip-label {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  color: var(--color-text-primary);
}

.cv-chip-status {
  font-size: 9px;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 600;
  text-transform: uppercase;
}

.cv-chip-status--completed {
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
}

.cv-chip-status--running {
  background-color: rgba(88, 166, 255, 0.15);
  color: var(--color-accent-blue);
}

.cv-chip-status--failed {
  background-color: rgba(244, 67, 54, 0.15);
  color: var(--color-danger);
}

.cv-no-runs {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 8px 0;
}

/* Content */
.cv-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
}

.cv-section {
  margin-bottom: 24px;
}

.cv-section-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 12px;
}

/* Quality Score Cards */
.cv-quality-cards {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 8px;
}

.cv-quality-card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 12px 16px;
  min-width: 130px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.cv-quality-card-header {
  display: flex;
  align-items: center;
  gap: 6px;
}

.cv-quality-run-id {
  font-family: ui-monospace, monospace;
  font-size: 12px;
  color: var(--color-text-secondary);
  font-weight: 600;
}

.cv-quality-score-ring {
  position: relative;
  width: 80px;
  height: 80px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.cv-score-svg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.cv-score-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--color-text-primary);
  z-index: 1;
}

/* Metric unit label in table */
.cv-metric-unit-label {
  font-size: 10px;
  color: var(--color-text-secondary);
  font-weight: 400;
}

/* Comparison Table */
.cv-table-wrapper {
  overflow-x: auto;
}

.cv-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.cv-table th {
  text-align: left;
  padding: 8px 12px;
  font-weight: 600;
  color: var(--color-text-secondary);
  border-bottom: 2px solid var(--color-border);
  font-size: 11px;
  white-space: nowrap;
}

.cv-table td {
  padding: 7px 12px;
  color: var(--color-text-primary);
  border-bottom: 1px solid rgba(48, 54, 61, 0.3);
}

.cv-th-metric {
  min-width: 160px;
}

.cv-th-run {
  font-family: ui-monospace, monospace;
}

.cv-th-delta {
  min-width: 120px;
}

.cv-td-metric {
  font-weight: 500;
  color: var(--color-text-secondary);
  cursor: help;
}

.cv-td-value {
  font-family: ui-monospace, monospace;
}

.cv-unit {
  font-size: 10px;
  color: var(--color-text-secondary);
  margin-left: 2px;
}

.cv-td-delta {
  font-family: ui-monospace, monospace;
  font-size: 11px;
}

.cv-delta--improved {
  color: var(--color-success);
}

.cv-delta--regressed {
  color: var(--color-danger);
}

.cv-delta--neutral {
  color: var(--color-text-secondary);
}

/* Chart */
.cv-chart-container {
  height: 360px;
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 10px;
}

.cv-chart {
  width: 100%;
  height: 100%;
}

/* Empty */
.cv-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  text-align: center;
  padding: 40px;
}

.cv-empty-icon {
  font-size: 40px;
  opacity: 0.4;
  margin-bottom: 12px;
}

.cv-empty-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 6px;
}

.cv-empty-desc {
  font-size: 13px;
  color: var(--color-text-secondary);
  margin: 0;
}
</style>
