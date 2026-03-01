<template>
  <div class="wqr-container">
    <!-- Overall Verdict Banner -->
    <div class="wqr-verdict" :class="verdictClass">
      <div class="wqr-verdict-icon">{{ verdictIcon }}</div>
      <div class="wqr-verdict-content">
        <div class="wqr-verdict-label">{{ t('weldQuality.overallVerdict') }}</div>
        <div class="wqr-verdict-status">{{ verdictText }}</div>
      </div>
      <div class="wqr-verdict-score">
        <div class="wqr-score-value">{{ qualityScore }}</div>
        <div class="wqr-score-label">/ 100</div>
      </div>
    </div>

    <!-- Criteria Checklist -->
    <div class="wqr-section">
      <h3 class="wqr-section-title">{{ t('weldQuality.criteriaChecklist') }}</h3>
      <div class="wqr-criteria-table">
        <div class="wqr-criteria-header">
          <span class="wqr-criteria-col-status"></span>
          <span class="wqr-criteria-col-name">{{ t('weldQuality.criterion') }}</span>
          <span class="wqr-criteria-col-value">{{ t('weldQuality.actualValue') }}</span>
          <span class="wqr-criteria-col-range">{{ t('weldQuality.requiredRange') }}</span>
          <span class="wqr-criteria-col-badge">{{ t('weldQuality.status') }}</span>
        </div>
        <div
          v-for="criterion in criteriaList"
          :key="criterion.key"
          class="wqr-criteria-row"
        >
          <span class="wqr-criteria-col-status">{{ criterion.icon }}</span>
          <span class="wqr-criteria-col-name">{{ criterion.label }}</span>
          <span class="wqr-criteria-col-value wqr-mono">
            {{ criterion.value !== null ? criterion.value : '--' }} {{ criterion.unit }}
          </span>
          <span class="wqr-criteria-col-range wqr-mono">
            {{ criterion.min }}{{ criterion.unit }} &ndash; {{ criterion.max }}{{ criterion.unit }}
          </span>
          <span class="wqr-criteria-col-badge">
            <span class="wqr-status-badge" :class="'wqr-badge-' + criterion.status">
              {{ criterion.statusText }}
            </span>
          </span>
        </div>
      </div>
    </div>

    <!-- Risk Assessment Summary -->
    <div v-if="riskItems.length > 0" class="wqr-section">
      <h3 class="wqr-section-title">{{ t('weldQuality.riskAssessment') }}</h3>
      <div class="wqr-risk-grid">
        <div
          v-for="(risk, index) in riskItems"
          :key="index"
          class="wqr-risk-card"
          :class="'wqr-risk-' + risk.level"
        >
          <div class="wqr-risk-header">
            <span class="wqr-risk-icon">{{ riskIcon(risk.level) }}</span>
            <span class="wqr-risk-label">{{ risk.name }}</span>
          </div>
          <div class="wqr-risk-level">{{ risk.level }}</div>
          <div v-if="risk.description" class="wqr-risk-desc">{{ risk.description }}</div>
        </div>
      </div>
    </div>

    <!-- Detailed Analysis -->
    <div class="wqr-section">
      <h3 class="wqr-section-title">{{ t('weldQuality.detailedAnalysis') }}</h3>
      <div class="wqr-analysis-grid">
        <!-- Energy Analysis -->
        <div class="wqr-analysis-card">
          <div class="wqr-analysis-header">
            <span>&#x26A1;</span>
            <span>{{ t('weldQuality.energyAnalysis') }}</span>
          </div>
          <div class="wqr-analysis-body">
            <div class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.energyInput') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatNum(params.energy_j) }} J</span>
            </div>
            <div v-if="params.interface_power_w" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.interfacePower') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatNum(params.interface_power_w) }} W</span>
            </div>
            <div v-if="metrics?.energy_efficiency != null" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.efficiency') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatPct(metrics.energy_efficiency) }}</span>
            </div>
          </div>
        </div>

        <!-- Mechanical Analysis -->
        <div class="wqr-analysis-card">
          <div class="wqr-analysis-header">
            <span>&#x2699;&#xFE0F;</span>
            <span>{{ t('weldQuality.mechanicalAnalysis') }}</span>
          </div>
          <div class="wqr-analysis-body">
            <div class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.pressure') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatNum(params.pressure_mpa) }} MPa</span>
            </div>
            <div v-if="metrics?.safety_factor != null" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.safetyFactor') }}</span>
              <span
                class="wqr-analysis-value wqr-mono"
                :class="{ 'wqr-text-danger': metrics.safety_factor < 1.0, 'wqr-text-success': metrics.safety_factor >= 1.5 }"
              >
                {{ formatNum(metrics.safety_factor) }}
              </span>
            </div>
            <div v-if="params.amplitude_um" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.amplitude') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatNum(params.amplitude_um) }} &micro;m</span>
            </div>
          </div>
        </div>

        <!-- Quality Prediction -->
        <div class="wqr-analysis-card">
          <div class="wqr-analysis-header">
            <span>&#x1F4CA;</span>
            <span>{{ t('weldQuality.qualityPrediction') }}</span>
          </div>
          <div class="wqr-analysis-body">
            <div v-if="qualityEstimate?.weld_strength != null" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.weldStrength') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatNum(qualityEstimate.weld_strength) }} N</span>
            </div>
            <div v-if="qualityEstimate?.failure_mode" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.failureMode') }}</span>
              <span class="wqr-analysis-value">{{ qualityEstimate.failure_mode }}</span>
            </div>
            <div v-if="qualityEstimate?.confidence != null" class="wqr-analysis-row">
              <span class="wqr-analysis-label">{{ t('weldQuality.confidence') }}</span>
              <span class="wqr-analysis-value wqr-mono">{{ formatPct(qualityEstimate.confidence) }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recommendations -->
    <div v-if="recommendationsList.length > 0" class="wqr-section">
      <h3 class="wqr-section-title">{{ t('weldQuality.recommendations') }}</h3>
      <ol class="wqr-rec-list">
        <li
          v-for="(rec, index) in recommendationsList"
          :key="index"
          class="wqr-rec-item"
        >
          <span class="wqr-rec-icon">{{ recSeverityIcon(rec) }}</span>
          <span class="wqr-rec-text">{{ typeof rec === 'string' ? rec : rec.message || rec.text || String(rec) }}</span>
        </li>
      </ol>
    </div>

    <!-- Export Button -->
    <div class="wqr-export">
      <button class="wqr-export-btn" @click="emit('export-report')">
        &#x1F4E4; {{ t('weldQuality.exportReport') }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

interface ResultParameters {
  amplitude_um?: number
  pressure_mpa?: number
  energy_j?: number
  time_ms?: number
  interface_power_w?: number
  [key: string]: unknown
}

interface RiskItem {
  name: string
  level: string
  description?: string
  [key: string]: unknown
}

interface QualityEstimate {
  weld_strength?: number
  failure_mode?: string
  confidence?: number
  quality_score?: number
  [key: string]: unknown
}

interface Recommendation {
  message?: string
  text?: string
  severity?: string
  [key: string]: unknown
}

interface SimulationResults {
  parameters: ResultParameters
  risk_assessment?: RiskItem[] | Record<string, RiskItem> | Record<string, string>
  quality_estimate?: QualityEstimate
  recommendations?: (string | Recommendation)[]
  validation?: Record<string, unknown>
  [key: string]: unknown
}

interface Metrics {
  energy_efficiency?: number
  safety_factor?: number
  [key: string]: unknown
}

const props = withDefaults(defineProps<{
  results: SimulationResults
  metrics?: Metrics
}>(), {
  metrics: undefined,
})

const emit = defineEmits<{
  (e: 'export-report'): void
}>()

const { t } = useI18n()

// ---------- Computed helpers ----------

const params = computed<ResultParameters>(() => props.results.parameters ?? {})

const qualityEstimate = computed<QualityEstimate | undefined>(
  () => props.results.quality_estimate
)

const qualityScore = computed<number>(() => {
  if (qualityEstimate.value?.quality_score != null) {
    return Math.round(qualityEstimate.value.quality_score)
  }
  // Derive a basic score from criteria pass rate
  const passed = criteriaList.value.filter((c) => c.status === 'pass').length
  const total = criteriaList.value.length
  return total > 0 ? Math.round((passed / total) * 100) : 0
})

const verdictClass = computed(() => {
  if (qualityScore.value >= 80) return 'wqr-verdict-pass'
  if (qualityScore.value >= 60) return 'wqr-verdict-warning'
  return 'wqr-verdict-fail'
})

const verdictIcon = computed(() => {
  if (qualityScore.value >= 80) return '\u2705'
  if (qualityScore.value >= 60) return '\u26A0\uFE0F'
  return '\u274C'
})

const verdictText = computed(() => {
  if (qualityScore.value >= 80) return t('weldQuality.verdictPass')
  if (qualityScore.value >= 60) return t('weldQuality.verdictWarning')
  return t('weldQuality.verdictFail')
})

// ---------- Criteria ----------

interface CriterionInfo {
  key: string
  label: string
  value: number | null
  unit: string
  min: number | string
  max: number | string
  status: 'pass' | 'warning' | 'fail'
  statusText: string
  icon: string
}

const criteriaList = computed<CriterionInfo[]>(() => {
  const p = params.value
  const defs: Array<{
    key: string
    labelKey: string
    value: number | undefined
    unit: string
    min: number
    max: number
    warnMin?: number
    warnMax?: number
  }> = [
    {
      key: 'amplitude',
      labelKey: 'weldQuality.criteriaAmplitude',
      value: p.amplitude_um as number | undefined,
      unit: '\u00B5m',
      min: 15,
      max: 50,
      warnMin: 12,
      warnMax: 55,
    },
    {
      key: 'pressure',
      labelKey: 'weldQuality.criteriaPressure',
      value: p.pressure_mpa as number | undefined,
      unit: 'MPa',
      min: 1.0,
      max: 15.0,
      warnMin: 0.5,
      warnMax: 18.0,
    },
    {
      key: 'energy',
      labelKey: 'weldQuality.criteriaEnergy',
      value: p.energy_j as number | undefined,
      unit: 'J',
      min: 50,
      max: 3000,
      warnMin: 20,
      warnMax: 4000,
    },
    {
      key: 'time',
      labelKey: 'weldQuality.criteriaTime',
      value: p.time_ms as number | undefined,
      unit: 'ms',
      min: 100,
      max: 1500,
      warnMin: 50,
      warnMax: 2000,
    },
    {
      key: 'power',
      labelKey: 'weldQuality.criteriaPower',
      value: p.interface_power_w as number | undefined,
      unit: 'W',
      min: 0,
      max: 10000,
      warnMin: 0,
      warnMax: 12000,
    },
  ]

  return defs.map((d) => {
    const val = d.value ?? null
    let status: 'pass' | 'warning' | 'fail' = 'pass'
    if (val === null) {
      status = 'warning'
    } else if (val < d.min || val > d.max) {
      if (d.warnMin !== undefined && d.warnMax !== undefined && val >= d.warnMin && val <= d.warnMax) {
        status = 'warning'
      } else {
        status = 'fail'
      }
    }

    const icon = status === 'pass' ? '\u2705' : status === 'warning' ? '\u26A0\uFE0F' : '\u274C'
    const statusText =
      status === 'pass'
        ? t('weldQuality.statusPass')
        : status === 'warning'
        ? t('weldQuality.statusWarning')
        : t('weldQuality.statusFail')

    return {
      key: d.key,
      label: t(d.labelKey),
      value: val,
      unit: d.unit,
      min: d.min,
      max: d.max,
      status,
      statusText,
      icon,
    }
  })
})

// ---------- Risk Assessment ----------

const riskItems = computed<RiskItem[]>(() => {
  const ra = props.results.risk_assessment
  if (!ra) return []
  if (Array.isArray(ra)) return ra
  return Object.entries(ra).map(([key, item]) => {
    if (typeof item === 'object' && item !== null) {
      return { name: key, ...item } as RiskItem
    }
    return { name: key, level: String(item) }
  })
})

function riskIcon(level: string): string {
  const l = level.toLowerCase()
  if (l === 'low' || l === 'safe') return '\u2705'
  if (l === 'medium' || l === 'moderate') return '\u26A0\uFE0F'
  return '\u274C'
}

// ---------- Recommendations ----------

const recommendationsList = computed<(string | Recommendation)[]>(() => {
  return props.results.recommendations ?? []
})

function recSeverityIcon(rec: string | Recommendation): string {
  if (typeof rec === 'string') return '\u2022'
  const sev = (rec.severity ?? '').toLowerCase()
  if (sev === 'critical' || sev === 'high') return '\u274C'
  if (sev === 'warning' || sev === 'medium') return '\u26A0\uFE0F'
  if (sev === 'info' || sev === 'low') return '\u2139\uFE0F'
  return '\u2022'
}

// ---------- Format helpers ----------

function formatNum(val: unknown): string {
  if (val == null) return '--'
  const num = Number(val)
  if (isNaN(num)) return String(val)
  return Number.isInteger(num) ? num.toString() : num.toFixed(2)
}

function formatPct(val: unknown): string {
  if (val == null) return '--'
  const num = Number(val)
  if (isNaN(num)) return String(val)
  return `${(num * 100).toFixed(1)}%`
}
</script>

<style scoped>
.wqr-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
  font-size: 13px;
}

/* ---------- Verdict Banner ---------- */
.wqr-verdict {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px 20px;
  border-radius: 8px;
  border: 1px solid transparent;
}

.wqr-verdict-pass {
  background-color: rgba(76, 175, 80, 0.1);
  border-color: rgba(76, 175, 80, 0.3);
}

.wqr-verdict-warning {
  background-color: rgba(255, 152, 0, 0.1);
  border-color: rgba(255, 152, 0, 0.3);
}

.wqr-verdict-fail {
  background-color: rgba(244, 67, 54, 0.1);
  border-color: rgba(244, 67, 54, 0.3);
}

.wqr-verdict-icon {
  font-size: 32px;
  line-height: 1;
}

.wqr-verdict-content {
  flex: 1;
}

.wqr-verdict-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--color-text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.wqr-verdict-status {
  font-size: 18px;
  font-weight: 700;
  color: var(--color-text-primary);
  margin-top: 2px;
}

.wqr-verdict-score {
  display: flex;
  align-items: baseline;
  gap: 2px;
}

.wqr-score-value {
  font-size: 36px;
  font-weight: 800;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  color: var(--color-text-primary);
  line-height: 1;
}

.wqr-score-label {
  font-size: 14px;
  color: var(--color-text-secondary);
  font-weight: 500;
}

/* ---------- Sections ---------- */
.wqr-section {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 14px 16px;
}

.wqr-section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 12px 0;
}

/* ---------- Criteria Table ---------- */
.wqr-criteria-table {
  overflow-x: auto;
}

.wqr-criteria-header,
.wqr-criteria-row {
  display: grid;
  grid-template-columns: 28px 1fr 120px 140px 80px;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
}

.wqr-criteria-header {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
}

.wqr-criteria-row {
  font-size: 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--color-border) 50%, transparent);
}

.wqr-criteria-row:last-child {
  border-bottom: none;
}

.wqr-criteria-col-status {
  text-align: center;
  font-size: 14px;
  line-height: 1;
}

.wqr-criteria-col-name {
  color: var(--color-text-primary);
  font-weight: 500;
}

.wqr-criteria-col-value {
  color: var(--color-text-primary);
}

.wqr-criteria-col-range {
  color: var(--color-text-secondary);
  font-size: 11px;
}

.wqr-criteria-col-badge {
  text-align: center;
}

.wqr-mono {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}

.wqr-status-badge {
  display: inline-block;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 10px;
  text-transform: uppercase;
}

.wqr-badge-pass {
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success, #4caf50);
}

.wqr-badge-warning {
  background-color: rgba(255, 152, 0, 0.15);
  color: var(--color-warning, #ff9800);
}

.wqr-badge-fail {
  background-color: rgba(244, 67, 54, 0.15);
  color: var(--color-danger, #f44336);
}

/* ---------- Risk Grid ---------- */
.wqr-risk-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}

.wqr-risk-card {
  padding: 10px 12px;
  border-radius: 6px;
  border: 1px solid transparent;
}

.wqr-risk-low,
.wqr-risk-safe {
  background-color: rgba(76, 175, 80, 0.08);
  border-color: rgba(76, 175, 80, 0.2);
}

.wqr-risk-medium,
.wqr-risk-moderate {
  background-color: rgba(255, 152, 0, 0.08);
  border-color: rgba(255, 152, 0, 0.2);
}

.wqr-risk-high,
.wqr-risk-critical {
  background-color: rgba(244, 67, 54, 0.08);
  border-color: rgba(244, 67, 54, 0.2);
}

.wqr-risk-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.wqr-risk-icon {
  font-size: 14px;
  line-height: 1;
}

.wqr-risk-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.wqr-risk-level {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  color: var(--color-text-secondary);
}

.wqr-risk-desc {
  font-size: 11px;
  color: var(--color-text-secondary);
  line-height: 1.5;
  margin-top: 4px;
}

/* ---------- Analysis Grid ---------- */
.wqr-analysis-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
}

.wqr-analysis-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  overflow: hidden;
}

.wqr-analysis-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
  background-color: color-mix(in srgb, var(--color-border) 30%, var(--color-bg-primary));
  border-bottom: 1px solid var(--color-border);
}

.wqr-analysis-body {
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.wqr-analysis-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
}

.wqr-analysis-label {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.wqr-analysis-value {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.wqr-text-danger {
  color: var(--color-danger, #f44336) !important;
}

.wqr-text-success {
  color: var(--color-success, #4caf50) !important;
}

/* ---------- Recommendations ---------- */
.wqr-rec-list {
  margin: 0;
  padding: 0;
  list-style: none;
  counter-reset: rec-counter;
}

.wqr-rec-item {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 6px 0;
  border-bottom: 1px solid color-mix(in srgb, var(--color-border) 50%, transparent);
  counter-increment: rec-counter;
}

.wqr-rec-item:last-child {
  border-bottom: none;
}

.wqr-rec-icon {
  font-size: 14px;
  line-height: 1.4;
  flex-shrink: 0;
}

.wqr-rec-text {
  font-size: 12px;
  line-height: 1.5;
  color: var(--color-text-secondary);
}

.wqr-rec-text::before {
  content: counter(rec-counter) ". ";
  font-weight: 600;
  color: var(--color-text-primary);
}

/* ---------- Export ---------- */
.wqr-export {
  display: flex;
  justify-content: flex-end;
}

.wqr-export-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 20px;
  border: 1px solid var(--color-accent-blue, #58a6ff);
  border-radius: 6px;
  background: none;
  color: var(--color-accent-blue, #58a6ff);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.wqr-export-btn:hover {
  background-color: var(--color-accent-blue, #58a6ff);
  color: #fff;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
  .wqr-criteria-header,
  .wqr-criteria-row {
    grid-template-columns: 24px 1fr 80px 80px;
  }

  .wqr-criteria-col-badge {
    display: none;
  }

  .wqr-analysis-grid {
    grid-template-columns: 1fr;
  }

  .wqr-risk-grid {
    grid-template-columns: 1fr;
  }

  .wqr-verdict {
    flex-wrap: wrap;
  }

  .wqr-verdict-score {
    width: 100%;
    justify-content: center;
    margin-top: 8px;
  }
}

@media (max-width: 480px) {
  .wqr-criteria-header,
  .wqr-criteria-row {
    grid-template-columns: 24px 1fr 70px;
  }

  .wqr-criteria-col-range {
    display: none;
  }
}
</style>
