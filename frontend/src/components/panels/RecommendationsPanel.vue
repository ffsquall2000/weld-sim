<template>
  <div class="recommendations-panel">
    <!-- Header -->
    <div class="rp-header">
      <h3 class="rp-title">{{ t('recommendations.title') }}</h3>
      <button class="rp-refresh-btn" :disabled="loading" @click="refresh">
        <span :class="{ 'rp-spin': loading }">&#8635;</span>
      </button>
    </div>

    <!-- Quality Trend Indicator -->
    <div v-if="qualityTrend" class="rp-trend-bar" :class="trendClass">
      <span class="rp-trend-icon">{{ trendIcon }}</span>
      <span class="rp-trend-text">{{ qualityTrend.message }}</span>
    </div>

    <!-- Suggestions List -->
    <div class="rp-section">
      <div class="rp-section-header">
        <span class="rp-section-label">{{ t('recommendations.suggestions') }}</span>
        <span class="rp-badge">{{ suggestions.length }}</span>
      </div>
      <div v-if="suggestions.length > 0" class="rp-suggestion-list">
        <div
          v-for="(item, index) in suggestions"
          :key="index"
          class="rp-suggestion-card"
        >
          <div class="rp-suggestion-top">
            <span class="rp-param-name">{{ item.parameter }}</span>
            <span
              class="rp-priority-badge"
              :class="'rp-priority--' + item.priority"
            >
              {{ t('recommendations.priority_' + item.priority) }}
            </span>
          </div>
          <div class="rp-suggestion-values">
            <div class="rp-value-row">
              <span class="rp-value-label">{{ t('recommendations.current') }}:</span>
              <span class="rp-value-num">{{ item.current_value }} {{ item.unit }}</span>
            </div>
            <span class="rp-arrow">&#8594;</span>
            <div class="rp-value-row">
              <span class="rp-value-label">{{ t('recommendations.suggested') }}:</span>
              <span class="rp-value-num rp-value-num--suggested">{{ item.suggested_value }} {{ item.unit }}</span>
            </div>
          </div>
          <div class="rp-suggestion-reason">{{ item.reason }}</div>
          <!-- Safety status for this parameter -->
          <div
            v-if="safetyStatus[item.parameter]"
            class="rp-safety-badge"
            :class="safetyStatus[item.parameter].in_window ? 'rp-safety--ok' : 'rp-safety--warn'"
          >
            {{
              safetyStatus[item.parameter].in_window
                ? t('recommendations.withinSafeRange')
                : t('recommendations.outsideSafeRange')
            }}
            ({{ safetyStatus[item.parameter].safe_min }} - {{ safetyStatus[item.parameter].safe_max }})
          </div>
          <button
            class="rp-apply-btn"
            @click="applySuggestion(item)"
          >
            {{ t('recommendations.apply') }}
          </button>
        </div>
      </div>
      <div v-else class="rp-empty">
        {{ t('recommendations.noSuggestions') }}
      </div>
    </div>

    <!-- Knowledge Recommendations -->
    <div v-if="knowledgeRecommendations.length > 0" class="rp-section">
      <div class="rp-section-header">
        <span class="rp-section-label">{{ t('recommendations.knowledgeRules') }}</span>
        <span class="rp-badge">{{ knowledgeRecommendations.length }}</span>
      </div>
      <div
        v-for="(rule, idx) in knowledgeRecommendations"
        :key="idx"
        class="rp-knowledge-card"
      >
        <button
          class="rp-knowledge-header"
          @click="toggleKnowledge(idx)"
        >
          <span class="rp-knowledge-toggle">{{ expandedRules.has(idx) ? '&#9660;' : '&#9654;' }}</span>
          <span class="rp-knowledge-desc">{{ rule.description }}</span>
          <span class="rp-knowledge-priority">P{{ rule.priority }}</span>
        </button>
        <div v-if="expandedRules.has(idx)" class="rp-knowledge-body">
          <div
            v-for="(rec, rIdx) in rule.recommendations"
            :key="rIdx"
            class="rp-knowledge-rec"
          >
            &#8226; {{ rec }}
          </div>
          <div
            v-if="Object.keys(rule.adjustments).length > 0"
            class="rp-knowledge-adj"
          >
            <span class="rp-adj-label">{{ t('recommendations.adjustments') }}:</span>
            <span
              v-for="(val, key) in rule.adjustments"
              :key="String(key)"
              class="rp-adj-item"
            >
              {{ key }}: {{ val }}
            </span>
          </div>
        </div>
      </div>
    </div>

    <!-- Deviations Summary -->
    <div v-if="Object.keys(deviations).length > 0" class="rp-section">
      <div class="rp-section-header">
        <span class="rp-section-label">{{ t('recommendations.deviations') }}</span>
      </div>
      <div class="rp-deviation-list">
        <div
          v-for="(dev, param) in deviations"
          :key="String(param)"
          class="rp-deviation-row"
        >
          <span class="rp-deviation-param">{{ param }}</span>
          <span
            class="rp-deviation-pct"
            :class="{
              'rp-deviation--high': Math.abs(dev.deviation_pct) > 20,
              'rp-deviation--medium': Math.abs(dev.deviation_pct) > 10 && Math.abs(dev.deviation_pct) <= 20,
              'rp-deviation--low': Math.abs(dev.deviation_pct) <= 10,
            }"
          >
            {{ dev.deviation_pct > 0 ? '+' : '' }}{{ dev.deviation_pct }}%
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useOptimizationStore } from '@/stores/optimization'
import { useSimulationStore } from '@/stores/simulation'
import axios from 'axios'

const { t } = useI18n()
const optimizationStore = useOptimizationStore()
const simulationStore = useSimulationStore()

const api = axios.create({
  baseURL: '/api/v2',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
})

// State
const loading = ref(false)
const suggestions = ref<Array<{
  parameter: string
  current_value: number
  suggested_value: number
  unit: string
  reason: string
  priority: string
  confidence: number
  direction: string
  source?: string
  rule_id?: string | null
}>>([])
const deviations = ref<Record<string, { current: number; baseline: number; deviation_pct: number }>>({})
const safetyStatus = ref<Record<string, { in_window: boolean; safe_min: number; safe_max: number }>>({})
const knowledgeRecommendations = ref<Array<{
  rule_id: string
  description: string
  adjustments: Record<string, unknown>
  recommendations: string[]
  priority: number
}>>([])
const qualityTrend = ref<{ trend: string; message: string } | null>(null)
const expandedRules = reactive(new Set<number>())

// Emits
const emit = defineEmits<{
  (e: 'apply-suggestion', suggestion: {
    parameter: string
    current_value: number
    suggested_value: number
    unit: string
  }): void
}>()

// Computed
const trendClass = computed(() => {
  if (!qualityTrend.value) return ''
  const trend = qualityTrend.value.trend
  if (trend === 'improving') return 'rp-trend--improving'
  if (trend === 'declining') return 'rp-trend--declining'
  if (trend === 'stable') return 'rp-trend--stable'
  return 'rp-trend--unknown'
})

const trendIcon = computed(() => {
  if (!qualityTrend.value) return '?'
  const trend = qualityTrend.value.trend
  if (trend === 'improving') return '\u2191'
  if (trend === 'declining') return '\u2193'
  if (trend === 'stable') return '\u2194'
  return '\u2026'
})

// Methods
function toggleKnowledge(idx: number) {
  if (expandedRules.has(idx)) {
    expandedRules.delete(idx)
  } else {
    expandedRules.add(idx)
  }
}

function applySuggestion(item: {
  parameter: string
  current_value: number
  suggested_value: number
  unit: string
}) {
  emit('apply-suggestion', {
    parameter: item.parameter,
    current_value: item.current_value,
    suggested_value: item.suggested_value,
    unit: item.unit,
  })
}

async function refresh() {
  loading.value = true
  try {
    // Build request from current optimization/simulation state
    const currentParams: Record<string, number> = {}
    const baselineParams: Record<string, number> = {}
    const context: Record<string, unknown> = {}

    // Attempt to gather params from the current study or simulation
    const study = optimizationStore.currentStudy
    if (study) {
      for (const dv of study.design_variables) {
        baselineParams[dv.name] = ((dv.min_value ?? 0) + (dv.max_value ?? 100)) / 2
        currentParams[dv.name] = baselineParams[dv.name]
      }
    }

    // Use the best iteration's design point if available
    const best = optimizationStore.bestIteration
    if (best) {
      Object.assign(currentParams, best.design_point)
    }

    // If no params available, skip the request
    if (Object.keys(baselineParams).length === 0 && Object.keys(currentParams).length === 0) {
      suggestions.value = []
      deviations.value = {}
      safetyStatus.value = {}
      knowledgeRecommendations.value = []
      qualityTrend.value = {
        trend: 'insufficient_data',
        message: t('recommendations.noData'),
      }
      return
    }

    const response = await api.post('/manual-optimization/suggest', {
      current_params: currentParams,
      baseline_params: baselineParams,
      context,
    })

    suggestions.value = response.data.suggestions ?? []
    deviations.value = response.data.deviations ?? {}
    safetyStatus.value = response.data.safety_status ?? {}
    knowledgeRecommendations.value = response.data.knowledge_recommendations ?? []
    qualityTrend.value = response.data.quality_trend ?? null
  } catch {
    suggestions.value = []
    qualityTrend.value = {
      trend: 'insufficient_data',
      message: t('recommendations.loadFailed'),
    }
  } finally {
    loading.value = false
  }
}

// Watch for study changes to auto-refresh
watch(
  () => optimizationStore.currentStudy,
  () => {
    refresh()
  },
  { immediate: true }
)
</script>

<style scoped>
.recommendations-panel {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  font-size: 12px;
}

/* Header */
.rp-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-bg-primary);
}

.rp-title {
  font-size: 12px;
  font-weight: 600;
  margin: 0;
  color: var(--color-text-primary);
}

.rp-refresh-btn {
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 16px;
  cursor: pointer;
  padding: 2px 4px;
  border-radius: 4px;
  line-height: 1;
}

.rp-refresh-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.rp-refresh-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.rp-spin {
  display: inline-block;
  animation: rp-spin-anim 0.8s linear infinite;
}

@keyframes rp-spin-anim {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* Trend bar */
.rp-trend-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  font-size: 11px;
  border-bottom: 1px solid var(--color-border);
}

.rp-trend--improving {
  background-color: rgba(76, 175, 80, 0.1);
  color: var(--color-success);
}

.rp-trend--declining {
  background-color: rgba(244, 67, 54, 0.1);
  color: var(--color-danger);
}

.rp-trend--stable {
  background-color: rgba(255, 152, 0, 0.1);
  color: var(--color-warning);
}

.rp-trend--unknown {
  background-color: var(--color-bg-card);
  color: var(--color-text-secondary);
}

.rp-trend-icon {
  font-size: 14px;
  font-weight: 700;
}

.rp-trend-text {
  flex: 1;
}

/* Section */
.rp-section {
  border-bottom: 1px solid var(--color-border);
  padding: 8px 12px;
}

.rp-section-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.rp-section-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.rp-badge {
  font-size: 10px;
  background-color: var(--color-accent-blue);
  color: #fff;
  padding: 1px 6px;
  border-radius: 8px;
  font-weight: 600;
}

/* Suggestion cards */
.rp-suggestion-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.rp-suggestion-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px;
}

.rp-suggestion-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.rp-param-name {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-accent-blue);
  font-family: ui-monospace, monospace;
}

.rp-priority-badge {
  font-size: 9px;
  padding: 1px 6px;
  border-radius: 4px;
  font-weight: 700;
  text-transform: uppercase;
}

.rp-priority--critical {
  background-color: rgba(244, 67, 54, 0.15);
  color: var(--color-danger);
}

.rp-priority--high {
  background-color: rgba(255, 152, 0, 0.15);
  color: var(--color-warning);
}

.rp-priority--medium {
  background-color: rgba(255, 235, 59, 0.15);
  color: #f9a825;
}

.rp-priority--low {
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
}

/* Suggestion values */
.rp-suggestion-values {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-family: ui-monospace, monospace;
}

.rp-value-row {
  display: flex;
  flex-direction: column;
  gap: 1px;
}

.rp-value-label {
  font-size: 9px;
  color: var(--color-text-secondary);
}

.rp-value-num {
  font-size: 12px;
  color: var(--color-text-primary);
}

.rp-value-num--suggested {
  color: var(--color-accent-orange);
  font-weight: 600;
}

.rp-arrow {
  color: var(--color-text-secondary);
  font-size: 14px;
}

.rp-suggestion-reason {
  font-size: 10px;
  color: var(--color-text-secondary);
  line-height: 1.4;
  margin-bottom: 4px;
}

/* Safety badge */
.rp-safety-badge {
  font-size: 9px;
  padding: 2px 6px;
  border-radius: 3px;
  margin-bottom: 6px;
  display: inline-block;
}

.rp-safety--ok {
  background-color: rgba(76, 175, 80, 0.1);
  color: var(--color-success);
}

.rp-safety--warn {
  background-color: rgba(244, 67, 54, 0.1);
  color: var(--color-danger);
}

/* Apply button */
.rp-apply-btn {
  width: 100%;
  padding: 4px 8px;
  border: 1px solid var(--color-accent-orange);
  border-radius: 4px;
  background: none;
  color: var(--color-accent-orange);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.15s, color 0.15s;
}

.rp-apply-btn:hover {
  background-color: var(--color-accent-orange);
  color: #fff;
}

/* Empty state */
.rp-empty {
  font-size: 11px;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 8px 0;
}

/* Knowledge cards */
.rp-knowledge-card {
  border: 1px solid var(--color-border);
  border-radius: 4px;
  margin-bottom: 4px;
  overflow: hidden;
}

.rp-knowledge-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 8px;
  border: none;
  background: none;
  color: var(--color-text-primary);
  font-size: 11px;
  cursor: pointer;
  text-align: left;
}

.rp-knowledge-header:hover {
  background-color: var(--color-bg-card);
}

.rp-knowledge-toggle {
  font-size: 8px;
  width: 10px;
  color: var(--color-text-secondary);
}

.rp-knowledge-desc {
  flex: 1;
  font-size: 11px;
}

.rp-knowledge-priority {
  font-size: 9px;
  color: var(--color-text-secondary);
  font-weight: 600;
}

.rp-knowledge-body {
  padding: 4px 8px 8px 24px;
  border-top: 1px solid var(--color-border);
}

.rp-knowledge-rec {
  font-size: 10px;
  color: var(--color-text-secondary);
  line-height: 1.6;
}

.rp-knowledge-adj {
  margin-top: 4px;
  font-size: 10px;
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  align-items: center;
}

.rp-adj-label {
  color: var(--color-text-secondary);
  font-weight: 600;
}

.rp-adj-item {
  background-color: var(--color-bg-card);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: ui-monospace, monospace;
  color: var(--color-text-primary);
}

/* Deviations */
.rp-deviation-list {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.rp-deviation-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 3px 6px;
  border-radius: 3px;
  background-color: var(--color-bg-primary);
}

.rp-deviation-param {
  font-size: 11px;
  font-family: ui-monospace, monospace;
  color: var(--color-text-primary);
}

.rp-deviation-pct {
  font-size: 11px;
  font-weight: 600;
  font-family: ui-monospace, monospace;
}

.rp-deviation--high {
  color: var(--color-danger);
}

.rp-deviation--medium {
  color: var(--color-warning);
}

.rp-deviation--low {
  color: var(--color-success);
}
</style>
