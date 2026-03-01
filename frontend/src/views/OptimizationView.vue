<template>
  <div class="optimization-view">
    <!-- Header -->
    <div class="ov-header">
      <div class="ov-header-left">
        <button class="ov-back-btn" @click="goBack">&#8592;</button>
        <h1 class="ov-title">{{ currentStudy?.name || t('optimization.title') }}</h1>
        <span v-if="currentStudy" class="ov-status-badge" :class="statusClass(currentStudy.status)">
          {{ currentStudy.status }}
        </span>
      </div>
      <div class="ov-header-right">
        <button
          v-if="isRunning"
          class="ov-action-btn ov-action-btn--pause"
          @click="handlePause"
        >
          &#9612;&#9612; {{ t('optimization.pause') }}
        </button>
        <button
          v-else-if="currentStudy?.status === 'paused'"
          class="ov-action-btn ov-action-btn--resume"
          @click="handleResume"
        >
          &#9654; {{ t('optimization.resume') }}
        </button>
        <button
          v-else
          class="ov-action-btn ov-action-btn--run"
          @click="showCreateDialog = true"
        >
          &#9654; {{ t('optimization.run') }}
        </button>
      </div>
    </div>

    <!-- Progress -->
    <div v-if="currentStudy" class="ov-progress-section">
      <div class="ov-progress-bar-container">
        <div
          class="ov-progress-bar"
          :style="{ width: optimizationStore.studyProgress + '%' }"
          :class="{ 'ov-progress-bar--active': isRunning }"
        />
      </div>
      <span class="ov-progress-text">
        {{ currentStudy.completed_iterations }} / {{ currentStudy.total_iterations }}
        {{ t('optimization.iterations') }}
      </span>
    </div>

    <div class="ov-content">
      <!-- Left Column: Configuration -->
      <div class="ov-config-panel">
        <!-- Design Variables -->
        <section class="ov-section">
          <h3 class="ov-section-title">{{ t('optimization.designVariables') }}</h3>
          <div v-for="dv in designVariables" :key="dv.name" class="ov-dv-card">
            <div class="ov-dv-header">
              <span class="ov-dv-name">{{ dv.name }}</span>
              <span class="ov-dv-type">{{ dv.var_type }}</span>
            </div>
            <div class="ov-dv-slider-row">
              <span class="ov-dv-min">{{ dv.min_value }}</span>
              <input
                type="range"
                class="ov-dv-slider"
                :min="dv.min_value ?? 0"
                :max="dv.max_value ?? 100"
                :step="dv.step ?? 0.1"
                :value="currentDesignPoint[dv.name] ?? ((dv.min_value ?? 0) + (dv.max_value ?? 100)) / 2"
                @input="updateDesignPoint(dv.name, ($event.target as HTMLInputElement).value)"
              />
              <span class="ov-dv-max">{{ dv.max_value }}</span>
            </div>
            <div class="ov-dv-value">
              {{ t('optimization.currentValue') }}: {{ (currentDesignPoint[dv.name] ?? ((dv.min_value ?? 0) + (dv.max_value ?? 100)) / 2).toFixed(2) }}
            </div>
          </div>
          <div v-if="designVariables.length === 0" class="ov-empty-hint">
            {{ t('optimization.noDesignVars') }}
          </div>
        </section>

        <!-- Constraints -->
        <section class="ov-section">
          <h3 class="ov-section-title">{{ t('optimization.constraints') }}</h3>
          <table class="ov-table" v-if="constraints.length > 0">
            <thead>
              <tr>
                <th>{{ t('optimization.metric') }}</th>
                <th>{{ t('optimization.operator') }}</th>
                <th>{{ t('optimization.constraintValue') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(c, i) in constraints" :key="i">
                <td>{{ c.metric }}</td>
                <td>
                  <select v-model="c.operator" class="ov-inline-select">
                    <option value="<=">&lt;=</option>
                    <option value=">=">&gt;=</option>
                    <option value="==">=</option>
                  </select>
                </td>
                <td>
                  <input v-model.number="c.value" type="number" class="ov-inline-input" step="0.1" />
                </td>
              </tr>
            </tbody>
          </table>
          <div v-else class="ov-empty-hint">{{ t('optimization.noConstraints') }}</div>
        </section>

        <!-- Objectives -->
        <section class="ov-section">
          <h3 class="ov-section-title">{{ t('optimization.objectives') }}</h3>
          <div v-for="(obj, i) in objectives" :key="i" class="ov-obj-card">
            <div class="ov-obj-metric">{{ obj.metric }}</div>
            <div class="ov-obj-config">
              <select v-model="obj.direction" class="ov-inline-select">
                <option value="minimize">{{ t('optimization.minimize') }}</option>
                <option value="maximize">{{ t('optimization.maximize') }}</option>
              </select>
              <span class="ov-obj-weight-label">{{ t('optimization.weight') }}:</span>
              <input v-model.number="obj.weight" type="number" class="ov-inline-input" step="0.1" min="0" max="1" />
            </div>
          </div>
          <div v-if="objectives.length === 0" class="ov-empty-hint">{{ t('optimization.noObjectives') }}</div>
        </section>
      </div>

      <!-- Right Column: Results -->
      <div class="ov-results-panel">
        <!-- Pareto Front Chart -->
        <section class="ov-section">
          <h3 class="ov-section-title">{{ t('optimization.paretoFront') }}</h3>
          <div class="ov-chart-container">
            <v-chart
              v-if="paretoData.length > 0"
              :option="paretoChartOption"
              autoresize
              class="ov-chart"
            />
            <div v-else class="ov-chart-empty">
              {{ t('optimization.noParetoData') }}
            </div>
          </div>
        </section>

        <!-- Iteration History -->
        <section class="ov-section">
          <h3 class="ov-section-title">{{ t('optimization.iterationHistory') }}</h3>
          <div class="ov-iteration-table-wrapper">
            <table class="ov-table" v-if="optimizationStore.iterations.length > 0">
              <thead>
                <tr>
                  <th>#</th>
                  <th>{{ t('optimization.feasible') }}</th>
                  <th>{{ t('optimization.pareto') }}</th>
                  <th v-for="obj in objectives" :key="obj.metric">{{ obj.metric }}</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="iter in displayedIterations"
                  :key="iter.iteration_number"
                  :class="{ 'ov-row--pareto': iter.pareto_optimal }"
                >
                  <td>{{ iter.iteration_number }}</td>
                  <td>
                    <span :class="iter.feasible ? 'ov-badge--yes' : 'ov-badge--no'">
                      {{ iter.feasible ? '\u2713' : '\u2717' }}
                    </span>
                  </td>
                  <td>
                    <span v-if="iter.pareto_optimal" class="ov-badge--star">&#10038;</span>
                  </td>
                  <td v-for="obj in objectives" :key="obj.metric">
                    {{ (iter.objective_values[obj.metric] ?? 0).toFixed(4) }}
                  </td>
                </tr>
              </tbody>
            </table>
            <div v-else class="ov-empty-hint">{{ t('optimization.noIterations') }}</div>
          </div>
        </section>
      </div>
    </div>

    <!-- Create Study Dialog placeholder -->
    <Teleport to="body">
      <div v-if="showCreateDialog" class="ov-modal-overlay" @click.self="showCreateDialog = false">
        <div class="ov-modal">
          <h2 class="ov-modal-title">{{ t('optimization.createStudy') }}</h2>
          <div class="ov-modal-body">
            <div class="ov-form-field">
              <label>{{ t('optimization.studyName') }}</label>
              <input v-model="newStudyName" type="text" class="ov-form-input" />
            </div>
            <div class="ov-form-field">
              <label>{{ t('optimization.strategy') }}</label>
              <select v-model="newStudyStrategy" class="ov-form-input">
                <option value="bayesian">Bayesian Optimization</option>
                <option value="genetic">Genetic Algorithm (NSGA-II)</option>
                <option value="grid">Grid Search</option>
                <option value="random">Random Search</option>
              </select>
            </div>
            <div class="ov-form-field">
              <label>{{ t('optimization.totalIterations') }}</label>
              <input v-model.number="newStudyIterations" type="number" class="ov-form-input" min="10" max="1000" />
            </div>
          </div>
          <div class="ov-modal-footer">
            <button class="ov-modal-btn" @click="showCreateDialog = false">{{ t('common.cancel') }}</button>
            <button class="ov-modal-btn ov-modal-btn--primary" @click="handleCreateStudy">{{ t('optimization.start') }}</button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { useOptimizationStore } from '@/stores/optimization'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { ScatterChart } from 'echarts/charts'
import { TooltipComponent, GridComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([ScatterChart, TooltipComponent, GridComponent, CanvasRenderer])

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const optimizationStore = useOptimizationStore()

const showCreateDialog = ref(false)
const newStudyName = ref('Optimization Study 1')
const newStudyStrategy = ref('bayesian')
const newStudyIterations = ref(100)
const currentDesignPoint = reactive<Record<string, number>>({})

const currentStudy = computed(() => optimizationStore.currentStudy)
const isRunning = computed(() => optimizationStore.isStudyRunning)
const paretoData = computed(() => optimizationStore.paretoFront)

const designVariables = computed(() => currentStudy.value?.design_variables ?? [])
const constraints = computed(() => currentStudy.value?.constraints ?? [])
const objectives = computed(() => currentStudy.value?.objectives ?? [])

const displayedIterations = computed(() => {
  return [...optimizationStore.iterations]
    .sort((a, b) => b.iteration_number - a.iteration_number)
    .slice(0, 50)
})

function statusClass(status: string) {
  return {
    'ov-status--running': status === 'running',
    'ov-status--paused': status === 'paused',
    'ov-status--completed': status === 'completed',
    'ov-status--failed': status === 'failed',
  }
}

function goBack() {
  router.back()
}

function updateDesignPoint(name: string, value: string) {
  currentDesignPoint[name] = parseFloat(value)
}

async function handlePause() {
  if (currentStudy.value) {
    await optimizationStore.pauseStudy(currentStudy.value.id)
  }
}

async function handleResume() {
  if (currentStudy.value) {
    await optimizationStore.resumeStudy(currentStudy.value.id)
  }
}

async function handleCreateStudy() {
  // This would need a projectId - using route or default
  showCreateDialog.value = false
}

// Pareto chart
const paretoChartOption = computed(() => {
  if (objectives.value.length < 2) return {}

  const obj1 = objectives.value[0]!.metric
  const obj2 = objectives.value[1]!.metric

  const scatterData = paretoData.value.map((p) => [
    p.objective_values[obj1] ?? 0,
    p.objective_values[obj2] ?? 0,
  ])

  const allIterData = optimizationStore.iterations.map((it) => [
    it.objective_values[obj1] ?? 0,
    it.objective_values[obj2] ?? 0,
  ])

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        return `${obj1}: ${params.value[0].toFixed(4)}<br/>${obj2}: ${params.value[1].toFixed(4)}`
      },
    },
    grid: {
      left: '15%',
      right: '10%',
      top: '10%',
      bottom: '15%',
    },
    xAxis: {
      name: obj1,
      nameLocation: 'middle',
      nameGap: 30,
      nameTextStyle: { color: '#8b949e', fontSize: 11 },
      axisLabel: { color: '#8b949e', fontSize: 10 },
      axisLine: { lineStyle: { color: '#30363d' } },
      splitLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.1)' } },
    },
    yAxis: {
      name: obj2,
      nameLocation: 'middle',
      nameGap: 40,
      nameTextStyle: { color: '#8b949e', fontSize: 11 },
      axisLabel: { color: '#8b949e', fontSize: 10 },
      axisLine: { lineStyle: { color: '#30363d' } },
      splitLine: { lineStyle: { color: 'rgba(139, 148, 158, 0.1)' } },
    },
    series: [
      {
        name: 'All Iterations',
        type: 'scatter',
        data: allIterData,
        symbolSize: 6,
        itemStyle: {
          color: 'rgba(139, 148, 158, 0.4)',
        },
      },
      {
        name: 'Pareto Front',
        type: 'scatter',
        data: scatterData,
        symbolSize: 10,
        itemStyle: {
          color: '#ff9800',
          borderColor: '#fff',
          borderWidth: 1,
        },
      },
    ],
  }
})

// Load data when route changes
watch(
  () => route.params.studyId as string,
  async (studyId) => {
    if (studyId) {
      await optimizationStore.fetchStudy(studyId)
      await Promise.all([
        optimizationStore.fetchIterations(studyId),
        optimizationStore.fetchPareto(studyId),
      ])
    }
  },
  { immediate: true }
)
</script>

<style scoped>
.optimization-view {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 0;
}

/* Header */
.ov-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 20px;
  border-bottom: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}

.ov-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.ov-back-btn {
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 18px;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
}

.ov-back-btn:hover {
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
}

.ov-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0;
}

.ov-status-badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 4px;
  font-weight: 600;
  text-transform: uppercase;
}

.ov-status--running {
  background-color: rgba(88, 166, 255, 0.15);
  color: var(--color-accent-blue);
}

.ov-status--paused {
  background-color: rgba(255, 152, 0, 0.15);
  color: var(--color-warning);
}

.ov-status--completed {
  background-color: rgba(76, 175, 80, 0.15);
  color: var(--color-success);
}

.ov-status--failed {
  background-color: rgba(244, 67, 54, 0.15);
  color: var(--color-danger);
}

.ov-action-btn {
  padding: 6px 16px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}

.ov-action-btn--run {
  background-color: var(--color-success);
  color: #fff;
}

.ov-action-btn--pause {
  background-color: var(--color-warning);
  color: #fff;
}

.ov-action-btn--resume {
  background-color: var(--color-accent-blue);
  color: #fff;
}

.ov-action-btn:hover {
  opacity: 0.9;
}

/* Progress */
.ov-progress-section {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 6px 20px;
  background-color: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
}

.ov-progress-bar-container {
  flex: 1;
  height: 6px;
  background-color: var(--color-bg-card);
  border-radius: 3px;
  overflow: hidden;
}

.ov-progress-bar {
  height: 100%;
  background-color: var(--color-accent-orange);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.ov-progress-bar--active {
  background-image: linear-gradient(
    45deg,
    rgba(255, 255, 255, 0.15) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.15) 50%,
    rgba(255, 255, 255, 0.15) 75%,
    transparent 75%
  );
  background-size: 16px 16px;
  animation: stripes 1s linear infinite;
}

@keyframes stripes {
  from { background-position: 16px 0; }
  to { background-position: 0 0; }
}

.ov-progress-text {
  font-size: 12px;
  color: var(--color-text-secondary);
  white-space: nowrap;
}

/* Content layout */
.ov-content {
  display: flex;
  flex: 1;
  overflow: hidden;
}

.ov-config-panel {
  width: 380px;
  min-width: 320px;
  overflow-y: auto;
  border-right: 1px solid var(--color-border);
  background-color: var(--color-bg-secondary);
}

.ov-results-panel {
  flex: 1;
  overflow-y: auto;
  background-color: var(--color-bg-primary);
}

/* Sections */
.ov-section {
  padding: 12px 16px;
  border-bottom: 1px solid var(--color-border);
}

.ov-section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 10px;
}

/* Design Variable Cards */
.ov-dv-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 10px;
  margin-bottom: 8px;
}

.ov-dv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
}

.ov-dv-name {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-accent-blue);
}

.ov-dv-type {
  font-size: 10px;
  color: var(--color-text-secondary);
  text-transform: capitalize;
}

.ov-dv-slider-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ov-dv-min,
.ov-dv-max {
  font-size: 10px;
  color: var(--color-text-secondary);
  font-family: ui-monospace, monospace;
  width: 40px;
}

.ov-dv-max {
  text-align: right;
}

.ov-dv-slider {
  flex: 1;
  -webkit-appearance: none;
  appearance: none;
  height: 4px;
  background-color: var(--color-bg-card);
  border-radius: 2px;
  outline: none;
}

.ov-dv-slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 14px;
  height: 14px;
  background-color: var(--color-accent-orange);
  border-radius: 50%;
  cursor: pointer;
}

.ov-dv-value {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-top: 4px;
  font-family: ui-monospace, monospace;
}

/* Tables */
.ov-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}

.ov-table th {
  text-align: left;
  padding: 6px 8px;
  font-weight: 600;
  color: var(--color-text-secondary);
  border-bottom: 1px solid var(--color-border);
  font-size: 11px;
}

.ov-table td {
  padding: 5px 8px;
  color: var(--color-text-primary);
  border-bottom: 1px solid rgba(48, 54, 61, 0.3);
}

.ov-row--pareto {
  background-color: rgba(255, 152, 0, 0.05);
}

.ov-inline-select,
.ov-inline-input {
  padding: 3px 6px;
  border: 1px solid var(--color-border);
  border-radius: 3px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 11px;
  font-family: ui-monospace, monospace;
}

.ov-inline-input {
  width: 70px;
}

.ov-inline-select:focus,
.ov-inline-input:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

/* Objective cards */
.ov-obj-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 6px;
  padding: 8px 10px;
  margin-bottom: 6px;
}

.ov-obj-metric {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.ov-obj-config {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ov-obj-weight-label {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.ov-badge--yes {
  color: var(--color-success);
  font-weight: 600;
}

.ov-badge--no {
  color: var(--color-danger);
  font-weight: 600;
}

.ov-badge--star {
  color: var(--color-accent-orange);
}

.ov-empty-hint {
  font-size: 12px;
  color: var(--color-text-secondary);
  font-style: italic;
  padding: 10px 0;
}

/* Chart */
.ov-chart-container {
  height: 320px;
}

.ov-chart {
  width: 100%;
  height: 100%;
}

.ov-chart-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: var(--color-text-secondary);
  font-style: italic;
  font-size: 13px;
}

.ov-iteration-table-wrapper {
  max-height: 400px;
  overflow-y: auto;
}

/* Modal */
.ov-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.6);
}

.ov-modal {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  width: 420px;
  max-width: 90vw;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.ov-modal-title {
  font-size: 16px;
  font-weight: 600;
  padding: 16px 20px 0;
  margin: 0;
  color: var(--color-text-primary);
}

.ov-modal-body {
  padding: 16px 20px;
}

.ov-form-field {
  margin-bottom: 12px;
}

.ov-form-field label {
  display: block;
  font-size: 12px;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.ov-form-input {
  width: 100%;
  padding: 7px 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 13px;
}

.ov-form-input:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.ov-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 0 20px 16px;
}

.ov-modal-btn {
  padding: 7px 16px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: none;
  color: var(--color-text-secondary);
  font-size: 13px;
  cursor: pointer;
}

.ov-modal-btn:hover {
  color: var(--color-text-primary);
}

.ov-modal-btn--primary {
  background-color: var(--color-accent-orange);
  border: none;
  color: #fff;
}

.ov-modal-btn--primary:hover {
  opacity: 0.9;
}
</style>
