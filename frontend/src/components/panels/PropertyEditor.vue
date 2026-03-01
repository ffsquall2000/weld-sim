<template>
  <div class="property-editor">
    <!-- No Selection State -->
    <div v-if="!activeContext" class="pe-empty">
      <span class="pe-empty-icon">&#9998;</span>
      <span class="pe-empty-text">{{ t('propertyEditor.noSelection') }}</span>
    </div>

    <!-- Geometry Properties -->
    <div v-else-if="activeContext === 'geometry'" class="pe-content">
      <h3 class="pe-title">{{ t('propertyEditor.geometryProperties') }}</h3>

      <!-- Horn Type -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('hornType')">
          <span class="pe-accordion-icon">{{ accordions.hornType ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.hornType') }}</span>
        </button>
        <div v-if="accordions.hornType" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.type') }}</label>
            <select v-model="geometryForm.hornType" class="pe-select">
              <option value="flat">{{ t('wizard.hornFlat') }}</option>
              <option value="cylindrical">{{ t('hornDesign.typeCylindrical') }}</option>
              <option value="exponential">{{ t('hornDesign.typeExponential') }}</option>
              <option value="blade">{{ t('wizard.hornBlade') }}</option>
              <option value="stepped">{{ t('hornDesign.typeStepped') }}</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Dimensions -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('dimensions')">
          <span class="pe-accordion-icon">{{ accordions.dimensions ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.dimensions') }}</span>
        </button>
        <div v-if="accordions.dimensions" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.widthMm') }}</label>
            <input v-model.number="geometryForm.width" type="number" class="pe-input" step="0.1" />
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.heightMm') }}</label>
            <input v-model.number="geometryForm.height" type="number" class="pe-input" step="0.1" />
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.lengthMm') }}</label>
            <input v-model.number="geometryForm.length" type="number" class="pe-input" step="0.1" />
          </div>
        </div>
      </div>

      <!-- Knurl -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('knurl')">
          <span class="pe-accordion-icon">{{ accordions.knurl ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('hornDesign.knurlParams') }}</span>
        </button>
        <div v-if="accordions.knurl" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.knurlType') }}</label>
            <select v-model="geometryForm.knurlType" class="pe-select">
              <option value="linear">{{ t('wizard.knurlLinear') }}</option>
              <option value="cross_hatch">{{ t('wizard.knurlCrossHatch') }}</option>
              <option value="diamond">{{ t('wizard.knurlDiamond') }}</option>
            </select>
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.knurlPitch') }}</label>
            <input v-model.number="geometryForm.knurlPitch" type="number" class="pe-input" step="0.1" min="0.1" />
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.knurlToothWidth') }}</label>
            <input v-model.number="geometryForm.knurlToothWidth" type="number" class="pe-input" step="0.1" min="0.1" />
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.knurlDepth') }}</label>
            <input v-model.number="geometryForm.knurlDepth" type="number" class="pe-input" step="0.05" min="0.05" />
          </div>
        </div>
      </div>

      <!-- Chamfer -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('chamfer')">
          <span class="pe-accordion-icon">{{ accordions.chamfer ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('hornDesign.chamferParams') }}</span>
        </button>
        <div v-if="accordions.chamfer" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.edgeTreatment') }}</label>
            <select v-model="geometryForm.edgeTreatment" class="pe-select">
              <option value="sharp">{{ t('hornDesign.edgeSharp') }}</option>
              <option value="chamfer">{{ t('hornDesign.edgeChamfer') }}</option>
              <option value="fillet">{{ t('hornDesign.edgeFillet') }}</option>
              <option value="compound">{{ t('hornDesign.edgeCompound') }}</option>
            </select>
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.chamferRadius') }}</label>
            <input v-model.number="geometryForm.chamferRadius" type="number" class="pe-input" step="0.1" min="0" />
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('hornDesign.chamferAngle') }}</label>
            <input v-model.number="geometryForm.chamferAngle" type="number" class="pe-input" step="1" min="0" max="90" />
          </div>
        </div>
      </div>
    </div>

    <!-- Simulation Properties -->
    <div v-else-if="activeContext === 'simulation'" class="pe-content">
      <h3 class="pe-title">{{ t('propertyEditor.simulationProperties') }}</h3>

      <!-- Solver -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('solver')">
          <span class="pe-accordion-icon">{{ accordions.solver ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.solver') }}</span>
        </button>
        <div v-if="accordions.solver" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.solverBackend') }}</label>
            <select v-model="simulationForm.solverBackend" class="pe-select">
              <option value="internal">Internal FEA</option>
              <option value="calculix">CalculiX</option>
              <option value="elmer">Elmer</option>
            </select>
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.analysisType') }}</label>
            <select v-model="simulationForm.analysisType" class="pe-select">
              <option value="modal">{{ t('propertyEditor.analysisModal') }}</option>
              <option value="harmonic">{{ t('propertyEditor.analysisHarmonic') }}</option>
              <option value="transient">{{ t('propertyEditor.analysisTransient') }}</option>
              <option value="thermal">{{ t('propertyEditor.analysisThermal') }}</option>
            </select>
          </div>
        </div>
      </div>

      <!-- Boundary Conditions -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('boundary')">
          <span class="pe-accordion-icon">{{ accordions.boundary ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.boundaryConditions') }}</span>
        </button>
        <div v-if="accordions.boundary" class="pe-accordion-body">
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.fixedSupport') }}</label>
            <select v-model="simulationForm.fixedSupport" class="pe-select">
              <option value="top">{{ t('propertyEditor.supportTop') }}</option>
              <option value="bottom">{{ t('propertyEditor.supportBottom') }}</option>
              <option value="both">{{ t('propertyEditor.supportBoth') }}</option>
            </select>
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.loadType') }}</label>
            <select v-model="simulationForm.loadType" class="pe-select">
              <option value="pressure">{{ t('propertyEditor.loadPressure') }}</option>
              <option value="force">{{ t('propertyEditor.loadForce') }}</option>
              <option value="displacement">{{ t('propertyEditor.loadDisplacement') }}</option>
            </select>
          </div>
          <div class="pe-field">
            <label class="pe-label">{{ t('propertyEditor.loadValue') }}</label>
            <input v-model.number="simulationForm.loadValue" type="number" class="pe-input" step="0.1" />
          </div>
        </div>
      </div>
    </div>

    <!-- Optimization Properties -->
    <div v-else-if="activeContext === 'optimization'" class="pe-content">
      <h3 class="pe-title">{{ t('propertyEditor.optimizationProperties') }}</h3>

      <!-- Design Variables -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('designVars')">
          <span class="pe-accordion-icon">{{ accordions.designVars ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.designVariables') }}</span>
        </button>
        <div v-if="accordions.designVars" class="pe-accordion-body">
          <div class="pe-info-text">{{ t('propertyEditor.designVarsHint') }}</div>
          <div
            v-for="(dv, index) in optimizationForm.designVariables"
            :key="index"
            class="pe-var-card"
          >
            <div class="pe-var-name">{{ dv.name }}</div>
            <div class="pe-var-range">
              <input v-model.number="dv.min_value" type="number" class="pe-input pe-input--small" placeholder="Min" />
              <span class="pe-var-sep">-</span>
              <input v-model.number="dv.max_value" type="number" class="pe-input pe-input--small" placeholder="Max" />
            </div>
          </div>
        </div>
      </div>

      <!-- Constraints -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('constraints')">
          <span class="pe-accordion-icon">{{ accordions.constraints ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.constraints') }}</span>
        </button>
        <div v-if="accordions.constraints" class="pe-accordion-body">
          <div
            v-for="(c, index) in optimizationForm.constraints"
            :key="index"
            class="pe-constraint-row"
          >
            <span class="pe-constraint-metric">{{ c.metric }}</span>
            <select v-model="c.operator" class="pe-select pe-select--small">
              <option value="<=">&lt;=</option>
              <option value=">=">&gt;=</option>
              <option value="==">=</option>
            </select>
            <input v-model.number="c.value" type="number" class="pe-input pe-input--small" />
          </div>
        </div>
      </div>

      <!-- Objectives -->
      <div class="pe-accordion">
        <button class="pe-accordion-header" @click="toggleAccordion('objectives')">
          <span class="pe-accordion-icon">{{ accordions.objectives ? '&#9660;' : '&#9654;' }}</span>
          <span>{{ t('propertyEditor.objectives') }}</span>
        </button>
        <div v-if="accordions.objectives" class="pe-accordion-body">
          <div
            v-for="(obj, index) in optimizationForm.objectives"
            :key="index"
            class="pe-objective-row"
          >
            <span class="pe-objective-metric">{{ obj.metric }}</span>
            <select v-model="obj.direction" class="pe-select pe-select--small">
              <option value="minimize">{{ t('propertyEditor.minimize') }}</option>
              <option value="maximize">{{ t('propertyEditor.maximize') }}</option>
            </select>
            <input v-model.number="obj.weight" type="number" class="pe-input pe-input--small" step="0.1" min="0" max="1" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useGeometryStore } from '@/stores/geometry'
import { useSimulationStore } from '@/stores/simulation'
import { useOptimizationStore } from '@/stores/optimization'

const { t } = useI18n()
const geometryStore = useGeometryStore()
const simulationStore = useSimulationStore()
const optimizationStore = useOptimizationStore()

// Determine active context based on what's selected
const activeContext = computed(() => {
  if (optimizationStore.currentStudy) return 'optimization'
  if (simulationStore.currentSimulation) return 'simulation'
  if (geometryStore.currentGeometry || geometryStore.selectedBody) return 'geometry'
  return null
})

// Accordion states
const accordions = reactive({
  hornType: true,
  dimensions: true,
  knurl: false,
  chamfer: false,
  solver: true,
  boundary: false,
  designVars: true,
  constraints: false,
  objectives: false,
})

function toggleAccordion(key: keyof typeof accordions) {
  accordions[key] = !accordions[key]
}

// Form data - geometry
const geometryForm = reactive({
  hornType: 'flat',
  width: 50.0,
  height: 25.0,
  length: 120.0,
  knurlType: 'cross_hatch',
  knurlPitch: 1.2,
  knurlToothWidth: 0.6,
  knurlDepth: 0.35,
  edgeTreatment: 'chamfer',
  chamferRadius: 1.0,
  chamferAngle: 45,
})

// Form data - simulation
const simulationForm = reactive({
  solverBackend: 'internal',
  analysisType: 'modal',
  fixedSupport: 'top',
  loadType: 'pressure',
  loadValue: 1.0,
})

// Form data - optimization
const optimizationForm = reactive({
  designVariables: [
    { name: 'horn_length', var_type: 'continuous', min_value: 80, max_value: 160, step: null },
    { name: 'horn_width', var_type: 'continuous', min_value: 30, max_value: 80, step: null },
    { name: 'knurl_pitch', var_type: 'continuous', min_value: 0.5, max_value: 3.0, step: 0.1 },
  ],
  constraints: [
    { metric: 'stress_safety_factor', operator: '>=', value: 1.5 },
    { metric: 'frequency_deviation_pct', operator: '<=', value: 2.0 },
  ],
  objectives: [
    { metric: 'amplitude_uniformity', direction: 'maximize', weight: 1.0 },
    { metric: 'stress_safety_factor', direction: 'maximize', weight: 0.5 },
  ],
})
</script>

<style scoped>
.property-editor {
  height: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  font-size: 12px;
}

.pe-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  gap: 8px;
  color: var(--color-text-secondary);
}

.pe-empty-icon {
  font-size: 24px;
  opacity: 0.5;
}

.pe-empty-text {
  font-size: 12px;
  font-style: italic;
}

.pe-content {
  padding: 0;
}

.pe-title {
  font-size: 12px;
  font-weight: 600;
  padding: 8px 12px;
  margin: 0;
  color: var(--color-text-primary);
  background-color: var(--color-bg-primary);
  border-bottom: 1px solid var(--color-border);
}

.pe-accordion {
  border-bottom: 1px solid var(--color-border);
}

.pe-accordion-header {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 6px 12px;
  border: none;
  background: none;
  color: var(--color-text-primary);
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.15s;
}

.pe-accordion-header:hover {
  background-color: var(--color-bg-card);
}

.pe-accordion-icon {
  font-size: 8px;
  width: 10px;
  color: var(--color-text-secondary);
}

.pe-accordion-body {
  padding: 6px 12px 10px;
}

.pe-field {
  margin-bottom: 8px;
}

.pe-label {
  display: block;
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-bottom: 3px;
}

.pe-input {
  width: 100%;
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 12px;
  font-family: ui-monospace, monospace;
}

.pe-input:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.pe-input--small {
  width: 70px;
  padding: 3px 6px;
  font-size: 11px;
}

.pe-select {
  width: 100%;
  padding: 4px 8px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 12px;
  cursor: pointer;
}

.pe-select:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.pe-select--small {
  width: auto;
  padding: 3px 6px;
  font-size: 11px;
}

.pe-info-text {
  font-size: 11px;
  color: var(--color-text-secondary);
  margin-bottom: 8px;
  font-style: italic;
}

.pe-var-card {
  background-color: var(--color-bg-primary);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  padding: 6px 8px;
  margin-bottom: 6px;
}

.pe-var-name {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-accent-blue);
  margin-bottom: 4px;
}

.pe-var-range {
  display: flex;
  align-items: center;
  gap: 4px;
}

.pe-var-sep {
  color: var(--color-text-secondary);
}

.pe-constraint-row,
.pe-objective-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 6px;
}

.pe-constraint-metric,
.pe-objective-metric {
  font-size: 11px;
  color: var(--color-text-primary);
  min-width: 60px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style>
