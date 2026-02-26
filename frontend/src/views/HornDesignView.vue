<template>
  <div class="p-6 max-w-7xl mx-auto">
    <h1 class="text-2xl font-bold mb-6">{{ $t('hornDesign.title') }}</h1>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left: Design Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('hornDesign.designPanel') }}</h2>

        <!-- Horn Type -->
        <div>
          <label class="label-text">{{ $t('hornDesign.hornType') }}</label>
          <select v-model="form.horn_type" class="input-field w-full">
            <option value="flat">{{ $t('hornDesign.typeFlat') }}</option>
            <option value="cylindrical">{{ $t('hornDesign.typeCylindrical') }}</option>
            <option value="exponential">{{ $t('hornDesign.typeExponential') }}</option>
            <option value="blade">{{ $t('hornDesign.typeBlade') }}</option>
            <option value="stepped">{{ $t('hornDesign.typeStepped') }}</option>
          </select>
        </div>

        <!-- Dimensions -->
        <div>
          <label class="label-text">{{ $t('hornDesign.dimensions') }}</label>
          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.widthMm') }}</label>
              <input v-model.number="form.width_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.heightMm') }}</label>
              <input v-model.number="form.height_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.lengthMm') }}</label>
              <input v-model.number="form.length_mm" type="number" class="input-field w-full" min="1" step="0.5" />
            </div>
          </div>
        </div>

        <!-- Material -->
        <div>
          <label class="label-text">{{ $t('hornDesign.material') }}</label>
          <select v-model="form.material" class="input-field w-full">
            <option v-for="mat in materials" :key="mat" :value="mat">{{ mat }}</option>
          </select>
        </div>

        <!-- Knurl Parameters -->
        <div>
          <label class="label-text">{{ $t('hornDesign.knurlParams') }}</label>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.knurlType') }}</label>
              <select v-model="form.knurl_type" class="input-field w-full">
                <option value="linear">{{ $t('hornDesign.knurlLinear') }}</option>
                <option value="cross_hatch">{{ $t('hornDesign.knurlCrossHatch') }}</option>
                <option value="diamond">{{ $t('hornDesign.knurlDiamond') }}</option>
              </select>
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.knurlPitch') }}</label>
              <input v-model.number="form.knurl_pitch" type="number" class="input-field w-full" min="0.1" step="0.1" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.knurlToothWidth') }}</label>
              <input v-model.number="form.knurl_tooth_width" type="number" class="input-field w-full" min="0.05" step="0.05" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.knurlDepth') }}</label>
              <input v-model.number="form.knurl_depth" type="number" class="input-field w-full" min="0.05" step="0.05" />
            </div>
          </div>
        </div>

        <!-- Chamfer Parameters -->
        <div>
          <label class="label-text">{{ $t('hornDesign.chamferParams') }}</label>
          <div class="grid grid-cols-3 gap-2">
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.chamferRadius') }}</label>
              <input v-model.number="form.chamfer_radius" type="number" class="input-field w-full" min="0" step="0.1" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.chamferAngle') }}</label>
              <input v-model.number="form.chamfer_angle" type="number" class="input-field w-full" min="0" step="1" />
            </div>
            <div>
              <label class="text-xs" style="color: var(--color-text-secondary)">{{ $t('hornDesign.edgeTreatment') }}</label>
              <select v-model="form.edge_treatment" class="input-field w-full">
                <option value="sharp">{{ $t('hornDesign.edgeSharp') }}</option>
                <option value="chamfer">{{ $t('hornDesign.edgeChamfer') }}</option>
                <option value="fillet">{{ $t('hornDesign.edgeFillet') }}</option>
              </select>
            </div>
          </div>
        </div>

        <button class="btn-primary w-full" :disabled="generating" @click="generateHorn">
          {{ generating ? $t('hornDesign.generating') : $t('hornDesign.generate') }}
        </button>
      </div>

      <!-- Center: 3D Preview -->
      <div class="card flex flex-col">
        <h2 class="text-lg font-semibold mb-4">{{ $t('hornDesign.preview3d') }}</h2>
        <div class="flex-1" style="min-height: 400px">
          <ThreeViewer
            ref="viewerRef"
            :mesh="meshData"
            :wireframe="wireframe"
            :show-axes="true"
            :placeholder="$t('hornDesign.viewerPlaceholder')"
          />
        </div>
        <div class="flex gap-2 mt-3">
          <button class="btn-small" @click="viewerRef?.resetCamera()">{{ $t('hornDesign.resetView') }}</button>
          <button class="btn-small" @click="wireframe = !wireframe">
            {{ wireframe ? $t('hornDesign.solid') : $t('hornDesign.wireframe') }}
          </button>
        </div>
      </div>

      <!-- Right: Analysis Panel -->
      <div class="card space-y-4">
        <h2 class="text-lg font-semibold">{{ $t('hornDesign.analysisPanel') }}</h2>

        <!-- Chamfer Analysis -->
        <div v-if="chamferResult" class="space-y-3">
          <h3 class="text-sm font-semibold" style="color: var(--color-accent-orange)">
            {{ $t('hornDesign.chamferAnalysis') }}
          </h3>

          <div class="grid grid-cols-2 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('hornDesign.stressKt') }}</span>
              <div class="font-bold">{{ chamferResult.kt?.toFixed(3) ?? '--' }}</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('hornDesign.riskLevel') }}</span>
              <div class="font-bold" :style="{ color: riskColor }">
                {{ chamferResult.risk_level ?? '--' }}
              </div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('hornDesign.areaCorrection') }}</span>
              <div class="font-bold">{{ chamferResult.area_correction?.toFixed(4) ?? '--' }}</div>
            </div>
          </div>
        </div>

        <!-- Volume / Surface Area -->
        <div v-if="generateResult" class="space-y-3">
          <h3 class="text-sm font-semibold">{{ $t('hornDesign.geometryInfo') }}</h3>
          <div class="grid grid-cols-1 gap-2 text-sm">
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('hornDesign.volume') }}</span>
              <div class="font-bold">{{ generateResult.volume_mm3?.toFixed(1) ?? '--' }} mm&sup3;</div>
            </div>
            <div class="p-2 rounded" style="background-color: var(--color-bg-card)">
              <span style="color: var(--color-text-secondary)">{{ $t('hornDesign.surfaceArea') }}</span>
              <div class="font-bold">{{ generateResult.surface_area_mm2?.toFixed(1) ?? '--' }} mm&sup2;</div>
            </div>
          </div>
        </div>

        <!-- Download -->
        <div v-if="generateResult?.id" class="space-y-2">
          <h3 class="text-sm font-semibold">{{ $t('hornDesign.download') }}</h3>
          <div class="flex gap-2">
            <button class="btn-small flex-1" @click="downloadFile('step')">
              STEP
            </button>
            <button class="btn-small flex-1" @click="downloadFile('stl')">
              STL
            </button>
          </div>
        </div>

        <!-- Error -->
        <div
          v-if="error"
          class="p-3 rounded text-sm"
          style="background-color: rgba(220, 38, 38, 0.1); color: #dc2626"
        >
          {{ error }}
        </div>

        <!-- No results placeholder -->
        <div
          v-if="!generateResult && !chamferResult && !error"
          class="text-sm text-center py-8"
          style="color: var(--color-text-secondary)"
        >
          {{ $t('hornDesign.noResults') }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import apiClient from '@/api/client'
import ThreeViewer from '@/components/viewer/ThreeViewer.vue'
import type { MeshData } from '@/components/viewer/ThreeViewer.vue'

const { t } = useI18n()

const viewerRef = ref<InstanceType<typeof ThreeViewer> | null>(null)
const wireframe = ref(false)
const generating = ref(false)
const error = ref<string | null>(null)

interface HornForm {
  horn_type: string
  width_mm: number
  height_mm: number
  length_mm: number
  material: string
  knurl_type: string
  knurl_pitch: number
  knurl_tooth_width: number
  knurl_depth: number
  chamfer_radius: number
  chamfer_angle: number
  edge_treatment: string
}

const form = ref<HornForm>({
  horn_type: 'cylindrical',
  width_mm: 25,
  height_mm: 80,
  length_mm: 25,
  material: 'Titanium Ti-6Al-4V',
  knurl_type: 'linear',
  knurl_pitch: 1.5,
  knurl_tooth_width: 0.5,
  knurl_depth: 0.3,
  chamfer_radius: 0.5,
  chamfer_angle: 45,
  edge_treatment: 'chamfer',
})

const materials = [
  'Titanium Ti-6Al-4V',
  'Steel D2',
  'Aluminum 7075-T6',
  'Steel H13',
  'Copper C110',
  'Tungsten Carbide',
]

interface GenerateResult {
  id?: string
  volume_mm3?: number
  surface_area_mm2?: number
  mesh?: MeshData | null
}

interface ChamferResult {
  kt?: number
  risk_level?: string
  area_correction?: number
}

const meshData = ref<MeshData | null>(null)
const generateResult = ref<GenerateResult | null>(null)
const chamferResult = ref<ChamferResult | null>(null)

const riskColor = computed(() => {
  const level = chamferResult.value?.risk_level
  if (level === 'low') return '#22c55e'
  if (level === 'medium') return '#eab308'
  return '#ef4444'
})

async function generateHorn() {
  generating.value = true
  error.value = null
  generateResult.value = null
  chamferResult.value = null
  meshData.value = null

  try {
    const res = await apiClient.post<GenerateResult>('/horn/generate', form.value, { timeout: 60000 })
    generateResult.value = res.data
    if (res.data.mesh) {
      meshData.value = res.data.mesh
    }
    // Run chamfer analysis
    try {
      const chamferRes = await apiClient.post<ChamferResult>('/horn/chamfer-analysis', {
        horn_type: form.value.horn_type,
        width_mm: form.value.width_mm,
        height_mm: form.value.height_mm,
        length_mm: form.value.length_mm,
        chamfer_radius: form.value.chamfer_radius,
        chamfer_angle: form.value.chamfer_angle,
        edge_treatment: form.value.edge_treatment,
      })
      chamferResult.value = chamferRes.data
    } catch {
      // Chamfer analysis is optional; don't block
    }
  } catch (err: any) {
    error.value = err.response?.data?.detail || err.message || t('hornDesign.generateFailed')
  } finally {
    generating.value = false
  }
}

function downloadFile(fmt: string) {
  if (!generateResult.value?.id) return
  const url = `/api/v1/horn/download/${generateResult.value.id}?fmt=${fmt}`
  window.open(url, '_blank')
}
</script>

<style scoped>
.card {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 0.5rem;
  padding: 1.25rem;
}
.label-text {
  display: block;
  font-size: 0.875rem;
  font-weight: 600;
  color: var(--color-text-secondary);
  margin-bottom: 0.25rem;
}
.input-field {
  display: block;
  padding: 0.375rem 0.5rem;
  border-radius: 0.375rem;
  font-size: 0.875rem;
  background-color: var(--color-bg-primary);
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  margin-top: 0.25rem;
}
.input-field:focus {
  outline: none;
  border-color: var(--color-accent-orange);
}
.btn-primary {
  padding: 0.5rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 600;
  font-size: 0.875rem;
  background-color: var(--color-accent-orange);
  color: #fff;
  border: none;
  cursor: pointer;
  transition: opacity 0.2s;
}
.btn-primary:hover:not(:disabled) { opacity: 0.9; }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-small {
  padding: 0.25rem 0.75rem;
  border-radius: 0.375rem;
  font-size: 0.75rem;
  background-color: transparent;
  color: var(--color-text-primary);
  border: 1px solid var(--color-border);
  cursor: pointer;
}
.btn-small:hover { border-color: var(--color-accent-orange); }
</style>
