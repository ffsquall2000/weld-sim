<template>
  <div class="material-selector">
    <!-- Search -->
    <div class="ms-search">
      <input
        v-model="searchQuery"
        type="text"
        class="ms-search-input"
        :placeholder="t('materialSelector.searchPlaceholder')"
      />
    </div>

    <!-- Category Tabs -->
    <div class="ms-tabs">
      <button
        v-for="cat in categories"
        :key="cat.key"
        class="ms-tab"
        :class="{ 'ms-tab--active': activeCategory === cat.key }"
        @click="activeCategory = cat.key"
      >
        {{ t(cat.label) }}
      </button>
    </div>

    <!-- Material List -->
    <div class="ms-list">
      <div
        v-for="mat in filteredMaterials"
        :key="mat.name"
        class="ms-item"
        :class="{ 'ms-item--selected': selectedMaterial?.name === mat.name }"
        @click="selectMaterial(mat)"
      >
        <div class="ms-item-name">{{ mat.name }}</div>
        <div class="ms-item-category">{{ mat.category }}</div>
      </div>
      <div v-if="filteredMaterials.length === 0" class="ms-empty">
        {{ t('materialSelector.noMaterials') }}
      </div>
    </div>

    <!-- Material Detail Card -->
    <div v-if="selectedMaterial" class="ms-detail">
      <h4 class="ms-detail-title">{{ selectedMaterial.name }}</h4>
      <div class="ms-detail-grid">
        <div class="ms-detail-row">
          <span class="ms-detail-label">E ({{ t('materialSelector.youngsModulus') }})</span>
          <span class="ms-detail-value">{{ formatNumber(selectedMaterial.E) }} GPa</span>
        </div>
        <div class="ms-detail-row">
          <span class="ms-detail-label">&nu; ({{ t('materialSelector.poissonsRatio') }})</span>
          <span class="ms-detail-value">{{ selectedMaterial.nu.toFixed(3) }}</span>
        </div>
        <div class="ms-detail-row">
          <span class="ms-detail-label">&rho; ({{ t('materialSelector.density') }})</span>
          <span class="ms-detail-value">{{ formatNumber(selectedMaterial.rho) }} kg/m&sup3;</span>
        </div>
        <div class="ms-detail-row">
          <span class="ms-detail-label">k ({{ t('materialSelector.thermalConductivity') }})</span>
          <span class="ms-detail-value">{{ formatNumber(selectedMaterial.k) }} W/(m&middot;K)</span>
        </div>
        <div class="ms-detail-row">
          <span class="ms-detail-label">c<sub>p</sub> ({{ t('materialSelector.specificHeat') }})</span>
          <span class="ms-detail-value">{{ formatNumber(selectedMaterial.cp) }} J/(kg&middot;K)</span>
        </div>
        <div class="ms-detail-row">
          <span class="ms-detail-label">&sigma;<sub>y</sub> ({{ t('materialSelector.yieldStrength') }})</span>
          <span class="ms-detail-value">{{ formatNumber(selectedMaterial.sigma_y) }} MPa</span>
        </div>
      </div>
      <button class="ms-select-btn" @click="applyMaterial">
        {{ t('materialSelector.apply') }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useGeometryStore } from '@/stores/geometry'

const { t } = useI18n()
const geometryStore = useGeometryStore()

interface Material {
  name: string
  category: string
  E: number       // Young's modulus in GPa
  nu: number      // Poisson's ratio
  rho: number     // Density in kg/m3
  k: number       // Thermal conductivity W/(m*K)
  cp: number      // Specific heat J/(kg*K)
  sigma_y: number // Yield strength MPa
}

const searchQuery = ref('')
const activeCategory = ref('horn')
const selectedMaterial = ref<Material | null>(null)

const categories = [
  { key: 'horn', label: 'materialSelector.hornMaterials' },
  { key: 'workpiece', label: 'materialSelector.workpieceMaterials' },
  { key: 'custom', label: 'materialSelector.customMaterials' },
]

// Built-in FEA materials library
const allMaterials = ref<Material[]>([
  // Horn / Tool Materials
  { name: 'Ti-6Al-4V', category: 'horn', E: 113.8, nu: 0.342, rho: 4430, k: 6.7, cp: 526, sigma_y: 880 },
  { name: 'Titanium Grade 2', category: 'horn', E: 105, nu: 0.34, rho: 4510, k: 16.4, cp: 523, sigma_y: 275 },
  { name: 'Tool Steel D2', category: 'horn', E: 210, nu: 0.3, rho: 7700, k: 20.0, cp: 460, sigma_y: 1650 },
  { name: 'Tool Steel H13', category: 'horn', E: 210, nu: 0.3, rho: 7800, k: 24.3, cp: 460, sigma_y: 1380 },
  { name: '7075-T6 Aluminum', category: 'horn', E: 71.7, nu: 0.33, rho: 2810, k: 130, cp: 960, sigma_y: 503 },
  { name: '2024-T3 Aluminum', category: 'horn', E: 73.1, nu: 0.33, rho: 2780, k: 121, cp: 875, sigma_y: 345 },
  { name: 'Maraging Steel C300', category: 'horn', E: 190, nu: 0.3, rho: 8100, k: 25.4, cp: 450, sigma_y: 2000 },
  { name: 'Tungsten Carbide', category: 'horn', E: 600, nu: 0.24, rho: 15630, k: 84, cp: 200, sigma_y: 4000 },

  // Workpiece Materials
  { name: 'Nickel 201', category: 'workpiece', E: 207, nu: 0.31, rho: 8890, k: 70, cp: 456, sigma_y: 148 },
  { name: 'Nickel 200', category: 'workpiece', E: 204, nu: 0.31, rho: 8890, k: 70.2, cp: 456, sigma_y: 193 },
  { name: 'Copper C110', category: 'workpiece', E: 117, nu: 0.34, rho: 8940, k: 388, cp: 385, sigma_y: 69 },
  { name: 'Copper C101', category: 'workpiece', E: 117, nu: 0.34, rho: 8940, k: 391, cp: 385, sigma_y: 55 },
  { name: 'Aluminum 1100', category: 'workpiece', E: 69, nu: 0.33, rho: 2710, k: 222, cp: 904, sigma_y: 34 },
  { name: 'Aluminum 3003', category: 'workpiece', E: 69, nu: 0.33, rho: 2730, k: 193, cp: 893, sigma_y: 145 },
  { name: 'Stainless Steel 304', category: 'workpiece', E: 193, nu: 0.29, rho: 7900, k: 16.2, cp: 500, sigma_y: 215 },
  { name: 'Brass C260', category: 'workpiece', E: 110, nu: 0.35, rho: 8530, k: 120, cp: 380, sigma_y: 200 },
  { name: 'Silver Ag', category: 'workpiece', E: 83, nu: 0.37, rho: 10500, k: 429, cp: 235, sigma_y: 170 },
  { name: 'Gold Au', category: 'workpiece', E: 79, nu: 0.44, rho: 19300, k: 317, cp: 129, sigma_y: 205 },
])

const filteredMaterials = computed(() => {
  let result = allMaterials.value
  if (activeCategory.value !== 'custom') {
    result = result.filter((m) => m.category === activeCategory.value)
  }
  if (searchQuery.value.trim()) {
    const q = searchQuery.value.toLowerCase()
    result = result.filter((m) => m.name.toLowerCase().includes(q))
  }
  return result
})

function selectMaterial(mat: Material) {
  selectedMaterial.value = mat
}

function applyMaterial() {
  if (!selectedMaterial.value) return
  const body = geometryStore.selectedBody
  if (body) {
    geometryStore.updateBodyMaterial(body, selectedMaterial.value.name)
  }
}

function formatNumber(val: number): string {
  if (val >= 1e6) return (val / 1e6).toFixed(1) + 'M'
  if (val >= 1e3) return val.toLocaleString()
  return val.toFixed(1)
}
</script>

<style scoped>
.material-selector {
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 12px;
}

/* Search */
.ms-search {
  padding: 8px;
  border-bottom: 1px solid var(--color-border);
}

.ms-search-input {
  width: 100%;
  padding: 5px 10px;
  border: 1px solid var(--color-border);
  border-radius: 4px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 12px;
}

.ms-search-input:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.ms-search-input::placeholder {
  color: var(--color-text-secondary);
}

/* Category Tabs */
.ms-tabs {
  display: flex;
  border-bottom: 1px solid var(--color-border);
}

.ms-tab {
  flex: 1;
  padding: 6px 8px;
  border: none;
  background: none;
  color: var(--color-text-secondary);
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: color 0.15s, border-color 0.15s;
}

.ms-tab:hover {
  color: var(--color-text-primary);
}

.ms-tab--active {
  color: var(--color-text-primary);
  border-bottom-color: var(--color-accent-orange);
}

/* Material List */
.ms-list {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
}

.ms-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 12px;
  cursor: pointer;
  border-bottom: 1px solid rgba(48, 54, 61, 0.5);
  transition: background-color 0.1s;
}

.ms-item:hover {
  background-color: var(--color-bg-card);
}

.ms-item--selected {
  background-color: var(--color-bg-card);
  border-left: 2px solid var(--color-accent-orange);
  padding-left: 10px;
}

.ms-item-name {
  font-size: 12px;
  color: var(--color-text-primary);
  font-weight: 500;
}

.ms-item-category {
  font-size: 10px;
  color: var(--color-text-secondary);
  text-transform: capitalize;
}

.ms-empty {
  padding: 20px;
  text-align: center;
  color: var(--color-text-secondary);
  font-style: italic;
}

/* Material Detail */
.ms-detail {
  border-top: 1px solid var(--color-border);
  padding: 10px 12px;
  background-color: var(--color-bg-primary);
}

.ms-detail-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-accent-orange);
  margin: 0 0 8px;
}

.ms-detail-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
  margin-bottom: 10px;
}

.ms-detail-row {
  display: flex;
  flex-direction: column;
  padding: 4px 6px;
  background-color: var(--color-bg-card);
  border-radius: 4px;
}

.ms-detail-label {
  font-size: 10px;
  color: var(--color-text-secondary);
  margin-bottom: 2px;
}

.ms-detail-value {
  font-size: 12px;
  font-family: ui-monospace, monospace;
  color: var(--color-text-primary);
  font-weight: 600;
}

.ms-select-btn {
  width: 100%;
  padding: 6px;
  border: 1px solid var(--color-accent-orange);
  border-radius: 4px;
  background-color: rgba(255, 152, 0, 0.1);
  color: var(--color-accent-orange);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color 0.15s;
}

.ms-select-btn:hover {
  background-color: rgba(255, 152, 0, 0.2);
}
</style>
