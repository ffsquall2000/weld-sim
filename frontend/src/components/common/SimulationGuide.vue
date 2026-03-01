<template>
  <div class="sg-container" :class="{ 'sg-collapsed': isCollapsed }">
    <!-- Header -->
    <button class="sg-header" @click="toggleCollapse">
      <div class="sg-header-left">
        <span class="sg-icon">&#x1F4A1;</span>
        <span class="sg-title">{{ t(`${guideKey}.title`) }}</span>
      </div>
      <span class="sg-toggle" :class="{ 'sg-toggle-open': !isCollapsed }">&#9660;</span>
    </button>

    <!-- Content -->
    <transition name="sg-slide">
      <div v-if="!isCollapsed" class="sg-body">
        <!-- Purpose -->
        <div class="sg-section">
          <div class="sg-section-header">
            <span class="sg-section-icon">&#x1F3AF;</span>
            <span class="sg-section-title">{{ t('simulationGuide.purpose') }}</span>
          </div>
          <p class="sg-section-text">{{ t(`${guideKey}.purpose`) }}</p>
        </div>

        <!-- Practical Significance -->
        <div class="sg-section">
          <div class="sg-section-header">
            <span class="sg-section-icon">&#x26A1;</span>
            <span class="sg-section-title">{{ t('simulationGuide.significance') }}</span>
          </div>
          <p class="sg-section-text">{{ t(`${guideKey}.significance`) }}</p>
        </div>

        <!-- Expected Effects -->
        <div class="sg-section">
          <div class="sg-section-header">
            <span class="sg-section-icon">&#x1F4CA;</span>
            <span class="sg-section-title">{{ t('simulationGuide.expectedEffects') }}</span>
          </div>
          <p class="sg-section-text">{{ t(`${guideKey}.expectedEffects`) }}</p>
        </div>

        <!-- Tips -->
        <div class="sg-section">
          <div class="sg-section-header">
            <span class="sg-section-icon">&#x1F4A1;</span>
            <span class="sg-section-title">{{ t('simulationGuide.tips') }}</span>
          </div>
          <ol class="sg-tips-list">
            <li v-for="(tip, index) in tipsList" :key="index" class="sg-tip-item">
              {{ tip }}
            </li>
          </ol>
        </div>

        <!-- Parameter Help (optional) -->
        <div v-if="hasParameterHelp" class="sg-section">
          <div class="sg-section-header">
            <span class="sg-section-icon">&#x1F4CB;</span>
            <span class="sg-section-title">{{ t('simulationGuide.parameterHelp') }}</span>
          </div>
          <div class="sg-param-grid">
            <template v-for="(entry, index) in parameterHelpEntries" :key="index">
              <div class="sg-param-name">{{ entry.name }}</div>
              <div class="sg-param-desc">{{ entry.description }}</div>
            </template>
          </div>
        </div>
      </div>
    </transition>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(defineProps<{
  guideKey: string
  collapsed?: boolean
}>(), {
  collapsed: false,
})

const { t, tm, te } = useI18n()

const isCollapsed = ref(props.collapsed)

function toggleCollapse() {
  isCollapsed.value = !isCollapsed.value
}

const tipsList = computed<string[]>(() => {
  const key = `${props.guideKey}.tips`
  if (!te(key)) return []
  const raw = tm(key)
  if (Array.isArray(raw)) {
    return raw.map((item: unknown) => {
      if (typeof item === 'string') return item
      if (item && typeof item === 'object' && 'value' in (item as Record<string, unknown>)) {
        return String((item as Record<string, unknown>).value)
      }
      return String(item)
    })
  }
  return [String(raw)]
})

const hasParameterHelp = computed(() => {
  return te(`${props.guideKey}.parameterHelp`)
})

interface ParamHelpEntry {
  name: string
  description: string
}

const parameterHelpEntries = computed<ParamHelpEntry[]>(() => {
  const key = `${props.guideKey}.parameterHelp`
  if (!te(key)) return []
  const raw = tm(key)
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    return Object.entries(raw as Record<string, unknown>).map(([name, desc]) => ({
      name,
      description: typeof desc === 'string' ? desc
        : (desc && typeof desc === 'object' && 'value' in (desc as Record<string, unknown>))
          ? String((desc as Record<string, unknown>).value)
          : String(desc),
    }))
  }
  if (Array.isArray(raw)) {
    return (raw as Array<Record<string, unknown>>).map((item) => ({
      name: String(item.name ?? item.param ?? ''),
      description: String(item.description ?? item.desc ?? item.help ?? ''),
    }))
  }
  return []
})
</script>

<style scoped>
.sg-container {
  border: 1px solid var(--color-accent-blue, #58a6ff);
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 16px;
  background-color: color-mix(in srgb, var(--color-accent-blue, #58a6ff) 6%, var(--color-bg-primary, #0d1117));
}

.sg-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 10px 14px;
  border: none;
  background-color: color-mix(in srgb, var(--color-accent-blue, #58a6ff) 12%, var(--color-bg-primary, #0d1117));
  cursor: pointer;
  transition: background-color 0.15s;
  color: var(--color-text-primary);
}

.sg-header:hover {
  background-color: color-mix(in srgb, var(--color-accent-blue, #58a6ff) 18%, var(--color-bg-primary, #0d1117));
}

.sg-header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.sg-icon {
  font-size: 16px;
  line-height: 1;
}

.sg-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-accent-blue, #58a6ff);
}

.sg-toggle {
  font-size: 10px;
  color: var(--color-text-secondary);
  transition: transform 0.25s ease;
  transform: rotate(-90deg);
}

.sg-toggle-open {
  transform: rotate(0deg);
}

/* Collapse/Expand transition */
.sg-slide-enter-active,
.sg-slide-leave-active {
  transition: all 0.25s ease;
  max-height: 2000px;
  opacity: 1;
  overflow: hidden;
}

.sg-slide-enter-from,
.sg-slide-leave-to {
  max-height: 0;
  opacity: 0;
  overflow: hidden;
}

/* Body */
.sg-body {
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

/* Sections */
.sg-section {
  padding: 0;
}

.sg-section-header {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.sg-section-icon {
  font-size: 14px;
  line-height: 1;
}

.sg-section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--color-text-primary);
}

.sg-section-text {
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-secondary);
  margin: 0;
  padding-left: 22px;
}

/* Tips list */
.sg-tips-list {
  margin: 0;
  padding-left: 22px;
  list-style-position: inside;
  counter-reset: tip-counter;
}

.sg-tip-item {
  font-size: 12px;
  line-height: 1.6;
  color: var(--color-text-secondary);
  counter-increment: tip-counter;
  list-style: decimal;
  padding: 1px 0;
}

/* Parameter Help grid */
.sg-param-grid {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 4px 12px;
  padding-left: 22px;
}

.sg-param-name {
  font-size: 11px;
  font-weight: 600;
  color: var(--color-accent-blue, #58a6ff);
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  white-space: nowrap;
}

.sg-param-desc {
  font-size: 11px;
  line-height: 1.5;
  color: var(--color-text-secondary);
}

/* Responsive */
@media (max-width: 640px) {
  .sg-param-grid {
    grid-template-columns: 1fr;
    gap: 2px;
  }

  .sg-param-name {
    margin-top: 4px;
  }
}
</style>
