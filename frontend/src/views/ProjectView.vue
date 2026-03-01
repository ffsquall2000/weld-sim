<template>
  <div class="project-view">
    <!-- Header -->
    <div class="pv-header">
      <div class="pv-header-left">
        <h1 class="pv-title">{{ t('projects.title') }}</h1>
        <span class="pv-count">{{ projectStore.projectCount }} {{ t('projects.projectsLabel') }}</span>
      </div>
      <div class="pv-header-right">
        <div class="pv-search">
          <input
            v-model="searchQuery"
            type="text"
            class="pv-search-input"
            :placeholder="t('projects.searchPlaceholder')"
          />
        </div>
        <select v-model="filterType" class="pv-filter-select">
          <option value="">{{ t('projects.allTypes') }}</option>
          <option value="li_battery_tab">{{ t('wizard.appLiBatteryTab') }}</option>
          <option value="li_battery_busbar">{{ t('wizard.appLiBatteryBusbar') }}</option>
          <option value="li_battery_collector">{{ t('wizard.appLiBatteryCollector') }}</option>
          <option value="general_metal">{{ t('wizard.appGeneralMetal') }}</option>
        </select>
        <button class="pv-new-btn" @click="showCreateModal = true">
          + {{ t('projects.createNew') }}
        </button>
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="projectStore.loading" class="pv-loading">
      <span class="pv-spinner" />
      <span>{{ t('common.loading') }}</span>
    </div>

    <!-- Error State -->
    <div v-else-if="projectStore.error" class="pv-error">
      <span class="pv-error-icon">&#9888;</span>
      <span>{{ projectStore.error }}</span>
      <button class="pv-retry-btn" @click="projectStore.fetchProjects()">
        {{ t('projects.retry') }}
      </button>
    </div>

    <!-- Empty State -->
    <div v-else-if="filteredProjects.length === 0" class="pv-empty">
      <div class="pv-empty-icon">&#128193;</div>
      <h3 class="pv-empty-title">{{ t('projects.noProjects') }}</h3>
      <p class="pv-empty-desc">{{ t('projects.noProjectsDesc') }}</p>
      <button class="pv-new-btn" @click="showCreateModal = true">
        + {{ t('projects.createFirst') }}
      </button>
    </div>

    <!-- Project Grid -->
    <div v-else class="pv-grid">
      <div
        v-for="project in filteredProjects"
        :key="project.id"
        class="pv-card"
        @click="openProject(project)"
      >
        <div class="pv-card-header">
          <span class="pv-card-icon">&#9641;</span>
          <div class="pv-card-meta">
            <h3 class="pv-card-name">{{ project.name }}</h3>
            <span class="pv-card-type">{{ appTypeLabel(project.application_type) }}</span>
          </div>
        </div>
        <p v-if="project.description" class="pv-card-desc">{{ project.description }}</p>
        <div class="pv-card-footer">
          <span class="pv-card-date">
            {{ formatDate(project.updated_at) }}
          </span>
          <div class="pv-card-tags" v-if="project.tags?.length">
            <span v-for="tag in project.tags.slice(0, 3)" :key="tag" class="pv-card-tag">
              {{ tag }}
            </span>
          </div>
        </div>
        <div class="pv-card-actions" @click.stop>
          <button
            class="pv-card-action-btn pv-card-action-btn--danger"
            :title="t('projects.delete')"
            @click="confirmDelete(project)"
          >
            &#10005;
          </button>
        </div>
      </div>
    </div>

    <!-- Create Project Modal -->
    <Teleport to="body">
      <div v-if="showCreateModal" class="pv-modal-overlay" @click.self="showCreateModal = false">
        <div class="pv-modal">
          <h2 class="pv-modal-title">{{ t('projects.createNew') }}</h2>
          <div class="pv-modal-body">
            <div class="pv-form-field">
              <label class="pv-form-label">{{ t('projects.name') }}</label>
              <input v-model="newProject.name" type="text" class="pv-form-input" :placeholder="t('projects.namePlaceholder')" />
            </div>
            <div class="pv-form-field">
              <label class="pv-form-label">{{ t('projects.description') }}</label>
              <textarea v-model="newProject.description" class="pv-form-textarea" rows="3" :placeholder="t('projects.descriptionPlaceholder')" />
            </div>
            <div class="pv-form-field">
              <label class="pv-form-label">{{ t('projects.applicationType') }}</label>
              <select v-model="newProject.application_type" class="pv-form-select">
                <option value="li_battery_tab">{{ t('wizard.appLiBatteryTab') }}</option>
                <option value="li_battery_busbar">{{ t('wizard.appLiBatteryBusbar') }}</option>
                <option value="li_battery_collector">{{ t('wizard.appLiBatteryCollector') }}</option>
                <option value="general_metal">{{ t('wizard.appGeneralMetal') }}</option>
              </select>
            </div>
            <div class="pv-form-field">
              <label class="pv-form-label">{{ t('projects.tags') }}</label>
              <input v-model="tagsInput" type="text" class="pv-form-input" :placeholder="t('projects.tagsPlaceholder')" />
            </div>
          </div>
          <div class="pv-modal-footer">
            <button class="pv-modal-btn pv-modal-btn--cancel" @click="showCreateModal = false">
              {{ t('common.cancel') }}
            </button>
            <button
              class="pv-modal-btn pv-modal-btn--create"
              :disabled="!newProject.name.trim()"
              @click="handleCreate"
            >
              {{ t('projects.create') }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useProjectStore, type Project } from '@/stores/project'

const { t } = useI18n()
const router = useRouter()
const projectStore = useProjectStore()

const searchQuery = ref('')
const filterType = ref('')
const showCreateModal = ref(false)
const tagsInput = ref('')

const newProject = reactive({
  name: '',
  description: '',
  application_type: 'li_battery_tab',
})

const filteredProjects = computed(() => {
  let result = projectStore.sortedProjects
  if (searchQuery.value.trim()) {
    const q = searchQuery.value.toLowerCase()
    result = result.filter(
      (p) =>
        p.name.toLowerCase().includes(q) ||
        (p.description && p.description.toLowerCase().includes(q))
    )
  }
  if (filterType.value) {
    result = result.filter((p) => p.application_type === filterType.value)
  }
  return result
})

function appTypeLabel(type: string): string {
  const map: Record<string, string> = {
    li_battery_tab: t('wizard.appLiBatteryTab'),
    li_battery_busbar: t('wizard.appLiBatteryBusbar'),
    li_battery_collector: t('wizard.appLiBatteryCollector'),
    general_metal: t('wizard.appGeneralMetal'),
  }
  return map[type] || type
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  return d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

function openProject(project: Project) {
  projectStore.setCurrentProject(project)
  router.push({ name: 'workbench', params: { projectId: project.id } })
}

async function handleCreate() {
  const tags = tagsInput.value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
  const created = await projectStore.createProject({
    name: newProject.name,
    description: newProject.description || undefined,
    application_type: newProject.application_type,
    tags: tags.length > 0 ? tags : undefined,
  })
  if (created) {
    showCreateModal.value = false
    newProject.name = ''
    newProject.description = ''
    newProject.application_type = 'li_battery_tab'
    tagsInput.value = ''
    router.push({ name: 'workbench', params: { projectId: created.id } })
  }
}

function confirmDelete(project: Project) {
  // Simple confirm - in production, use a modal dialog
  if (window.confirm(t('projects.confirmDelete', { name: project.name }))) {
    projectStore.deleteProject(project.id)
  }
}

onMounted(() => {
  projectStore.fetchProjects()
})
</script>

<style scoped>
.project-view {
  padding: 24px 32px;
  max-width: 1200px;
  margin: 0 auto;
}

/* Header */
.pv-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.pv-header-left {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.pv-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--color-text-primary);
  margin: 0;
}

.pv-count {
  font-size: 14px;
  color: var(--color-text-secondary);
}

.pv-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.pv-search-input {
  padding: 7px 14px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 13px;
  width: 200px;
}

.pv-search-input:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.pv-search-input::placeholder {
  color: var(--color-text-secondary);
}

.pv-filter-select {
  padding: 7px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 13px;
  cursor: pointer;
}

.pv-filter-select:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.pv-new-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background-color: var(--color-accent-orange);
  color: #fff;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
  white-space: nowrap;
}

.pv-new-btn:hover {
  opacity: 0.9;
}

/* Loading / Error / Empty */
.pv-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 60px 0;
  color: var(--color-text-secondary);
  font-size: 14px;
}

.pv-spinner {
  width: 20px;
  height: 20px;
  border: 2px solid var(--color-border);
  border-top-color: var(--color-accent-orange);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.pv-error {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 60px 0;
  color: var(--color-danger);
  font-size: 14px;
}

.pv-error-icon {
  font-size: 20px;
}

.pv-retry-btn {
  padding: 6px 12px;
  border: 1px solid var(--color-danger);
  border-radius: 4px;
  background: none;
  color: var(--color-danger);
  font-size: 12px;
  cursor: pointer;
}

.pv-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  text-align: center;
}

.pv-empty-icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.4;
}

.pv-empty-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 8px;
}

.pv-empty-desc {
  font-size: 14px;
  color: var(--color-text-secondary);
  margin: 0 0 20px;
}

/* Project Grid */
.pv-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.pv-card {
  position: relative;
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 8px;
  padding: 16px;
  cursor: pointer;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.pv-card:hover {
  border-color: var(--color-accent-blue);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
}

.pv-card-header {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 10px;
}

.pv-card-icon {
  font-size: 24px;
  color: var(--color-accent-orange);
  flex-shrink: 0;
  line-height: 1;
}

.pv-card-meta {
  min-width: 0;
}

.pv-card-name {
  font-size: 15px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin: 0 0 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pv-card-type {
  font-size: 12px;
  color: var(--color-text-secondary);
}

.pv-card-desc {
  font-size: 12px;
  color: var(--color-text-secondary);
  margin: 0 0 12px;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.pv-card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.pv-card-date {
  font-size: 11px;
  color: var(--color-text-secondary);
}

.pv-card-tags {
  display: flex;
  gap: 4px;
}

.pv-card-tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  background-color: var(--color-bg-card);
  color: var(--color-text-secondary);
}

.pv-card-actions {
  position: absolute;
  top: 10px;
  right: 10px;
  opacity: 0;
  transition: opacity 0.15s;
}

.pv-card:hover .pv-card-actions {
  opacity: 1;
}

.pv-card-action-btn {
  width: 24px;
  height: 24px;
  border: none;
  background: none;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--color-text-secondary);
  transition: background-color 0.15s, color 0.15s;
}

.pv-card-action-btn:hover {
  background-color: var(--color-bg-card);
}

.pv-card-action-btn--danger:hover {
  background-color: rgba(244, 67, 54, 0.1);
  color: var(--color-danger);
}

/* Modal */
.pv-modal-overlay {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: rgba(0, 0, 0, 0.6);
}

.pv-modal {
  background-color: var(--color-bg-secondary);
  border: 1px solid var(--color-border);
  border-radius: 12px;
  width: 480px;
  max-width: 90vw;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.pv-modal-title {
  font-size: 18px;
  font-weight: 600;
  padding: 20px 24px 0;
  margin: 0;
  color: var(--color-text-primary);
}

.pv-modal-body {
  padding: 16px 24px;
}

.pv-form-field {
  margin-bottom: 14px;
}

.pv-form-label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--color-text-secondary);
  margin-bottom: 4px;
}

.pv-form-input,
.pv-form-select,
.pv-form-textarea {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background-color: var(--color-bg-card);
  color: var(--color-text-primary);
  font-size: 13px;
  font-family: inherit;
}

.pv-form-input:focus,
.pv-form-select:focus,
.pv-form-textarea:focus {
  outline: none;
  border-color: var(--color-accent-blue);
}

.pv-form-textarea {
  resize: vertical;
  min-height: 60px;
}

.pv-modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 0 24px 20px;
}

.pv-modal-btn {
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 0.15s;
}

.pv-modal-btn--cancel {
  border: 1px solid var(--color-border);
  background: none;
  color: var(--color-text-secondary);
}

.pv-modal-btn--cancel:hover {
  color: var(--color-text-primary);
}

.pv-modal-btn--create {
  border: none;
  background-color: var(--color-accent-orange);
  color: #fff;
}

.pv-modal-btn--create:hover {
  opacity: 0.9;
}

.pv-modal-btn--create:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
