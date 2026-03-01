import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { projectApi } from '@/api/v2'

export interface Project {
  id: string
  name: string
  description: string | null
  application_type: string
  settings: Record<string, any> | null
  tags: string[] | null
  created_at: string
  updated_at: string
}

export interface ProjectState {
  projects: Project[]
  currentProject: Project | null
  loading: boolean
  error: string | null
}

export const useProjectStore = defineStore('project', () => {
  // State
  const projects = ref<Project[]>([])
  const currentProject = ref<Project | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Getters
  const sortedProjects = computed(() =>
    [...projects.value].sort(
      (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
    )
  )

  const projectCount = computed(() => projects.value.length)

  // Actions
  async function fetchProjects() {
    loading.value = true
    error.value = null
    try {
      const response = await projectApi.list()
      projects.value = response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  async function fetchProject(id: string) {
    loading.value = true
    error.value = null
    try {
      const response = await projectApi.get(id)
      currentProject.value = response.data
      // Also update in the list if present
      const index = projects.value.findIndex((p) => p.id === id)
      if (index !== -1) {
        projects.value[index] = response.data
      }
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function createProject(data: {
    name: string
    description?: string
    application_type: string
    settings?: Record<string, any>
    tags?: string[]
  }) {
    loading.value = true
    error.value = null
    try {
      const response = await projectApi.create(data)
      projects.value.push(response.data)
      currentProject.value = response.data
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function updateProject(
    id: string,
    data: Partial<Pick<Project, 'name' | 'description' | 'application_type' | 'settings' | 'tags'>>
  ) {
    loading.value = true
    error.value = null
    try {
      const response = await projectApi.update(id, data)
      const index = projects.value.findIndex((p) => p.id === id)
      if (index !== -1) {
        projects.value[index] = response.data
      }
      if (currentProject.value?.id === id) {
        currentProject.value = response.data
      }
      return response.data
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return null
    } finally {
      loading.value = false
    }
  }

  async function deleteProject(id: string) {
    loading.value = true
    error.value = null
    try {
      await projectApi.delete(id)
      projects.value = projects.value.filter((p) => p.id !== id)
      if (currentProject.value?.id === id) {
        currentProject.value = null
      }
      return true
    } catch (err: unknown) {
      error.value = err instanceof Error ? err.message : String(err)
      return false
    } finally {
      loading.value = false
    }
  }

  function setCurrentProject(project: Project | null) {
    currentProject.value = project
  }

  return {
    // State
    projects,
    currentProject,
    loading,
    error,
    // Getters
    sortedProjects,
    projectCount,
    // Actions
    fetchProjects,
    fetchProject,
    createProject,
    updateProject,
    deleteProject,
    setCurrentProject,
  }
})
