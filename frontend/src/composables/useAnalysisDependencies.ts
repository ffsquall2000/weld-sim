import { ref, computed } from 'vue'

const DEPS: Record<string, string[]> = {
  modal: [],
  static: [],
  harmonic: ['modal'],
  stress: ['harmonic'],
  uniformity: ['harmonic'],
  fatigue: ['stress'],
}

export function useAnalysisDependencies() {
  const selected = ref<Set<string>>(new Set(['modal']))

  function toggle(mod: string) {
    const s = new Set(selected.value)
    if (s.has(mod)) {
      s.delete(mod)
      // Remove dependents recursively
      const removeDependents = (target: string) => {
        for (const [m, deps] of Object.entries(DEPS)) {
          if (deps.includes(target) && s.has(m)) {
            s.delete(m)
            removeDependents(m)
          }
        }
      }
      removeDependents(mod)
    } else {
      s.add(mod)
      // Add dependencies recursively
      const addDeps = (m: string) => {
        for (const dep of DEPS[m] || []) {
          if (!s.has(dep)) {
            s.add(dep)
            addDeps(dep)
          }
        }
      }
      addDeps(mod)
    }
    selected.value = s
  }

  const orderedModules = computed(() => {
    const order = ['modal', 'static', 'harmonic', 'uniformity', 'stress', 'fatigue']
    return order.filter(m => selected.value.has(m))
  })

  function getDependencyLabel(mod: string): string {
    const deps = DEPS[mod]
    if (!deps || deps.length === 0) return ''
    return `depends on: ${deps.join(', ')}`
  }

  return { selected, toggle, orderedModules, getDependencyLabel, DEPS }
}
