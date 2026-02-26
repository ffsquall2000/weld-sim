import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const useSettingsStore = defineStore('settings', () => {
  const theme = ref<'dark' | 'light'>(
    (localStorage.getItem('theme') as 'dark' | 'light') || 'dark'
  )
  const locale = ref(localStorage.getItem('locale') || 'zh-CN')

  watch(theme, (val) => {
    localStorage.setItem('theme', val)
    document.documentElement.classList.toggle('light', val === 'light')
  }, { immediate: true })

  watch(locale, (val) => {
    localStorage.setItem('locale', val)
  })

  function toggleTheme() {
    theme.value = theme.value === 'dark' ? 'light' : 'dark'
  }

  return { theme, locale, toggleTheme }
})
