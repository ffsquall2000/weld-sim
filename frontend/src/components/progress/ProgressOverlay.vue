<!-- frontend/src/components/progress/ProgressOverlay.vue -->
<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="visible" class="progress-overlay">
        <div class="progress-card">
          <!-- Header -->
          <div class="progress-header">
            <h3>{{ title }}</h3>
            <button v-if="cancellable" class="cancel-btn" @click="$emit('cancel')" title="Cancel">
              ✕
            </button>
          </div>

          <!-- Steps -->
          <div class="steps-container">
            <div
              v-for="(step, idx) in steps"
              :key="idx"
              :class="['step', { active: idx === currentStep, completed: idx < currentStep }]"
            >
              <div class="step-indicator">
                <span v-if="idx < currentStep" class="check">✓</span>
                <span v-else-if="idx === currentStep" class="spinner"></span>
                <span v-else class="dot">○</span>
              </div>
              <span class="step-label">{{ step }}</span>
            </div>
          </div>

          <!-- Progress bar -->
          <div class="progress-bar-container">
            <div class="progress-bar" :style="{ width: `${progress * 100}%` }"></div>
          </div>
          <div class="progress-text">
            <span>{{ message || 'Processing...' }}</span>
            <span>{{ Math.round(progress * 100) }}%</span>
          </div>

          <!-- Error state -->
          <div v-if="error" class="error-message">
            <span class="error-icon">⚠</span>
            {{ error }}
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  visible: boolean
  title?: string
  steps?: string[]
  currentStep?: number
  progress?: number
  message?: string
  error?: string | null
  cancellable?: boolean
}>(), {
  title: 'Processing',
  steps: () => [],
  currentStep: 0,
  progress: 0,
  message: 'Processing...',
  error: null,
  cancellable: false,
})

defineEmits<{
  (e: 'cancel'): void
}>()
</script>

<style scoped>
.progress-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.progress-card {
  background: #1e1e2e;
  border-radius: 12px;
  padding: 24px;
  min-width: 400px;
  max-width: 500px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.progress-header h3 {
  color: #fff;
  margin: 0;
  font-size: 18px;
}

.cancel-btn {
  background: transparent;
  border: none;
  color: #888;
  font-size: 18px;
  cursor: pointer;
  padding: 4px 8px;
  border-radius: 4px;
}
.cancel-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }

.steps-container {
  display: flex;
  flex-direction: column;
  gap: 12px;
  margin-bottom: 20px;
}

.step {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #666;
  font-size: 14px;
}
.step.active { color: #ff9800; }
.step.completed { color: #4caf50; }

.step-indicator {
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.check { color: #4caf50; font-weight: bold; }
.dot { font-size: 10px; }

.spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #ff9800;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.progress-bar-container {
  height: 6px;
  background: #333;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 8px;
}

.progress-bar {
  height: 100%;
  background: linear-gradient(90deg, #ff9800, #f44336);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.progress-text {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #888;
}

.error-message {
  margin-top: 16px;
  padding: 12px;
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid rgba(244, 67, 54, 0.3);
  border-radius: 6px;
  color: #f44336;
  font-size: 13px;
  display: flex;
  gap: 8px;
  align-items: center;
}

.fade-enter-active, .fade-leave-active { transition: opacity 0.2s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
