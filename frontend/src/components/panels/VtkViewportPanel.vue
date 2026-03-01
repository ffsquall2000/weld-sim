<template>
  <div class="panel-placeholder">
    <div class="panel-placeholder__icon">&#128302;</div>
    <div class="panel-placeholder__title">{{ $t('layout.panels.vtkViewport') }}</div>
    <div class="panel-placeholder__hint">{{ $t('layout.placeholders.vtkViewport') }}</div>
    <div class="viewport-controls">
      <button class="viewport-btn" disabled>Rotate</button>
      <button class="viewport-btn" disabled>Pan</button>
      <button class="viewport-btn" disabled>Zoom</button>
      <button class="viewport-btn" disabled>Reset</button>
    </div>
  </div>
</template>

<script setup lang="ts">
</script>

<style scoped>
.panel-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  padding: 16px;
  color: var(--color-text-secondary);
  text-align: center;
  background-color: var(--color-bg-primary);
}

.panel-placeholder__icon {
  font-size: 48px;
  margin-bottom: 12px;
  opacity: 0.3;
}

.panel-placeholder__title {
  font-size: 14px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.panel-placeholder__hint {
  font-size: 11px;
  max-width: 240px;
  line-height: 1.4;
  margin-bottom: 16px;
}

.viewport-controls {
  display: flex;
  gap: 8px;
}

.viewport-btn {
  padding: 4px 10px;
  font-size: 11px;
  color: var(--color-text-secondary);
  background-color: var(--color-bg-card);
  border: 1px solid var(--color-border);
  border-radius: 3px;
  cursor: not-allowed;
  opacity: 0.5;
}
</style>
