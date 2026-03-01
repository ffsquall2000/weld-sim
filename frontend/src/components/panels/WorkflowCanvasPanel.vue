<template>
  <div class="panel-placeholder">
    <div class="panel-placeholder__icon">&#128736;</div>
    <div class="panel-placeholder__title">{{ $t('layout.panels.workflowCanvas') }}</div>
    <div class="panel-placeholder__hint">{{ $t('layout.placeholders.workflowCanvas') }}</div>
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
}
</style>
