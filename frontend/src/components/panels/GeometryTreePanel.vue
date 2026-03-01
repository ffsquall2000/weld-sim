<template>
  <div class="panel-placeholder">
    <div class="panel-placeholder__icon">&#128316;</div>
    <div class="panel-placeholder__title">{{ $t('layout.panels.geometryTree') }}</div>
    <div class="panel-placeholder__hint">{{ $t('layout.placeholders.geometryTree') }}</div>
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
}

.panel-placeholder__icon {
  font-size: 32px;
  margin-bottom: 8px;
  opacity: 0.5;
}

.panel-placeholder__title {
  font-size: 13px;
  font-weight: 600;
  color: var(--color-text-primary);
  margin-bottom: 4px;
}

.panel-placeholder__hint {
  font-size: 11px;
  max-width: 200px;
  line-height: 1.4;
}
</style>
