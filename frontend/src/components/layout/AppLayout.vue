<template>
  <div class="flex h-screen overflow-hidden">
    <Sidebar />
    <main class="flex-1 overflow-auto">
      <slot />
    </main>
  </div>
</template>

<script setup lang="ts">
import Sidebar from './Sidebar.vue'
</script>
