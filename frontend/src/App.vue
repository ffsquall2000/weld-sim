<template>
  <AppShell v-if="route.meta.useShell" />
  <AppLayout v-else>
    <router-view />
  </AppLayout>
</template>

<script setup lang="ts">
import { useRoute } from 'vue-router'
import AppLayout from '@/components/layout/AppLayout.vue'
import AppShell from '@/components/layout/AppShell.vue'

const route = useRoute()
</script>
