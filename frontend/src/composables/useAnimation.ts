// frontend/src/composables/useAnimation.ts
import { ref, computed } from 'vue'

export interface ModeData {
  index: number
  frequency_hz: number
  mode_type: string
  shape: Float32Array // (N*3) displacement vector
}

export function useAnimation() {
  const isPlaying = ref(false)
  const phase = ref(0) // 0-360 degrees
  const amplitudeScale = ref(1.0)
  const speed = ref(1.0)
  const loop = ref(true)
  const currentModeIndex = ref(0)
  const modes = ref<ModeData[]>([])

  let lastTime = 0
  let animationId: number | null = null

  const currentMode = computed(() => modes.value[currentModeIndex.value] ?? null)

  function setModes(modeList: ModeData[]) {
    modes.value = modeList
    currentModeIndex.value = 0
    phase.value = 0
  }

  function play() {
    isPlaying.value = true
    lastTime = performance.now()
    tick()
  }

  function pause() {
    isPlaying.value = false
    if (animationId !== null) {
      cancelAnimationFrame(animationId)
      animationId = null
    }
  }

  function togglePlay() {
    isPlaying.value ? pause() : play()
  }

  function stepForward() {
    phase.value = (phase.value + 10) % 360
  }

  function stepBackward() {
    phase.value = (phase.value - 10 + 360) % 360
  }

  function tick() {
    if (!isPlaying.value) return
    animationId = requestAnimationFrame(tick)

    const now = performance.now()
    const dt = (now - lastTime) / 1000 // seconds
    lastTime = now

    // Advance phase: speed = 1 means one full cycle per second
    phase.value = (phase.value + dt * speed.value * 360) % 360
    if (!loop.value && phase.value < dt * speed.value * 360 - 360) {
      pause()
      phase.value = 0
    }
  }

  /** Get deformed positions: pos + scale * mode_shape * sin(phase) */
  function getDeformation(): Float32Array | null {
    const mode = currentMode.value
    if (!mode) return null
    const factor = amplitudeScale.value * Math.sin((phase.value * Math.PI) / 180)
    const deformed = new Float32Array(mode.shape.length)
    for (let i = 0; i < mode.shape.length; i++) {
      deformed[i] = mode.shape[i]! * factor
    }
    return deformed
  }

  function dispose() {
    pause()
    modes.value = []
  }

  return {
    isPlaying,
    phase,
    amplitudeScale,
    speed,
    loop,
    currentModeIndex,
    modes,
    currentMode,
    setModes,
    play,
    pause,
    togglePlay,
    stepForward,
    stepBackward,
    getDeformation,
    dispose,
  }
}
