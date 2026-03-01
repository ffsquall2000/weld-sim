// frontend/src/composables/useColormap.ts
import * as THREE from 'three'
import vertexShader from '@/shaders/colormap.vert.glsl?raw'
import fragmentShader from '@/shaders/colormap.frag.glsl?raw'

export type ColormapName = 'jet' | 'viridis' | 'coolwarm' | 'rainbow' | 'grayscale'

// Generate colormap lookup tables (256 RGB values)
function jetLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    let r: number, g: number, b: number
    if (t < 0.125) { r = 0; g = 0; b = 0.5 + t * 4 }
    else if (t < 0.375) { r = 0; g = (t - 0.125) * 4; b = 1 }
    else if (t < 0.625) { r = (t - 0.375) * 4; g = 1; b = 1 - (t - 0.375) * 4 }
    else if (t < 0.875) { r = 1; g = 1 - (t - 0.625) * 4; b = 0 }
    else { r = 1 - (t - 0.875) * 4; g = 0; b = 0 }
    data[i * 4] = Math.round(r * 255)
    data[i * 4 + 1] = Math.round(g * 255)
    data[i * 4 + 2] = Math.round(b * 255)
    data[i * 4 + 3] = 255
  }
  return data
}

function viridisLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  // Simplified viridis: purple → teal → yellow
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const r = Math.round(255 * (0.267 + t * (0.993 - 0.267)))
    const g = Math.round(255 * (0.004 + t * 0.906 * (1 - 0.4 * (t - 0.5) ** 2)))
    const b = Math.round(255 * (0.329 + 0.5 * Math.sin(Math.PI * t) * (1 - t)))
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255
  }
  return data
}

function coolwarmLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    // Blue → White → Red
    const r = Math.round(255 * Math.min(1, 0.2 + 1.6 * t))
    const g = Math.round(255 * (1 - 2 * Math.abs(t - 0.5)))
    const b = Math.round(255 * Math.min(1, 1.8 - 1.6 * t))
    data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255
  }
  return data
}

function rainbowLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    const t = i / 255
    const h = t * 300 / 360 // hue 0-300 degrees
    const s = 1, l = 0.5
    // HSL to RGB
    const c = (1 - Math.abs(2 * l - 1)) * s
    const x = c * (1 - Math.abs((h * 6) % 2 - 1))
    const m = l - c / 2
    let r = 0, g = 0, b = 0
    const h6 = h * 6
    if (h6 < 1) { r = c; g = x }
    else if (h6 < 2) { r = x; g = c }
    else if (h6 < 3) { g = c; b = x }
    else if (h6 < 4) { g = x; b = c }
    else if (h6 < 5) { r = x; b = c }
    else { r = c; b = x }
    data[i * 4] = Math.round((r + m) * 255)
    data[i * 4 + 1] = Math.round((g + m) * 255)
    data[i * 4 + 2] = Math.round((b + m) * 255)
    data[i * 4 + 3] = 255
  }
  return data
}

function grayscaleLUT(): Uint8Array {
  const data = new Uint8Array(256 * 4)
  for (let i = 0; i < 256; i++) {
    data[i * 4] = i; data[i * 4 + 1] = i; data[i * 4 + 2] = i; data[i * 4 + 3] = 255
  }
  return data
}

const LUT_GENERATORS: Record<ColormapName, () => Uint8Array> = {
  jet: jetLUT,
  viridis: viridisLUT,
  coolwarm: coolwarmLUT,
  rainbow: rainbowLUT,
  grayscale: grayscaleLUT,
}

export function useColormap() {
  const textureCache = new Map<ColormapName, THREE.DataTexture>()

  function getTexture(name: ColormapName): THREE.DataTexture {
    if (textureCache.has(name)) return textureCache.get(name)!
    const lut = LUT_GENERATORS[name]()
    const tex = new THREE.DataTexture(lut, 256, 1, THREE.RGBAFormat)
    tex.needsUpdate = true
    tex.minFilter = THREE.LinearFilter
    tex.magFilter = THREE.LinearFilter
    textureCache.set(name, tex)
    return tex
  }

  function createColormapMaterial(
    colormapName: ColormapName = 'jet',
    scalarMin = 0,
    scalarMax = 1,
    opacity = 1.0,
  ): THREE.ShaderMaterial {
    return new THREE.ShaderMaterial({
      vertexShader,
      fragmentShader,
      uniforms: {
        colormapTexture: { value: getTexture(colormapName) },
        scalarMin: { value: scalarMin },
        scalarMax: { value: scalarMax },
        opacity: { value: opacity },
      },
      transparent: opacity < 1.0,
      side: THREE.DoubleSide,
      clipping: true,
    })
  }

  function updateMaterial(
    material: THREE.ShaderMaterial,
    opts: { colormap?: ColormapName; min?: number; max?: number; opacity?: number },
  ) {
    if (opts.colormap) material.uniforms['colormapTexture']!.value = getTexture(opts.colormap)
    if (opts.min !== undefined) material.uniforms['scalarMin']!.value = opts.min
    if (opts.max !== undefined) material.uniforms['scalarMax']!.value = opts.max
    if (opts.opacity !== undefined) {
      material.uniforms['opacity']!.value = opts.opacity
      material.transparent = opts.opacity < 1.0
    }
  }

  /** Get RGB array [r,g,b] for a normalized value t in [0,1] using given colormap */
  function sampleColor(name: ColormapName, t: number): [number, number, number] {
    const lut = LUT_GENERATORS[name]()
    const idx = Math.round(Math.max(0, Math.min(1, t)) * 255) * 4
    return [lut[idx]! / 255, lut[idx + 1]! / 255, lut[idx + 2]! / 255]
  }

  function disposeAll() {
    textureCache.forEach((tex) => tex.dispose())
    textureCache.clear()
  }

  return { getTexture, createColormapMaterial, updateMaterial, sampleColor, disposeAll }
}
