import { ref, onBeforeUnmount, type Ref, watch, nextTick } from 'vue'

// VTK.js rendering profile must be imported first to register backends
import '@kitware/vtk.js/Rendering/Profiles/Geometry'

import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow'
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor'
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper'
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData'
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray'
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray'
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction'
import vtkAxesActor from '@kitware/vtk.js/Rendering/Core/AxesActor'
import { Representation } from '@kitware/vtk.js/Rendering/Core/Property/Constants'
import { ColorMode, ScalarMode } from '@kitware/vtk.js/Rendering/Core/Mapper/Constants'

import type { MeshData, DisplayMode, ColorMapPreset } from '@/stores/viewer3d'

// Color map preset definitions: arrays of [position, r, g, b] control points (0-1 range)
type ColorStop = [number, number, number, number]

const COLOR_MAP_PRESETS: Record<ColorMapPreset, ColorStop[]> = {
  jet: [
    [0.0, 0.0, 0.0, 0.5],
    [0.1, 0.0, 0.0, 1.0],
    [0.35, 0.0, 1.0, 1.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.65, 1.0, 1.0, 0.0],
    [0.9, 1.0, 0.0, 0.0],
    [1.0, 0.5, 0.0, 0.0],
  ],
  rainbow: [
    [0.0, 1.0, 0.0, 0.0],
    [0.17, 1.0, 0.5, 0.0],
    [0.33, 1.0, 1.0, 0.0],
    [0.5, 0.0, 1.0, 0.0],
    [0.67, 0.0, 0.5, 1.0],
    [0.83, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
  ],
  cool_warm: [
    [0.0, 0.23, 0.30, 0.75],
    [0.5, 0.87, 0.87, 0.87],
    [1.0, 0.71, 0.016, 0.15],
  ],
  viridis: [
    [0.0, 0.267, 0.005, 0.329],
    [0.25, 0.283, 0.141, 0.458],
    [0.5, 0.127, 0.567, 0.551],
    [0.75, 0.544, 0.773, 0.248],
    [1.0, 0.993, 0.906, 0.144],
  ],
}

/**
 * Composable that manages a VTK.js rendering pipeline in a given container element.
 * Handles WebGL context, actors, scalar field coloring, display modes, and lifecycle cleanup.
 */
export function useVtkViewer(containerRef: Ref<HTMLElement | null>) {
  // --- State refs ---
  const isInitialized = ref(false)
  const webGLSupported = ref(true)
  const currentScalarField = ref<string | null>(null)
  const currentDisplayMode = ref<DisplayMode>('solid')
  const currentColorMap = ref<ColorMapPreset>('cool_warm')

  // --- Internal VTK objects (not reactive) ---
  let genericRenderWindow: ReturnType<typeof vtkGenericRenderWindow.newInstance> | null = null
  let mainActor: ReturnType<typeof vtkActor.newInstance> | null = null
  let mainMapper: ReturnType<typeof vtkMapper.newInstance> | null = null
  let wireframeActor: ReturnType<typeof vtkActor.newInstance> | null = null
  let wireframeMapper: ReturnType<typeof vtkMapper.newInstance> | null = null
  let axesActor: ReturnType<typeof vtkAxesActor.newInstance> | null = null
  let polyData: ReturnType<typeof vtkPolyData.newInstance> | null = null
  let colorTransferFunction: ReturnType<typeof vtkColorTransferFunction.newInstance> | null = null
  let resizeObserver: ResizeObserver | null = null

  // --- Check WebGL availability ---
  function checkWebGL(): boolean {
    try {
      const canvas = document.createElement('canvas')
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl')
      return gl !== null
    } catch {
      return false
    }
  }

  // --- Initialize VTK rendering pipeline ---
  function initialize(): boolean {
    const container = containerRef.value
    if (!container || isInitialized.value) return isInitialized.value

    if (!checkWebGL()) {
      webGLSupported.value = false
      return false
    }

    try {
      // Create the generic render window bound to our container
      genericRenderWindow = vtkGenericRenderWindow.newInstance({
        background: [0.1, 0.1, 0.14, 1.0],
      })
      genericRenderWindow.setContainer(container)

      const renderer = genericRenderWindow.getRenderer()
      const renderWindow = genericRenderWindow.getRenderWindow()

      // Create main surface actor & mapper
      mainMapper = vtkMapper.newInstance()
      mainActor = vtkActor.newInstance()
      mainActor.setMapper(mainMapper)
      mainActor.getProperty().setRepresentation(Representation.SURFACE)
      mainActor.getProperty().setAmbient(0.2)
      mainActor.getProperty().setDiffuse(0.7)
      mainActor.getProperty().setSpecular(0.3)
      mainActor.getProperty().setSpecularPower(20)
      mainActor.getProperty().setColor(0.45, 0.55, 0.7)

      // Create wireframe overlay actor (for solid_wireframe mode)
      wireframeMapper = vtkMapper.newInstance()
      wireframeMapper.setScalarVisibility(false)
      wireframeActor = vtkActor.newInstance()
      wireframeActor.setMapper(wireframeMapper)
      wireframeActor.getProperty().setRepresentation(Representation.WIREFRAME)
      wireframeActor.getProperty().setColor(0.0, 0.0, 0.0)
      wireframeActor.getProperty().setOpacity(0.15)
      wireframeActor.getProperty().setLineWidth(1)
      wireframeActor.setVisibility(false)

      // Create axes actor
      axesActor = vtkAxesActor.newInstance()
      renderer.addActor(axesActor)

      // Color transfer function for scalar fields
      colorTransferFunction = vtkColorTransferFunction.newInstance()

      // Add actors to renderer
      renderer.addActor(mainActor)
      renderer.addActor(wireframeActor)

      // Setup interactor
      const interactor = renderWindow.getInteractor()
      interactor.setDesiredUpdateRate(15.0)

      // Handle container resize
      resizeObserver = new ResizeObserver(() => {
        handleResize()
      })
      resizeObserver.observe(container)

      // Initial resize
      handleResize()

      isInitialized.value = true
      return true
    } catch (err) {
      console.error('[useVtkViewer] Failed to initialize VTK pipeline:', err)
      webGLSupported.value = false
      return false
    }
  }

  // --- Handle container resize ---
  function handleResize() {
    if (!genericRenderWindow) return
    const container = containerRef.value
    if (!container) return

    const { width, height } = container.getBoundingClientRect()
    if (width === 0 || height === 0) return

    const apiWindow = genericRenderWindow.getApiSpecificRenderWindow()
    apiWindow.setSize(Math.floor(width), Math.floor(height))
    genericRenderWindow.resize()
    render()
  }

  // --- Render ---
  function render() {
    if (!genericRenderWindow) return
    genericRenderWindow.getRenderWindow().render()
  }

  // --- Build polydata from MeshData ---
  function buildPolyDataFromMeshData(meshData: MeshData): ReturnType<typeof vtkPolyData.newInstance> {
    const pd = vtkPolyData.newInstance()

    // Set points
    const pointsArray = vtkDataArray.newInstance({
      numberOfComponents: 3,
      values: meshData.points,
      name: 'Points',
    })
    pd.getPoints().setData(pointsArray.getData(), 3)

    // Build cell arrays from the connectivity and cell types.
    // meshData.cells is expected to be in VTK connectivity format:
    // [npts, p0, p1, ..., npts, p0, p1, ...]
    // We separate triangles and other polygons into the polys array.
    const cellArray = vtkCellArray.newInstance()
    cellArray.setData(meshData.cells)
    pd.setPolys(cellArray)

    // Add point data (scalar fields)
    for (const [name, values] of Object.entries(meshData.pointData)) {
      const da = vtkDataArray.newInstance({
        numberOfComponents: 1,
        values,
        name,
      })
      pd.getPointData().addArray(da)
    }

    return pd
  }

  // --- Load mesh data ---
  function loadMeshFromData(meshData: MeshData) {
    if (!isInitialized.value) {
      if (!initialize()) return
    }
    if (!genericRenderWindow || !mainMapper || !mainActor || !wireframeMapper) return

    // Build polydata
    polyData = buildPolyDataFromMeshData(meshData)

    // Connect to main mapper
    mainMapper.setInputData(polyData)

    // Connect to wireframe mapper
    wireframeMapper.setInputData(polyData)

    // Auto-select first scalar field if available
    const fieldKeys = Object.keys(meshData.pointData)
    if (fieldKeys.length > 0) {
      setScalarField(fieldKeys[0]!)
    } else {
      mainMapper.setScalarVisibility(false)
    }

    // Reset camera to fit
    resetCamera()
  }

  // --- Set scalar field for coloring ---
  function setScalarField(fieldName: string | null, range?: [number, number]) {
    currentScalarField.value = fieldName

    if (!mainMapper || !polyData || !colorTransferFunction) return

    if (!fieldName) {
      mainMapper.setScalarVisibility(false)
      render()
      return
    }

    const dataArray = polyData.getPointData().getArrayByName(fieldName)
    if (!dataArray) {
      mainMapper.setScalarVisibility(false)
      render()
      return
    }

    // Determine scalar range
    const dataRange = range ?? dataArray.getRange()

    // Apply color map to the color transfer function
    applyColorMap(currentColorMap.value, dataRange)

    // Configure mapper for scalar coloring
    mainMapper.setScalarVisibility(true)
    mainMapper.setColorMode(ColorMode.MAP_SCALARS)
    mainMapper.setScalarMode(ScalarMode.USE_POINT_DATA)
    mainMapper.setLookupTable(colorTransferFunction)
    mainMapper.setScalarRange(dataRange[0], dataRange[1])

    // Set the active scalars
    polyData.getPointData().setActiveScalars(fieldName)

    render()
  }

  // --- Apply color map preset ---
  function applyColorMap(preset: ColorMapPreset, range: [number, number] | number[]) {
    if (!colorTransferFunction) return
    currentColorMap.value = preset

    const stops = COLOR_MAP_PRESETS[preset]
    if (!stops) return

    const min = range[0]!
    const max = range[1]!
    const span = max - min

    colorTransferFunction.removeAllPoints()
    for (const [pos, r, g, b] of stops) {
      colorTransferFunction.addRGBPoint(min + pos * span, r, g, b)
    }
    colorTransferFunction.modified()
  }

  // --- Set color map without range override ---
  function setColorMap(preset: ColorMapPreset) {
    currentColorMap.value = preset
    if (!mainMapper || !currentScalarField.value || !polyData) return

    const dataArray = polyData.getPointData().getArrayByName(currentScalarField.value)
    if (!dataArray) return

    const dataRange = mainMapper.getScalarRange()
    applyColorMap(preset, dataRange)
    render()
  }

  // --- Set display mode ---
  function setDisplayMode(mode: DisplayMode) {
    currentDisplayMode.value = mode
    if (!mainActor || !wireframeActor) return

    switch (mode) {
      case 'solid':
        mainActor.getProperty().setRepresentation(Representation.SURFACE)
        mainActor.getProperty().setEdgeVisibility(false)
        mainActor.setVisibility(true)
        wireframeActor.setVisibility(false)
        break
      case 'wireframe':
        mainActor.getProperty().setRepresentation(Representation.WIREFRAME)
        mainActor.getProperty().setEdgeVisibility(false)
        mainActor.setVisibility(true)
        wireframeActor.setVisibility(false)
        break
      case 'solid_wireframe':
        mainActor.getProperty().setRepresentation(Representation.SURFACE)
        mainActor.getProperty().setEdgeVisibility(false)
        mainActor.setVisibility(true)
        wireframeActor.setVisibility(true)
        break
    }
    render()
  }

  // --- Toggle edge visibility ---
  function setEdgeVisibility(visible: boolean) {
    if (!mainActor) return
    mainActor.getProperty().setEdgeVisibility(visible)
    if (visible) {
      mainActor.getProperty().setEdgeColor(0.0, 0.0, 0.0)
    }
    render()
  }

  // --- Toggle axes actor ---
  function setAxesVisibility(visible: boolean) {
    if (!axesActor) return
    axesActor.setVisibility(visible)
    render()
  }

  // --- Reset camera to fit all actors ---
  function resetCamera() {
    if (!genericRenderWindow) return
    const renderer = genericRenderWindow.getRenderer()
    renderer.resetCamera()
    renderer.resetCameraClippingRange()
    render()
  }

  // --- Take a screenshot and return a data URL ---
  function takeScreenshot(): string | null {
    if (!genericRenderWindow) return null
    try {
      render()
      const apiWindow = genericRenderWindow.getApiSpecificRenderWindow()
      // Get the canvas element from the OpenGL render window
      const canvas = apiWindow.getCanvas()
      if (canvas) {
        return canvas.toDataURL('image/png')
      }
    } catch (err) {
      console.error('[useVtkViewer] Screenshot failed:', err)
    }
    return null
  }

  // --- Get color for a normalized value (0-1) using current color map ---
  function getColorForValue(normalizedValue: number): [number, number, number] {
    if (!colorTransferFunction) return [0.5, 0.5, 0.5]
    const range = mainMapper?.getScalarRange() ?? [0, 1]
    const value = range[0] + normalizedValue * (range[1] - range[0])
    const rgb: [number, number, number] = [0, 0, 0]
    colorTransferFunction.getColor(value, rgb)
    return rgb
  }

  // --- Cleanup ---
  function dispose() {
    resizeObserver?.disconnect()
    resizeObserver = null

    if (genericRenderWindow) {
      const renderer = genericRenderWindow.getRenderer()
      renderer.removeAllActors()
      genericRenderWindow.delete()
      genericRenderWindow = null
    }

    mainActor = null
    mainMapper = null
    wireframeActor = null
    wireframeMapper = null
    axesActor = null
    polyData = null
    colorTransferFunction = null
    isInitialized.value = false
  }

  // Watch for container ref to become available and auto-initialize
  const stopWatch = watch(containerRef, async (newContainer) => {
    if (newContainer && !isInitialized.value) {
      await nextTick()
      initialize()
    }
  }, { immediate: true })

  // Clean up on component unmount
  onBeforeUnmount(() => {
    stopWatch()
    dispose()
  })

  return {
    // Reactive state
    isInitialized,
    webGLSupported,
    currentScalarField,
    currentDisplayMode,
    currentColorMap,

    // Methods
    initialize,
    loadMeshFromData,
    setScalarField,
    setColorMap,
    setDisplayMode,
    setEdgeVisibility,
    setAxesVisibility,
    resetCamera,
    takeScreenshot,
    getColorForValue,
    handleResize,
    render,
    dispose,
  }
}
