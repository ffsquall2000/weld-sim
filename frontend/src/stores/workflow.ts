import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { Node, Edge } from '@vue-flow/core'
import {
  type SimNodeData,
  type SimNodeType,
  topologicalSort,
  validateDAG,
  type ValidationError,
} from '@/composables/useWorkflowGraph'

let nextNodeId = 1

function createNodeId(): string {
  return `node_${nextNodeId++}`
}

function createNode(
  type: SimNodeType,
  label: string,
  position: { x: number; y: number }
): Node<SimNodeData> {
  return {
    id: createNodeId(),
    type: 'simNode',
    position,
    data: {
      type,
      label,
      status: 'idle',
      config: {},
      outputs: {},
    },
  }
}

function createEdge(source: string, target: string): Edge {
  return {
    id: `edge_${source}_${target}`,
    source,
    target,
    animated: false,
  }
}

// ---- Predefined templates ----

function basicModalAnalysisTemplate(): { nodes: Node<SimNodeData>[]; edges: Edge[] } {
  const geo = createNode('geometry', 'Geometry', { x: 50, y: 200 })
  const mesh = createNode('mesh', 'Mesh', { x: 280, y: 200 })
  const mat = createNode('material', 'Material', { x: 280, y: 50 })
  const solver = createNode('solver', 'Modal Solver', { x: 520, y: 150 })
  const post = createNode('post_process', 'Post Process', { x: 760, y: 150 })

  return {
    nodes: [geo, mesh, mat, solver, post],
    edges: [
      createEdge(geo.id, mesh.id),
      createEdge(mesh.id, solver.id),
      createEdge(mat.id, solver.id),
      createEdge(solver.id, post.id),
    ],
  }
}

function thermalStructuralTemplate(): { nodes: Node<SimNodeData>[]; edges: Edge[] } {
  const geo = createNode('geometry', 'Geometry', { x: 50, y: 200 })
  const mesh = createNode('mesh', 'Mesh', { x: 280, y: 200 })
  const mat = createNode('material', 'Material', { x: 280, y: 50 })
  const thermalSolver = createNode('solver', 'Thermal Solver', { x: 520, y: 100 })
  const structSolver = createNode('solver', 'Structural Solver', { x: 520, y: 300 })
  const post = createNode('post_process', 'Post Process', { x: 760, y: 200 })

  return {
    nodes: [geo, mesh, mat, thermalSolver, structSolver, post],
    edges: [
      createEdge(geo.id, mesh.id),
      createEdge(mesh.id, thermalSolver.id),
      createEdge(mesh.id, structSolver.id),
      createEdge(mat.id, thermalSolver.id),
      createEdge(mat.id, structSolver.id),
      createEdge(thermalSolver.id, post.id),
      createEdge(structSolver.id, post.id),
    ],
  }
}

function optimizationLoopTemplate(): { nodes: Node<SimNodeData>[]; edges: Edge[] } {
  const geo = createNode('geometry', 'Geometry', { x: 50, y: 200 })
  const mesh = createNode('mesh', 'Mesh', { x: 280, y: 200 })
  const mat = createNode('material', 'Material', { x: 280, y: 50 })
  const solver = createNode('solver', 'Solver', { x: 520, y: 150 })
  const post = createNode('post_process', 'Post Process', { x: 760, y: 150 })
  const opt = createNode('optimize', 'Optimizer', { x: 760, y: 320 })

  return {
    nodes: [geo, mesh, mat, solver, post, opt],
    edges: [
      createEdge(geo.id, mesh.id),
      createEdge(mesh.id, solver.id),
      createEdge(mat.id, solver.id),
      createEdge(solver.id, post.id),
      createEdge(post.id, opt.id),
      createEdge(opt.id, geo.id),
    ],
  }
}

const templates: Record<string, () => { nodes: Node<SimNodeData>[]; edges: Edge[] }> = {
  basicModalAnalysis: basicModalAnalysisTemplate,
  thermalStructural: thermalStructuralTemplate,
  optimizationLoop: optimizationLoopTemplate,
}

export const templateNames: Record<string, string> = {
  basicModalAnalysis: 'Basic Modal Analysis',
  thermalStructural: 'Thermal-Structural Coupled',
  optimizationLoop: 'Optimization Loop',
}

// ---- Store ----

export const useWorkflowStore = defineStore('workflow', () => {
  const nodes = ref<Node<SimNodeData>[]>([])
  const edges = ref<Edge[]>([])
  const selectedNodeId = ref<string | null>(null)
  const executionStatus = ref<'idle' | 'running' | 'completed' | 'error'>('idle')
  const executionOrder = ref<string[]>([])
  const validationErrors = ref<ValidationError[]>([])

  const selectedNode = computed(() => {
    if (!selectedNodeId.value) return null
    return nodes.value.find((n) => n.id === selectedNodeId.value) ?? null
  })

  // --- Actions ---

  function addNode(type: SimNodeType, label: string, position: { x: number; y: number }) {
    const node = createNode(type, label, position)
    nodes.value = [...nodes.value, node]
    return node.id
  }

  function removeNode(nodeId: string) {
    nodes.value = nodes.value.filter((n) => n.id !== nodeId)
    edges.value = edges.value.filter((e) => e.source !== nodeId && e.target !== nodeId)
    if (selectedNodeId.value === nodeId) {
      selectedNodeId.value = null
    }
  }

  function updateNodeConfig(nodeId: string, config: Record<string, any>) {
    const node = nodes.value.find((n) => n.id === nodeId)
    if (node) {
      node.data = {
        ...node.data,
        config: { ...node.data.config, ...config },
        status: 'configured',
      }
      // Trigger reactivity
      nodes.value = [...nodes.value]
    }
  }

  function updateNodeStatus(nodeId: string, status: SimNodeData['status']) {
    const node = nodes.value.find((n) => n.id === nodeId)
    if (node) {
      node.data = { ...node.data, status }
      nodes.value = [...nodes.value]
    }
  }

  function connectNodes(sourceId: string, targetId: string) {
    // Prevent duplicate edges
    const exists = edges.value.some((e) => e.source === sourceId && e.target === targetId)
    if (exists) return

    const edge = createEdge(sourceId, targetId)
    edges.value = [...edges.value, edge]
  }

  function disconnectNodes(edgeId: string) {
    edges.value = edges.value.filter((e) => e.id !== edgeId)
  }

  function setSelectedNode(nodeId: string | null) {
    selectedNodeId.value = nodeId
  }

  function validateWorkflow(): ValidationError[] {
    const errors = validateDAG(nodes.value, edges.value)
    validationErrors.value = errors
    return errors
  }

  async function executeWorkflow(): Promise<boolean> {
    const errors = validateWorkflow()
    if (errors.length > 0) {
      executionStatus.value = 'error'
      return false
    }

    const order = topologicalSort(nodes.value, edges.value)
    if (!order) {
      executionStatus.value = 'error'
      validationErrors.value = [{ message: 'Cannot determine execution order (cycle detected).' }]
      return false
    }

    executionOrder.value = order
    executionStatus.value = 'running'

    // Simulate execution: iterate through topological order
    for (const nodeId of order) {
      updateNodeStatus(nodeId, 'running')
      // Simulate processing delay
      await new Promise((resolve) => setTimeout(resolve, 800 + Math.random() * 400))
      updateNodeStatus(nodeId, 'completed')
    }

    executionStatus.value = 'completed'
    return true
  }

  function stopExecution() {
    executionStatus.value = 'idle'
    // Reset running nodes back to configured/idle
    for (const node of nodes.value) {
      if (node.data.status === 'running') {
        node.data = { ...node.data, status: 'configured' }
      }
    }
    nodes.value = [...nodes.value]
  }

  function clearWorkflow() {
    nodes.value = []
    edges.value = []
    selectedNodeId.value = null
    executionStatus.value = 'idle'
    executionOrder.value = []
    validationErrors.value = []
    nextNodeId = 1
  }

  function loadTemplate(templateKey: string) {
    const factory = templates[templateKey]
    if (!factory) return

    clearWorkflow()
    const template = factory()
    nodes.value = template.nodes
    edges.value = template.edges
  }

  return {
    // State
    nodes,
    edges,
    selectedNodeId,
    executionStatus,
    executionOrder,
    validationErrors,
    // Computed
    selectedNode,
    // Actions
    addNode,
    removeNode,
    updateNodeConfig,
    updateNodeStatus,
    connectNodes,
    disconnectNodes,
    setSelectedNode,
    validateWorkflow,
    executeWorkflow,
    stopExecution,
    clearWorkflow,
    loadTemplate,
  }
})
