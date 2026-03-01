import type { Node, Edge } from '@vue-flow/core'

export interface SimNodeData {
  type: 'geometry' | 'mesh' | 'material' | 'boundary_condition' | 'solver' | 'post_process' | 'compare' | 'optimize'
  label: string
  status: 'idle' | 'configured' | 'running' | 'completed' | 'error'
  config: Record<string, any>
  outputs: Record<string, any>
}

export type SimNodeType = SimNodeData['type']

/**
 * Connection rules: which source node types can connect to which target node types.
 */
const connectionRules: Record<SimNodeType, SimNodeType[]> = {
  geometry: ['mesh'],
  mesh: ['solver', 'boundary_condition'],
  material: ['solver'],
  boundary_condition: ['solver'],
  solver: ['post_process', 'compare', 'optimize'],
  post_process: ['compare', 'optimize'],
  compare: [],
  optimize: ['geometry'],
}

/**
 * Check if a connection from sourceType to targetType is allowed
 * by the simulation pipeline rules.
 */
export function isValidConnection(sourceType: SimNodeType, targetType: SimNodeType): boolean {
  const allowed = connectionRules[sourceType]
  return allowed ? allowed.includes(targetType) : false
}

/**
 * Perform topological sort on the DAG defined by nodes and edges.
 * Returns an ordered array of node IDs, or null if the graph contains a cycle.
 */
export function topologicalSort(nodes: Node<SimNodeData>[], edges: Edge[]): string[] | null {
  const nodeIds = new Set(nodes.map((n) => n.id))
  const adjacency = new Map<string, string[]>()
  const inDegree = new Map<string, number>()

  for (const id of nodeIds) {
    adjacency.set(id, [])
    inDegree.set(id, 0)
  }

  for (const edge of edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) continue
    adjacency.get(edge.source)!.push(edge.target)
    inDegree.set(edge.target, (inDegree.get(edge.target) || 0) + 1)
  }

  const queue: string[] = []
  for (const [id, deg] of inDegree) {
    if (deg === 0) queue.push(id)
  }

  const sorted: string[] = []
  while (queue.length > 0) {
    const current = queue.shift()!
    sorted.push(current)
    for (const neighbor of adjacency.get(current)!) {
      const newDeg = (inDegree.get(neighbor) || 1) - 1
      inDegree.set(neighbor, newDeg)
      if (newDeg === 0) queue.push(neighbor)
    }
  }

  if (sorted.length !== nodeIds.size) {
    return null // cycle detected
  }

  return sorted
}

export interface ValidationError {
  nodeId?: string
  message: string
}

/**
 * Validate the DAG for structural correctness.
 * Checks for: cycles, invalid connections, unconnected required inputs.
 */
export function validateDAG(nodes: Node<SimNodeData>[], edges: Edge[]): ValidationError[] {
  const errors: ValidationError[] = []

  if (nodes.length === 0) {
    errors.push({ message: 'Workflow is empty. Add at least one node.' })
    return errors
  }

  // Check for cycles
  const sorted = topologicalSort(nodes, edges)
  if (sorted === null) {
    errors.push({ message: 'Workflow contains a cycle. Remove circular dependencies.' })
  }

  // Build node map for quick lookup
  const nodeMap = new Map<string, Node<SimNodeData>>()
  for (const node of nodes) {
    nodeMap.set(node.id, node)
  }

  // Check each edge for valid connection type
  for (const edge of edges) {
    const sourceNode = nodeMap.get(edge.source)
    const targetNode = nodeMap.get(edge.target)
    if (!sourceNode || !targetNode) continue

    const sourceType = sourceNode.data!.type
    const targetType = targetNode.data!.type
    if (!isValidConnection(sourceType, targetType)) {
      errors.push({
        message: `Invalid connection: ${sourceType} cannot connect to ${targetType}.`,
      })
    }
  }

  // Required inputs check: solver needs at least one mesh input
  const solverNodes = nodes.filter((n) => n.data?.type === 'solver')
  for (const solver of solverNodes) {
    const incomingTypes = edges
      .filter((e) => e.target === solver.id)
      .map((e) => nodeMap.get(e.source)?.data?.type)
      .filter(Boolean)

    if (!incomingTypes.includes('mesh')) {
      errors.push({
        nodeId: solver.id,
        message: `Solver "${solver.data!.label}" requires a Mesh input.`,
      })
    }
    if (!incomingTypes.includes('material')) {
      errors.push({
        nodeId: solver.id,
        message: `Solver "${solver.data!.label}" requires a Material input.`,
      })
    }
  }

  // Mesh nodes need geometry input
  const meshNodes = nodes.filter((n) => n.data?.type === 'mesh')
  for (const mesh of meshNodes) {
    const hasGeometry = edges.some(
      (e) => e.target === mesh.id && nodeMap.get(e.source)?.data?.type === 'geometry'
    )
    if (!hasGeometry) {
      errors.push({
        nodeId: mesh.id,
        message: `Mesh "${mesh.data!.label}" requires a Geometry input.`,
      })
    }
  }

  return errors
}

/**
 * Get all nodes that feed into the given node (its inputs).
 */
export function getNodeInputs(
  nodeId: string,
  nodes: Node<SimNodeData>[],
  edges: Edge[]
): Node<SimNodeData>[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  return edges
    .filter((e) => e.target === nodeId)
    .map((e) => nodeMap.get(e.source))
    .filter((n): n is Node<SimNodeData> => n !== undefined)
}

/**
 * Get all nodes that the given node feeds into (its outputs).
 */
export function getNodeOutputs(
  nodeId: string,
  nodes: Node<SimNodeData>[],
  edges: Edge[]
): Node<SimNodeData>[] {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]))
  return edges
    .filter((e) => e.source === nodeId)
    .map((e) => nodeMap.get(e.target))
    .filter((n): n is Node<SimNodeData> => n !== undefined)
}
