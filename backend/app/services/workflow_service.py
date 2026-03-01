"""Service layer for Workflow validation and execution."""
from __future__ import annotations

import uuid
from collections import defaultdict, deque

from backend.app.schemas.workflow import (
    WorkflowDefinition,
    WorkflowExecuteResponse,
    WorkflowValidateResponse,
)

# Valid connection rules: source_type -> set of valid target_types
VALID_CONNECTIONS: dict[str, set[str]] = {
    "geometry": {"mesh"},
    "mesh": {"solver", "boundary_condition"},
    "material": {"solver"},
    "boundary_condition": {"solver"},
    "solver": {"post_process", "compare", "optimize"},
    "post_process": {"compare", "optimize"},
    "optimize": {"geometry"},
}


class WorkflowService:
    def validate(self, definition: WorkflowDefinition) -> WorkflowValidateResponse:
        """Validate a workflow DAG."""
        errors: list[str] = []
        warnings: list[str] = []

        if not definition.nodes:
            errors.append("Workflow must contain at least one node.")
            return WorkflowValidateResponse(valid=False, errors=errors, warnings=warnings, execution_order=[])

        node_map = {n.id: n for n in definition.nodes}

        # Check for invalid connections
        for edge in definition.edges:
            if edge.source not in node_map:
                errors.append(f"Edge source '{edge.source}' not found in nodes")
                continue
            if edge.target not in node_map:
                errors.append(f"Edge target '{edge.target}' not found in nodes")
                continue
            src_type = node_map[edge.source].type
            tgt_type = node_map[edge.target].type
            valid_targets = VALID_CONNECTIONS.get(src_type, set())
            if tgt_type not in valid_targets:
                errors.append(f"Invalid connection: {src_type} -> {tgt_type}")

        # Topological sort (Kahn's algorithm) to detect cycles
        in_degree: dict[str, int] = {n.id: 0 for n in definition.nodes}
        adjacency: dict[str, list[str]] = defaultdict(list)
        for edge in definition.edges:
            adjacency[edge.source].append(edge.target)
            if edge.target in in_degree:
                in_degree[edge.target] += 1

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        execution_order: list[str] = []
        while queue:
            node_id = queue.popleft()
            execution_order.append(node_id)
            for neighbor in adjacency[node_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(execution_order) != len(definition.nodes):
            errors.append("Workflow contains a cycle")

        # Check solver nodes have mesh input
        for node in definition.nodes:
            if node.type == "solver":
                has_mesh_input = any(
                    e.target == node.id and node_map.get(e.source, None) and node_map[e.source].type == "mesh"
                    for e in definition.edges
                )
                if not has_mesh_input:
                    errors.append(f"Solver node '{node.id}' has no mesh input")

        return WorkflowValidateResponse(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_order=execution_order,
        )

    def execute(self, definition: WorkflowDefinition) -> WorkflowExecuteResponse:
        """Start workflow execution."""
        validation = self.validate(definition)
        if not validation.valid:
            return WorkflowExecuteResponse(
                workflow_id=str(uuid.uuid4()),
                status="failed",
                node_statuses={n.id: "error" for n in definition.nodes},
            )
        workflow_id = str(uuid.uuid4())
        node_statuses = {n.id: "pending" for n in definition.nodes}
        # In production, dispatch Celery task for each node in topological order
        return WorkflowExecuteResponse(
            workflow_id=workflow_id,
            status="submitted",
            node_statuses=node_statuses,
        )
