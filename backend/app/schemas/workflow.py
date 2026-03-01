"""Pydantic schemas for Workflow resources."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class WorkflowNode(BaseModel):
    """Schema for a single node in a simulation workflow DAG."""

    id: str
    type: str  # geometry, mesh, material, boundary_condition, solver, post_process, compare, optimize
    position: dict  # {x, y}
    data: dict  # node-specific configuration


class WorkflowEdge(BaseModel):
    """Schema for an edge connecting two workflow nodes."""

    id: str
    source: str
    target: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None


class WorkflowDefinition(BaseModel):
    """Schema for a complete workflow definition."""

    nodes: List[WorkflowNode]
    edges: List[WorkflowEdge]


class WorkflowValidateResponse(BaseModel):
    """Schema for workflow validation results."""

    valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    execution_order: List[str] = []


class WorkflowExecuteResponse(BaseModel):
    """Schema for workflow execution status."""

    workflow_id: str
    status: str
    node_statuses: Dict[str, str] = {}
