"""Endpoints for Workflow resources."""

from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from backend.app.schemas.workflow import (
    WorkflowDefinition,
    WorkflowExecuteResponse,
    WorkflowValidateResponse,
)

router = APIRouter(prefix="/workflows", tags=["workflows"])


@router.post("/validate", response_model=WorkflowValidateResponse)
async def validate_workflow(
    body: WorkflowDefinition,
) -> WorkflowValidateResponse:
    """Validate a workflow DAG for correctness (cycles, missing edges, etc.)."""
    # TODO: call workflow validation service
    errors: List[str] = []
    warnings: List[str] = []
    execution_order: List[str] = []

    if not body.nodes:
        errors.append("Workflow must contain at least one node.")
    else:
        # Placeholder: return nodes in definition order
        execution_order = [node.id for node in body.nodes]

    return WorkflowValidateResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        execution_order=execution_order,
    )


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(
    body: WorkflowDefinition,
) -> WorkflowExecuteResponse:
    """Execute a validated workflow."""
    # TODO: call workflow execution service / task queue
    workflow_id = str(uuid.uuid4())
    node_statuses = {node.id: "pending" for node in body.nodes}
    return WorkflowExecuteResponse(
        workflow_id=workflow_id,
        status="submitted",
        node_statuses=node_statuses,
    )


@router.get(
    "/{workflow_id}/status",
    response_model=WorkflowExecuteResponse,
)
async def get_workflow_status(
    workflow_id: str,
) -> WorkflowExecuteResponse:
    """Get the execution status of a workflow."""
    # TODO: call workflow service to fetch status
    raise HTTPException(status_code=404, detail="Workflow not found")
