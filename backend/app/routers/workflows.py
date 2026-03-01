"""Endpoints for Workflow resources."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.workflow import WorkflowDefinition, WorkflowExecuteResponse, WorkflowValidateResponse
from backend.app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/workflows", tags=["workflows"])
_workflow_store: dict[str, WorkflowExecuteResponse] = {}


@router.post("/validate", response_model=WorkflowValidateResponse)
async def validate_workflow(body: WorkflowDefinition) -> WorkflowValidateResponse:
    svc = WorkflowService()
    return svc.validate(body)


@router.post("/execute", response_model=WorkflowExecuteResponse)
async def execute_workflow(body: WorkflowDefinition) -> WorkflowExecuteResponse:
    svc = WorkflowService()
    result = svc.execute(body)
    _workflow_store[result.workflow_id] = result
    return result


@router.get("/{workflow_id}/status", response_model=WorkflowExecuteResponse)
async def get_workflow_status(workflow_id: str) -> WorkflowExecuteResponse:
    if workflow_id not in _workflow_store:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _workflow_store[workflow_id]
