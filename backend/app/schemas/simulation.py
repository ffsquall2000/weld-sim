"""Pydantic schemas for SimulationCase resources."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class SimulationCaseCreate(BaseModel):
    """Schema for creating a new simulation case."""

    name: str
    description: Optional[str] = None
    analysis_type: str  # modal, harmonic, thermal_steady, etc.
    solver_backend: str = "preview"
    configuration: Optional[dict] = None
    boundary_conditions: Optional[dict] = None
    material_assignments: Optional[dict] = None
    assembly_components: Optional[List[str]] = None  # geometry version IDs
    workflow_dag: Optional[dict] = None


class SimulationCaseUpdate(BaseModel):
    """Schema for partially updating a simulation case."""

    name: Optional[str] = None
    description: Optional[str] = None
    configuration: Optional[dict] = None
    boundary_conditions: Optional[dict] = None
    material_assignments: Optional[dict] = None
    assembly_components: Optional[List[str]] = None
    workflow_dag: Optional[dict] = None


class SimulationCaseResponse(BaseModel):
    """Schema for returning a simulation case."""

    id: UUID
    project_id: UUID
    name: str
    description: Optional[str]
    analysis_type: str
    solver_backend: str
    configuration: Optional[dict]
    boundary_conditions: Optional[dict]
    material_assignments: Optional[dict]
    assembly_components: Optional[dict]
    workflow_dag: Optional[dict]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
