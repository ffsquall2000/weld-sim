"""Pydantic schemas for Optimization resources."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class DesignVariable(BaseModel):
    """Schema for a single design variable in an optimization study."""

    name: str
    var_type: str = "continuous"  # continuous, categorical
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[List[str]] = None  # for categorical


class Constraint(BaseModel):
    """Schema for a constraint on an optimization study."""

    metric: str
    operator: str  # <=, >=, ==, within_percent
    value: float
    tolerance_pct: Optional[float] = None


class Objective(BaseModel):
    """Schema for an optimization objective."""

    metric: str
    direction: str = "minimize"  # minimize, maximize
    weight: float = 1.0


class OptimizationCreate(BaseModel):
    """Schema for creating an optimization study."""

    name: str
    strategy: str = "parametric_sweep"
    design_variables: List[DesignVariable]
    constraints: Optional[List[Constraint]] = None
    objectives: List[Objective]
    max_iterations: int = 50


class OptimizationResponse(BaseModel):
    """Schema for returning an optimization study."""

    id: UUID
    simulation_case_id: UUID
    name: str
    strategy: str
    design_variables: List[dict]
    constraints: Optional[List[dict]]
    objectives: Optional[List[dict]]
    status: str
    total_iterations: int
    completed_iterations: int
    best_run_id: Optional[UUID]
    pareto_front_run_ids: Optional[List[UUID]]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class IterationResult(BaseModel):
    """Schema for a single iteration result in an optimization study."""

    iteration: int
    run_id: UUID
    parameters: dict
    metrics: Dict[str, float]
    feasible: bool
