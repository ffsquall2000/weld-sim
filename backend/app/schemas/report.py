"""Pydantic schemas for Report generation endpoints."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ReportFormat(str, Enum):
    """Supported report output formats."""

    pdf = "pdf"
    excel = "excel"
    json = "json"
    all = "all"


class ReportRequest(BaseModel):
    """Request body for generating a report."""

    run_ids: List[str] = Field(
        ...,
        min_length=1,
        description="One or more Run UUIDs to include in the report.",
    )
    format: ReportFormat = Field(
        default=ReportFormat.pdf,
        description="Output format: pdf, excel, json, or all.",
    )
    include_screenshots: bool = Field(
        default=False,
        description="If True, embed VTK screenshot images (paths resolved from artifacts).",
    )
    title: Optional[str] = Field(
        default=None,
        description="Custom title for the report. Defaults to project/run info.",
    )


class ReportMetricRow(BaseModel):
    """A single metric row in the report."""

    metric_name: str
    display_name: str
    value: float
    unit: Optional[str] = None
    status: str = Field(
        description="pass / warn / fail based on threshold evaluation."
    )
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None


class ReportResponse(BaseModel):
    """Response after generating a single-format report."""

    status: str = "ok"
    format: str
    file_path: str
    file_size_bytes: int
    generated_at: datetime


class MultiReportResponse(BaseModel):
    """Response when format='all' is requested."""

    status: str = "ok"
    files: Dict[str, str] = Field(
        description="Mapping of format name to file path, e.g. {'pdf': '/path/to/file.pdf', ...}",
    )
    file_sizes: Dict[str, int] = Field(
        default_factory=dict,
        description="Mapping of format name to file size in bytes.",
    )
    generated_at: datetime


class ReportListItem(BaseModel):
    """An entry when listing available report files."""

    filename: str
    format: str
    file_size_bytes: int
    created_at: datetime


class ReportListResponse(BaseModel):
    """Response for the list-reports endpoint."""

    reports: List[ReportListItem]
    total: int
