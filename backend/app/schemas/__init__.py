"""Pydantic schemas for the Ultrasonic Metal Welding Virtual Simulation Platform."""

from backend.app.schemas.artifact import ArtifactResponse
from backend.app.schemas.comparison import (
    ComparisonCreate,
    ComparisonResponse,
    ComparisonResultItem,
)
from backend.app.schemas.geometry import (
    GeometryCreate,
    GeometryResponse,
    HornParams,
    MeshPreviewResponse,
)
from backend.app.schemas.material import MaterialCreate, MaterialResponse
from backend.app.schemas.metric import MetricResponse, MetricSummary
from backend.app.schemas.optimization import (
    Constraint,
    DesignVariable,
    IterationResult,
    Objective,
    OptimizationCreate,
    OptimizationResponse,
)
from backend.app.schemas.project import (
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
)
from backend.app.schemas.run import RunCreate, RunDetailResponse, RunResponse
from backend.app.schemas.simulation import (
    SimulationCaseCreate,
    SimulationCaseResponse,
    SimulationCaseUpdate,
)
from backend.app.schemas.websocket import (
    WSCompleted,
    WSError,
    WSMessage,
    WSMetricUpdate,
    WSProgress,
)
from backend.app.schemas.report import (
    MultiReportResponse,
    ReportFormat,
    ReportListItem,
    ReportListResponse,
    ReportMetricRow,
    ReportRequest,
    ReportResponse,
)
from backend.app.schemas.workflow import (
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowExecuteResponse,
    WorkflowNode,
    WorkflowValidateResponse,
)

__all__ = [
    # Project
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectResponse",
    "ProjectListResponse",
    # Geometry
    "HornParams",
    "GeometryCreate",
    "GeometryResponse",
    "MeshPreviewResponse",
    # Simulation
    "SimulationCaseCreate",
    "SimulationCaseUpdate",
    "SimulationCaseResponse",
    # Run
    "RunCreate",
    "RunResponse",
    "RunDetailResponse",
    # Artifact
    "ArtifactResponse",
    # Metric
    "MetricResponse",
    "MetricSummary",
    # Comparison
    "ComparisonCreate",
    "ComparisonResultItem",
    "ComparisonResponse",
    # Optimization
    "DesignVariable",
    "Constraint",
    "Objective",
    "OptimizationCreate",
    "OptimizationResponse",
    "IterationResult",
    # Material
    "MaterialResponse",
    "MaterialCreate",
    # Workflow
    "WorkflowNode",
    "WorkflowEdge",
    "WorkflowDefinition",
    "WorkflowValidateResponse",
    "WorkflowExecuteResponse",
    # Report
    "ReportFormat",
    "ReportRequest",
    "ReportResponse",
    "MultiReportResponse",
    "ReportMetricRow",
    "ReportListItem",
    "ReportListResponse",
    # WebSocket
    "WSMessage",
    "WSProgress",
    "WSMetricUpdate",
    "WSCompleted",
    "WSError",
]
