"""SimulationCase model - defines analysis configuration and solver settings."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.optimization_study import OptimizationStudy
    from backend.app.models.project import Project
    from backend.app.models.run import Run


class SimulationCase(UUIDMixin, TimestampMixin, Base):
    """A simulation case containing analysis type, solver, and configuration."""

    __tablename__ = "simulation_cases"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    analysis_type: Mapped[str] = mapped_column(String(100), nullable=False)
    solver_backend: Mapped[str] = mapped_column(String(100), nullable=False)
    configuration: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    boundary_conditions: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    material_assignments: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    assembly_components: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    workflow_dag: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )

    # Relationships
    project: Mapped[Project] = relationship(
        "Project",
        back_populates="simulation_cases",
    )
    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="simulation_case",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    optimization_studies: Mapped[list[OptimizationStudy]] = relationship(
        "OptimizationStudy",
        back_populates="simulation_case",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
