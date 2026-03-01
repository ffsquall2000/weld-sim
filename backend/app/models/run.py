"""Run model - a single execution of a simulation case on a geometry version."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.artifact import Artifact
    from backend.app.models.geometry_version import GeometryVersion
    from backend.app.models.metric import Metric
    from backend.app.models.optimization_study import OptimizationStudy
    from backend.app.models.simulation_case import SimulationCase


class Run(UUIDMixin, TimestampMixin, Base):
    """A single simulation run tying a case to a geometry version."""

    __tablename__ = "runs"

    simulation_case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("simulation_cases.id", ondelete="CASCADE"),
        nullable=False,
    )
    geometry_version_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("geometry_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    optimization_study_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(),
        ForeignKey("optimization_studies.id", ondelete="SET NULL"),
        nullable=True,
    )
    iteration_number: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(50), default="queued", nullable=False
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    solver_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    compute_time_s: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    input_snapshot: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )

    # Relationships
    simulation_case: Mapped[SimulationCase] = relationship(
        "SimulationCase",
        back_populates="runs",
    )
    geometry_version: Mapped[GeometryVersion] = relationship(
        "GeometryVersion",
        back_populates="runs",
    )
    optimization_study: Mapped[Optional[OptimizationStudy]] = relationship(
        "OptimizationStudy",
        back_populates="runs",
        foreign_keys=[optimization_study_id],
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        "Artifact",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    metrics: Mapped[list[Metric]] = relationship(
        "Metric",
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
