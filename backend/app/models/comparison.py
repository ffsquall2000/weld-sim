"""Comparison and ComparisonResult models - cross-run comparison data."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.project import Project
    from backend.app.models.run import Run


class Comparison(UUIDMixin, TimestampMixin, Base):
    """A comparison across multiple simulation runs."""

    __tablename__ = "comparisons"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    run_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    metric_names: Mapped[Optional[list[str]]] = mapped_column(
        ARRAY(String), nullable=True
    )
    baseline_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    configuration: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )

    # Relationships
    project: Mapped[Project] = relationship(
        "Project",
        back_populates="comparisons",
    )
    baseline_run: Mapped[Optional[Run]] = relationship(
        "Run",
        foreign_keys=[baseline_run_id],
        lazy="selectin",
    )
    results: Mapped[list[ComparisonResult]] = relationship(
        "ComparisonResult",
        back_populates="comparison",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class ComparisonResult(UUIDMixin, TimestampMixin, Base):
    """A single metric result row within a comparison."""

    __tablename__ = "comparison_results"

    comparison_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("comparisons.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    delta_from_baseline: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    delta_percent: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )

    # Relationships
    comparison: Mapped[Comparison] = relationship(
        "Comparison",
        back_populates="results",
    )
