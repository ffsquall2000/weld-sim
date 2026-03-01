"""OptimizationStudy model - parameter optimization over simulation runs."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.run import Run
    from backend.app.models.simulation_case import SimulationCase


class OptimizationStudy(UUIDMixin, TimestampMixin, Base):
    """An optimization study that drives multiple simulation runs."""

    __tablename__ = "optimization_studies"

    simulation_case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("simulation_cases.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    design_variables: Mapped[dict] = mapped_column(JSON, nullable=False)
    constraints: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    objectives: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    total_iterations: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    completed_iterations: Mapped[int] = mapped_column(
        Integer, default=0, nullable=False
    )
    best_run_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(),
        ForeignKey("runs.id", ondelete="SET NULL"),
        nullable=True,
    )
    pareto_front_run_ids: Mapped[Optional[list[uuid.UUID]]] = mapped_column(
        JSON, nullable=True
    )

    # Relationships
    simulation_case: Mapped[SimulationCase] = relationship(
        "SimulationCase",
        back_populates="optimization_studies",
    )
    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="optimization_study",
        foreign_keys="[Run.optimization_study_id]",
        lazy="selectin",
    )
    best_run: Mapped[Optional[Run]] = relationship(
        "Run",
        foreign_keys=[best_run_id],
        lazy="selectin",
    )
