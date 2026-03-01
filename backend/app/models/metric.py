"""Metric model - scalar results from a simulation run."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Float, ForeignKey, JSON, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.run import Run


class Metric(UUIDMixin, TimestampMixin, Base):
    """A scalar metric result from a simulation run."""

    __tablename__ = "metrics"
    __table_args__ = (
        UniqueConstraint("run_id", "metric_name", name="uq_run_metric"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    metric_name: Mapped[str] = mapped_column(String(255), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    unit: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="metrics",
    )
