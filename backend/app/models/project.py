"""Project model - top-level container for simulation work."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.comparison import Comparison
    from backend.app.models.geometry_version import GeometryVersion
    from backend.app.models.simulation_case import SimulationCase


class Project(UUIDMixin, TimestampMixin, Base):
    """A welding simulation project."""

    __tablename__ = "projects"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    application_type: Mapped[str] = mapped_column(String(100), nullable=False)
    settings: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(
        JSON, nullable=True
    )
    owner: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Relationships
    geometry_versions: Mapped[list[GeometryVersion]] = relationship(
        "GeometryVersion",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    simulation_cases: Mapped[list[SimulationCase]] = relationship(
        "SimulationCase",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    comparisons: Mapped[list[Comparison]] = relationship(
        "Comparison",
        back_populates="project",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
