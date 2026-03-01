"""GeometryVersion model - versioned geometry data for a project."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.project import Project
    from backend.app.models.run import Run


class GeometryVersion(UUIDMixin, TimestampMixin, Base):
    """A versioned geometry configuration within a project."""

    __tablename__ = "geometry_versions"

    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # parametric / imported_step / imported_stl
    parametric_params: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
    file_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    mesh_config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    mesh_file_path: Mapped[Optional[str]] = mapped_column(
        String, nullable=True
    )
    parent_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("geometry_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )

    # Relationships
    project: Mapped[Project] = relationship(
        "Project",
        back_populates="geometry_versions",
    )
    parent_version: Mapped[Optional[GeometryVersion]] = relationship(
        "GeometryVersion",
        remote_side="GeometryVersion.id",
        lazy="selectin",
    )
    runs: Mapped[list[Run]] = relationship(
        "Run",
        back_populates="geometry_version",
        lazy="selectin",
    )
