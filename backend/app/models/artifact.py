"""Artifact model - output files produced by a simulation run."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, ForeignKey, JSON, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.models.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from backend.app.models.run import Run


class Artifact(UUIDMixin, TimestampMixin, Base):
    """An output artifact (file) produced by a simulation run."""

    __tablename__ = "artifacts"

    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type: Mapped[str] = mapped_column(String(100), nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True
    )
    mime_type: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )

    # Relationships
    run: Mapped[Run] = relationship(
        "Run",
        back_populates="artifacts",
    )
