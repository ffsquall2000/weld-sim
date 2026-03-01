"""Material model - custom material property definitions."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Float, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.models.base import Base, TimestampMixin, UUIDMixin


class Material(UUIDMixin, TimestampMixin, Base):
    """A custom material with physical and acoustic properties."""

    __tablename__ = "materials"

    name: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )
    category: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True
    )
    density_kg_m3: Mapped[float] = mapped_column(Float, nullable=False)
    youngs_modulus_pa: Mapped[float] = mapped_column(Float, nullable=False)
    poisson_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    yield_strength_mpa: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    thermal_conductivity: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    specific_heat: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    acoustic_impedance: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    properties_json: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True
    )
