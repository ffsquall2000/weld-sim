"""Abstract solver interface for FEA backends."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ultrasonic_weld_master.plugins.geometry_analyzer.fea.config import (
    ModalConfig,
    HarmonicConfig,
    StaticConfig,
)
from ultrasonic_weld_master.plugins.geometry_analyzer.fea.results import (
    ModalResult,
    HarmonicResult,
    StaticResult,
)


class SolverInterface(ABC):
    """Abstract base for FEA solvers."""

    @abstractmethod
    def modal_analysis(self, config: ModalConfig) -> ModalResult:
        """Run eigenvalue analysis."""
        ...

    @abstractmethod
    def harmonic_analysis(self, config: HarmonicConfig) -> HarmonicResult:
        """Run harmonic response analysis."""
        ...

    @abstractmethod
    def static_analysis(self, config: StaticConfig) -> StaticResult:
        """Run static stress analysis."""
        ...
