"""Service layer for Comparison operations."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.comparison import Comparison, ComparisonResult
from backend.app.models.metric import Metric
from backend.app.schemas.comparison import ComparisonCreate


class ComparisonService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, project_id: UUID, data: ComparisonCreate) -> Comparison:
        comp = Comparison(
            project_id=project_id,
            name=data.name,
            description=data.description,
            run_ids=data.run_ids,
            metric_names=data.metric_names,
            baseline_run_id=data.baseline_run_id,
        )
        self.session.add(comp)
        await self.session.flush()
        # Compute comparison results
        await self._compute_results(comp)
        await self.session.refresh(comp)
        return comp

    async def get(self, comparison_id: UUID) -> Comparison | None:
        query = (
            select(Comparison)
            .where(Comparison.id == comparison_id)
            .options(selectinload(Comparison.results))
        )
        result = await self.session.execute(query)
        return result.scalar_one_or_none()

    async def refresh(self, comparison_id: UUID) -> Comparison | None:
        comp = await self.get(comparison_id)
        if not comp:
            return None
        # Delete old results
        for r in comp.results:
            await self.session.delete(r)
        await self.session.flush()
        # Recompute
        await self._compute_results(comp)
        await self.session.refresh(comp)
        return comp

    async def _compute_results(self, comp: Comparison) -> None:
        """Compute comparison results from run metrics."""
        baseline_metrics: dict[str, float] = {}
        if comp.baseline_run_id:
            query = select(Metric).where(Metric.run_id == comp.baseline_run_id)
            result = await self.session.execute(query)
            for m in result.scalars().all():
                baseline_metrics[m.metric_name] = m.value

        for run_id in (comp.run_ids or []):
            query = select(Metric).where(Metric.run_id == run_id)
            result = await self.session.execute(query)
            for m in result.scalars().all():
                if comp.metric_names and m.metric_name not in comp.metric_names:
                    continue
                delta = None
                delta_pct = None
                if m.metric_name in baseline_metrics:
                    baseline_val = baseline_metrics[m.metric_name]
                    delta = m.value - baseline_val
                    delta_pct = (delta / baseline_val * 100) if baseline_val != 0 else None
                cr = ComparisonResult(
                    comparison_id=comp.id,
                    run_id=run_id,
                    metric_name=m.metric_name,
                    value=m.value,
                    delta_from_baseline=delta,
                    delta_percent=delta_pct,
                )
                self.session.add(cr)
        await self.session.flush()
