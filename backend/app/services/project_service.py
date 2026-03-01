"""Service layer for Project CRUD operations."""
from __future__ import annotations

from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.project import Project
from backend.app.schemas.project import ProjectCreate, ProjectUpdate


class ProjectService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: ProjectCreate) -> Project:
        project = Project(
            name=data.name,
            description=data.description,
            application_type=data.application_type,
            settings=data.settings or {},
            tags=data.tags or [],
        )
        self.session.add(project)
        await self.session.flush()
        await self.session.refresh(project)
        return project

    async def get(self, project_id: UUID) -> Project | None:
        return await self.session.get(Project, project_id)

    async def list_projects(
        self,
        skip: int = 0,
        limit: int = 20,
        application_type: str | None = None,
        search: str | None = None,
    ) -> tuple[list[Project], int]:
        query = select(Project)
        count_query = select(func.count()).select_from(Project)
        if application_type:
            query = query.where(Project.application_type == application_type)
            count_query = count_query.where(Project.application_type == application_type)
        if search:
            query = query.where(Project.name.ilike(f"%{search}%"))
            count_query = count_query.where(Project.name.ilike(f"%{search}%"))
        total = (await self.session.execute(count_query)).scalar() or 0
        query = query.offset(skip).limit(limit).order_by(Project.updated_at.desc())
        result = await self.session.execute(query)
        return list(result.scalars().all()), total

    async def update(self, project_id: UUID, data: ProjectUpdate) -> Project | None:
        project = await self.get(project_id)
        if not project:
            return None
        for field, value in data.model_dump(exclude_unset=True).items():
            setattr(project, field, value)
        await self.session.flush()
        await self.session.refresh(project)
        return project

    async def delete(self, project_id: UUID) -> bool:
        project = await self.get(project_id)
        if not project:
            return False
        await self.session.delete(project)
        await self.session.flush()
        return True
