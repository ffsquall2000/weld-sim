"""Service layer for GeometryVersion operations."""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models.geometry_version import GeometryVersion
from backend.app.schemas.geometry import GeometryCreate


class GeometryService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, project_id: UUID, data: GeometryCreate) -> GeometryVersion:
        # Auto-increment version number
        count_q = select(func.count()).select_from(GeometryVersion).where(
            GeometryVersion.project_id == project_id
        )
        count = (await self.session.execute(count_q)).scalar() or 0
        geom = GeometryVersion(
            project_id=project_id,
            version_number=count + 1,
            label=data.label,
            source_type=data.source_type or "parametric",
            parametric_params=data.parametric_params.model_dump() if data.parametric_params else None,
            mesh_config=data.mesh_config,
        )
        self.session.add(geom)
        await self.session.flush()
        await self.session.refresh(geom)
        return geom

    async def get(self, geometry_id: UUID) -> GeometryVersion | None:
        return await self.session.get(GeometryVersion, geometry_id)

    async def list_by_project(self, project_id: UUID) -> list[GeometryVersion]:
        query = (
            select(GeometryVersion)
            .where(GeometryVersion.project_id == project_id)
            .order_by(GeometryVersion.version_number.desc())
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def generate_parametric(self, geometry_id: UUID) -> GeometryVersion | None:
        """Generate CAD model from parametric parameters."""
        geom = await self.get(geometry_id)
        if not geom or not geom.parametric_params:
            return None
        # Use domain horn generator
        from backend.app.domain.horn_generator import HornGenerator, HornParams
        params = HornParams(**geom.parametric_params)
        generator = HornGenerator()
        result = generator.generate(params)
        # Store files
        storage_dir = Path(settings.STORAGE_PATH) / "geometries" / str(geometry_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        if result.step_data:
            step_path = storage_dir / "model.step"
            step_path.write_bytes(result.step_data)
            geom.file_path = str(step_path)
        elif result.stl_data:
            stl_path = storage_dir / "model.stl"
            stl_path.write_bytes(result.stl_data)
            geom.file_path = str(stl_path)
        geom.metadata_json = {
            "dimensions": result.dimensions,
            "volume_mm3": result.volume_mm3,
            "surface_area_mm2": result.surface_area_mm2,
            "has_cad_export": result.has_cad_export,
        }
        if result.knurl_info:
            geom.metadata_json["knurl_info"] = result.knurl_info
        if result.chamfer_info:
            geom.metadata_json["chamfer_info"] = result.chamfer_info
        await self.session.flush()
        await self.session.refresh(geom)
        return geom

    async def generate_mesh(self, geometry_id: UUID, mesh_config: dict | None = None) -> GeometryVersion | None:
        """Generate FEA mesh for a geometry."""
        geom = await self.get(geometry_id)
        if not geom:
            return None
        storage_dir = Path(settings.STORAGE_PATH) / "geometries" / str(geometry_id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        # Use mesh converter if geometry file exists
        from backend.app.solvers.mesh_converter import MeshConfig, MeshConverter
        converter = MeshConverter()
        config = MeshConfig(**(mesh_config or geom.mesh_config or {}))
        if geom.file_path and Path(geom.file_path).exists():
            mesh_result = converter.step_to_mesh(
                step_path=geom.file_path,
                output_path=str(storage_dir / "mesh.msh"),
                config=config,
            )
        else:
            # No geometry file available; return with empty mesh info
            mesh_result = {
                "mesh_path": None,
                "stats": {"error": "No geometry file available for meshing"},
            }
        geom.mesh_file_path = mesh_result.get("mesh_path")
        geom.mesh_config = mesh_config or geom.mesh_config
        if not geom.metadata_json:
            geom.metadata_json = {}
        geom.metadata_json["mesh_info"] = mesh_result.get("stats", mesh_result)
        await self.session.flush()
        await self.session.refresh(geom)
        return geom

    async def get_preview(self, geometry_id: UUID) -> dict | None:
        """Get mesh preview data for VTK.js visualization."""
        geom = await self.get(geometry_id)
        if not geom:
            return None
        # Try to load from generated mesh or parametric params
        if geom.parametric_params:
            from backend.app.domain.horn_generator import HornGenerator, HornParams
            params = HornParams(**geom.parametric_params)
            generator = HornGenerator()
            result = generator.generate(params)
            mesh = result.mesh
            return {
                "vertices": mesh.get("vertices", []),
                "faces": mesh.get("faces", []),
                "scalar_field": None,
                "node_count": len(mesh.get("vertices", [])),
                "element_count": len(mesh.get("faces", [])),
            }
        return None

    async def upload(self, project_id: UUID, file_content: bytes, filename: str, label: str | None = None) -> GeometryVersion:
        """Upload a STEP/STL file as a new geometry version."""
        count_q = select(func.count()).select_from(GeometryVersion).where(
            GeometryVersion.project_id == project_id
        )
        count = (await self.session.execute(count_q)).scalar() or 0
        source_type = "imported_step" if filename.endswith(".step") or filename.endswith(".stp") else "imported_stl"
        geom = GeometryVersion(
            project_id=project_id,
            version_number=count + 1,
            label=label or filename,
            source_type=source_type,
        )
        self.session.add(geom)
        await self.session.flush()
        # Save file
        storage_dir = Path(settings.STORAGE_PATH) / "geometries" / str(geom.id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        file_path = storage_dir / filename
        file_path.write_bytes(file_content)
        geom.file_path = str(file_path)
        await self.session.flush()
        await self.session.refresh(geom)
        return geom
