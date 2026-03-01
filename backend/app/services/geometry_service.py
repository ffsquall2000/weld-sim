"""Service layer for GeometryVersion operations."""
from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models.geometry_version import GeometryVersion
from backend.app.schemas.geometry import GeometryCreate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helper functions for mesh preview
# ---------------------------------------------------------------------------

def _load_mesh_preview(mesh_path: str, max_faces: int = 5000) -> dict | None:
    """Load mesh preview data from a .msh file.

    Reads the mesh using meshio and extracts vertices and faces suitable
    for VTK.js visualization.  Tetrahedra are decomposed into their four
    triangular faces.  The result is capped at *max_faces* to keep the
    preview payload small.
    """
    try:
        import meshio  # type: ignore[import-untyped]

        mesh = meshio.read(mesh_path)
        vertices = mesh.points.tolist()
        faces: list[list[int]] = []
        for cell_block in mesh.cells:
            if cell_block.type in ("triangle", "triangle3"):
                faces.extend(cell_block.data.tolist())
            elif cell_block.type in ("tetra", "tetra4", "tetra10"):
                # Extract surface triangles from tetrahedra
                for tet in cell_block.data:
                    faces.extend([
                        [tet[0], tet[1], tet[2]],
                        [tet[0], tet[1], tet[3]],
                        [tet[0], tet[2], tet[3]],
                        [tet[1], tet[2], tet[3]],
                    ])
        # Limit faces for preview
        if len(faces) > max_faces:
            import random

            faces = random.sample(faces, max_faces)
        return {
            "vertices": vertices,
            "faces": faces,
            "scalar_field": None,
            "node_count": len(vertices),
            "element_count": len(faces),
        }
    except ImportError:
        logger.warning("meshio not installed, cannot load mesh preview")
        return None
    except Exception as exc:
        logger.warning("Failed to load mesh preview from %s: %s", mesh_path, exc)
        return None


def _generate_step_preview(step_path: str) -> dict | None:
    """Generate a coarse preview mesh from a STEP file.

    Uses :class:`MeshConverter` with a large element size so that the mesh
    is generated quickly and stays small enough for a preview payload.
    """
    try:
        from backend.app.solvers.mesh_converter import MeshConverter, MeshConfig

        converter = MeshConverter()
        # Use coarse mesh for quick preview
        config = MeshConfig(element_size=8.0, min_element_size=4.0, max_element_size=12.0)
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
            tmp_path = tmp.name
        converter.step_to_mesh(
            step_path=step_path,
            output_path=tmp_path,
            config=config,
        )
        preview = _load_mesh_preview(tmp_path)
        # Clean up temporary file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
        return preview
    except Exception as exc:
        logger.warning("STEP preview generation failed for %s: %s", step_path, exc)
        return None


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
        """Get mesh preview data for VTK.js visualization.

        Supports four sources of preview data, tried in order:

        1. **Parametric geometry** -- regenerate a lightweight mesh via the
           horn generator (fast, no disk I/O).
        2. **Existing mesh file** -- read from the ``.msh`` file produced by
           a prior meshing step.
        3. **Mesh info in metadata** -- if a previous mesh generation stored
           vertex / face data in ``metadata_json``.
        4. **STEP / STL file on disk** -- generate a coarse preview mesh
           on-the-fly from the imported CAD file.
        """
        geom = await self.get(geometry_id)
        if not geom:
            return None

        # Case 1: Parametric geometry (horn generator)
        if geom.parametric_params:
            try:
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
            except Exception as exc:
                logger.warning("Parametric preview failed for %s: %s", geometry_id, exc)

        # Case 2: Has generated mesh file -- extract preview from mesh
        if geom.mesh_file_path and Path(geom.mesh_file_path).exists():
            try:
                preview = _load_mesh_preview(geom.mesh_file_path)
                if preview:
                    return preview
            except Exception as exc:
                logger.warning("Mesh preview load failed for %s: %s", geometry_id, exc)

        # Case 3: Has mesh info in metadata (from mesh generation)
        if geom.metadata_json:
            mesh_info = geom.metadata_json.get("mesh_info", {})
            if "vertices" in mesh_info or "node_count" in mesh_info:
                return {
                    "vertices": mesh_info.get("vertices", []),
                    "faces": mesh_info.get("faces", []),
                    "scalar_field": None,
                    "node_count": mesh_info.get("node_count", 0),
                    "element_count": mesh_info.get("element_count", 0),
                }

        # Case 4: Has STEP file but no mesh yet -- generate a quick coarse preview
        if geom.file_path and Path(geom.file_path).exists():
            try:
                preview = _generate_step_preview(geom.file_path)
                if preview:
                    return preview
            except Exception as exc:
                logger.warning("STEP preview generation failed for %s: %s", geometry_id, exc)

        return None

    async def upload(self, project_id: UUID, file_content: bytes, filename: str, label: str | None = None) -> GeometryVersion:
        """Upload a STEP/STL file as a new geometry version.

        For STEP/STP files the service automatically runs CAD analysis and
        stores the extracted dimensions (horn_type, width_mm, height_mm,
        length_mm, volume, surface_area) in ``metadata_json`` so that the
        FEA endpoint can use them via ``geometry_id``.
        """
        import logging

        _logger = logging.getLogger(__name__)

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

        # Attempt CAD analysis for STEP/STP files to extract dimensions
        if source_type == "imported_step":
            try:
                from web.services.geometry_service import GeometryService as WebGeometryService

                web_svc = WebGeometryService()
                cad_result = web_svc.analyze_step_file(str(file_path))
                dims = cad_result.get("dimensions", {})
                # Include horn_type in dimensions for FEA lookup
                if "horn_type" in cad_result and dims:
                    dims["horn_type"] = cad_result["horn_type"]
                geom.metadata_json = {
                    "dimensions": dims,
                    "horn_type": cad_result.get("horn_type"),
                    "gain_estimate": cad_result.get("gain_estimate"),
                    "confidence": cad_result.get("confidence"),
                    "volume_mm3": cad_result.get("volume_mm3"),
                    "surface_area_mm2": cad_result.get("surface_area_mm2"),
                    "bounding_box": cad_result.get("bounding_box"),
                }
                _logger.info(
                    "CAD analysis stored for geometry %s: %s %s",
                    geom.id,
                    cad_result.get("horn_type"),
                    dims,
                )
            except Exception as exc:
                _logger.warning(
                    "CAD analysis failed for uploaded geometry %s: %s. "
                    "FEA will use default parametric params.",
                    geom.id,
                    exc,
                )

        await self.session.flush()
        await self.session.refresh(geom)
        return geom
