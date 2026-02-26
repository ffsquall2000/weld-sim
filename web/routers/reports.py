"""Report export and download endpoints."""
from __future__ import annotations

import logging
import os
import traceback

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from web.dependencies import get_engine_service
from web.services.engine_service import EngineService
from web.services.file_service import FileService
from ultrasonic_weld_master.core.models import WeldRecipe, ValidationResult, ValidationStatus

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


class ExportRequest(BaseModel):
    """Request body for report export."""
    recipe_id: str
    format: str = "json"  # json | excel | pdf | all


@router.post("/reports/export")
async def export_report(
    request: ExportRequest,
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Export a report for a given recipe."""
    try:
        db = svc.engine.database
        recipe_data = db.get_recipe(request.recipe_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Reconstruct WeldRecipe from DB data
    recipe = WeldRecipe(
        recipe_id=recipe_data["id"],
        application=recipe_data["application"],
        inputs=recipe_data.get("inputs", {}),
        parameters=recipe_data.get("parameters", {}),
        safety_window=recipe_data.get("safety_window", {}),
        quality_estimate=recipe_data.get("quality_estimate", {}),
        risk_assessment=recipe_data.get("risk_assessment", {}),
        recommendations=recipe_data.get("recommendations", []),
        created_at=recipe_data.get("created_at", ""),
    )

    # Reconstruct ValidationResult if present
    vr_data = recipe_data.get("validation_result", {})
    validation = None
    if vr_data:
        validation = ValidationResult(
            status=ValidationStatus(vr_data.get("status", "pass")),
            validators=vr_data.get("validators", {}),
            messages=vr_data.get("messages", []),
        )

    output_dir = FileService.get_reports_dir()
    fmt = request.format

    try:
        if fmt == "all":
            reporter = svc.engine.plugin_manager.get_plugin("reporter")
            paths = reporter.export_all(recipe, validation, output_dir)
            return {"status": "ok", "files": paths}
        else:
            path = svc.export_report(recipe, validation, fmt, output_dir)
            return {"status": "ok", "file": path}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Export error: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/reports/download/{filename}")
async def download_report(filename: str) -> FileResponse:
    """Download a previously exported report file."""
    filepath = FileService.get_report_path(filename)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"Report file '{filename}' not found")
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream",
    )
