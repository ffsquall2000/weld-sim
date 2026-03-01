"""Endpoints for report generation, download, listing, and deletion."""

from __future__ import annotations

import logging
import os
import traceback
from typing import Union

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.dependencies import get_db
from backend.app.schemas.report import (
    MultiReportResponse,
    ReportListResponse,
    ReportRequest,
    ReportResponse,
)
from backend.app.services import report_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


# ------------------------------------------------------------------
# POST /reports/generate
# ------------------------------------------------------------------

@router.post(
    "/reports/generate",
    response_model=Union[ReportResponse, MultiReportResponse],
    status_code=201,
    summary="Generate a PDF, Excel, or JSON report for one or more runs.",
)
async def generate_report(
    body: ReportRequest,
    session: AsyncSession = Depends(get_db),
) -> Union[ReportResponse, MultiReportResponse]:
    """Generate a report from simulation run data.

    Supports single-run and multi-run (comparison) modes.
    When ``format`` is ``"all"``, PDF, Excel, **and** JSON are generated.
    """
    try:
        result = await report_service.generate_report(
            session=session,
            run_ids=body.run_ids,
            fmt=body.format.value,
            include_screenshots=body.include_screenshots,
            title=body.title,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Report generation failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if body.format.value == "all":
        return MultiReportResponse(
            status=result["status"],
            files=result["files"],
            file_sizes=result["file_sizes"],
            generated_at=result["generated_at"],
        )

    return ReportResponse(
        status=result["status"],
        format=result["format"],
        file_path=result["file_path"],
        file_size_bytes=result["file_size_bytes"],
        generated_at=result["generated_at"],
    )


# ------------------------------------------------------------------
# GET /reports/download/{filename}
# ------------------------------------------------------------------

@router.get(
    "/reports/download/{filename}",
    summary="Download a previously generated report file.",
)
async def download_report(filename: str) -> FileResponse:
    """Serve a report file for download."""
    filepath = report_service.get_report_path(filename)
    if filepath is None:
        raise HTTPException(
            status_code=404, detail=f"Report file '{filename}' not found."
        )

    # Determine media type based on extension
    ext = os.path.splitext(filename)[1].lower()
    media_types = {
        ".pdf": "application/pdf",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".json": "application/json",
    }
    media_type = media_types.get(ext, "application/octet-stream")

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type=media_type,
    )


# ------------------------------------------------------------------
# GET /reports/list
# ------------------------------------------------------------------

@router.get(
    "/reports/list",
    response_model=ReportListResponse,
    summary="List all available report files.",
)
async def list_reports() -> ReportListResponse:
    """Return a list of report files stored on disk."""
    items = report_service.list_reports()
    return ReportListResponse(reports=items, total=len(items))


# ------------------------------------------------------------------
# DELETE /reports/{filename}
# ------------------------------------------------------------------

@router.delete(
    "/reports/{filename}",
    summary="Delete a report file.",
)
async def delete_report(filename: str) -> dict:
    """Delete a previously generated report file."""
    deleted = report_service.delete_report(filename)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Report file '{filename}' not found."
        )
    return {"status": "ok", "deleted": filename}
