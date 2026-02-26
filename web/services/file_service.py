"""File upload/download helper for reports and uploads directories."""
from __future__ import annotations

import os

from web.config import WebConfig


class FileService:
    """Simple static helper to ensure reports/uploads directories exist."""

    @staticmethod
    def get_reports_dir() -> str:
        """Return the reports directory path, creating it if needed."""
        path = os.path.abspath(WebConfig.REPORTS_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_uploads_dir() -> str:
        """Return the uploads directory path, creating it if needed."""
        path = os.path.abspath(WebConfig.UPLOADS_DIR)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def get_report_path(filename: str) -> str | None:
        """Return the full path for a report file, or None if it does not exist."""
        reports_dir = FileService.get_reports_dir()
        filepath = os.path.join(reports_dir, filename)
        if os.path.isfile(filepath):
            return filepath
        return None
