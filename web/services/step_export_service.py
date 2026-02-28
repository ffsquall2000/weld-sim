"""STEP file export service for CadQuery solids.

Manages exporting CadQuery solids to STEP files in a temporary directory,
with automatic cleanup of old files to prevent disk usage from growing
unboundedly.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class StepExportService:
    """Export CadQuery solids to STEP files and manage the export directory.

    Files are written to :attr:`EXPORT_DIR` and automatically cleaned up
    when they exceed *max_age_seconds* (default 1 hour).
    """

    EXPORT_DIR = "/tmp/weld-sim-exports"

    def export(self, solid, filename: str) -> str:
        """Export a CadQuery solid (Workplane) to a STEP file.

        Parameters
        ----------
        solid
            A CadQuery Workplane or Shape object.
        filename : str
            Target filename (e.g. ``"horn_knurl.step"``).  If it does not
            end with ``.step``, the suffix is appended.

        Returns
        -------
        str
            Absolute path to the written STEP file.

        Raises
        ------
        RuntimeError
            If CadQuery is not installed or the export fails.
        """
        # Ensure the filename has the correct extension
        if not filename.lower().endswith(".step"):
            filename = f"{filename}.step"

        # Create the export directory if it does not exist
        os.makedirs(self.EXPORT_DIR, exist_ok=True)

        # Cleanup old files on each export call
        self.cleanup()

        file_path = os.path.join(self.EXPORT_DIR, filename)

        try:
            import cadquery as cq
            from cadquery import exporters

            exporters.export(solid, file_path)
        except ImportError as exc:
            raise RuntimeError(
                "CadQuery is required for STEP export but is not installed."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"STEP export failed: {exc}"
            ) from exc

        logger.info("Exported STEP file: %s", file_path)
        return file_path

    def get_file(self, filename: str) -> str | None:
        """Get the path to an exported file if it exists.

        Parameters
        ----------
        filename : str
            The filename to look up (e.g. ``"horn_knurl.step"``).

        Returns
        -------
        str | None
            Absolute path if the file exists, otherwise ``None``.
        """
        if not filename.lower().endswith(".step"):
            filename = f"{filename}.step"

        file_path = os.path.join(self.EXPORT_DIR, filename)
        if os.path.isfile(file_path):
            return file_path
        return None

    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """Remove files older than *max_age_seconds* from the export directory.

        Parameters
        ----------
        max_age_seconds : int
            Maximum file age in seconds.  Files older than this are deleted.
            Defaults to 3600 (1 hour).

        Returns
        -------
        int
            Number of files removed.
        """
        if not os.path.isdir(self.EXPORT_DIR):
            return 0

        now = time.time()
        removed = 0

        for entry in os.scandir(self.EXPORT_DIR):
            if not entry.is_file():
                continue
            try:
                age = now - entry.stat().st_mtime
                if age > max_age_seconds:
                    os.remove(entry.path)
                    removed += 1
                    logger.debug("Cleaned up old export: %s", entry.name)
            except OSError as exc:
                logger.warning("Failed to remove %s: %s", entry.path, exc)

        if removed > 0:
            logger.info("Cleaned up %d old STEP export(s)", removed)

        return removed
