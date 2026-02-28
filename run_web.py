#!/usr/bin/env python3
"""Uvicorn entry point for the UltrasonicWeldMaster web service.

NOTE: We always use **workers=1** because the AnalysisManager singleton
(which bridges HTTP task creation → WebSocket progress streaming) lives
in-process.  Multiple Uvicorn workers would each get their own isolated
AnalysisManager, causing WebSocket connections to miss task updates when
the WebSocket lands on a different worker than the HTTP handler.

Heavy FEA computation is already off-loaded to child processes via
``FEAProcessRunner``, so a single async event-loop worker handles
concurrent HTTP + WebSocket traffic well.
"""
from __future__ import annotations

import os

import uvicorn

from web.config import WebConfig

if __name__ == "__main__":
    is_dev = os.environ.get("UWM_ENV", "production") == "development"

    uvicorn.run(
        "web.app:app",
        host=WebConfig.HOST,
        port=WebConfig.PORT,
        reload=is_dev,
        workers=1,
        timeout_keep_alive=30,
        limit_concurrency=50,
    )
