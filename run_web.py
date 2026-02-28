#!/usr/bin/env python3
"""Uvicorn entry point for the UltrasonicWeldMaster web service."""
from __future__ import annotations

import os

import uvicorn

from web.config import WebConfig

if __name__ == "__main__":
    is_dev = os.environ.get("UWM_ENV", "production") == "development"
    workers = int(os.environ.get("UWM_WORKERS", "1" if is_dev else "4"))

    uvicorn.run(
        "web.app:app",
        host=WebConfig.HOST,
        port=WebConfig.PORT,
        reload=is_dev,
        workers=1 if is_dev else workers,
        timeout_keep_alive=30,
        limit_concurrency=50,
    )
