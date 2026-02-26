#!/usr/bin/env python3
"""Uvicorn entry point for the UltrasonicWeldMaster web service."""
from __future__ import annotations

import uvicorn

from web.config import WebConfig

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host=WebConfig.HOST,
        port=WebConfig.PORT,
        reload=True,
    )
