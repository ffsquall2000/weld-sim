"""Recipe listing endpoints."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from web.dependencies import get_engine_service
from web.services.engine_service import EngineService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recipes"])


@router.get("/recipes")
async def list_recipes(
    limit: int = Query(default=50, ge=1, le=500),
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """List recent recipes from the database."""
    try:
        db = svc.engine.database
        rows = db.execute(
            "SELECT id, application, inputs, created_at FROM recipes "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        recipes = []
        for row in rows:
            d = dict(row)
            d["inputs"] = json.loads(d["inputs"]) if isinstance(d["inputs"], str) else d["inputs"]
            recipes.append(d)
        return {"recipes": recipes, "count": len(recipes)}
    except Exception as exc:
        logger.error("Error listing recipes: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/recipes/{recipe_id}")
async def get_recipe(
    recipe_id: str,
    svc: EngineService = Depends(get_engine_service),
) -> dict:
    """Get a single recipe by its ID."""
    try:
        db = svc.engine.database
        recipe = db.get_recipe(recipe_id)
        return recipe
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error getting recipe: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
