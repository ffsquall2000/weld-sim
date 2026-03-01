"""V2 recipe management endpoints.

Provides read access to simulation recipes stored by the V1 engine service
database, exposed under the /api/v2 prefix with Pydantic v2 schemas.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.recipes import RecipeDetailResponse, RecipeListResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/recipes", tags=["recipes"])

# ---------------------------------------------------------------------------
# Lazy-load the V1 engine service singleton.
# ---------------------------------------------------------------------------
try:
    from web.dependencies import get_engine_service

    _engine_service = get_engine_service()
except (ImportError, Exception):
    _engine_service = None
    logger.warning(
        "V1 EngineService not available; /recipes endpoints will return 503"
    )


def _get_engine_service():
    """Return the cached engine service or raise 503."""
    if _engine_service is None:
        raise HTTPException(
            status_code=503,
            detail="Recipe database is not available",
        )
    return _engine_service


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=RecipeListResponse)
async def list_recipes(
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of recipes"),
) -> RecipeListResponse:
    """List recent simulation recipes ordered by creation date (newest first).

    Returns a compact summary suitable for table/list UIs.
    """
    svc = _get_engine_service()
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
            # inputs may be stored as a JSON string
            if isinstance(d.get("inputs"), str):
                try:
                    d["inputs"] = json.loads(d["inputs"])
                except (json.JSONDecodeError, TypeError):
                    pass
            recipes.append(d)

        return RecipeListResponse(recipes=recipes, count=len(recipes))
    except Exception as exc:
        logger.error("Error listing recipes: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{recipe_id}", response_model=RecipeDetailResponse)
async def get_recipe(recipe_id: str) -> RecipeDetailResponse:
    """Retrieve a single recipe by its unique identifier.

    Returns the full recipe detail including parameters, safety window,
    risk assessment, quality estimates, and recommendations.
    """
    svc = _get_engine_service()
    try:
        db = svc.engine.database
        recipe_data = db.get_recipe(recipe_id)
        return RecipeDetailResponse(**recipe_data)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Error getting recipe %s: %s", recipe_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
