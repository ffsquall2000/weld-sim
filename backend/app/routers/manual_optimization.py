"""Manual optimization API endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.manual_optimization import (
    ManualSessionCreate,
    ManualSessionResponse,
    RecordIterationRequest,
    SuggestionRequest,
    SuggestionResponse,
    TrendAnalysisResponse,
)
from backend.app.services.manual_optimization_service import (
    ManualOptimizationService,
)

router = APIRouter(
    prefix="/manual-optimization", tags=["manual-optimization"]
)

# In-memory session store (would be DB-backed in production)
_sessions: dict[str, dict] = {}
_service = ManualOptimizationService()


@router.post("/sessions", response_model=ManualSessionResponse)
async def create_session(req: ManualSessionCreate):
    """Create a new manual optimization session."""
    session = _service.create_session(req.project_id, req.baseline_params)
    _sessions[session["session_id"]] = session
    return ManualSessionResponse(**session)


@router.get("/sessions/{session_id}", response_model=ManualSessionResponse)
async def get_session(session_id: str):
    """Get manual optimization session details."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return ManualSessionResponse(**session)


@router.post(
    "/sessions/{session_id}/suggestions",
    response_model=SuggestionResponse,
)
async def get_suggestions(session_id: str, req: SuggestionRequest):
    """Generate parameter adjustment suggestions for the current state."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = _service.generate_suggestions(
        current_params=req.current_params,
        baseline_params=req.baseline_params,
        context=req.context,
        trial_history=req.trial_history,
    )
    return result


@router.post("/suggest", response_model=SuggestionResponse)
async def suggest_without_session(req: SuggestionRequest):
    """Generate suggestions without a session (stateless)."""
    result = _service.generate_suggestions(
        current_params=req.current_params,
        baseline_params=req.baseline_params,
        context=req.context,
        trial_history=req.trial_history,
    )
    return result


@router.post("/sessions/{session_id}/iterations")
async def record_iteration(session_id: str, req: RecordIterationRequest):
    """Record a manual optimization iteration."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    updated = _service.record_iteration(
        session, req.params, req.results, req.notes
    )
    _sessions[session_id] = updated
    return {
        "iteration": updated["iteration_count"],
        "status": "recorded",
    }


@router.get(
    "/sessions/{session_id}/trends",
    response_model=TrendAnalysisResponse,
)
async def get_trends(session_id: str):
    """Get trend analysis for a manual optimization session."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    analysis = _service.get_trend_analysis(session)
    return analysis


@router.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a manual optimization session."""
    session = _sessions.pop(session_id, None)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "status": "closed",
        "total_iterations": session["iteration_count"],
    }
