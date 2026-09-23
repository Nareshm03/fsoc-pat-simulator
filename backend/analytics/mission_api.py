"""Mission REST API (thin FastAPI layer over analytics, no simulation logic).

Endpoint:
    POST /mission/run    run the scripted mission and return its report

All mission work is delegated to analytics.mission_runner.run_mission;
this module adds no mission logic, saves no files, and returns JSON only.
WebSocket and experiment endpoints are untouched.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .mission_runner import run_mission

# Hang guard for long runs (notably YOLO inference per tick). Generous
# default so legitimate runs are unaffected; override via
# MISSION_RUN_TIMEOUT_S. Non-positive or unparsable values disable it.
try:
    RUN_TIMEOUT_S = float(os.environ.get("MISSION_RUN_TIMEOUT_S", "1800"))
except ValueError:
    RUN_TIMEOUT_S = 0.0

router = APIRouter(prefix="/mission", tags=["mission"])


class MissionRequest(BaseModel):
    """POST /mission/run body."""

    detector: str = Field(default="classical", description="'classical' or 'yolo'")
    seed: int = Field(default=7)
    dt: float = Field(default=1.0 / 30.0, gt=0)
    phase_budgets: Optional[Dict[str, int]] = Field(
        default=None,
        description="Optional per-phase tick-budget overrides",
    )


@router.post("/run")
async def run_mission_endpoint(body: MissionRequest) -> Dict[str, Any]:
    """Run the scripted mission in a worker thread; return its report."""
    try:
        # CPU-bound (notably YOLO inference): keep the event loop free.
        coro = asyncio.to_thread(
            run_mission,
            body.detector,
            body.seed,
            body.dt,
            body.phase_budgets,
        )
        if RUN_TIMEOUT_S > 0:
            result = await asyncio.wait_for(coro, timeout=RUN_TIMEOUT_S)
        else:
            result = await coro
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Mission exceeded {RUN_TIMEOUT_S:g}s limit",
        )
    return result
