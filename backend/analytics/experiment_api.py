"""Experiment REST API (thin FastAPI layer over analytics, no simulation logic).

Endpoints:
    POST /experiments/run            run + save + return one experiment
    GET  /experiments                list saved experiments
    GET  /experiments/{experiment_id} load one saved experiment

All simulation work is delegated to analytics.experiment_runner and
analytics.experiment_store. JSON responses only. WebSocket endpoints
are untouched.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .experiment_runner import run_experiment
from .experiment_store import list_experiments, load_experiment, save_experiment

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiment_results"

# Hang guard for long runs (e.g. max ticks with YOLO). Generous default so
# legitimate runs are unaffected; override via EXPERIMENT_RUN_TIMEOUT_S.
# Non-positive or unparsable values disable the timeout.
RUN_TIMEOUT_S = float(os.environ.get("EXPERIMENT_RUN_TIMEOUT_S", "1800"))

router = APIRouter(prefix="/experiments", tags=["experiments"])


class ExperimentRequest(BaseModel):
    """POST /experiments/run body."""

    detector: str = Field(default="classical", description="'classical' or 'yolo'")
    scenario: Union[str, Dict[str, Any]] = Field(
        default="stationary_ideal",
        description="Scenario name or scenario dict",
    )
    seed: int = Field(default=0)
    ticks: int = Field(default=60, gt=0, le=2000)
    dt: float = Field(default=1.0 / 30.0, gt=0)


def _storage_dir() -> Path:
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return EXPERIMENTS_DIR


@router.post("/run")
async def run_experiment_endpoint(body: ExperimentRequest) -> Dict[str, Any]:
    """Run one experiment, persist it, and return the saved result."""
    try:
        # CPU-bound (notably YOLO inference): keep the event loop free.
        coro = asyncio.to_thread(
            run_experiment,
            body.detector,
            body.scenario,
            body.seed,
            body.ticks,
            body.dt,
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
            detail=f"Experiment exceeded {RUN_TIMEOUT_S:g}s limit",
        )
    saved = save_experiment(result, _storage_dir())
    return saved


@router.get("")
async def list_experiments_endpoint() -> List[Dict[str, Any]]:
    """Return all saved experiments (deterministically ordered)."""
    return list_experiments(_storage_dir())


@router.get("/{experiment_id}")
async def get_experiment_endpoint(experiment_id: str) -> Dict[str, Any]:
    """Load and return one saved experiment by id."""
    if not experiment_id or "/" in experiment_id or "\\" in experiment_id \
            or ".." in experiment_id:
        raise HTTPException(status_code=404, detail="Experiment not found")
    file_path = _storage_dir() / f"{experiment_id}.json"
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return load_experiment(file_path)
