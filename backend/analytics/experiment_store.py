"""JSON experiment result persistence (pure analytics I/O, no database).

Stores one JSON file per experiment. Each saved experiment contains:
    experiment_id, timestamp, detector, scenario, dt/duration,
    metrics (from compute_experiment_metrics), raw per-frame records
    (when supplied; older files without records still load), and
    basic run config.

Metric definitions live in experiment_metrics.py and are NOT modified
here; this module only calls compute_experiment_metrics() when the
caller supplies raw records instead of precomputed metrics.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Union

from .experiment_metrics import compute_experiment_metrics

PathLike = Union[str, Path]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_json_serializable(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate (and normalize) payload via a JSON round-trip."""
    return json.loads(json.dumps(payload, sort_keys=True))


def save_experiment(result: Dict[str, Any], path: PathLike) -> Dict[str, Any]:
    """Save one experiment result as JSON.

    Args:
        result: dict with any of:
            experiment_id (str, optional; generated when missing),
            timestamp (ISO str, optional; generated when missing),
            detector (str/dict, default "unknown"),
            scenario (str/dict, default "unknown"),
            dt (float, default 1/30),
            duration_s/duration (float, optional; derived as
                frames*dt from metrics when missing),
            config (dict, default {}),
            records (list, optional; raw per-frame telemetry used to
                derive metrics via compute_experiment_metrics),
            metrics (dict, optional; used as-is when records absent;
                recomputed from records when records are present).
        path: directory (file stored as <experiment_id>.json) or
            explicit .json file path. Parent directories are created.

    Returns:
        The saved payload dict (exactly what was written to disk).
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    experiment_id = result.get("experiment_id") or uuid.uuid4().hex
    experiment_id = str(experiment_id)
    timestamp = result.get("timestamp") or _utc_now_iso()
    timestamp = str(timestamp)
    detector = result.get("detector", "unknown")
    scenario = result.get("scenario", "unknown")
    dt = float(result.get("dt", 1.0 / 30.0))
    config = result.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("result['config'] must be a dict")

    records = result.get("records")
    stored_records = None
    if records is not None:
        if not isinstance(records, list):
            raise TypeError("result['records'] must be a list")
        metrics = compute_experiment_metrics(records, dt=dt)
        # Persist raw per-frame records so GET detail can serve them
        # (e.g. for convergence graphs). Old files without records
        # remain loadable; readers must tolerate a missing key.
        stored_records = records
    else:
        metrics = result.get("metrics", {})
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise TypeError("result['metrics'] must be a dict")
        metrics = dict(metrics)

    duration = result.get("duration_s", result.get("duration"))
    if duration is None:
        frames = metrics.get("frames")
        duration_s = float(frames * dt) if isinstance(frames, int) else 0.0
    else:
        duration_s = float(duration)

    payload = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "detector": detector,
        "scenario": scenario,
        "dt": dt,
        "duration_s": duration_s,
        "duration": duration_s,
        "config": config,
        "metrics": metrics,
    }
    if stored_records is not None:
        payload["records"] = stored_records
    payload = _ensure_json_serializable(payload)

    dest = Path(path)
    if dest.suffix.lower() == ".json":
        dest.parent.mkdir(parents=True, exist_ok=True)
        file_path = dest
    else:
        dest.mkdir(parents=True, exist_ok=True)
        file_path = dest / f"{experiment_id}.json"
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_experiment(path: PathLike) -> Dict[str, Any]:
    """Load one saved experiment JSON file."""
    file_path = Path(path)
    return json.loads(file_path.read_text())


def list_experiments(path: PathLike) -> List[Dict[str, Any]]:
    """List saved experiments as loaded dicts, sorted deterministically.

    Args:
        path: directory containing *.json files (sorted by filename),
            or a single .json file (returns a one-element list).

    Returns:
        List of experiment dicts sorted by (timestamp, experiment_id).
    """
    base = Path(path)
    if base.is_file():
        return [load_experiment(base)]
    if not base.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for file_path in sorted(base.glob("*.json")):
        items.append(load_experiment(file_path))
    items.sort(key=lambda d: (str(d.get("timestamp", "")), str(d.get("experiment_id", ""))))
    return items
