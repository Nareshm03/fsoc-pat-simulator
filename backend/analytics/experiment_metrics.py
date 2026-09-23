"""Reusable experiment metrics (pure analytics only).

Computes summary metrics from a finished simulation run. No imports from
runtime/PAT/physics/detector modules, no thresholds changed here -- the
default convergence threshold (5.0px) only mirrors the existing
PATStateMachine.lock_threshold for reporting; callers may override it.

Accepted input: a list of per-frame records. Each record may be either
flat or nested (backend telemetry shape). Missing keys default safely:

Flat record:
    {"detected": bool, "confidence": float, "error_px": float,
     "pat_state": str, "time": float (optional)}

Nested backend telemetry (from SimulationManager._build_telemetry):
    {"detection": {"detected": bool, "confidence": float},
     "error": {"pixel": float}, "pat": {"state": str},
     "simulation_time": float}

Returns a JSON-serializable dict with:
    detection_rate, avg_confidence, initial_error, final_error,
    minimum_error, convergence_frame/time, locked_achieved,
    lock_retention, lost/reacquired_count (+ aliases for compat).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LOCK_THRESHOLD_PX = 5.0  # mirrors PATStateMachine.lock_threshold (report only)


def _as_bool(value: Any) -> bool:
    return bool(value) if value is not None else False


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one flat or nested telemetry record to canonical form."""
    if not isinstance(record, dict):
        return {"detected": False, "confidence": 0.0, "error_px": 0.0,
                "pat_state": None, "time": None}
    detection = record.get("detection")
    if isinstance(detection, dict):
        detected = _as_bool(detection.get("detected", record.get("detected", False)))
        confidence = _as_float(detection.get("confidence", record.get("confidence", 0.0)))
    else:
        detected = _as_bool(record.get("detected", False))
        confidence = _as_float(record.get("confidence", 0.0))

    error = record.get("error")
    if isinstance(error, dict):
        error_px = _as_float(error.get("pixel", error.get("px", record.get("error_px", 0.0))))
    else:
        error_px = _as_float(record.get("error_px", record.get("error", 0.0)))

    pat = record.get("pat")
    if isinstance(pat, dict):
        pat_state = pat.get("state", record.get("pat_state"))
    else:
        pat_state = record.get("pat_state", record.get("state"))
    if pat_state is not None:
        pat_state = str(pat_state)

    time_value: Optional[float] = None
    for key in ("time", "simulation_time", "t"):
        if record.get(key) is not None:
            try:
                time_value = float(record[key])
                break
            except (TypeError, ValueError):
                continue

    if not detected:
        confidence = 0.0
    return {
        "detected": detected,
        "confidence": confidence,
        "error_px": error_px,
        "pat_state": pat_state,
        "time": time_value,
    }


def compute_experiment_metrics(
    records: List[Dict[str, Any]],
    dt: float = 1.0 / 30.0,
    convergence_threshold: float = LOCK_THRESHOLD_PX,
) -> Dict[str, Any]:
    """Compute reproducible summary metrics from per-frame records.

    Args:
        records: list of flat or nested telemetry dicts (see module docstring).
        dt: seconds per frame, used for time-derived fields when a record
            has no explicit time.
        convergence_threshold: first detected frame with error strictly
            below this value defines convergence (default 5.0px).

    Returns:
        JSON-serializable dict (only ints/floats/bools/None/str).
    """
    frames = list(records) if records else []
    total = len(frames)
    normalized = [normalize_record(r) for r in frames]

    detections = sum(1 for r in normalized if r["detected"])
    misses = total - detections
    detection_rate = (detections / total) if total else 0.0

    detected_confs = [r["confidence"] for r in normalized if r["detected"]]
    detected_errors = [r["error_px"] for r in normalized if r["detected"]]

    if detected_confs:
        avg_confidence = sum(detected_confs) / len(detected_confs)
        min_confidence = min(detected_confs)
        max_confidence = max(detected_confs)
    else:
        avg_confidence = 0.0
        min_confidence = 0.0
        max_confidence = 0.0

    initial_error: Optional[float] = detected_errors[0] if detected_errors else None
    final_error: Optional[float] = detected_errors[-1] if detected_errors else None
    minimum_error: Optional[float] = min(detected_errors) if detected_errors else None

    convergence_frame: Optional[int] = None
    for i, r in enumerate(normalized, start=1):
        if r["detected"] and r["error_px"] < convergence_threshold:
            convergence_frame = i
            break
    if convergence_frame is not None:
        rec_time = normalized[convergence_frame - 1]["time"]
        convergence_time_s = rec_time if rec_time is not None else convergence_frame * dt
    else:
        convergence_time_s = None

    locked_frames = sum(1 for r in normalized if r["pat_state"] == "LOCKED")
    locked_achieved = locked_frames > 0
    lock_retention_s = locked_frames * dt
    lock_time_s: Optional[float] = None
    if locked_achieved:
        for r in normalized:
            if r["pat_state"] == "LOCKED":
                lock_time_s = r["time"] if r["time"] is not None else None
                break
        if lock_time_s is None:
            # No explicit timestamps: derive from first LOCKED frame index.
            for i, r in enumerate(normalized, start=1):
                if r["pat_state"] == "LOCKED":
                    lock_time_s = i * dt
                    break

    # A "loss" is a detected -> undetected transition (pure, reproducible).
    lost_count = 0
    prev_detected = False
    seen_detection = False
    for r in normalized:
        if r["detected"]:
            seen_detection = True
        elif seen_detection and prev_detected:
            lost_count += 1
        prev_detected = r["detected"]

    return {
        "frames": total,
        "detections": detections,
        "misses": misses,
        "detection_rate": detection_rate,
        "avg_confidence": avg_confidence,
        "min_confidence": min_confidence,
        "max_confidence": max_confidence,
        "initial_error": initial_error,
        "final_error": final_error,
        "minimum_error": minimum_error,
        # Alias kept for compat with benchmark reports.
        "min_error": minimum_error,
        "convergence_frame": convergence_frame,
        "convergence_time_s": convergence_time_s,
        "locked_achieved": locked_achieved,
        "lock_time_s": lock_time_s,
        "lock_retention_s": lock_retention_s,
        "lost_count": lost_count,
        # Aliases for the requested lost/reacquired count naming.
        "lost_reacquired_count": lost_count,
        "reacquired_count": lost_count,
        "dt": dt,
        "convergence_threshold": convergence_threshold,
    }
