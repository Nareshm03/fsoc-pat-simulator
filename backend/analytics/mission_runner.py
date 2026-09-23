"""Deterministic end-to-end Mission Mode (orchestration only, no I/O).

Runs a scripted scenario through the existing closed PAT loop only::

    Target.pixel_from_angles -> Disturbance offsets -> VirtualCamera.render
    -> detector.detect -> Tracker -> error_px -> PATStateMachine
    -> RasterSearch (search mode) | PATController (otherwise)
    -> PanTilt.update, with LinkBudget evaluated from the same geometry.

Scripted phases, in order:

    1. acquisition       - stationary beacon, ends on first ACQUIRE+
    2. stationary_lock   - stationary beacon, ends on first LOCKED
    3. moving            - moving target, fixed duration
    4. disturbed         - moving target + disturbances, fixed duration
    5. loss              - forced blank frames (real detector sees nothing),
                           ends on first REACQUIRE/SEARCH
    6. search_reacquire  - blank until SEARCH (raster sweep proven), then the
                           stationary beacon is restored and tracked until
                           ACQUIRE+ via the real detector
    7. final_lock        - stationary beacon until LOCKED, plus a retention tail

No control/physics/detector logic is duplicated or modified here; every
algorithm is called from its existing module. Detections are never faked
(blank frames go through the real detector, which must return None) and
lock states are never forced (they emerge from PATStateMachine). The only
randomness source is the supplied seed (Python + NumPy RNGs are reseeded
on entry). This module never writes files.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from simulation.camera import VirtualCamera
from simulation.disturbances import DisturbanceModel
from simulation.link_budget import LinkBudgetConfig
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATState, PATStateMachine
from simulation.search_pattern import RasterSearch
from simulation.sim_step import simulation_step
from simulation.target import Target
from simulation.tracker import Tracker

from .experiment_metrics import compute_experiment_metrics
from .experiment_runner import _resolve_detector

# Scripted beacon geometry (degrees). The restore/final-lock legs reuse the
# stationary beacon, mirroring the proven reacquisition tests.
STATIONARY_AZ_DEG = 5.0
STATIONARY_EL_DEG = -8.0

# Disturbance regime of the disturbed leg (mirrors the moving-target tests).
DISTURBED = {"turbulence": 2.0, "vibration": 2.0,
             "camera_motion": 2.0, "sensor_noise": 1.0}
QUIET = {"turbulence": 0.0, "vibration": 0.0,
         "camera_motion": 0.0, "sensor_noise": 0.0}

PHASE_ORDER = ["acquisition", "stationary_lock", "moving", "disturbed",
               "loss", "search_reacquire", "final_lock"]

_TRACK_STATES = ("ACQUIRE", "FINE_TRACK", "LOCKED")


def _phase_budgets(overrides: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Per-phase tick budgets (safety guards, not behavior)."""
    budgets = {
        "acquisition": 300,        # 10 s to first ACQUIRE+
        "stationary_lock": 600,    # 20 s to first LOCKED
        "moving": 300,             # fixed 10 s moving leg
        "disturbed": 300,          # fixed 10 s disturbed leg
        "loss": 300,               # 10 s to first REACQUIRE/SEARCH
        "search_blank": 300,       # 10 s of sweep to reach SEARCH
        "reacquire": 600,          # 20 s from restore to ACQUIRE+
        "final_lock": 600,         # 20 s back to LOCKED
        "retention_tail": 150,     # 5 s lock-retention observation
    }
    for key, value in (overrides or {}).items():
        if key not in budgets:
            raise ValueError(f"Unknown phase budget: {key!r}")
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"Budget {key!r} must be a positive int")
        budgets[key] = value
    return budgets


def run_mission(
    detector: Any = "classical",
    seed: int = 7,
    dt: float = 1.0 / 30.0,
    phase_budgets: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Run the scripted mission and return the machine-readable report.

    Args:
        detector: "classical" / "yolo" or an instance with detect(frame).
        seed: RNG seed; the only randomness source (sensor noise).
        dt: seconds per tick (> 0).
        phase_budgets: optional per-phase tick-budget overrides.

    Returns:
        JSON-serializable dict with mission status, phase timeline and
        durations, final PAT/link state, recovery time, final pointing
        error, aggregate metrics, and complete per-tick records. Never
        writes files.
    """
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    seed = int(seed)
    budgets = _phase_budgets(phase_budgets)

    detector_name, detector_instance = _resolve_detector(detector)

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass

    target = Target(width=1280, height=720)
    camera = VirtualCamera(width=1280, height=720)
    pan_tilt = PanTilt()
    controller = PATController()
    state_machine = PATStateMachine()
    search = RasterSearch()
    tracker = Tracker()
    link_config = LinkBudgetConfig()

    fixed_az = math.radians(STATIONARY_AZ_DEG)
    fixed_el = math.radians(STATIONARY_EL_DEG)

    records: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    phases: List[Dict[str, Any]] = []

    sim_time = 0.0
    current_phase: Optional[str] = None
    phase_start_t = 0.0
    phase_start_tick = 0

    def note(event: str, extra: Optional[Dict[str, Any]] = None) -> None:
        entry: Dict[str, Any] = {"time": sim_time, "phase": current_phase,
                                 "event": event}
        if extra:
            entry.update(extra)
        timeline.append(entry)

    def close_phase(end_state: str, met: bool) -> None:
        phases.append({
            "name": current_phase,
            "start_t": phase_start_t,
            "end_t": sim_time,
            "duration_s": sim_time - phase_start_t,
            "ticks": len(records) - phase_start_tick,
            "end_state": end_state,
            "end_condition_met": met,
        })

    # Mission event markers (filled as the run unfolds).
    marks: Dict[str, Optional[float]] = {
        "loss_onset": None,
        "search_entered": None,
        "beacon_restored": None,
        "redetected": None,
        "reacquired": None,
        "final_lock": None,
    }

    def tick(phase: str, moving: bool, levels: Dict[str, float],
             blank: bool) -> Dict[str, Any]:
        """One closed-loop tick via the shared simulation step."""
        nonlocal sim_time
        sim_time += dt

        disturbances = DisturbanceModel(
            turbulence=levels["turbulence"], vibration=levels["vibration"],
            camera_motion=levels["camera_motion"],
            sensor_noise=levels["sensor_noise"],
        )
        # Shared closed-loop tick (mission semantics: raster search,
        # feedforward, tracker, link budget, out-of-FOV render, blank guard).
        out = simulation_step(
            target=target,
            target_override=None if moving else (fixed_az, fixed_el),
            disturbances=disturbances,
            camera=camera,
            detector=detector_instance,
            tracker=tracker,
            pan_tilt=pan_tilt,
            controller=controller,
            state_machine=state_machine,
            search=search,
            link_config=link_config,
            sim_time=sim_time,
            dt=dt,
            simulation_running=True,
            blank=blank,
            detect_when_out_of_fov=True,
            use_raster_search=True,
            feedforward=True,
        )
        detected = out["detected"]
        dx, dy = float(out["dx"]), float(out["dy"])
        confidence = float(out["confidence"]) if detected else 0.0
        error_px = out["error_px"]
        pat_state = out["pat_state"]
        tracked = out["tracking"]
        if tracked is not None:
            track_vx, track_vy = float(tracked["vx"]), float(tracked["vy"])
        else:
            track_vx, track_vy = 0.0, 0.0
        link = out["link"]
        target_az, target_el = out["target_az_rad"], out["target_el_rad"]
        pointing_error_rad = out["pointing_error_rad"]
        az_cmd, el_cmd = out["az_rate_cmd"], out["el_rate_cmd"]

        rec = {
            "time": sim_time,
            "phase": phase,
            "detected": detected,
            "confidence": confidence,
            "error_px": error_px,
            "pat_state": pat_state.value,
            "target_az_deg": math.degrees(target_az),
            "target_el_deg": math.degrees(target_el),
            "gimbal_az_deg": pan_tilt.azimuth,
            "gimbal_el_deg": pan_tilt.elevation,
            "cmd_az_deg_s": az_cmd,
            "cmd_el_deg_s": el_cmd,
            "act_az_deg_s": pan_tilt.azimuth_velocity,
            "act_el_deg_s": pan_tilt.elevation_velocity,
            "track_vx_px": track_vx,
            "track_vy_px": track_vy,
            "pointing_error_rad": pointing_error_rad,
            "link_margin_db": link["link_margin_db"],
            "link_state": link["link_state"],
            "blanked": blank,
        }
        records.append(rec)
        return rec

    def begin(phase: str) -> None:
        nonlocal current_phase, phase_start_t, phase_start_tick
        current_phase = phase
        phase_start_t = sim_time
        phase_start_tick = len(records)
        note("phase_entered")

    status = "COMPLETE"

    # Phase 1: initial acquisition (stationary, quiet).
    begin("acquisition")
    met = False
    for _ in range(budgets["acquisition"]):
        rec = tick("acquisition", moving=False, levels=QUIET, blank=False)
        if rec["pat_state"] in _TRACK_STATES:
            note("acquired", {"pat_state": rec["pat_state"]})
            met = True
            break
    close_phase(records[-1]["pat_state"], met)
    if not met:
        status = "TIMEOUT_acquisition"

    # Phase 2: stationary lock.
    if status == "COMPLETE":
        begin("stationary_lock")
        met = False
        for _ in range(budgets["stationary_lock"]):
            rec = tick("stationary_lock", moving=False, levels=QUIET,
                       blank=False)
            if rec["pat_state"] == PATState.LOCKED.value:
                note("locked", {"pat_state": rec["pat_state"]})
                met = True
                break
        close_phase(records[-1]["pat_state"], met)
        if not met:
            status = "TIMEOUT_stationary_lock"

    # Phase 3: moving target.
    if status == "COMPLETE":
        begin("moving")
        for _ in range(budgets["moving"]):
            tick("moving", moving=True, levels=QUIET, blank=False)
        close_phase(records[-1]["pat_state"], True)

    # Phase 4: moving target + disturbances.
    if status == "COMPLETE":
        begin("disturbed")
        for _ in range(budgets["disturbed"]):
            tick("disturbed", moving=True, levels=DISTURBED, blank=False)
        close_phase(records[-1]["pat_state"], True)

    # Phase 5: forced real beacon loss.
    if status == "COMPLETE":
        begin("loss")
        marks["loss_onset"] = sim_time
        note("beacon_blanked")
        met = False
        for _ in range(budgets["loss"]):
            rec = tick("loss", moving=True, levels=DISTURBED, blank=True)
            if rec["pat_state"] in ("REACQUIRE", "SEARCH"):
                note("loss_detected", {"pat_state": rec["pat_state"]})
                met = True
                break
        close_phase(records[-1]["pat_state"], met)
        if not met:
            status = "TIMEOUT_loss"

    # Phase 6: search sweep, then real reacquisition.
    if status == "COMPLETE":
        begin("search_reacquire")
        met_search = False
        for _ in range(budgets["search_blank"]):
            rec = tick("search_reacquire", moving=True, levels=DISTURBED,
                       blank=True)
            if rec["detected"]:
                raise RuntimeError("detection during blanked search sweep")
            if rec["pat_state"] == PATState.SEARCH.value:
                if marks["search_entered"] is None:
                    marks["search_entered"] = rec["time"]
                    note("search_entered", {
                        "gimbal_az_deg": rec["gimbal_az_deg"],
                        "gimbal_el_deg": rec["gimbal_el_deg"]})
                met_search = True
                break
        if not met_search:
            # The loss leg may already have reached SEARCH; accept that.
            met_search = any(r["pat_state"] == PATState.SEARCH.value
                             for r in records if r["phase"] == "loss")
            if met_search and marks["search_entered"] is None:
                first = next(r["time"] for r in records
                             if r["phase"] == "loss"
                             and r["pat_state"] == PATState.SEARCH.value)
                marks["search_entered"] = first
        marks["beacon_restored"] = sim_time
        note("beacon_restored")
        met_reacq = False
        for _ in range(budgets["reacquire"]):
            # Stationary quiet beacon, as in the proven reacquisition tests.
            rec = tick("search_reacquire", moving=False, levels=QUIET,
                       blank=False)
            if rec["detected"] and marks["redetected"] is None:
                marks["redetected"] = rec["time"]
                note("redetected", {"confidence": rec["confidence"],
                                    "error_px": rec["error_px"]})
            if rec["pat_state"] in _TRACK_STATES and marks["redetected"] is not None:
                marks["reacquired"] = rec["time"]
                note("reacquired", {"pat_state": rec["pat_state"]})
                met_reacq = True
                break
        met = met_search and met_reacq
        close_phase(records[-1]["pat_state"], met)
        if not met:
            status = "TIMEOUT_search_reacquire"

    # Phase 7: final lock + retention tail.
    if status == "COMPLETE":
        begin("final_lock")
        met = False
        for _ in range(budgets["final_lock"]):
            rec = tick("final_lock", moving=False, levels=QUIET, blank=False)
            if rec["pat_state"] == PATState.LOCKED.value:
                marks["final_lock"] = rec["time"]
                note("locked", {"pat_state": rec["pat_state"]})
                met = True
                break
        if met:
            for _ in range(budgets["retention_tail"]):
                tick("final_lock", moving=False, levels=QUIET, blank=False)
        close_phase(records[-1]["pat_state"], met)
        if not met:
            status = "TIMEOUT_final_lock"

    detected_recs = [r for r in records if r["detected"]]
    final_rec = records[-1]
    final_lock_recs = [r for r in records
                       if r["phase"] == "final_lock"
                       and r["pat_state"] == PATState.LOCKED.value]

    def _span(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return (b - a) if (a is not None and b is not None) else None

    report: Dict[str, Any] = {
        "mission": "pat_mission",
        "status": status,
        "detector": detector_name,
        "seed": seed,
        "dt": dt,
        "phase_order": [p["name"] for p in phases],
        "phases": phases,
        "phase_durations_s": {p["name"]: p["duration_s"] for p in phases},
        "timeline": timeline,
        "final_pat_state": final_rec["pat_state"],
        "final_link_state": final_rec["link_state"],
        "final_link_margin_db": final_rec["link_margin_db"],
        "final_pointing_error_px": (detected_recs[-1]["error_px"]
                                    if detected_recs else None),
        "final_pointing_error_rad": (detected_recs[-1]["pointing_error_rad"]
                                     if detected_recs else None),
        "recovery_time_s": _span(marks["loss_onset"], marks["final_lock"]),
        "search_to_reacquire_s": _span(marks["search_entered"],
                                       marks["reacquired"]),
        "restore_to_redetect_s": _span(marks["beacon_restored"],
                                       marks["redetected"]),
        "restore_to_reacquire_s": _span(marks["beacon_restored"],
                                        marks["reacquired"]),
        "final_lock_retention_s": len(final_lock_recs) * dt,
        "markers": marks,
        "metrics": compute_experiment_metrics(records, dt=dt),
        "records": records,
    }
    return report
