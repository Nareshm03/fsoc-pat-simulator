"""Mission Mode verification: scripted end-to-end scenario.

Proves, using the real pipeline on every tick (render -> real detector ->
tracker -> PAT state machine -> raster/PID -> gimbal -> link budget), that:
  1. all seven phases occur in order,
  2. loss -> SEARCH -> reacquisition -> LOCK happens with genuine
     (unfaked) detections,
  3. the machine-readable report is complete and reproducible.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analytics.mission_runner import PHASE_ORDER, run_mission

_cached = {}


def _mission():
    if "report" not in _cached:
        _cached["report"] = run_mission(detector="classical", seed=7)
    return _cached["report"]


def test_phases_occur_in_order():
    """Each of the seven phases occurs exactly once, in script order."""
    report = _mission()
    assert report["status"] == "COMPLETE", report["status"]
    assert report["phase_order"] == PHASE_ORDER
    assert [p["name"] for p in report["phases"]] == PHASE_ORDER
    # Timeline carries a phase_entered marker per phase, time-ordered.
    entered = [e for e in report["timeline"] if e["event"] == "phase_entered"]
    assert [e["phase"] for e in entered] == PHASE_ORDER
    times = [e["time"] for e in entered]
    assert times == sorted(times)
    # Every record belongs to exactly one phase; durations are consistent.
    for phase in report["phases"]:
        assert phase["end_condition_met"], phase["name"]
        assert phase["duration_s"] >= 0
    total_ticks = sum(p["ticks"] for p in report["phases"])
    assert total_ticks == len(report["records"])
    assert report["phase_durations_s"] == {
        p["name"]: p["duration_s"] for p in report["phases"]}


def test_loss_search_reacquire_lock_real_pipeline():
    """Loss -> SEARCH -> ACQUIRE -> LOCKED with genuine detections only."""
    report = _mission()
    recs = report["records"]

    loss_start = next(i for i, r in enumerate(recs) if r["phase"] == "loss")
    # 1. Forced loss: every blanked tick is undetected (real detector).
    blanked = [r for r in recs if r["blanked"]]
    assert len(blanked) > 0
    assert all(not r["detected"] for r in blanked)

    # 2. State order after loss onset: REACQUIRE ... SEARCH ... ACQUIRE ... LOCKED.
    states = [r["pat_state"] for r in recs[loss_start:]]
    i_reacq = states.index("REACQUIRE")
    i_search = states.index("SEARCH")
    i_acq = next(i for i, s in enumerate(states)
                 if s in ("ACQUIRE", "FINE_TRACK", "LOCKED"))
    i_lock = states.index("LOCKED")
    assert i_reacq < i_search < i_acq < i_lock

    # 3. The sweep actually moved the gimbal while blind.
    sweep = [r for r in recs[loss_start:]
             if r["phase"] in ("loss", "search_reacquire") and r["blanked"]]
    azs = [r["gimbal_az_deg"] for r in sweep]
    assert max(azs) - min(azs) > 1.0, "raster must sweep during loss"

    # 4. Redetection comes from the real imaging geometry (nothing faked):
    # the first post-restore detection matches the true projection.
    redetect = next(r for r in recs[loss_start:]
                    if r["phase"] == "search_reacquire" and r["detected"])
    assert redetect["confidence"] > 0.0
    taz = math.radians(redetect["target_az_deg"])
    tel = math.radians(redetect["target_el_deg"])
    gaz = math.radians(redetect["gimbal_az_deg"])
    gel = math.radians(redetect["gimbal_el_deg"])
    fov_h, fov_v = math.radians(40.0), math.radians(30.0)
    rel_az, rel_el = taz - gaz, tel - gel
    assert abs(rel_az) <= fov_h / 2 and abs(rel_el) <= fov_v / 2
    exp_x = 640.0 + rel_az * (1280.0 / fov_h)
    exp_y = 360.0 - rel_el * (720.0 / fov_v)
    assert abs(redetect["error_px"] - math.hypot(exp_x - 640.0,
                                                 exp_y - 360.0)) < 30.0


def test_report_is_complete_and_machine_readable():
    """Status, timeline, durations, final states, recovery, error, records."""
    report = _mission()
    for key in ("status", "phase_order", "phases", "phase_durations_s",
                "timeline", "final_pat_state", "final_link_state",
                "final_link_margin_db", "recovery_time_s",
                "search_to_reacquire_s", "restore_to_redetect_s",
                "restore_to_reacquire_s", "final_pointing_error_px",
                "final_lock_retention_s", "markers", "metrics", "records"):
        assert key in report, key
    assert report["final_pat_state"] == "LOCKED"
    assert report["final_link_state"] in ("NOMINAL", "MARGINAL", "OUTAGE")
    m = report["markers"]
    assert m["loss_onset"] < m["search_entered"] < m["redetected"] \
        <= m["reacquired"] < m["final_lock"]
    assert report["recovery_time_s"] == \
        m["final_lock"] - m["loss_onset"]
    assert report["final_pointing_error_px"] is not None
    assert report["final_lock_retention_s"] > 3.0
    assert report["metrics"]["locked_achieved"] is True
    # Strictly machine-readable: JSON round-trip preserves the report.
    assert json.loads(json.dumps(report))["status"] == "COMPLETE"


def test_mission_is_deterministic():
    """Same seed reproduces the identical mission (records + timeline)."""
    first = _mission()
    second = run_mission(detector="classical", seed=7)
    assert second["status"] == first["status"] == "COMPLETE"
    assert second["records"] == first["records"]
    assert second["timeline"] == first["timeline"]
    assert second["phase_durations_s"] == first["phase_durations_s"]
