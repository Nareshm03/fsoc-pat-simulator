"""
Forced beacon-loss search and reacquisition tests.

Proves the full recovery flow with the real simulation loop and a real
(Classical) detector - detections are never faked:
  LOCKED/FINE_TRACK -> beacon blanked -> REACQUIRE/SEARCH
  -> raster sweep -> beacon restored -> ACQUIRE -> FINE_TRACK -> LOCKED

Loss is forced by rendering the beacon off-screen (a genuinely empty
frame, so the detector truly returns None). Thresholds, gains, and
detector algorithms are untouched.
"""
import math

import pytest

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATState, PATStateMachine
from simulation.search_pattern import RasterSearch
from simulation.target import Target

DT = 1.0 / 30.0
TARGET_AZ_DEG = 5.0
TARGET_EL_DEG = -8.0
TARGET_AZ = math.radians(TARGET_AZ_DEG)
TARGET_EL = math.radians(TARGET_EL_DEG)


class LoopSim:
    """Manual harness mirroring SimulationManager._simulation_tick."""

    def __init__(self):
        self.target = Target(width=1280, height=720)
        self.camera = VirtualCamera(width=1280, height=720)
        self.detector = BeaconDetector()
        self.pan_tilt = PanTilt()
        self.controller = PATController()
        self.state_machine = PATStateMachine()
        self.search = RasterSearch()
        self.sim_time = 0.0
        self.records = []

    def tick(self, beacon_present=True):
        self.sim_time += DT
        gaz = math.radians(self.pan_tilt.azimuth)
        gel = math.radians(self.pan_tilt.elevation)
        pos = self.target.pixel_from_angles(TARGET_AZ, TARGET_EL, gaz, gel)
        if pos is not None and beacon_present:
            frame = self.camera.render(*pos)
        else:
            frame = self.camera.render(-100.0, -100.0)  # genuinely empty frame
        detection = self.detector.detect(frame)
        if detection is not None:
            detected, dx, dy = True, detection["x"], detection["y"]
            err = math.hypot(dx - 640.0, dy - 360.0)
            conf = detection["confidence"]
        else:
            detected, dx, dy, err, conf = False, 0.0, 0.0, 0.0, 0.0
        state = self.state_machine.update(detected, err, self.sim_time, True)
        mode = self.state_machine.get_control_mode()
        if mode == "search":
            self.controller.reset()
            az_cmd, el_cmd = self.search.command(
                self.pan_tilt.azimuth, self.pan_tilt.elevation, self.sim_time, DT
            )
        else:
            az_cmd, el_cmd = self.controller.compute_command(
                detected, dx, dy, mode, DT
            )
        self.pan_tilt.update(az_cmd, el_cmd, DT)
        self.records.append({
            "t": self.sim_time, "detected": detected, "err": err,
            "conf": conf, "state": state,
            "az": self.pan_tilt.azimuth, "el": self.pan_tilt.elevation,
            "cmd": (az_cmd, el_cmd),
        })
        return self.records[-1]

    def run_until(self, predicate, max_ticks, beacon_present=True):
        for _ in range(max_ticks):
            rec = self.tick(beacon_present=beacon_present)
            if predicate(rec):
                return rec
        raise AssertionError("condition not met within tick budget")


def test_raster_stays_bounded_and_deterministic():
    pattern = RasterSearch()
    seen = set()
    for i in range(600):
        t = i * 0.5
        az, el = pattern.target_at(t)
        assert -20.0 <= az <= 20.0, (t, az)
        assert -10.0 <= el <= 10.0, (t, el)
        seen.add((round(az, 1), round(el, 1)))
        assert pattern.target_at(t) == (az, el)  # deterministic
    assert len(seen) > 50  # it actually scans, not a fixed point
    assert pattern.target_at(0.0) != pattern.target_at(
        pattern.row_period / 2.0
    )


def test_raster_commands_respect_rate_limits():
    pattern = RasterSearch()
    az_cmd, el_cmd = pattern.command(-20.0, -10.0, 1000.0, DT)
    assert abs(az_cmd) <= 10.0 + 1e-9
    assert abs(el_cmd) <= 10.0 + 1e-9
    # Far from waypoint -> saturated at the cap; near it -> proportional.
    assert abs(az_cmd) == pytest.approx(10.0) or abs(el_cmd) == pytest.approx(10.0)
    tax, tel = pattern.target_at(5.0)
    az_cmd, el_cmd = pattern.command(tax, tel, 5.0, DT)
    assert az_cmd == pytest.approx(0.0, abs=1e-9)
    assert el_cmd == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        RasterSearch(az_min_deg=10.0, az_max_deg=-10.0)
    with pytest.raises(ValueError):
        RasterSearch(az_sweep_rate_deg_s=0.0)
    with pytest.raises(ValueError):
        pattern.command(0.0, 0.0, 1.0, 0.0)


def _lock_sim():
    sim = LoopSim()
    rec = sim.run_until(lambda r: r["state"] == PATState.LOCKED, 360)
    return sim, rec


def test_forced_loss_enters_reacquire_search():
    """1. Forced beacon loss drives LOCKED -> REACQUIRE/SEARCH."""
    sim, _ = _lock_sim()
    assert all(r["detected"] for r in sim.records[-30:])  # solid lock first
    for _ in range(30):  # 1 s of genuinely empty frames
        rec = sim.tick(beacon_present=False)
        assert not rec["detected"]
    assert rec["state"] in (PATState.REACQUIRE, PATState.SEARCH)


def test_search_moves_gimbal_within_bounds():
    """2. While lost, the gimbal sweeps (never holds) inside raster bounds."""
    sim, _ = _lock_sim()
    az0, el0 = sim.pan_tilt.azimuth, sim.pan_tilt.elevation
    path, moved = 0.0, False
    prev = (az0, el0)
    for _ in range(120):  # 4 s blanked: REACQUIRE timeout -> SEARCH sweep
        rec = sim.tick(beacon_present=False)
        assert not rec["detected"]
        assert -20.5 <= rec["az"] <= 20.5, rec
        assert -10.5 <= rec["el"] <= 10.5, rec
        path += math.hypot(rec["az"] - prev[0], rec["el"] - prev[1])
        prev = (rec["az"], rec["el"])
        if abs(rec["cmd"][0]) > 1e-9 or abs(rec["cmd"][1]) > 1e-9:
            moved = True
    assert moved, "gimbal must issue sweep commands while lost"
    assert path > 2.0, f"sweep path too short: {path:.2f} deg"
    assert PATState.SEARCH in {r["state"] for r in sim.records}


def test_reappearance_reacquire_and_relock():
    """3-5. Real redetection -> ACQUIRE -> FINE_TRACK -> LOCKED with metrics."""
    sim, _ = _lock_sim()
    for _ in range(120):  # force loss deep into SEARCH sweep
        sim.tick(beacon_present=False)
    t_loss_start = sim.sim_time
    assert sim.records[-1]["state"] == PATState.SEARCH

    # 3. Real beacon reappearance: first detection comes from the detector.
    rec = sim.run_until(lambda r: r["detected"], 600, beacon_present=True)
    t_redetect = rec["t"]
    assert rec["conf"] > 0.0
    gaz = math.radians(sim.pan_tilt.azimuth)
    gel = math.radians(sim.pan_tilt.elevation)
    exp_x, exp_y = sim.target.pixel_from_angles(TARGET_AZ, TARGET_EL, gaz, gel)
    assert exp_x is not None
    # Detection matches the true projection (nothing faked).
    assert abs(rec["err"] - math.hypot(exp_x - 640.0, exp_y - 360.0)) < 30.0

    # 4. Reacquisition through ACQUIRE. The redetect tick itself lands in
    # ACQUIRE (SEARCH/REACQUIRE + detected -> ACQUIRE); error < 50px moves
    # on to FINE_TRACK the very next tick, so assert it directly.
    assert rec["state"] == PATState.ACQUIRE, rec["state"]
    t_acquire = rec["t"]

    # 5. Return to LOCKED.
    rec = sim.run_until(
        lambda r: r["state"] == PATState.LOCKED, 600, beacon_present=True
    )
    t_relock = rec["t"]
    follow_start = len(sim.records)
    for _ in range(150):  # 5 s follow window for retention
        sim.tick(beacon_present=True)
    locked_time = sum(
        DT for r in sim.records[follow_start:] if r["state"] == PATState.LOCKED
    )
    final_err = sim.records[-1]["err"]

    print(
        f"\nSearch duration (loss->SEARCH already covered): "
        f"reacquisition t_loss->relock = {t_relock - t_loss_start:.2f}s "
        f"(redetect +{t_redetect - t_loss_start:.2f}s, "
        f"acquire +{t_acquire - t_loss_start:.2f}s)"
    )
    print(f"Final error: {final_err:.2f}px, "
          f"lock retention after relock: {locked_time:.2f}s")

    assert t_relock - t_loss_start < 25.0
    assert final_err < 8.0
    assert locked_time > 3.0


def test_no_loss_case_unchanged():
    """6. Without loss: no SEARCH/REACQUIRE, no sweep excursion."""
    sim = LoopSim()
    for _ in range(240):  # 8 s, beacon always present
        sim.tick(beacon_present=True)
    # Tick 1 legitimately passes through SEARCH on startup; after that,
    # a lossless run must never re-enter SEARCH/REACQUIRE.
    states = {r["state"] for r in sim.records[10:]}
    assert PATState.SEARCH not in states
    assert PATState.REACQUIRE not in states
    losses = sum(
        1 for a, b in zip(sim.records, sim.records[1:])
        if a["detected"] and not b["detected"]
    )
    assert losses == 0
    locked = [r for r in sim.records if r["state"] == PATState.LOCKED]
    assert locked, "must reach LOCKED without loss"
    azs = [r["az"] for r in sim.records[len(sim.records) // 2:]]
    els = [r["el"] for r in sim.records[len(sim.records) // 2:]]
    assert max(azs) - min(azs) < 3.0  # quiescent tracking, no sweep
    assert max(els) - min(els) < 3.0


@pytest.mark.asyncio
async def test_manager_tick_sweeps_on_real_loss():
    """main.py wiring: genuinely empty frames drive gimbal sweep via tick."""
    import numpy as np

    from main import SimulationManager

    manager = SimulationManager()
    for name in ("turbulence", "vibration", "camera_motion", "sensor_noise"):
        manager.set_disturbance(name, 0.0)
    # Ticks alone don't mark the sim running; the flag mirrors a live loop
    # (state machine needs simulation_running=True to leave IDLE).
    manager.running = True
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    manager.camera.render = lambda *args, **kwargs: black
    az0, el0 = manager.pan_tilt.azimuth, manager.pan_tilt.elevation
    for _ in range(100):
        await manager._simulation_tick()
    moved = math.hypot(
        manager.pan_tilt.azimuth - az0, manager.pan_tilt.elevation - el0
    )
    assert moved > 1.0, f"gimbal must sweep on loss, moved {moved:.2f} deg"
    assert manager.pat_state_machine.state in (PATState.SEARCH, PATState.REACQUIRE)
