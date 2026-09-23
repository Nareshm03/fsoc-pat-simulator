"""
Moving-target PAT tracking: measured-rate feedforward regression tests.

Evidence (instrumented identical before/after runs, classical detector,
real render->detect pipeline):
- Before: moving target never LOCKED (min 5.8px, mean ~73px); error grows
  monotonically with target speed (0.5dps->35px ... 10dps->~100-170px),
  matching err_ss ~= rate/kp; commands saturate at the 10 deg/s cap while
  the target peaks at 10 deg/s azimuth.
- After (feedforward of measured target rate + 15 deg/s cap, still 2x
  below the 30/20 deg/s gimbal limits): moving LOCKS at ~6.8s with
  lock_frac 0.53, mean ~16px, final ~2px; moving+disturbed LOCKS at ~5.7s;
  stationary bit-identical (lock 5.07s).

Constraints honored: detectors untouched, search/reacquisition untouched,
thresholds asserted at defaults, stationary preserved, no faked
detections (FF uses only real detections + measured gimbal rates; without
either it contributes exactly 0).
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATState, PATStateMachine
from simulation.search_pattern import RasterSearch
from simulation.target import Target

DT = 1.0 / 30.0
raster = RasterSearch()


def run_scenario(kind, duration_s, seed=7, loss_at=None, loss_dur=0.0):
    """Closed loop mirroring SimulationManager._simulation_tick.

    kind: 'stationary_ideal' | 'moving' | 'moving_disturbed'.
    loss_at/loss_dur: optional forced blank-frame window (real None detections).
    """
    random.seed(seed)
    target = Target(1280, 720)
    camera = VirtualCamera(1280, 720)
    detector = BeaconDetector()
    if kind == "stationary_ideal":
        dist = DisturbanceModel(0.0, 0.0, 0.0, 0.0)
        fixed = (math.radians(5.0), math.radians(-8.0))
    elif kind == "moving":
        dist = DisturbanceModel(0.0, 0.0, 0.0, 0.0)
        fixed = None
    elif kind == "moving_disturbed":
        dist = DisturbanceModel(2.0, 2.0, 2.0, 1.0)
        fixed = None
    else:
        raise ValueError(kind)
    pt = PanTilt()
    ctrl = PATController()
    sm = PATStateMachine()
    assert sm.acquire_threshold == 50.0
    assert sm.lock_threshold == 5.0
    assert sm.unlock_threshold == 10.0
    assert sm.lock_duration == 2.0
    assert sm.reacquire_timeout == 3.0

    recs = []
    t = 0.0
    n = int(duration_s / DT)
    for _ in range(n):
        t += DT
        taz, tel = fixed if fixed else target.angular_position(t)
        blank = loss_at is not None and loss_at <= t < loss_at + loss_dur
        gaz, gel = math.radians(pt.azimuth), math.radians(pt.elevation)
        pos = target.pixel_from_angles(taz, tel, gaz, gel)
        if pos is not None and not blank:
            turb = dist.turbulence_offset(t)
            vib = dist.vibration_offset(t)
            cam = dist.camera_offset(t)
            sen = dist.sensor_offset()
            bx = pos[0] + turb[0] + vib[0] + cam[0] + sen[0]
            by = pos[1] + turb[1] + vib[1] + cam[1] + sen[1]
            det = detector.detect(camera.render(bx, by))
        else:
            if blank:
                # Forced loss is genuine: an empty frame through the real
                # detector yields no detection (never a faked boolean).
                assert detector.detect(camera.render(-1, -1)) is None
            det = None
        if det:
            detected, dx, dy = True, det["x"], det["y"]
            err = math.hypot(dx - 640.0, dy - 360.0)
        else:
            detected, dx, dy, err = False, 0.0, 0.0, 0.0
        state = sm.update(detected, err, t, True)
        mode = sm.get_control_mode()
        if mode == "search":
            ctrl.reset()
            az_cmd, el_cmd = raster.command(pt.azimuth, pt.elevation, t, DT)
        else:
            az_cmd, el_cmd = ctrl.compute_command(
                detected, dx, dy, mode, DT,
                gimbal_az_rate_deg_s=pt.azimuth_velocity,
                gimbal_el_rate_deg_s=pt.elevation_velocity,
            )
        pt.update(az_cmd, el_cmd, DT)
        recs.append({"t": t, "detected": detected, "err": err,
                     "state": state, "cmd": (az_cmd, el_cmd),
                     "az": pt.azimuth, "el": pt.elevation})
    return recs


def metrics(recs):
    det = [r for r in recs if r["detected"]]
    errs = [r["err"] for r in det]
    locked = [r for r in recs if r["state"] == PATState.LOCKED]
    return {
        "final": errs[-1],
        "min": min(errs),
        "mean": sum(errs) / len(errs),
        "locked": bool(locked),
        "lock_time": locked[0]["t"] if locked else None,
        "lock_frac": len(locked) / len(recs),
        "max_cmd": max(max(abs(c[0]), abs(c[1])) for r in recs for c in [r["cmd"]]),
        "states": {r["state"] for r in recs},
    }


class TestMovingTargetImproves:
    def test_stationary_ideal_unchanged(self):
        m = metrics(run_scenario("stationary_ideal", 15.0))
        assert m["locked"] and m["lock_time"] < 6.0
        assert m["min"] < 5.0 and m["final"] < 8.0
        assert m["lock_frac"] * 15.0 > 5.0
        print(f"\nstationary: lock_t={m['lock_time']:.2f}s final={m['final']:.2f}px")

    def test_moving_target_locks(self):
        m = metrics(run_scenario("moving", 25.0))
        # Before fix: never LOCKED, min 5.8px, mean ~73px, final ~98px.
        assert m["locked"], "moving target must now reach LOCKED"
        assert m["lock_time"] < 15.0
        assert m["min"] < 5.0
        assert m["mean"] < 30.0
        assert m["final"] < 15.0
        print(f"\nmoving: lock_t={m['lock_time']:.2f}s frac={m['lock_frac']:.3f} "
              f"min={m['min']:.2f}px mean={m['mean']:.1f}px final={m['final']:.2f}px")

    def test_moving_disturbed_locks(self):
        m = metrics(run_scenario("moving_disturbed", 25.0))
        # Before fix: never LOCKED, mean ~80px, final ~113px.
        assert m["locked"], "moving+disturbance must now reach LOCKED"
        assert m["min"] < 5.0
        assert m["mean"] < 35.0
        assert m["final"] < 30.0
        print(f"\ndisturbed: lock_t={m['lock_time']:.2f}s frac={m['lock_frac']:.3f} "
              f"min={m['min']:.2f}px mean={m['mean']:.1f}px final={m['final']:.2f}px")

    def test_commands_within_limits(self):
        for kind in ("stationary_ideal", "moving", "moving_disturbed"):
            recs = run_scenario(kind, 10.0)
            ctrl_cap = PATController().max_rate
            assert ctrl_cap <= 15.0 + 1e-9
            for r in recs:
                assert abs(r["cmd"][0]) <= ctrl_cap + 1e-9
                assert abs(r["cmd"][1]) <= ctrl_cap + 1e-9
                assert -180.0 <= r["az"] <= 180.0
                assert -30.0 <= r["el"] <= 90.0


class TestFeedforwardUsesOnlyMeasuredData:
    def test_legacy_signature_is_pure_pid(self):
        a = PATController()
        b = PATController()
        c1 = a.compute_command(True, 700.0, 400.0, "track", DT)
        c2 = b.compute_command(True, 700.0, 400.0, "track", DT,
                               gimbal_az_rate_deg_s=None,
                               gimbal_el_rate_deg_s=None)
        assert c1 == c2  # explicit None == omitted: exact legacy behavior

    def test_estimate_converges_to_measured_rate(self):
        c = PATController()
        # Synthetic constant-velocity beacon: +320 px/s in x at 30 Hz with a
        # measured gimbal rate of 0 -> true target rate 320/32 = 10 deg/s az.
        x = 640.0
        for _ in range(60):
            x += 320.0 * DT
            c.compute_command(True, x, 360.0, "track", DT,
                              gimbal_az_rate_deg_s=0.0,
                              gimbal_el_rate_deg_s=0.0)
        assert abs(c._ff_waz - 10.0) < 0.5, c._ff_waz
        assert abs(c._ff_wel) < 0.5, c._ff_wel

    def test_no_estimate_without_previous_detection(self):
        c = PATController()
        az, el = c.compute_command(True, 700.0, 400.0, "track", DT,
                                   gimbal_az_rate_deg_s=5.0,
                                   gimbal_el_rate_deg_s=1.0)
        legacy = PATController().compute_command(True, 700.0, 400.0, "track", DT)
        assert (az, el) == legacy  # first detection: FF contributes 0

    def test_dropout_clears_estimate(self):
        c = PATController()
        for i in range(10):
            c.compute_command(True, 640.0 + i, 360.0, "track", DT,
                              gimbal_az_rate_deg_s=0.0,
                              gimbal_el_rate_deg_s=0.0)
        assert c._ff_prev_x is not None
        c.compute_command(False, 0.0, 0.0, "track", DT)
        assert c._ff_prev_x is None
        assert c._ff_waz == 0.0 and c._ff_wel == 0.0

    def test_ff_applies_only_in_precision_modes(self):
        # Small errors keep commands clear of rate clamps so the FF term is
        # observable. Beacon drifts +5px x / +2px y per tick at 30 Hz with a
        # stationary gimbal: true target rate = +4.69 deg/s az, -2.50 deg/s el.
        def run(mode, with_rates):
            c = PATController()
            kw = {} if not with_rates else {
                "gimbal_az_rate_deg_s": 0.0, "gimbal_el_rate_deg_s": 0.0}
            c.compute_command(True, 640.0, 360.0, mode, DT, **kw)
            kw = {} if not with_rates else {
                "gimbal_az_rate_deg_s": 0.0, "gimbal_el_rate_deg_s": 0.0}
            return c.compute_command(True, 645.0, 362.0, mode, DT, **kw)

        # acquire: measured rates must not change the command (FF gated off)
        assert run("acquire", True) == run("acquire", False)
        # track/locked: command carries the measured target-rate term
        for mode in ("track", "locked"):
            with_ff, without = run(mode, True), run(mode, False)
            assert abs(with_ff[0] - without[0]) > 0.5, (mode, with_ff, without)
            assert abs(with_ff[1] - without[1]) > 0.5, (mode, with_ff, without)


class TestReacquisitionPreservedOnMovingTarget:
    def test_loss_during_tracking_reacquires_and_relocks(self):
        recs = run_scenario("moving", 30.0, loss_at=8.0, loss_dur=2.0)
        states = [r["state"] for r in recs]
        assert PATState.REACQUIRE in states or PATState.SEARCH in states
        post = [r for r in recs if r["t"] > 10.5]
        assert any(r["detected"] for r in post), "real redetection required"
        assert any(r["state"] == PATState.LOCKED for r in post), \
            "must re-lock after loss on moving target"
