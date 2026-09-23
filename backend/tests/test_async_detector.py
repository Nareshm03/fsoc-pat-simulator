"""Bounded async YOLO detection pipeline: queue, lifecycle, staleness.

The live tick runs at control cadence while YOLO inference (~80 ms)
runs newest-frame-only in a worker. These tests prove, without
wall-clock assertions (robust on loaded machines):
- queue: bounded single pending slot, newest wins, no per-tick pileup
- lifecycle: pump launches/harvests, shutdown cancels cleanly
- staleness: expiry, generation drops, fresh negatives -> undetected;
  coasted ticks carry explicit age and motion-compensated (not copied)
  positions matching true projection
- switching: rapid yolo/classical leaves no stale YOLO state behind
- tracking: moving + disturbed YOLO lock per the Classical bars
- loss: blank frames -> REACQUIRE/SEARCH -> restore -> LOCKED (YOLO)
- telemetry: observation_age_ticks/observation_fresh fields, both modes

Conventions follow test_yolo_detector.py (requires_yolo marker) and
pytest.ini (asyncio marked explicitly per test).
"""
import asyncio
import math

import pytest

from main import SimulationManager
from simulation.async_detector import (
    AsyncDetector,
    MAX_OBSERVATION_AGE_TICKS,
)
from simulation.target import Target
from simulation.yolo_detector import resolve_model_path


def _yolo_available():
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return False
    return resolve_model_path() is not None


requires_yolo = pytest.mark.skipif(
    not _yolo_available(),
    reason="YOLO unavailable (ultralytics not installed or best.pt missing)",
)


class FakeDetector:
    """Scripted sync detector: returns queued results, counts calls."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0
        self.frames_seen = []

    def detect(self, frame):
        self.calls += 1
        self.frames_seen.append(frame)
        if self._results:
            return self._results.pop(0)
        return None


def _det(x, y, conf=0.9):
    return {"x": float(x), "y": float(y), "confidence": conf,
            "bbox": [x - 5, y - 5, x + 5, y + 5]}


def _target():
    return Target(width=1280, height=720)


def _obs(det, fn, t, gaz=0.0, gel=0.0, gen=0):
    return {"detection": det, "sim_time": t, "frame_number": fn,
            "gimbal_az": gaz, "gimbal_el": gel, "generation": gen}


class TestQueueBounds:
    def test_newest_wins_single_pending_slot(self):
        """Rapid submits keep only the newest frame (bounded by design)."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.0, 1, 0.0, 0.0)
        ad.detect(object())
        ad.begin_tick(1.0 / 30.0, 2, 0.0, 0.0)
        ad.detect(object())
        assert ad.frames_submitted == 2
        assert ad._pending["frame_number"] == 2
        assert ad._task is None  # nothing launched without pump()

    def test_invalid_max_age_rejected(self):
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(), max_age_s=-1)

    def test_expiry_reports_undetected(self):
        """Positive obs older than the window is never applied."""
        ad = AsyncDetector(FakeDetector([]), _target(), max_age_s=0.1)
        ad.begin_tick(10 / 30.0, 10, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 10 / 30.0)
        assert ad.detect(object()) is not None
        ad.begin_tick(14 / 30.0, 10 + 3 + 1, 0.0, 0.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None
        assert ad.last_fresh is False

    def test_fresh_negative_reports_undetected(self):
        """A real miss is never masked by coasting, even within window."""
        ad = AsyncDetector(FakeDetector([]), _target(), max_age_s=8 / 30.0)
        ad.begin_tick(10 / 30.0, 10, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 10 / 30.0)
        assert ad.detect(object()) is not None
        ad.begin_tick(11 / 30.0, 11, 0.0, 0.0)
        ad._latest = _obs(None, 11, 11 / 30.0)
        assert ad.detect(object()) is None

    def test_stale_generation_dropped(self):
        """Completions tagged pre-switch/reset are discarded, not applied."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.0, 10, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 10 / 30.0, gen=0)
        ad.reset_generation()
        ad.begin_tick(0.0, 11, 0.0, 0.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None

    def test_out_of_frame_prediction_reports_undetected(self):
        """Propagated points leaving the frame take the loss direction."""
        ad = AsyncDetector(FakeDetector([]), _target(), max_age_s=8 / 30.0)
        ad.begin_tick(10 / 30.0, 10, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 10 / 30.0)
        assert ad.detect(object()) is not None
        # Gimbal slews ~30 deg away: prediction must leave the frame.
        ad.begin_tick(12 / 30.0, 12, math.radians(30.0), 0.0)
        assert ad.detect(object()) is None

    def test_coasted_matches_projection(self):
        """Coasted position tracks true projection within a few px.

        Fixes arrive every other tick so the smoothed rate is engaged;
        the coast then follows true target motion (a single fix can
        only hold world position).
        """
        tgt = _target()
        ad = AsyncDetector(FakeDetector([]), tgt, max_age_s=8 / 30.0)
        t0, gaz0, gel0 = 5.0, math.radians(3.0), math.radians(-5.0)
        for i, fn in enumerate((100, 102, 104)):
            t = t0 + (fn - 100) / 30.0
            az, el = tgt.angular_position(t)
            gaz = gaz0 + math.radians(0.1) * (fn - 100)
            gel = gel0 - math.radians(0.05) * (fn - 100)
            x, y = tgt.pixel_from_angles(az, el, gaz, gel)
            ad.begin_tick(t, fn, gaz, gel)
            ad._latest = _obs(_det(x, y), fn, t, gaz, gel)
            got = ad.detect(object())
            assert got is not None
            assert set(got.keys()) == {"x", "y", "confidence", "bbox"}
        t2 = t0 + 6.0 / 30.0
        az2, el2 = tgt.angular_position(t2)
        gaz2, gel2 = gaz0 + math.radians(0.3), gel0 - math.radians(0.15)
        exp = tgt.pixel_from_angles(az2, el2, gaz2, gel2)
        ad.begin_tick(t2, 106, gaz2, gel2)
        got = ad.detect(object())
        assert got is not None
        assert ad.last_age_ticks == 2
        assert ad.last_fresh is False
        # Within lock-threshold scale (a static hold would err ~22 px
        # here at peak target rate); end-to-end lock is proven live.
        assert math.hypot(got["x"] - exp[0], got["y"] - exp[1]) < 5.0


class TestWorkerLifecycle:
    @pytest.mark.asyncio
    async def test_pump_launches_harvests_and_decouples(self):
        """One inference in flight max; ticks outrun inferences."""
        fake = FakeDetector([_det(640, 360)] * 10)
        ad = AsyncDetector(fake, _target())
        for fn in range(1, 6):
            ad.begin_tick(fn / 30.0, fn, 0.0, 0.0)
            ad.pump()
            ad.detect(object())
        assert ad.inferences_started >= 1
        assert ad.inferences_started <= 5  # never one-per-tick pileup
        assert ad.frames_submitted == 5
        # Drain: pump harvests the completion without blocking.
        for _ in range(100):
            ad.pump()
            if ad._latest is not None:
                break
            await asyncio.sleep(0.01)
        assert ad._latest is not None
        assert ad._latest["detection"]["x"] == 640.0
        await ad.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_cancels_worker(self):
        """Shutdown leaves no live task; second shutdown is safe."""
        slow = FakeDetector([_det(1, 1)] * 100)

        orig = slow.detect

        def hanging(frame):
            import time as _t
            _t.sleep(5.0)
            return orig(frame)

        slow.detect = hanging
        ad = AsyncDetector(slow, _target())
        ad.begin_tick(0.0, 1, 0.0, 0.0)
        ad.detect(object())
        ad.pump()
        assert ad._task is not None
        await ad.shutdown()
        assert ad._task is None
        assert ad._latest is None
        await ad.shutdown()  # idempotent


def _quiet_manager():
    m = SimulationManager()
    for name in ("turbulence", "vibration", "camera_motion", "sensor_noise"):
        m.set_disturbance(name, 0.0)
    return m


async def _paced_ticks(m, n):
    for _ in range(n):
        await m._simulation_tick()
        await asyncio.sleep(m.dt)


def _lock_metrics(m, recs):
    errs = [r["err"] for r in recs if r["det"]]
    locked = [r for r in recs if r["state"] == "LOCKED"]
    return {
        "locked": bool(locked),
        "lock_frac": len(locked) / len(recs),
        "min": min(errs), "mean": sum(errs) / len(errs),
        "final": errs[-1],
        "det_rate": len(errs) / len(recs),
    }


async def _run_closed_loop(m, n, seed=7):
    import random

    random.seed(seed)
    recs = []
    for _ in range(n):
        await m._simulation_tick()
        recs.append({"det": m.detected, "err": m.tracking_error_px,
                     "state": m.pat_state_machine.state.value,
                     "age": m.obs_age_ticks})
        await asyncio.sleep(m.dt)
    return recs


class TestLiveAsyncPipeline:
    @pytest.mark.asyncio
    @requires_yolo
    async def test_moving_target_locks(self):
        """YOLO async: moving target locks (Classical bars apply)."""
        m = _quiet_manager()
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        recs = await _run_closed_loop(m, 600)
        await m.stop()
        met = _lock_metrics(m, recs)
        assert met["locked"], "async YOLO must lock on moving target"
        assert met["min"] < 5.0
        assert met["mean"] < 35.0
        assert met["final"] < 30.0
        ages = [r["age"] for r in recs if r["det"] and r["age"] is not None]
        assert max(ages) <= MAX_OBSERVATION_AGE_TICKS

    @pytest.mark.asyncio
    @requires_yolo
    async def test_disturbed_moving_target_locks(self):
        """YOLO async: disturbed moving target locks (Classical bars)."""
        m = SimulationManager()  # default disturbances = disturbed regime
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        recs = await _run_closed_loop(m, 600)
        await m.stop()
        met = _lock_metrics(m, recs)
        assert met["locked"], "async YOLO must lock on disturbed target"
        assert met["min"] < 5.0
        assert met["mean"] < 35.0
        assert met["final"] < 30.0

    @pytest.mark.asyncio
    @requires_yolo
    async def test_loss_reacquire_relock(self):
        """Blank frames -> REACQUIRE/SEARCH -> restore -> LOCKED (YOLO)."""
        import random

        import numpy as np

        random.seed(7)
        m = _quiet_manager()
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        for _ in range(150):
            await m._simulation_tick()
            await asyncio.sleep(m.dt)
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        real_render = m.camera.render
        m.camera.render = lambda *a, **k: black
        seen = set()
        for _ in range(200):
            await m._simulation_tick()
            seen.add(m.pat_state_machine.state.value)
            if "SEARCH" in seen:
                break
            await asyncio.sleep(m.dt)
        assert "REACQUIRE" in seen
        assert "SEARCH" in seen
        m.camera.render = real_render
        seen2, relocked = set(), False
        for _ in range(600):
            await m._simulation_tick()
            seen2.add(m.pat_state_machine.state.value)
            if m.pat_state_machine.state.value == "LOCKED":
                relocked = True
                break
            await asyncio.sleep(m.dt)
        assert "ACQUIRE" in seen2 or "FINE_TRACK" in seen2
        assert relocked, "must re-lock after real beacon return"
        await m.stop()

    @pytest.mark.asyncio
    @requires_yolo
    async def test_telemetry_age_fields(self):
        """Telemetry carries explicit observation-age provenance."""
        m = _quiet_manager()
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        saw_detected = False
        for _ in range(120):
            await m._simulation_tick()
            tele = m._build_telemetry(
                m.pat_state_machine.state.value, m._calculate_fsoc_status())
            det = tele["detection"]
            assert "observation_age_ticks" in det
            assert "observation_fresh" in det
            assert isinstance(det["observation_fresh"], bool)
            if m.detected:
                saw_detected = True
                assert det["observation_age_ticks"] is not None
                assert det["observation_age_ticks"] <= \
                    MAX_OBSERVATION_AGE_TICKS
            await asyncio.sleep(m.dt)
        assert saw_detected, "expected detections within 120 paced ticks"
        await m.stop()

    @pytest.mark.asyncio
    @requires_yolo
    async def test_no_stale_yolo_after_classical_switch(self):
        """Rapid switches: classical ticks never apply stale YOLO data."""
        m = _quiet_manager()
        for _ in range(3):
            m.set_detector("yolo")
            m.set_detector("classical")
        assert m.detector_type == "classical"
        assert m.detector is m.classical_detector
        m.running = True
        for _ in range(60):
            await m._simulation_tick()
            assert m.obs_age_ticks == 0
            assert m.obs_fresh is True
            assert m.detection_bbox is None  # classical carries no bbox
            await asyncio.sleep(m.dt)
        await m.stop()

    @pytest.mark.asyncio
    @requires_yolo
    async def test_inference_decoupled_from_ticks(self):
        """Structural proof of async operation: inferences lag ticks.

        Over rapid unpaced ticks, completed inferences must trail
        submitted frames (a synchronous design would complete 1:1).
        No wall-clock assertion -- robust on loaded machines.
        """
        m = _quiet_manager()
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        for _ in range(10):
            await m._simulation_tick()
        w = m._async_yolo
        assert w.frames_submitted == 10
        assert w.inferences_completed < w.frames_submitted
        await m.stop()

    @pytest.mark.asyncio
    @requires_yolo
    async def test_stop_shuts_down_worker(self):
        """stop() leaves no live async-detector task behind."""
        m = _quiet_manager()
        m.set_detector("yolo")
        if m._yolo_warmup_task is not None:
            await m._yolo_warmup_task
        m.running = True
        for _ in range(10):
            await m._simulation_tick()
            await asyncio.sleep(m.dt)
        await m.stop()
        w = m._async_yolo
        assert w is None or w._task is None or w._task.done()


def _perfect_latency_run(kind="moving", latency_ticks=3, duration_s=25.0,
                         seed=7, extra_dist=None):
    """Deterministic closed loop, perfect sensor at fixed tick latency.

    True detections complete with an exact L-tick delay via direct
    _latest injection (no worker, no wall-clock nondeterminism), so this
    isolates latency handling from sensor noise entirely. Loop order
    mirrors run_scenario in test_pat_tracking_moving.py.
    """
    import random

    from simulation.disturbances import DisturbanceModel
    from simulation.pan_tilt import PanTilt
    from simulation.pat_controller import PATController
    from simulation.pat_state_machine import PATState, PATStateMachine
    from simulation.search_pattern import RasterSearch

    random.seed(seed)
    target = Target(width=1280, height=720)
    if extra_dist is not None:
        dist = DisturbanceModel(**extra_dist)
    elif kind == "moving":
        dist = DisturbanceModel(0.0, 0.0, 0.0, 0.0)
    else:
        raise ValueError(kind)
    pt, ctrl, sm = PanTilt(), PATController(), PATStateMachine()
    raster = RasterSearch()
    ad = AsyncDetector(FakeDetector([]), target)
    dt = 1.0 / 30.0
    history = {}
    recs, t = [], 0.0
    n = int(duration_s / dt)
    for i in range(n):
        t += dt
        fn = i + 1
        gaz = math.radians(pt.azimuth)
        gel = math.radians(pt.elevation)
        taz, tel = target.angular_position(t)
        pos = target.pixel_from_angles(taz, tel, gaz, gel)
        if pos is not None:
            turb = dist.turbulence_offset(t)
            vib = dist.vibration_offset(t)
            cam = dist.camera_offset(t)
            sen = dist.sensor_offset()
            bx = pos[0] + turb[0] + vib[0] + cam[0] + sen[0]
            by = pos[1] + turb[1] + vib[1] + cam[1] + sen[1]
            if 0 <= bx < 1280 and 0 <= by < 720:
                history[fn] = ({"x": bx, "y": by, "confidence": 0.99,
                                "bbox": [bx - 5, by - 5, bx + 5, by + 5]},
                               gaz, gel)
            else:
                history[fn] = None
        else:
            history[fn] = None
        old = history.get(fn - latency_ticks)
        if old is not None:
            det_old, gaz_old, gel_old = old
            ad._latest = {"detection": det_old,
                          "sim_time": t - latency_ticks * dt,
                          "frame_number": fn - latency_ticks,
                          "gimbal_az": gaz_old, "gimbal_el": gel_old,
                          "generation": 0}
        ad.begin_tick(t, fn, gaz, gel)
        det = ad.detect(None)
        if det:
            detected, dx, dy = True, det["x"], det["y"]
            err = math.hypot(dx - 640.0, dy - 360.0)
        else:
            detected, dx, dy, err = False, 0.0, 0.0, 0.0
        state = sm.update(detected, err, t, True)
        mode = sm.get_control_mode()
        if mode == "search":
            ctrl.reset()
            az_cmd, el_cmd = raster.command(pt.azimuth, pt.elevation, t, dt)
        else:
            az_cmd, el_cmd = ctrl.compute_command(
                detected, dx, dy, mode, dt,
                gimbal_az_rate_deg_s=pt.azimuth_velocity,
                gimbal_el_rate_deg_s=pt.elevation_velocity)
        pt.update(az_cmd, el_cmd, dt)
        recs.append({"t": t, "detected": detected, "err": err,
                     "state": state})
    return recs


def _perfect_metrics(recs):
    from simulation.pat_state_machine import PATState

    det = [r for r in recs if r["detected"]]
    errs = [r["err"] for r in det]
    locked = [r for r in recs if r["state"] == PATState.LOCKED]
    return {
        "locked": bool(locked),
        "lock_frac": len(locked) / len(recs),
        "min": min(errs), "mean": sum(errs) / len(errs),
        "final": errs[-1],
        "det_rate": len(errs) / len(recs),
    }


class TestLatencyCompensation:
    """Latency-compensated fusion: corrections target where the fix
    should be *now* (fix + rate x age), not where it was captured.

    Without compensation the track trails ~one latency window behind
    the target; with a perfect sensor and fixed 3-tick latency the
    pre-fix code never locked (frac 0.000, mean ~72px). These tests
    are fully deterministic (no worker threads, no YOLO)."""

    def test_constant_velocity_no_lag(self):
        """Steady 10 px/tick motion: reported track must not trail.

        Each injected fix carries the position truthfully valid at its
        own timestamp (2-tick lag); the track must keep pace on coast
        (lag << one latency window), not freeze or jump.
        """
        tgt = _target()
        ad = AsyncDetector(FakeDetector([]), tgt, max_age_s=8 / 30.0)
        for fn in range(10, 30):
            t = fn / 30.0
            x = 640.0 + 10.0 * (fn - 2 - 10)  # truth at capture tick
            ad.begin_tick(t, fn, 0.0, 0.0)
            ad._latest = _obs(_det(x, 360.0), fn - 2, t - 2.0 / 30.0)
            got = ad.detect(object())
            assert got is not None
        # After the stream ends, coast 3 ticks: track must keep pace
        # (lag << one latency window), not freeze or jump.
        errs = []
        for fn in range(30, 33):
            t = fn / 30.0
            ad.begin_tick(t, fn, 0.0, 0.0)
            got = ad.detect(object())
            assert got is not None
            errs.append(abs(got["x"] - (640.0 + 10.0 * (fn - 10))))
        assert max(errs) < 8.0, errs

    def test_perfect_fixed_latency_three_locks(self):
        recs = _perfect_latency_run("moving", latency_ticks=3)
        m = _perfect_metrics(recs)
        assert m["locked"], "compensated L=3 latency must lock"
        assert m["lock_frac"] >= 0.30, m
        assert m["min"] < 5.0
        assert m["mean"] < 25.0
        assert m["final"] < 15.0
        assert m["det_rate"] > 0.95

    def test_perfect_fixed_latency_six_locks(self):
        recs = _perfect_latency_run("moving", latency_ticks=6)
        m = _perfect_metrics(recs)
        assert m["locked"], "compensated L=6 latency must lock"
        assert m["lock_frac"] >= 0.25, m
        assert m["min"] < 5.0
        assert m["mean"] < 25.0

    def test_perfect_vibration_only_locks(self):
        """6-7 Hz vibration at fixed latency: aliasing stays bounded."""
        recs = _perfect_latency_run(
            "moving", latency_ticks=3,
            extra_dist={"turbulence": 0.0, "vibration": 2.0,
                        "camera_motion": 0.0, "sensor_noise": 0.0})
        m = _perfect_metrics(recs)
        assert m["locked"], "vibration must not break compensated lock"
        assert m["min"] < 5.0
        assert m["mean"] < 30.0


def _done_future(obs):
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    fut.set_result(obs)
    return fut


class TestHarvestRegularization:
    """Jitter root-cause regression: harvest on a ~0.1 s simulation-time
    grid instead of a fixed tick count, so correction jumps stay periodic
    when dt changes and the 2 s lock timer can sustain.

    Live YOLO uses the default 0.1 s (≈10 Hz: every 3rd tick at 30 Hz,
    every 6th at 60 Hz, every 2nd at 20 Hz). Fresh negatives still apply
    immediately (genuine loss never masked); expiry and generation safety
    are unchanged (honest age, no faked freshness). All tests drive the
    real pump() with pre-resolved completions: fully deterministic, no
    wall-clock nondeterminism.
    """

    def test_invalid_harvest_interval_rejected(self):
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(), harvest_interval_s=0)
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(), harvest_interval_s=-0.5)
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(),
                          harvest_interval_s=float("inf"))
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(),
                          harvest_interval_s=float("nan"))

    def test_default_harvest_interval_is_tenth_second(self):
        ad = AsyncDetector(FakeDetector([]), _target())
        assert ad.harvest_interval_s == pytest.approx(0.1)
        assert ad._held is None
        assert ad._next_harvest_time == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_positives_held_to_time_grid(self):
        """Positive completions wait in _held until the 0.1 s grid."""
        ad = AsyncDetector(FakeDetector([]), _target())
        # First grid point harvests immediately...
        ad.begin_tick(0.1, 3, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(640, 360), 2, 2 / 30.0))
        ad.pump()
        assert ad._latest is not None
        assert ad._latest["frame_number"] == 2
        assert ad._held is None
        assert ad._next_harvest_time == pytest.approx(0.2)
        # ...mid-grid completions are held...
        ad.begin_tick(0.15, 4, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(650, 360), 4, 0.15))
        ad.pump()
        assert ad._latest["frame_number"] == 2
        assert ad._held is not None
        assert ad._held["frame_number"] == 4
        assert ad.inferences_completed == 2
        # ...and surface on the next grid point (newest wins).
        ad.begin_tick(0.2, 6, 0.0, 0.0)
        ad.pump()
        assert ad._held is None
        assert ad._latest is not None
        assert ad._latest["frame_number"] == 4
        assert ad._next_harvest_time == pytest.approx(0.3)

    @pytest.mark.asyncio
    async def test_fresh_negative_applies_immediately_off_grid(self):
        """Real misses are never held, even mid-grid (no loss masking)."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.05, 1, 0.0, 0.0)
        ad._task = _done_future(_obs(None, 1, 0.05))
        ad.pump()
        assert ad._latest is not None
        assert ad._latest["detection"] is None
        assert ad._held is None
        ad.begin_tick(0.05, 1, 0.0, 0.0)
        assert ad.detect(object()) is None

    @pytest.mark.asyncio
    async def test_newer_negative_supersedes_held_positive(self):
        """Newest wins: a real miss discards older held data."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.1, 3, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(640, 360), 3, 0.1))
        ad.pump()  # grid point: applied, next grid 0.2
        assert ad._latest["frame_number"] == 3
        ad.begin_tick(0.15, 4, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(650, 360), 4, 0.15))
        ad.pump()  # mid-grid -> held
        assert ad._held is not None
        ad.begin_tick(0.16, 5, 0.0, 0.0)
        ad._task = _done_future(_obs(None, 5, 0.16))
        ad.pump()  # negative immediate, clears older held
        assert ad._latest is not None
        assert ad._latest["detection"] is None
        assert ad._held is None

    @pytest.mark.asyncio
    async def test_reset_clears_held_and_reanchors_grid(self):
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.1, 3, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(640, 360), 3, 0.1))
        ad.pump()  # grid point: applied, next grid 0.2
        assert ad._latest["frame_number"] == 3
        ad.begin_tick(0.15, 4, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(650, 360), 4, 0.15))
        ad.pump()  # mid-grid -> held
        assert ad._held is not None
        ad.reset_generation()
        assert ad._held is None
        assert ad._latest is None
        assert ad._next_harvest_time == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_cadence_follows_sim_time_not_tick_count_dt_60(self):
        """At dt=1/60 harvests land on 0.1 s multiples, not frame%3.

        Completions every tick (frames 1..18). Tick-based interval 3
        would harvest at frames 3,9,15; time-based harvests at frames
        6,12,18 (t=0.1,0.2,0.3).
        """
        ad = AsyncDetector(FakeDetector([]), _target())
        seen = []
        for fn in range(1, 19):
            ad.begin_tick(fn / 60.0, fn, 0.0, 0.0)
            ad._task = _done_future(_obs(_det(640, 360), fn, fn / 60.0))
            ad.pump()
            seen.append(ad._latest["frame_number"] if ad._latest else None)
        assert seen == [None] * 5 + [6] * 6 + [12] * 6 + [18]
        assert ad.inferences_completed == 18

    @pytest.mark.asyncio
    async def test_cadence_follows_sim_time_not_tick_count_dt_20(self):
        """At dt=1/20 harvests land on 0.1 s multiples (every 2nd tick)."""
        ad = AsyncDetector(FakeDetector([]), _target())
        seen = []
        for fn in range(1, 10):
            ad.begin_tick(fn / 20.0, fn, 0.0, 0.0)
            ad._task = _done_future(_obs(_det(640, 360), fn, fn / 20.0))
            ad.pump()
            seen.append(ad._latest["frame_number"] if ad._latest else None)
        assert seen == [None, 2, 2, 4, 4, 6, 6, 8, 8]

    @pytest.mark.asyncio
    async def test_no_cumulative_drift_with_irregular_dt(self):
        """Irregular ticks: harvests stay on the absolute 0.1 s grid.

        ~2.1 s of irregular dt in [0.017..0.05] s. Asserts harvest count
        (one per 0.1 s boundary), absolute grid alignment, and that the
        grid offset does not grow over time (no cumulative drift).
        """
        # All steps < 0.1 s, so every 0.1 s boundary gets its own tick
        # (large skipped steps are covered by test_large_dt_skip_*).
        dts = [1 / 60.0, 1 / 30.0, 1 / 20.0, 0.043, 0.017, 0.025]
        max_dt = max(dts)
        ticks = []
        t, fn = 0.0, 0
        i = 0
        while t < 2.1 - 1e-12:
            dt = dts[i % len(dts)]
            i += 1
            t += dt
            fn += 1
            ticks.append((t, fn))
        ad = AsyncDetector(FakeDetector([]), _target())
        harvest_times = []
        prev_latest = None
        for t, fn in ticks:
            ad.begin_tick(t, fn, 0.0, 0.0)
            ad._task = _done_future(_obs(_det(640, 360), fn, t))
            ad.pump()
            cur = ad._latest["frame_number"] if ad._latest else None
            if cur != prev_latest and cur == fn:
                # Fresh harvest applied exactly on this tick.
                harvest_times.append(t)
            prev_latest = cur
        assert len(harvest_times) == 21, harvest_times
        # Absolute grid alignment: never early, never later than one tick.
        for k, h in enumerate(harvest_times, start=1):
            assert h >= k * 0.1 - 1e-9, (k, h)
            assert h - k * 0.1 < max_dt, (k, h)
        # No cumulative drift: grid offset of last 5 == first 5 (within a tick).
        first = sum(h - k * 0.1 for k, h in zip(range(1, 6), harvest_times[:5])) / 5
        last = sum(h - k * 0.1 for k, h in
                   zip(range(17, 22), harvest_times[16:21])) / 5
        assert abs(last - first) < max_dt, (first, last)
        # Newest-wins across the large jump: latest is the jump frame.
        assert ad.inferences_completed == len(ticks)

    @pytest.mark.asyncio
    async def test_large_dt_skip_harvests_newest_without_drift(self):
        """A 0.35 s skipped step harvests newest once; grid stays on 0.1 s."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(0.1, 3, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(640, 360), 3, 0.1))
        ad.pump()
        assert ad._latest["frame_number"] == 3
        ad.begin_tick(0.15, 4, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(650, 360), 4, 0.15))
        ad.pump()
        assert ad._held["frame_number"] == 4
        # Jump 0.35 s: crosses 0.2,0.3,0.4,0.5 at once.
        ad.begin_tick(0.5, 30, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(660, 360), 30, 0.5))
        ad.pump()
        assert ad._latest["frame_number"] == 30
        assert ad._held is None
        assert ad._next_harvest_time == pytest.approx(0.6)
        # Next grid point behaves normally again.
        ad.begin_tick(0.55, 31, 0.0, 0.0)
        ad._task = _done_future(_obs(_det(670, 360), 31, 0.55))
        ad.pump()
        assert ad._held["frame_number"] == 31
        ad.begin_tick(0.6, 32, 0.0, 0.0)
        ad.pump()
        assert ad._latest["frame_number"] == 31


class TestTimeBasedExpiry:
    """Observation timeout in simulation seconds (default 0.4 s).

    0.4 s == 12 ticks at the default dt=1/30, so default-dt behavior is
    decision-equivalent to the previous tick rule at every boundary age.
    All tests use direct _latest injection (no workers): fully
    deterministic, no wall-clock involvement.
    """

    def test_invalid_max_age_s_rejected(self):
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(), max_age_s="0.4")
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(), max_age_s=-1)
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(),
                          max_age_s=float("inf"))
        with pytest.raises(ValueError):
            AsyncDetector(FakeDetector([]), _target(),
                          max_age_s=float("nan"))

    def test_default_timeout_is_point_four_seconds(self):
        ad = AsyncDetector(FakeDetector([]), _target())
        assert ad.max_age_s == pytest.approx(0.4)

    def test_just_inside_timeout_usable(self):
        """Age 0.399 s coasts honestly with tick-count telemetry."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.0, 10, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 5.0)
        assert ad.detect(object()) is not None
        assert ad.last_fresh is True
        ad.begin_tick(5.0 + 0.399, 22, 0.0, 0.0)
        assert ad.detect(object()) is not None
        assert ad.last_age_ticks == 12
        assert ad.last_fresh is False

    def test_just_beyond_timeout_expires(self):
        """Age 0.41 s reports undetected with honest provenance."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.0 + 0.41, 23, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 5.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None
        assert ad.last_fresh is False

    def test_exact_timeout_boundary_usable(self):
        """Age exactly 0.4 s is usable (mirrors old age==12 usable)."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.0 + 0.4, 22, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 5.0)
        assert ad.detect(object()) is not None

    def test_tick_equivalence_at_default_dt(self):
        """At dt=1/30, usable iff frame gap <= 12 (old rule, all ages)."""
        for gap in range(0, 14):
            ad = AsyncDetector(FakeDetector([]), _target())
            ad.begin_tick(5.0 + gap / 30.0, 100 + gap, 0.0, 0.0)
            ad._latest = _obs(_det(640, 360), 100, 5.0)
            got = ad.detect(object())
            assert (got is not None) == (gap <= 12), gap

    def test_expiry_boundary_in_seconds_at_dt_60(self):
        """24 ticks (0.4 s) usable, 25 ticks expired: seconds, not ticks."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.0 + 24 / 60.0, 124, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 100, 5.0)
        assert ad.detect(object()) is not None
        ad2 = AsyncDetector(FakeDetector([]), _target())
        ad2.begin_tick(5.0 + 25 / 60.0, 125, 0.0, 0.0)
        ad2._latest = _obs(_det(640, 360), 100, 5.0)
        assert ad2.detect(object()) is None
        assert ad2.last_age_ticks is None

    def test_expiry_boundary_in_seconds_at_dt_20(self):
        """8 ticks (0.4 s) usable, 9 ticks expired: seconds, not ticks."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.0 + 8 / 20.0, 108, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 100, 5.0)
        assert ad.detect(object()) is not None
        ad2 = AsyncDetector(FakeDetector([]), _target())
        ad2.begin_tick(5.0 + 9 / 20.0, 109, 0.0, 0.0)
        ad2._latest = _obs(_det(640, 360), 100, 5.0)
        assert ad2.detect(object()) is None
        assert ad2.last_age_ticks is None

    def test_large_dt_jump_expires(self):
        """A 5 s skipped step expires instead of erroring."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(10.0, 300, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 150, 5.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None
        assert ad.last_fresh is False

    def test_reset_clears_observations(self):
        """Detector reset retires observations even within the window."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(5.05, 11, 0.0, 0.0)
        ad._latest = _obs(_det(640, 360), 10, 5.0)
        assert ad.detect(object()) is not None
        ad.reset_generation()
        ad.begin_tick(5.06, 12, 0.0, 0.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None

    def test_negative_immediate_despite_old_timestamp(self):
        """A real miss reports loss even with an ancient timestamp."""
        ad = AsyncDetector(FakeDetector([]), _target())
        ad.begin_tick(50.0, 1500, 0.0, 0.0)
        ad._latest = _obs(None, 1400, 5.0)
        assert ad.detect(object()) is None
        assert ad.last_age_ticks is None
        assert ad.last_fresh is False
