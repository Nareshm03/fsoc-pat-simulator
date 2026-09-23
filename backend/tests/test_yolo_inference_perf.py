"""
YOLO inference resolution regression + performance tests.

Default imgsz=416 (was 640): profiling showed the torch forward pass is
~90% of per-tick cost, and same-seed closed-loop runs at 640 vs 416 both
lock on moving and disturbed targets with means within 0.2 px. imgsz=320
was rejected (static center error ~6.5 px vs the 5 px lock scale).

Covers: default/override behavior, output schema + conf threshold,
inference speedup, detection-quality parity, and closed-loop lock
(moving + disturbed) at the shipped default.
"""
import math
import random
import time

import pytest

from simulation.camera import VirtualCamera
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATState, PATStateMachine
from simulation.search_pattern import RasterSearch
from simulation.target import Target
from simulation.yolo_detector import YOLO11Detector, resolve_model_path


def _yolo_available():
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return False
    return resolve_model_path() is not None


requires_yolo = pytest.mark.skipif(
    not _yolo_available(),
    reason="YOLO unavailable (ultralytics not installed or best.pt missing)",
)

DT = 1.0 / 30.0
TRUTHS = [(640, 360), (700, 400), (900, 500), (300, 200),
          (1100, 650), (150, 620), (1000, 120)]


def _frames(camera):
    return [(x, y, camera.render(x, y)) for x, y in TRUTHS]


def _median(fn, n=10):
    ts = sorted((lambda t0: (fn(), (time.perf_counter() - t0) * 1000)[1])(
        time.perf_counter()) for _ in range(n))
    return ts[n // 2]


def test_default_imgsz_and_overrides():
    """Shipped default is 416; explicit imgsz still honored; schema intact."""
    det = YOLO11Detector()
    assert det.imgsz == 416
    assert det.conf == 0.25  # confidence threshold unchanged
    legacy = YOLO11Detector(imgsz=640)
    assert legacy.imgsz == 640
    assert det.get_info()["imgsz"] == 416


@requires_yolo
def test_inference_faster_than_legacy_default():
    """Same frames, warmed models: 416 median well below 640 median."""
    camera = VirtualCamera(width=1280, height=720)
    frames = [f for _, _, f in _frames(camera)]
    new = YOLO11Detector(imgsz=416)
    old = YOLO11Detector(imgsz=640)
    assert new.is_available and old.is_available
    for _ in range(3):
        for f in frames:
            new.detect(f)
            old.detect(f)
    t_new = _median(lambda: [new.detect(f) for f in frames])
    t_old = _median(lambda: [old.detect(f) for f in frames])
    print(f"\n416 median {t_new:.1f} ms/frame vs 640 median {t_old:.1f} ms/frame")
    assert t_new <= t_old * 0.95


@requires_yolo
def test_detection_quality_preserved():
    """416 detects every static frame with centers near truth and 640."""
    camera = VirtualCamera(width=1280, height=720)
    new = YOLO11Detector(imgsz=416)
    old = YOLO11Detector(imgsz=640)
    assert new.is_available and old.is_available
    max_err_new = 0.0
    max_pair_diff = 0.0
    for x, y, f in _frames(camera):
        rn, ro = new.detect(f), old.detect(f)
        assert (rn is None) == (ro is None), (x, y)
        assert rn is not None, f"missed at {(x, y)}"
        assert set(rn.keys()) == {"x", "y", "confidence", "bbox"}
        max_err_new = max(max_err_new, abs(rn["x"] - x) + abs(rn["y"] - y))
        max_pair_diff = max(max_pair_diff,
                            abs(rn["x"] - ro["x"]) + abs(rn["y"] - ro["y"]))
    print(f"\n416 max center err {max_err_new:.2f}px, "
          f"max 416-vs-640 diff {max_pair_diff:.2f}px")
    assert max_err_new < 3.0
    assert max_pair_diff < 4.0


def _run_closed_loop(detector, kind, duration_s=20.0, seed=7):
    """YOLO closed loop mirroring test_pat_tracking_moving.run_scenario."""
    random.seed(seed)
    target = Target(1280, 720)
    camera = VirtualCamera(1280, 720)
    dist = DisturbanceModel(0.0, 0.0, 0.0, 0.0) if kind == "moving" else \
        DisturbanceModel(2.0, 2.0, 2.0, 1.0)
    pt, ctrl, sm = PanTilt(), PATController(), PATStateMachine()
    raster = RasterSearch()
    recs, t = [], 0.0
    for _ in range(int(duration_s / DT)):
        t += DT
        taz, tel = target.angular_position(t)
        gaz, gel = math.radians(pt.azimuth), math.radians(pt.elevation)
        pos = target.pixel_from_angles(taz, tel, gaz, gel)
        if pos is not None:
            turb = dist.turbulence_offset(t)
            vib = dist.vibration_offset(t)
            cam = dist.camera_offset(t)
            sen = dist.sensor_offset()
            det = detector.detect(camera.render(
                pos[0] + turb[0] + vib[0] + cam[0] + sen[0],
                pos[1] + turb[1] + vib[1] + cam[1] + sen[1]))
        else:
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
                gimbal_el_rate_deg_s=pt.elevation_velocity)
        pt.update(az_cmd, el_cmd, DT)
        recs.append({"t": t, "detected": detected, "err": err, "state": state})
    return recs


def _metrics(recs):
    det = [r for r in recs if r["detected"]]
    errs = [r["err"] for r in det]
    locked = [r for r in recs if r["state"] == PATState.LOCKED]
    return {
        "locked": bool(locked),
        "lock_time": locked[0]["t"] if locked else None,
        "lock_frac": len(locked) / len(recs),
        "min": min(errs), "mean": sum(errs) / len(errs), "final": errs[-1],
        "det_rate": len(det) / len(recs),
    }


@requires_yolo
def test_yolo_moving_lock_at_default():
    """Moving target locks at shipped default (Classical bars apply)."""
    m = _metrics(_run_closed_loop(YOLO11Detector(), "moving"))
    print(f"\nyolo416 moving: locked={m['locked']} t={m['lock_time']} "
          f"frac={m['lock_frac']:.3f} min={m['min']:.2f} "
          f"mean={m['mean']:.1f} final={m['final']:.2f}")
    assert m["locked"]
    assert m["lock_time"] < 15.0
    assert m["min"] < 5.0
    assert m["mean"] < 30.0
    assert m["final"] < 15.0
    assert m["det_rate"] == 1.0


@requires_yolo
def test_yolo_disturbed_lock_at_default():
    """Disturbed moving target locks at shipped default."""
    m = _metrics(_run_closed_loop(YOLO11Detector(), "moving_disturbed"))
    print(f"\nyolo416 disturbed: locked={m['locked']} t={m['lock_time']} "
          f"frac={m['lock_frac']:.3f} min={m['min']:.2f} "
          f"mean={m['mean']:.1f} final={m['final']:.2f}")
    assert m["locked"]
    assert m["min"] < 5.0
    assert m["mean"] < 35.0
    assert m["final"] < 30.0
