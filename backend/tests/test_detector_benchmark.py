"""
Fair Classical CV vs YOLO11 benchmark.

Runs BOTH detectors on IDENTICAL scenarios (same seeds, trajectory,
disturbances, duration) through the unmodified PAT loop:
  detection -> error -> PAT controller -> pan/tilt -> new projection.

Scenarios (all stationary az=5deg el=-8deg unless noted):
  1. stationary_ideal         (all disturbances 0)
  2. stationary_turbulence    (turbulence=2.0)
  3. stationary_vibration     (vibration=2.0)
  4. stationary_camera_motion (camera_motion=2.0)
  5. stationary_high_noise    (sensor_noise=2.0)
  6. moving_target            (Target.angular_position(t), disturbances 0.5)

No runtime/PAT/physics/controller changes. Both detectors use the same
detect(frame)->{x,y,confidence,bbox?}|None interface.

Saves machine-readable backend/benchmark_results/detector_comparison.json
and .csv, prints a concise summary.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import pytest

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATState, PATStateMachine
from simulation.sim_step import simulation_step
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
TICKS = 300  # 10s @30Hz - enough for 2s LOCKED dwell + retention
BASE_SEED = 12345

SCENARIOS = [
    {
        "name": "stationary_ideal",
        "kind": "stationary",
        "az_deg": 5.0,
        "el_deg": -8.0,
        "turbulence": 0.0,
        "vibration": 0.0,
        "camera_motion": 0.0,
        "sensor_noise": 0.0,
    },
    {
        "name": "stationary_turbulence",
        "kind": "stationary",
        "az_deg": 5.0,
        "el_deg": -8.0,
        "turbulence": 2.0,
        "vibration": 0.0,
        "camera_motion": 0.0,
        "sensor_noise": 0.0,
    },
    {
        "name": "stationary_vibration",
        "kind": "stationary",
        "az_deg": 5.0,
        "el_deg": -8.0,
        "turbulence": 0.0,
        "vibration": 2.0,
        "camera_motion": 0.0,
        "sensor_noise": 0.0,
    },
    {
        "name": "stationary_camera_motion",
        "kind": "stationary",
        "az_deg": 5.0,
        "el_deg": -8.0,
        "turbulence": 0.0,
        "vibration": 0.0,
        "camera_motion": 2.0,
        "sensor_noise": 0.0,
    },
    {
        "name": "stationary_high_noise",
        "kind": "stationary",
        "az_deg": 5.0,
        "el_deg": -8.0,
        "turbulence": 0.0,
        "vibration": 0.0,
        "camera_motion": 0.0,
        "sensor_noise": 2.0,
    },
    {
        "name": "moving_target",
        "kind": "moving",
        "turbulence": 0.5,
        "vibration": 0.5,
        "camera_motion": 0.5,
        "sensor_noise": 0.5,
    },
]

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


def run_scenario(detector, scenario, seed, ticks=TICKS):
    """Run one detector on one scenario with a fixed seed.

    Fresh Target/Camera/Disturbance/PAT/PanTilt every call; only the
    detector instance is injected. Identical seed => identical
    sensor_noise sequence for both detectors.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass

    target = Target(width=1280, height=720)
    camera = VirtualCamera(width=1280, height=720)
    disturbances = DisturbanceModel(
        turbulence=scenario["turbulence"],
        vibration=scenario["vibration"],
        camera_motion=scenario["camera_motion"],
        sensor_noise=scenario["sensor_noise"],
    )
    pan_tilt = PanTilt()
    controller = PATController()
    state_machine = PATStateMachine()

    stationary = scenario["kind"] == "stationary"
    if stationary:
        t_az = math.radians(scenario["az_deg"])
        t_el = math.radians(scenario["el_deg"])

    sim_time = 0.0
    errors, confidences = [], []
    detections = 0
    lock_time = None
    time_locked = 0.0
    convergence_frame = None  # first frame with error < 5px (lock threshold)

    for tick in range(1, ticks + 1):
        sim_time += DT

        # Shared closed-loop tick (legacy benchmark semantics: no raster
        # branch, no feedforward, no tracker/link, no out-of-FOV render).
        out = simulation_step(
            target=target,
            target_override=(t_az, t_el) if stationary else None,
            disturbances=disturbances,
            camera=camera,
            detector=detector,
            tracker=None,
            pan_tilt=pan_tilt,
            controller=controller,
            state_machine=state_machine,
            search=None,
            link_config=None,
            sim_time=sim_time,
            dt=DT,
            simulation_running=True,
            blank=False,
            detect_when_out_of_fov=False,
            use_raster_search=False,
            feedforward=False,
        )
        detected = out["detected"]
        dx, dy = float(out["dx"]), float(out["dy"])
        conf = float(out["confidence"]) if detected else 0.0
        err = out["error_px"]
        pat_state = out["pat_state"]
        if detected:
            detections += 1
            errors.append(err)
            confidences.append(conf)
            if convergence_frame is None and err < 5.0:
                convergence_frame = tick

        if pat_state == PATState.LOCKED:
            if lock_time is None:
                lock_time = sim_time
            time_locked += DT

    loss_count = state_machine._loss_count
    return {
        "detections": detections,
        "ticks": ticks,
        "detection_rate": detections / ticks if ticks else 0.0,
        "avg_confidence": (sum(confidences) / len(confidences)) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "initial_error": errors[0] if errors else None,
        "final_error": errors[-1] if errors else None,
        "min_error": min(errors) if errors else None,
        "convergence_frame": convergence_frame,
        "convergence_time_s": (convergence_frame * DT) if convergence_frame else None,
        "locked_achieved": lock_time is not None,
        "lock_time_s": lock_time,
        "lock_retention_s": time_locked,
        "lost_count": loss_count,
        "missed_frames": ticks - detections,
        "final_gimbal_az": pan_tilt.azimuth,
        "final_gimbal_el": pan_tilt.elevation,
    }


def run_benchmark(ticks=TICKS):
    """Run all 6 scenarios x 2 detectors with identical seeds/duration."""
    classical = BeaconDetector()
    yolo = YOLO11Detector()
    assert yolo.is_available, f"YOLO unavailable: {yolo.load_error}"

    rows = []
    started = time.time()
    for i, scenario in enumerate(SCENARIOS):
        seed = BASE_SEED + i  # identical seed for BOTH detectors
        for det_name, detector in (("classical", classical), ("yolo", yolo)):
            m = run_scenario(detector, scenario, seed=seed, ticks=ticks)
            rows.append({
                "scenario": scenario["name"],
                "detector": det_name,
                "seed": seed,
                "ticks": ticks,
                "duration_s": round(ticks * DT, 3),
                **{k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in m.items()},
            })
    elapsed = time.time() - started
    return rows, elapsed


def save_report(rows, elapsed_s, out_dir=RESULTS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "detector_comparison.json"
    csv_path = out_dir / "detector_comparison.csv"
    payload = {
        "meta": {
            "ticks": TICKS,
            "dt": DT,
            "duration_s_per_run": round(TICKS * DT, 3),
            "base_seed": BASE_SEED,
            "scenarios": [s["name"] for s in SCENARIOS],
            "detectors": ["classical", "yolo"],
            "benchmark_wall_s": round(elapsed_s, 1),
            "note": "Identical seeds/trajectory/disturbances/duration per scenario. "
                    "No PAT/physics/controller changes.",
        },
        "results": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def print_summary(rows):
    print("\n=== Classical CV vs YOLO11 (identical scenarios) ===")
    print(f"{'scenario':<26}{'det':<10}{'det%':>7}{'avgConf':>9}"
          f"{'init':>9}{'final':>9}{'min':>9}{'convF':>7}{'LOCKED':>8}"
          f"{'ret(s)':>8}{'lost':>6}")
    for scenario in SCENARIOS:
        for det in ("classical", "yolo"):
            r = next(x for x in rows if x["scenario"] == scenario["name"]
                     and x["detector"] == det)
            conv = str(r["convergence_frame"]) if r["convergence_frame"] else "-"
            init = f"{r['initial_error']:.1f}" if r["initial_error"] is not None else "-"
            final = f"{r['final_error']:.1f}" if r["final_error"] is not None else "-"
            minimum = f"{r['min_error']:.1f}" if r["min_error"] is not None else "-"
            print(f"{r['scenario']:<26}{r['detector']:<10}"
                  f"{r['detection_rate'] * 100:>6.1f}%"
                  f"{r['avg_confidence']:>9.3f}"
                  f"{init:>9}{final:>9}{minimum:>9}"
                  f"{conv:>7}{str(r['locked_achieved']):>8}"
                  f"{r['lock_retention_s']:>8.2f}{r['lost_count']:>6}")


class TestDetectorBenchmark:
    """Fair comparison: same seeds/trajectory/disturbances/duration."""

    @requires_yolo
    def test_classical_vs_yolo_all_scenarios(self):
        rows, elapsed = run_benchmark(ticks=TICKS)
        json_path, csv_path = save_report(rows, elapsed)
        print_summary(rows)
        print(f"\nReport: {json_path}\nReport: {csv_path} "
              f"({elapsed:.0f}s wall)")

        # Structural guarantees (no threshold weakening, no winner asserted).
        assert len(rows) == len(SCENARIOS) * 2 == 12
        for r in rows:
            assert r["ticks"] == TICKS
            assert 0.0 <= r["detection_rate"] <= 1.0
            assert 0.0 <= r["avg_confidence"] <= 1.0
            assert r["missed_frames"] == r["ticks"] - r["detections"]
            assert r["lost_count"] >= 0
            if r["locked_achieved"]:
                assert r["lock_time_s"] is not None
                assert r["lock_retention_s"] > 0
        # Identical conditions proof: same seed/ticks/duration per pair.
        for scenario in SCENARIOS:
            pair = [x for x in rows if x["scenario"] == scenario["name"]]
            assert len(pair) == 2
            assert pair[0]["seed"] == pair[1]["seed"] == BASE_SEED + SCENARIOS.index(scenario)
            assert pair[0]["ticks"] == pair[1]["ticks"] == TICKS
        assert json_path.exists() and csv_path.exists()
