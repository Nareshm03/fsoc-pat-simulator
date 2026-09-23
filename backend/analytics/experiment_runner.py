"""Reusable single-experiment runner (orchestration only, no I/O).

Builds a fresh simulation from existing components and runs the existing
closed PAT loop per tick:
  Target.pixel_from_angles -> Disturbance offsets -> VirtualCamera.render
  -> detector.detect -> error -> PATStateMachine -> PATController
  -> PanTilt.update.

No control/physics/detector logic is duplicated or modified here; every
algorithm is called from its existing module. The only randomness source
is the supplied seed (Python + NumPy RNGs are reseeded on entry).

Returns one complete experiment result compatible with
analytics.experiment_store.save_experiment (which adds experiment_id /
timestamp when missing). This module never writes files.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Union

from simulation.camera import VirtualCamera
from simulation.disturbances import DisturbanceModel
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATStateMachine
from simulation.sim_step import simulation_step
from simulation.target import Target

from .experiment_metrics import compute_experiment_metrics

DetectorLike = Union[str, Any]
ScenarioLike = Union[str, Dict[str, Any]]

_DEFAULT_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "stationary_ideal": {
        "name": "stationary_ideal", "kind": "stationary",
        "az_deg": 5.0, "el_deg": -8.0,
        "turbulence": 0.0, "vibration": 0.0,
        "camera_motion": 0.0, "sensor_noise": 0.0,
    },
}


def _resolve_detector(detector: DetectorLike):
    """Return (name, instance) reusing existing detector classes."""
    if isinstance(detector, str):
        key = detector.strip().lower()
        if key == "classical":
            from simulation.detector import BeaconDetector

            return "classical", BeaconDetector()
        if key in ("yolo", "yolo11", "yolo11n"):
            from simulation.yolo_detector import YOLO11Detector

            instance = YOLO11Detector()
            if not instance.is_available:
                raise RuntimeError(f"YOLO detector unavailable: {instance.load_error}")
            return "yolo", instance
        raise ValueError(f"Unknown detector: {detector!r} (use 'classical' or 'yolo')")
    instance = detector
    name = getattr(instance, "detector_name", None)
    if name is None:
        cls = type(instance).__name__.lower()
        if "yolo" in cls:
            name = "yolo"
        elif "beacon" in cls or "classical" in cls:
            name = "classical"
        else:
            name = cls or "unknown"
    if not hasattr(instance, "detect") or not callable(instance.detect):
        raise TypeError("detector must provide a callable detect(frame) method")
    return str(name), instance


def _resolve_scenario(scenario: ScenarioLike) -> Dict[str, Any]:
    if isinstance(scenario, str):
        if scenario not in _DEFAULT_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario!r}")
        return dict(_DEFAULT_SCENARIOS[scenario])
    if not isinstance(scenario, dict):
        raise TypeError("scenario must be a dict or known scenario name")
    resolved = {
        "name": scenario.get("name", "custom"),
        "kind": scenario.get("kind", "stationary"),
        "az_deg": float(scenario.get("az_deg", 5.0)),
        "el_deg": float(scenario.get("el_deg", -8.0)),
        "turbulence": float(scenario.get("turbulence", 0.0)),
        "vibration": float(scenario.get("vibration", 0.0)),
        "camera_motion": float(scenario.get("camera_motion", 0.0)),
        "sensor_noise": float(scenario.get("sensor_noise", 0.0)),
    }
    if resolved["kind"] not in ("stationary", "moving"):
        raise ValueError(f"Unknown scenario kind: {resolved['kind']!r}")
    return resolved


def run_experiment(
    detector: DetectorLike,
    scenario: ScenarioLike,
    seed: int,
    ticks: int,
    dt: float = 1.0 / 30.0,
) -> Dict[str, Any]:
    """Run one experiment and return the complete result (no file I/O).

    Args:
        detector: "classical" / "yolo" or an instance with detect(frame).
        scenario: scenario dict (kind, az/el_deg, turbulence, vibration,
            camera_motion, sensor_noise) or known scenario name.
        seed: RNG seed; the only randomness source (sensor noise).
        ticks: number of frames to simulate (> 0).
        dt: seconds per tick.

    Returns:
        Dict with detector, scenario, dt, ticks, duration_s, seed,
        config, records (per-frame telemetry), and metrics from
        compute_experiment_metrics(). JSON-serializable; suitable for
        save_experiment() which fills experiment_id/timestamp.
    """
    if not isinstance(ticks, int) or ticks <= 0:
        raise ValueError("ticks must be a positive int")
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be positive")
    seed = int(seed)

    detector_name, detector_instance = _resolve_detector(detector)
    resolved = _resolve_scenario(scenario)

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass

    target = Target(width=1280, height=720)
    camera = VirtualCamera(width=1280, height=720)
    disturbances = DisturbanceModel(
        turbulence=resolved["turbulence"],
        vibration=resolved["vibration"],
        camera_motion=resolved["camera_motion"],
        sensor_noise=resolved["sensor_noise"],
    )
    pan_tilt = PanTilt()
    controller = PATController()
    state_machine = PATStateMachine()

    stationary = resolved["kind"] == "stationary"
    fixed_az = math.radians(resolved["az_deg"])
    fixed_el = math.radians(resolved["el_deg"])

    records = []
    sim_time = 0.0
    for _ in range(ticks):
        sim_time += dt
        # Shared closed-loop tick (legacy experiment semantics: no raster
        # branch, no feedforward, no tracker/link, no out-of-FOV render).
        out = simulation_step(
            target=target,
            target_override=(fixed_az, fixed_el) if stationary else None,
            disturbances=disturbances,
            camera=camera,
            detector=detector_instance,
            tracker=None,
            pan_tilt=pan_tilt,
            controller=controller,
            state_machine=state_machine,
            search=None,
            link_config=None,
            sim_time=sim_time,
            dt=dt,
            simulation_running=True,
            blank=False,
            detect_when_out_of_fov=False,
            use_raster_search=False,
            feedforward=False,
        )
        detected = out["detected"]
        dx, dy = float(out["dx"]), float(out["dy"])
        confidence = float(out["confidence"]) if detected else 0.0
        error_px = out["error_px"]

        records.append({
            "detected": detected,
            "confidence": confidence,
            "error_px": error_px,
            "pat_state": out["pat_state"].value,
            "time": sim_time,
        })

    metrics = compute_experiment_metrics(records, dt=dt)
    return {
        "detector": detector_name,
        "scenario": resolved.get("name", "custom"),
        "scenario_config": resolved,
        "dt": dt,
        "ticks": ticks,
        "duration_s": ticks * dt,
        "duration": ticks * dt,
        "seed": seed,
        "config": {"ticks": ticks, "dt": dt, "seed": seed, "scenario": resolved},
        "records": records,
        "metrics": metrics,
    }
