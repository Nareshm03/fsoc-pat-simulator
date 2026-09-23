"""Focused tests for analytics/experiment_runner.py (synthetic scenarios)."""
import json

import pytest

from analytics.experiment_metrics import compute_experiment_metrics
from analytics.experiment_runner import run_experiment
from analytics.experiment_store import save_experiment
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

IDEAL = {
    "name": "stationary_ideal", "kind": "stationary",
    "az_deg": 5.0, "el_deg": -8.0,
    "turbulence": 0.0, "vibration": 0.0,
    "camera_motion": 0.0, "sensor_noise": 0.0,
}


class TestExperimentRunner:
    def test_classical_run_and_reproducibility(self, tmp_path):
        first = run_experiment("classical", IDEAL, seed=11, ticks=60)
        second = run_experiment("classical", IDEAL, seed=11, ticks=60)

        assert first["detector"] == "classical"
        assert first["ticks"] == 60
        assert len(first["records"]) == 60
        assert first["metrics"] == second["metrics"]
        assert first["records"] == second["records"]
        assert first["metrics"] == compute_experiment_metrics(first["records"], dt=first["dt"])
        assert first["metrics"]["detection_rate"] == 1.0
        json.dumps(first)  # JSON-serializable, store-compatible
        saved = save_experiment(first, tmp_path)
        assert saved["metrics"] == first["metrics"]

    @requires_yolo
    def test_yolo_run_and_reproducibility(self):
        first = run_experiment("yolo", IDEAL, seed=11, ticks=60)
        second = run_experiment("yolo", IDEAL, seed=11, ticks=60)

        assert first["detector"] == "yolo"
        assert len(first["records"]) == 60
        assert first["metrics"] == second["metrics"]
        assert first["records"] == second["records"]
        assert first["metrics"]["detection_rate"] == 1.0
        assert first["metrics"]["avg_confidence"] > 0.5
        json.dumps(first)

    def test_invalid_inputs(self):
        import pytest as _pytest

        with _pytest.raises(ValueError):
            run_experiment("classical", IDEAL, seed=1, ticks=0)
        with _pytest.raises(ValueError):
            run_experiment("nope", IDEAL, seed=1, ticks=5)
