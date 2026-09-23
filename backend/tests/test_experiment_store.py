"""Focused tests for analytics/experiment_store.py (tmp dir, synthetic only)."""
import json

from analytics.experiment_metrics import compute_experiment_metrics
from analytics.experiment_store import (
    list_experiments,
    load_experiment,
    save_experiment,
)


def _records():
    return [
        {"detected": True, "confidence": 0.9, "error_px": 120.0,
         "pat_state": "ACQUIRE", "time": 1 / 30},
        {"detected": True, "confidence": 0.92, "error_px": 40.0,
         "pat_state": "FINE_TRACK", "time": 2 / 30},
        {"detected": True, "confidence": 0.95, "error_px": 3.0,
         "pat_state": "LOCKED", "time": 3 / 30},
        {"detected": False, "confidence": 0.0, "error_px": 0.0,
         "pat_state": "REACQUIRE", "time": 4 / 30},
        {"detected": True, "confidence": 0.93, "error_px": 2.0,
         "pat_state": "LOCKED", "time": 5 / 30},
    ]


def _result(**overrides):
    base = {
        "experiment_id": "exp_test_001",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "detector": "yolo",
        "scenario": "stationary_ideal",
        "dt": 1.0 / 30.0,
        "config": {"ticks": 5, "seed": 7},
        "records": _records(),
    }
    base.update(overrides)
    return base


class TestExperimentStore:
    def test_save_load_round_trips_exactly(self, tmp_path):
        saved = save_experiment(_result(), tmp_path)
        assert (tmp_path / "exp_test_001.json").exists()
        loaded = load_experiment(tmp_path / "exp_test_001.json")
        assert loaded == saved
        # Exact JSON round-trip (no float drift, sorted keys stable).
        assert json.loads(json.dumps(loaded, sort_keys=True)) == loaded

    def test_saved_contents(self, tmp_path):
        saved = save_experiment(_result(), tmp_path)
        for key in ("experiment_id", "timestamp", "detector", "scenario",
                    "dt", "duration_s", "metrics", "config"):
            assert key in saved
        assert saved["experiment_id"] == "exp_test_001"
        assert saved["detector"] == "yolo"
        assert saved["scenario"] == "stationary_ideal"
        assert saved["config"] == {"ticks": 5, "seed": 7}
        expected = compute_experiment_metrics(_records(), dt=1.0 / 30.0)
        assert saved["metrics"] == expected
        assert saved["metrics"]["detection_rate"] == 4 / 5
        assert saved["metrics"]["locked_achieved"] is True

    def test_save_to_explicit_file(self, tmp_path):
        target = tmp_path / "custom" / "run.json"
        saved = save_experiment(_result(experiment_id="exp_file"), target)
        assert target.exists()
        assert load_experiment(target) == saved

    def test_precomputed_metrics_preserved(self, tmp_path):
        metrics = compute_experiment_metrics(_records())
        saved = save_experiment({
            "experiment_id": "exp_pre",
            "timestamp": "2026-02-02T00:00:00+00:00",
            "detector": "classical",
            "scenario": "moving_target",
            "dt": 1.0 / 30.0,
            "config": {},
            "metrics": metrics,
        }, tmp_path)
        assert saved["metrics"] == metrics
        assert load_experiment(tmp_path / "exp_pre.json") == saved

    def test_list_experiments(self, tmp_path):
        first = save_experiment(_result(experiment_id="exp_b",
                                        timestamp="2026-01-02T00:00:00+00:00"), tmp_path)
        second = save_experiment(_result(experiment_id="exp_a",
                                         timestamp="2026-01-01T00:00:00+00:00"), tmp_path)
        listed = list_experiments(tmp_path)
        assert [d["experiment_id"] for d in listed] == ["exp_a", "exp_b"]
        assert listed[0] == second
        assert listed[1] == first
        # Single-file path returns a one-element list.
        assert list_experiments(tmp_path / "exp_a.json") == [second]
        # Missing dir returns empty (no crash).
        assert list_experiments(tmp_path / "absent") == []

    def test_generated_id_and_timestamp(self, tmp_path):
        saved = save_experiment({"detector": "yolo", "records": _records()}, tmp_path)
        assert saved["experiment_id"]
        assert saved["timestamp"]
        assert (tmp_path / f"{saved['experiment_id']}.json").exists()
        assert load_experiment(tmp_path / f"{saved['experiment_id']}.json") == saved
