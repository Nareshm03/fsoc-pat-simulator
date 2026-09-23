"""Focused unit tests for analytics/experiment_metrics.py (synthetic only)."""
import json
import math

from analytics.experiment_metrics import (
    compute_experiment_metrics,
    normalize_record,
)


def _flat(detected, confidence=0.9, error=10.0, state="ACQUIRE", time=None):
    rec = {"detected": detected, "confidence": confidence,
           "error_px": error, "pat_state": state}
    if time is not None:
        rec["time"] = time
    return rec


class TestNormalizeRecord:
    def test_flat_passthrough(self):
        r = normalize_record(_flat(True, 0.8, 12.5, "FINE_TRACK", time=1.0))
        assert r == {"detected": True, "confidence": 0.8, "error_px": 12.5,
                     "pat_state": "FINE_TRACK", "time": 1.0}

    def test_nested_backend_telemetry(self):
        nested = {"detection": {"detected": True, "confidence": 0.77},
                  "error": {"pixel": 9.5}, "pat": {"state": "LOCKED"},
                  "simulation_time": 2.5}
        r = normalize_record(nested)
        assert r["detected"] is True
        assert r["confidence"] == 0.77
        assert r["error_px"] == 9.5
        assert r["pat_state"] == "LOCKED"
        assert r["time"] == 2.5

    def test_undetected_confidence_zeroed(self):
        r = normalize_record(_flat(False, confidence=0.9, error=0.0))
        assert r["detected"] is False
        assert r["confidence"] == 0.0

    def test_non_dict_safe(self):
        r = normalize_record(None)
        assert r["detected"] is False
        assert r["time"] is None


class TestExperimentMetrics:
    def test_empty_run(self):
        m = compute_experiment_metrics([])
        assert m["frames"] == 0
        assert m["detections"] == 0
        assert m["detection_rate"] == 0.0
        assert m["initial_error"] is None
        assert m["final_error"] is None
        assert m["minimum_error"] is None
        assert m["min_error"] is None
        assert m["convergence_frame"] is None
        assert m["locked_achieved"] is False
        assert m["lock_retention_s"] == 0.0
        assert m["lost_count"] == 0
        json.dumps(m)  # JSON-serializable

    def test_converging_run(self):
        records = [
            _flat(True, 0.9, 250.0, "ACQUIRE", time=1 / 30),
            _flat(True, 0.9, 100.0, "ACQUIRE", time=2 / 30),
            _flat(True, 0.95, 4.0, "FINE_TRACK", time=3 / 30),
            _flat(True, 0.93, 1.5, "LOCKED", time=4 / 30),
            _flat(True, 0.93, 1.2, "LOCKED", time=5 / 30),
        ]
        m = compute_experiment_metrics(records)
        assert m["frames"] == 5
        assert m["detections"] == 5
        assert m["detection_rate"] == 1.0
        assert m["avg_confidence"] == (0.9 + 0.9 + 0.95 + 0.93 + 0.93) / 5
        assert m["initial_error"] == 250.0
        assert m["final_error"] == 1.2
        assert m["minimum_error"] == 1.2
        assert m["min_error"] == 1.2
        assert m["convergence_frame"] == 3
        assert m["convergence_time_s"] == 3 / 30
        assert m["locked_achieved"] is True
        assert m["lock_time_s"] == 4 / 30
        assert m["lock_retention_s"] == 2 / 30
        assert m["lost_count"] == 0
        assert m["lost_reacquired_count"] == 0
        json.dumps(m)

    def test_losses_and_misses(self):
        records = [
            _flat(True, 0.9, 20.0, "ACQUIRE"),
            _flat(True, 0.9, 10.0, "FINE_TRACK"),
            _flat(False, 0.0, 0.0, "REACQUIRE"),   # loss 1
            _flat(False, 0.0, 0.0, "REACQUIRE"),
            _flat(True, 0.8, 30.0, "ACQUIRE"),     # reacquired
            _flat(False, 0.0, 0.0, "REACQUIRE"),   # loss 2
            _flat(True, 0.85, 3.0, "FINE_TRACK"),
        ]
        m = compute_experiment_metrics(records)
        assert m["frames"] == 7
        assert m["detections"] == 4
        assert m["misses"] == 3
        assert m["detection_rate"] == 4 / 7
        assert m["lost_count"] == 2
        assert m["reacquired_count"] == 2
        assert m["convergence_frame"] == 7
        assert m["locked_achieved"] is False

    def test_no_convergence(self):
        records = [_flat(True, 0.9, 50.0 + i, "ACQUIRE") for i in range(5)]
        m = compute_experiment_metrics(records)
        assert m["convergence_frame"] is None
        assert m["convergence_time_s"] is None
        assert m["minimum_error"] == 50.0

    def test_nested_telemetry_input(self):
        records = [
            {"detection": {"detected": True, "confidence": 0.7},
             "error": {"pixel": 8.0}, "pat": {"state": "ACQUIRE"},
             "simulation_time": 0.1},
            {"detection": {"detected": True, "confidence": 0.8},
             "error": {"pixel": 2.0}, "pat": {"state": "LOCKED"},
             "simulation_time": 0.2},
        ]
        m = compute_experiment_metrics(records)
        assert m["detection_rate"] == 1.0
        assert m["initial_error"] == 8.0
        assert m["final_error"] == 2.0
        assert m["convergence_frame"] == 2
        assert m["locked_achieved"] is True

    def test_reproducible(self):
        records = [_flat(i % 3 != 0, 0.5 + (i % 5) / 10.0, float(60 - i),
                         "LOCKED" if i > 8 else "ACQUIRE") for i in range(12)]
        first = compute_experiment_metrics(records)
        second = compute_experiment_metrics(records)
        assert first == second
        assert json.loads(json.dumps(first)) == first
        for value in first.values():
            assert value is None or isinstance(value, (int, float, bool, str))
            if isinstance(value, float):
                assert not math.isnan(value) and not math.isinf(value)
