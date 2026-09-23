"""API tests for the experiment REST endpoints (isolated storage)."""
import pytest
from fastapi.testclient import TestClient

import analytics.experiment_api as experiment_api
from main import app


def _yolo_available():
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return False
    from simulation.yolo_detector import resolve_model_path

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


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with experiment storage isolated to tmp_path."""
    monkeypatch.setattr(experiment_api, "EXPERIMENTS_DIR", tmp_path)
    return TestClient(app)


class TestExperimentApi:
    def test_run_classical(self, client):
        resp = client.post("/experiments/run", json={
            "detector": "classical", "scenario": IDEAL,
            "seed": 11, "ticks": 10,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for key in ("experiment_id", "timestamp", "detector", "scenario",
                    "dt", "duration_s", "metrics", "config"):
            assert key in data, f"missing {key}"
        assert data["detector"] == "classical"
        assert data["metrics"]["frames"] == 10
        assert data["metrics"]["detection_rate"] == 1.0

    def test_list_and_fetch(self, client):
        first = client.post("/experiments/run", json={
            "detector": "classical", "scenario": IDEAL,
            "seed": 11, "ticks": 10,
        }).json()
        second = client.post("/experiments/run", json={
            "detector": "classical", "scenario": IDEAL,
            "seed": 12, "ticks": 10,
        }).json()

        listed = client.get("/experiments")
        assert listed.status_code == 200
        ids = [d["experiment_id"] for d in listed.json()]
        assert first["experiment_id"] in ids
        assert second["experiment_id"] in ids

        fetched = client.get(f"/experiments/{first['experiment_id']}")
        assert fetched.status_code == 200
        assert fetched.json() == first

    def test_fetch_invalid_id(self, client):
        assert client.get("/experiments/does_not_exist").status_code == 404
        assert client.get("/experiments/..%2Fsecret").status_code in (404, 422)

    def test_run_invalid_inputs(self, client):
        # Unknown detector -> 400 (runner ValueError).
        bad_detector = client.post("/experiments/run", json={
            "detector": "nope", "scenario": IDEAL, "seed": 1, "ticks": 5,
        })
        assert bad_detector.status_code == 400
        # Non-positive ticks -> 422 (request validation).
        bad_ticks = client.post("/experiments/run", json={
            "detector": "classical", "scenario": IDEAL, "seed": 1, "ticks": 0,
        })
        assert bad_ticks.status_code == 422

    @requires_yolo
    def test_run_yolo(self, client):
        resp = client.post("/experiments/run", json={
            "detector": "yolo", "scenario": IDEAL,
            "seed": 11, "ticks": 5,
        })
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["detector"] == "yolo"
        assert data["metrics"]["frames"] == 5
        assert data["metrics"]["detection_rate"] == 1.0
