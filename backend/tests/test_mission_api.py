"""API tests for POST /mission/run (thin layer over run_mission)."""
import pytest
from fastapi.testclient import TestClient

from analytics.mission_runner import PHASE_ORDER, run_mission
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

# Small budgets keep endpoint tests fast while exercising the real path.
SMALL_BUDGETS = {
    "acquisition": 10, "stationary_lock": 10, "moving": 10,
    "disturbed": 10, "loss": 10, "search_blank": 10,
    "reacquire": 10, "final_lock": 10, "retention_tail": 5,
}
TINY_BUDGETS = {k: 5 for k in SMALL_BUDGETS}

REPORT_KEYS = ("mission", "status", "detector", "seed", "dt",
               "phase_order", "phases", "phase_durations_s", "timeline",
               "final_pat_state", "final_link_state", "final_link_margin_db",
               "final_pointing_error_px", "recovery_time_s",
               "search_to_reacquire_s", "markers", "metrics", "records")


@pytest.fixture
def client():
    return TestClient(app)


class TestMissionApi:
    def test_run_classical_full_mission(self, client):
        """Default request returns the complete real mission report."""
        resp = client.post("/mission/run", json={"detector": "classical",
                                                 "seed": 7})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for key in REPORT_KEYS:
            assert key in data, f"missing {key}"
        assert data["mission"] == "pat_mission"
        assert data["detector"] == "classical"
        assert data["seed"] == 7
        assert data["status"] == "COMPLETE"
        assert data["phase_order"] == PHASE_ORDER
        assert data["final_pat_state"] == "LOCKED"
        assert len(data["records"]) > 0
        assert data["markers"]["loss_onset"] < data["markers"]["final_lock"]

    def test_response_matches_run_mission(self, client):
        """No duplicated logic: the endpoint returns run_mission() output."""
        body = {"detector": "classical", "seed": 11,
                "phase_budgets": SMALL_BUDGETS}
        resp = client.post("/mission/run", json=body)
        assert resp.status_code == 200, resp.text
        assert resp.json() == run_mission(
            detector="classical", seed=11,
            dt=1.0 / 30.0, phase_budgets=SMALL_BUDGETS)

    def test_deterministic_repeated_request(self, client):
        """Identical requests return identical reports."""
        body = {"detector": "classical", "seed": 21,
                "phase_budgets": SMALL_BUDGETS}
        first = client.post("/mission/run", json=body)
        second = client.post("/mission/run", json=body)
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        assert first.json() == second.json()

    def test_run_invalid_detector(self, client):
        resp = client.post("/mission/run", json={"detector": "nope",
                                                 "seed": 1,
                                                 "phase_budgets": TINY_BUDGETS})
        assert resp.status_code == 400

    def test_run_invalid_budgets(self, client):
        # Unknown phase key -> 400 (runner ValueError).
        bad_key = client.post("/mission/run", json={
            "detector": "classical", "seed": 1,
            "phase_budgets": {"nonsense": 5}})
        assert bad_key.status_code == 400
        # Non-positive budget -> 400 (runner ValueError).
        bad_value = client.post("/mission/run", json={
            "detector": "classical", "seed": 1,
            "phase_budgets": {"moving": 0}})
        assert bad_value.status_code == 400
        # Non-positive dt -> 422 (request validation).
        bad_dt = client.post("/mission/run", json={
            "detector": "classical", "seed": 1, "dt": 0.0,
            "phase_budgets": TINY_BUDGETS})
        assert bad_dt.status_code == 422

    @requires_yolo
    def test_run_yolo_mission(self, client):
        """YOLO detector flows through the endpoint and returns a report."""
        resp = client.post("/mission/run", json={"detector": "yolo",
                                                 "seed": 7,
                                                 "phase_budgets": TINY_BUDGETS})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        for key in REPORT_KEYS:
            assert key in data, f"missing {key}"
        assert data["detector"] == "yolo"
        # Tiny budgets time out early by design; phases run must be a
        # prefix of the scripted order with records behind them.
        assert len(data["phase_order"]) > 0
        assert data["phase_order"] == PHASE_ORDER[:len(data["phase_order"])]
        assert len(data["records"]) > 0
