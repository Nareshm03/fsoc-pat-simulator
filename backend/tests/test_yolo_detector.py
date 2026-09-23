"""
YOLO11 detector load/inference test.

Verifies:
- YOLO11Detector loads backend/models/best.pt
- Inference on synthetic VirtualCamera frames works
- Same interface as Classical: detect(frame) -> {x, y, confidence, bbox} | None
- SimulationManager detector selection (classical <-> yolo)
"""
import math
import pytest

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.yolo_detector import (
    YOLO11Detector,
    AVAILABLE_DETECTORS,
    resolve_model_path,
)


def _yolo_available():
    """Check quickly without importing torch unless needed."""
    import importlib.util

    if importlib.util.find_spec("ultralytics") is None:
        return False
    return resolve_model_path() is not None


requires_yolo = pytest.mark.skipif(
    not _yolo_available(),
    reason="YOLO unavailable (ultralytics not installed or best.pt missing)",
)


class TestYoloInterface:
    """YOLO and Classical share the same detect() contract."""

    def test_available_detectors(self):
        assert "classical" in AVAILABLE_DETECTORS
        assert "yolo" in AVAILABLE_DETECTORS

    def test_model_path_resolves(self):
        path = resolve_model_path()
        # In this repo best.pt (reconstructed from best.pt.zip) must exist.
        # If env lacks the file, skip rather than fail classical pipeline.
        if path is None:
            pytest.skip("best.pt / best.pt.zip not found")
        assert path.exists()

    def test_classical_unchanged(self):
        """Classical CV still works as before (no bbox required)."""
        camera = VirtualCamera()
        detector = BeaconDetector()
        frame = camera.render(640, 360)
        result = detector.detect(frame)
        assert result is not None
        assert "x" in result and "y" in result and "confidence" in result
        assert abs(result["x"] - 640) < 5
        assert abs(result["y"] - 360) < 5

    @requires_yolo
    def test_yolo_load(self):
        det = YOLO11Detector()
        assert det.is_available, f"YOLO failed to load: {det.load_error}"
        info = det.get_info()
        assert info["type"] == "yolo"
        assert info["available"] is True

    @requires_yolo
    def test_yolo_inference_same_fields(self):
        """Return detected, x, y, confidence, bbox on a synthetic frame."""
        camera = VirtualCamera()
        det = YOLO11Detector()
        assert det.is_available, f"YOLO unavailable: {det.load_error}"

        frame = camera.render(640, 360)
        result = det.detect(frame)

        assert result is not None, "YOLO should detect centered synthetic beacon"
        # Required unified fields
        for field in ("x", "y", "confidence", "bbox"):
            assert field in result, f"Missing field: {field}"
        assert not math.isnan(result["x"])
        assert not math.isnan(result["y"])
        assert 0.0 <= result["confidence"] <= 1.0
        x1, y1, x2, y2 = result["bbox"]
        assert x2 > x1 and y2 > y1
        # Center of bbox matches reported x/y
        assert abs((x1 + x2) / 2 - result["x"]) < 1e-3
        assert abs((y1 + y2) / 2 - result["y"]) < 1e-3
        # Roughly near ground truth (allow YOLO tolerance)
        assert abs(result["x"] - 640) < 60, f"x off: {result['x']}"
        assert abs(result["y"] - 360) < 60, f"y off: {result['y']}"
        print(f"YOLO detection: x={result['x']:.1f} y={result['y']:.1f} "
              f"conf={result['confidence']:.3f} bbox={result['bbox']}")

    @requires_yolo
    def test_yolo_none_on_empty_frame(self):
        """Blank frame should yield None (same as Classical)."""
        import numpy as np

        det = YOLO11Detector()
        assert det.is_available
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert det.detect(blank) is None

    @requires_yolo
    def test_yolo_feeds_tracker(self):
        """YOLO output plugs into existing Tracker without changes."""
        from simulation.tracker import Tracker

        camera = VirtualCamera()
        det = YOLO11Detector()
        tracker = Tracker()
        frame = camera.render(500, 300)
        detection = det.detect(frame)
        assert detection is not None
        tracked = tracker.update(detection)
        assert tracked is not None
        assert tracked["x"] == detection["x"]
        assert tracked["y"] == detection["y"]


class TestDetectorSelection:
    """SimulationManager can use either detector through same path."""

    def test_default_is_classical(self):
        from main import SimulationManager

        manager = SimulationManager()
        assert manager.detector_type == "classical"
        assert manager.detector is manager.classical_detector

    def test_set_classical_always_works(self):
        from main import SimulationManager

        manager = SimulationManager()
        info = manager.set_detector("classical")
        assert info["active"] == "classical"
        assert manager.detector_type == "classical"

    def test_set_invalid_raises(self):
        from main import SimulationManager

        manager = SimulationManager()
        with pytest.raises(ValueError):
            manager.set_detector("invalid_detector")

    @requires_yolo
    def test_set_yolo_and_back(self):
        from main import SimulationManager

        manager = SimulationManager()
        info = manager.set_detector("yolo")
        assert info["active"] == "yolo"
        assert manager.detector_type == "yolo"

        # Both detectors work through the same manager.detector.detect(frame)
        frame = manager.camera.render(640, 360)
        detection = manager.detector.detect(frame)
        assert detection is not None
        assert "x" in detection and "confidence" in detection

        # Switch back — Classical unchanged
        manager.set_detector("classical")
        assert manager.detector_type == "classical"
        detection2 = manager.detector.detect(frame)
        assert detection2 is not None

    @requires_yolo
    def test_websocket_set_detector(self):
        """End-to-end: WS set_detector switches backend detector."""
        import main
        from fastapi.testclient import TestClient

        if main.simulation_manager is None:
            from main import SimulationManager
            main.simulation_manager = SimulationManager()
        # Ensure stopped so no telemetry races with ack
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            main.simulation_manager.stop()
        ) if main.simulation_manager.running else None

        client = TestClient(main.app)
        with client.websocket_connect("/ws/simulation") as ws:
            ws.send_json({"command": "set_detector", "detector_type": "yolo"})
            resp = ws.receive_json()
            assert resp["type"] == "ack"
            assert resp["command"] == "set_detector"
            assert resp["detector_type"] == "yolo"

            ws.send_json({"command": "set_detector", "detector_type": "classical"})
            resp = ws.receive_json()
            assert resp["type"] == "ack"

            ws.send_json({"command": "set_detector", "detector_type": "nope"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
