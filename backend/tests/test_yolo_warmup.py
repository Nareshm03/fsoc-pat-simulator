"""
YOLO background warmup regression tests.

Switching to YOLO used to block the asyncio event loop twice: ~2.1 s of
model initialization inside set_detector(), then ~2.1 s of first-inference
cost inside the first live tick (measured 2026-09-20). set_detector() now
constructs the detector cheaply (load deferred) and warms the cached model
in a worker thread; the lazy load inside detect() remains the fallback.

Conventions follow test_yolo_detector.py (requires_yolo skip marker) and
pytest.ini (asyncio_mode = auto).
"""
import asyncio
import time

import pytest

from main import SimulationManager
from simulation.yolo_detector import YOLO11Detector, resolve_model_path


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


@requires_yolo
def test_switch_to_yolo_stays_responsive():
    """1. set_detector() returns quickly; heavy work is deferred."""
    manager = SimulationManager()
    t0 = time.perf_counter()
    info = manager.set_detector("yolo")
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"set_detector blocked: {elapsed:.2f}s"
    assert manager.detector_type == "yolo"
    assert manager.detector is manager.yolo_detector
    assert info["active"] == "yolo"
    # Sync context has no running loop: nothing scheduled, nothing loaded.
    assert manager._yolo_warmup_task is None
    assert manager.yolo_detector.model is None


@requires_yolo
async def test_warmup_runs_off_event_loop():
    """2. Model init + first inference happen in a worker thread."""
    manager = SimulationManager()
    manager.set_detector("yolo")
    assert manager._yolo_warmup_task is not None

    beats = 0
    stop = False

    async def heartbeat():
        nonlocal beats
        while not stop:
            await asyncio.sleep(0)
            beats += 1

    task = asyncio.create_task(heartbeat())
    try:
        warmed = await manager._yolo_warmup_task
    finally:
        stop = True
    await task
    await manager.stop()

    assert warmed is True
    assert beats > 0, "event loop starved during warmup"
    assert manager.yolo_detector.model is not None


@requires_yolo
async def test_first_tick_reuses_warmup():
    """3. A completed warmup is reused: first tick skips init cost."""
    manager = SimulationManager()
    manager.set_detector("yolo")
    t0 = time.perf_counter()
    await manager._yolo_warmup_task
    warmup_wall = time.perf_counter() - t0

    model_before = manager.yolo_detector.model
    assert model_before is not None

    manager.running = True
    t0 = time.perf_counter()
    await manager._simulation_tick()
    tick_wall = time.perf_counter() - t0
    await manager.stop()

    assert manager.yolo_detector.model is model_before
    assert manager.frame_number == 1
    assert manager.simulation_time > 0
    assert tick_wall < warmup_wall, (
        f"first tick ({tick_wall:.2f}s) did not reuse "
        f"warmup ({warmup_wall:.2f}s)"
    )


@requires_yolo
def test_fallback_without_warmup():
    """4. Without warmup, the lazy load inside detect() still works."""
    detector = YOLO11Detector(defer_load=True)
    assert detector.model is None

    import numpy as np

    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = detector.detect(black)  # must not raise
    assert result is None  # empty frame: no detection, not an error
    assert detector.model is not None  # lazy fallback engaged

    # Manager-level: sync context schedules nothing, yet switching works.
    manager = SimulationManager()
    manager.set_detector("yolo")
    assert manager._yolo_warmup_task is None
    assert manager.detector_type == "yolo"


@requires_yolo
async def test_rapid_switching_no_stale_state():
    """5. Rapid yolo/classical switching leaves consistent active state."""
    manager = SimulationManager()
    for _ in range(3):
        manager.set_detector("yolo")
        manager.set_detector("classical")
    manager.set_detector("yolo")

    # Concurrent warmups on the shared instance stay consistent.
    detector = manager.yolo_detector
    model_before = detector.model
    first, second = await asyncio.gather(
        asyncio.to_thread(detector.warmup),
        asyncio.to_thread(detector.warmup),
    )
    assert first is True and second is True
    assert detector.model is not None
    if model_before is not None:
        assert detector.model is model_before

    assert manager.detector_type == "yolo"
    assert manager.detector is manager.yolo_detector

    manager.running = True
    await manager._simulation_tick()
    assert manager.frame_number == 1

    manager.set_detector("classical")
    assert manager.detector_type == "classical"
    assert manager.detector is manager.classical_detector
    await manager.stop()


@requires_yolo
async def test_classical_unchanged():
    """6. Classical path never touches YOLO state."""
    manager = SimulationManager()
    manager.set_detector("classical")
    assert manager.detector_type == "classical"
    assert manager.yolo_detector is None

    manager.running = True
    for _ in range(5):
        await manager._simulation_tick()
    await manager.stop()

    assert manager.frame_number == 5
    assert manager.simulation_time > 0
    assert manager.yolo_detector is None
