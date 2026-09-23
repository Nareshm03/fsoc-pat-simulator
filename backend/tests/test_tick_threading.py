"""
Live-loop responsiveness: the CPU-bound closed-loop tick must not block
the asyncio event loop.

Background (measured 2026-09-20 through simulation_step directly):
  classical tick  ~5.5 ms median  (within the 33.3 ms frame budget)
  yolo tick     ~151.5 ms median, ~5.3 s first tick (model warmup)

Before this change _simulation_tick() ran simulation_step() inline, so
every YOLO tick starved the loop: a heartbeat coroutine got 0 beats
during a 3.8 s first tick and 0 beats during a 128 ms steady tick,
stalling WS commands, telemetry broadcast, and /health for all clients.
The tick now runs in a worker thread (same function, order, and dt).

Conventions follow test_yolo_closed_loop.py (requires_yolo skip marker)
and pytest.ini (asyncio_mode = auto).
"""
import asyncio
import random

import pytest

from main import SimulationManager
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


class _FakeClient:
    """Minimal WS client double: only send_json is used by _broadcast."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


def _fresh_manager(detector="classical"):
    manager = SimulationManager()
    if detector == "yolo":
        manager.set_detector("yolo")
    manager.running = True
    return manager


def _snapshot(manager):
    return (
        round(manager.simulation_time, 9),
        manager.frame_number,
        manager.detected,
        round(manager.detected_x, 9),
        round(manager.detected_y, 9),
        round(manager.tracking_error_px, 9),
        manager.pat_state_machine.state.value,
        round(manager.pan_tilt.azimuth, 9),
        round(manager.pan_tilt.elevation, 9),
        round(manager.pointing_error_rad, 12),
    )


@requires_yolo
async def test_event_loop_responsive_during_yolo_ticks():
    """Heartbeat coroutines make progress while a YOLO tick computes."""
    manager = _fresh_manager("yolo")
    beats = 0
    stop = False

    async def heartbeat():
        nonlocal beats
        while not stop:
            await asyncio.sleep(0)
            beats += 1

    task = asyncio.create_task(heartbeat())
    try:
        await manager._simulation_tick()
        await manager._simulation_tick()
    finally:
        stop = True
    await task
    await manager.stop()

    assert beats > 0, "event loop starved during YOLO tick"
    assert manager.frame_number == 2
    assert manager.simulation_time == pytest.approx(2.0 / 30.0)


async def test_classical_tick_parity_seeded():
    """Threaded ticks are deterministic: identical seeds, identical runs."""
    trajectories = []
    for _ in range(2):
        random.seed(123)
        manager = _fresh_manager("classical")
        trajectory = []
        for _ in range(30):
            await manager._simulation_tick()
            trajectory.append(_snapshot(manager))
        await manager.stop()
        trajectories.append(trajectory)
    assert trajectories[0] == trajectories[1]
    assert trajectories[0][-1][0] == pytest.approx(1.0)
    assert trajectories[0][-1][1] == 30


async def test_telemetry_schema_unchanged():
    """Broadcast telemetry + frame messages keep their exact schema."""
    manager = _fresh_manager("classical")
    client = _FakeClient()
    manager.add_client(client)
    random.seed(7)
    for _ in range(3):
        await manager._simulation_tick()
    await manager.stop()

    assert len(client.messages) == 4  # 3 telemetry + 1 frame (every 3rd)
    telemetries = [m for m in client.messages if m["type"] == "telemetry"]
    frames = [m for m in client.messages if m["type"] == "frame"]
    assert len(telemetries) == 3
    assert len(frames) == 1
    assert set(telemetries[0].keys()) == {
        "type", "simulation_time", "frame", "simulation", "target",
        "detection", "detector", "tracking", "error", "pat", "gimbal",
        "disturbances", "fsoc", "link", "performance",
    }
    assert set(telemetries[0]["link"].keys()) == {
        "received_power_w", "pointing_loss_db", "atmospheric_loss_db",
        "link_margin_db", "snr_db", "link_state",
    }
    assert set(frames[0].keys()) == {"type", "frame", "simulation_time", "image"}
    assert [t["frame"] for t in telemetries] == [1, 2, 3]
