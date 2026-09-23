"""Focused tests for live link-budget telemetry in the 30 Hz tick.

Drives SimulationManager._simulation_tick directly (classical detector,
zero disturbances, frozen geometry via tiny dt) so every assertion runs
through the real wiring: state angles -> pointing_error_rad ->
link_budget() -> WS telemetry "link" block. No PAT/controller/detector
behaviour is altered by the wiring under test.
"""
import math

import pytest

from main import SimulationManager, app
from simulation.link_budget import LinkBudgetConfig

LINK_KEYS = {
    "received_power_w",
    "pointing_loss_db",
    "atmospheric_loss_db",
    "link_margin_db",
    "snr_db",
    "link_state",
}


def make_manager(dt=1e-6):
    """Fresh manager with frozen geometry and no disturbances."""
    manager = SimulationManager()
    manager.dt = dt
    for name in ("turbulence", "vibration", "camera_motion", "sensor_noise"):
        manager.set_disturbance(name, 0.0)
    return manager


def aim_at_target(manager, az_offset_deg=0.0, el_offset_deg=0.0):
    """Point the gimbal at the target position of the upcoming tick."""
    t_az, t_el = manager.target.angular_position(manager.simulation_time + manager.dt)
    manager.pan_tilt.azimuth = math.degrees(t_az) + az_offset_deg
    manager.pan_tilt.elevation = math.degrees(t_el) + el_offset_deg


@pytest.mark.asyncio
async def test_aligned_target_strong_link():
    """Aimed gimbal -> negligible error -> NOMINAL link in telemetry."""
    manager = make_manager()
    aim_at_target(manager)
    await manager._simulation_tick()

    assert manager.pointing_error_rad < LinkBudgetConfig().divergence_rad
    telemetry = manager._build_telemetry("FINE_TRACK", "ACTIVE")
    link = telemetry["link"]
    assert link is not None
    assert set(link) == LINK_KEYS
    assert link["link_state"] == "NOMINAL"
    assert link["link_margin_db"] > 40.0
    assert link["pointing_loss_db"] < 1.0
    assert link["received_power_w"] > 0.0


@pytest.mark.asyncio
async def test_increasing_angular_error_lowers_margin():
    """Link margin strictly decreases as gimbal offset grows (radians intact)."""
    margins = []
    # Offsets are fractions of the ~0.001 deg beam divergence so all
    # three margins stay above the SNR floor and remain comparable.
    for offset_deg in (0.0, 0.0005, 0.002):
        manager = make_manager()
        aim_at_target(manager, az_offset_deg=offset_deg)
        await manager._simulation_tick()
        # Error is the true angular offset: no new approximation.
        assert manager.pointing_error_rad == pytest.approx(
            math.radians(offset_deg), abs=1e-6
        )
        margins.append(manager._build_telemetry("ACQUIRE", "LOST")["link"]["link_margin_db"])
    assert margins[0] > margins[1] > margins[2]


@pytest.mark.asyncio
async def test_large_mispointing_outage_independent_of_detection():
    """30 deg off-axis -> OUTAGE, computed even though nothing is detected."""
    manager = make_manager()
    aim_at_target(manager, az_offset_deg=30.0)
    await manager._simulation_tick()

    assert manager.detected is False  # out of FOV
    link = manager._build_telemetry("SEARCH", "LOST")["link"]
    assert link is not None  # link needs no detection
    assert link["link_state"] == "OUTAGE"
    assert link["link_margin_db"] < 0.0
    # Unbounded loss exceeds float range: strict-JSON null on the wire,
    # while the internal model still holds +inf (no detection needed).
    assert link["pointing_loss_db"] is None
    assert manager.link_result["pointing_loss_db"] == math.inf


@pytest.mark.asyncio
async def test_reset_clears_link_result():
    """Reset returns link state to pre-tick (None) without touching config."""
    manager = make_manager()
    aim_at_target(manager)
    await manager._simulation_tick()
    assert manager.link_result is not None
    config_before = manager.link_config
    manager.reset()
    assert manager.link_result is None
    assert manager.pointing_error_rad == 0.0
    assert manager.link_config is config_before
    assert manager._build_telemetry("IDLE", "LOST")["link"] is None


def test_pat_convergence_unchanged():
    """Stationary classical loop still converges (PAT path untouched)."""
    from simulation.camera import VirtualCamera
    from simulation.detector import BeaconDetector
    from simulation.pan_tilt import PanTilt
    from simulation.pat_controller import PATController
    from simulation.pat_state_machine import PATStateMachine
    from simulation.target import Target

    camera = VirtualCamera(width=1280, height=720)
    target = Target(width=1280, height=720)
    detector = BeaconDetector()
    pan_tilt = PanTilt()
    controller = PATController()
    state_machine = PATStateMachine()
    t_az, t_el = math.radians(5.0), math.radians(-8.0)
    dt, sim_time, errors = 1.0 / 30.0, 0.0, []

    for _ in range(150):
        sim_time += dt
        gaz, gel = math.radians(pan_tilt.azimuth), math.radians(pan_tilt.elevation)
        px, py = target.pixel_from_angles(t_az, t_el, gaz, gel)
        detection = detector.detect(camera.render(px, py))
        err = math.hypot(detection["x"] - 640, detection["y"] - 360)
        errors.append(err)
        state_machine.update(True, err, sim_time, True)
        az_cmd, el_cmd = controller.compute_command(
            True, detection["x"], detection["y"],
            state_machine.get_control_mode(), dt,
        )
        pan_tilt.update(az_cmd, el_cmd, dt)

    initial, final, minimum = errors[0], errors[-1], min(errors)
    print(f"\nPAT convergence (post-change): initial={initial:.2f} "
          f"final={final:.2f} min={minimum:.2f}")
    assert initial > 100.0
    assert final < 8.0
    assert minimum < 5.0


@pytest.mark.asyncio
async def test_websocket_telemetry_carries_link():
    """Live WS telemetry includes the link block on every tick."""
    import main as main_module
    from fastapi.testclient import TestClient

    main_module.simulation_manager = SimulationManager()
    client = TestClient(app)
    with client.websocket_connect("/ws/simulation") as websocket:
        websocket.send_json({"command": "start"})
        assert websocket.receive_json()["type"] == "ack"
        seen = 0
        for _ in range(12):
            msg = websocket.receive_json()
            if msg["type"] == "telemetry":
                assert set(msg["link"]) == LINK_KEYS
                seen += 1
                if seen >= 3:
                    break
        assert seen >= 3
        websocket.send_json({"command": "stop"})
    await main_module.simulation_manager.stop()
