"""Strict-JSON wire guarantee for WebSocket telemetry.

Proves the backend itself emits valid strict JSON on every frame, even with
a huge pointing error that drives pointing loss to +inf: non-finite floats
become null, finite values pass through exactly.
"""
import json
import math

import pytest

from main import SimulationManager, app, sanitize_json_numbers


def strict_loads(text):
    """Parse like a browser: reject Infinity/-Infinity/NaN tokens."""

    def _reject(value):
        raise ValueError(f"non-JSON number token: {value}")

    return json.loads(text, parse_constant=_reject)


class TestSanitizeJsonNumbers:
    def test_non_finite_become_null(self):
        assert sanitize_json_numbers(float("inf")) is None
        assert sanitize_json_numbers(float("-inf")) is None
        assert sanitize_json_numbers(float("nan")) is None

    def test_finite_values_preserved_exactly(self):
        payload = {
            "int": 3,
            "zero": 0.0,
            "neg_zero": -0.0,
            "pi": 3.141592653589793,
            "bool": True,
            "none": None,
            "str": "Infinity inside strings is untouched",
            "image": "SW5maW5pdHk=/9j/4AAQ",  # base64 incl. 'Infinity' text
            "nested": {"list": [1.5, float("nan"), {"deep": float("inf")}]},
            "tuple": (2.5, float("-inf")),
        }
        out = sanitize_json_numbers(payload)
        assert out["int"] == 3
        assert out["zero"] == 0.0
        assert out["pi"] == 3.141592653589793
        assert out["bool"] is True
        assert out["none"] is None
        assert out["str"] == "Infinity inside strings is untouched"
        assert out["image"] == "SW5maW5pdHk=/9j/4AAQ"
        assert out["nested"] == {"list": [1.5, None, {"deep": None}]}
        assert out["tuple"] == [2.5, None]

    def test_idempotent(self):
        payload = {"a": float("inf"), "b": [1.0, {"c": float("nan")}]}
        once = sanitize_json_numbers(payload)
        assert sanitize_json_numbers(once) == once


@pytest.mark.asyncio
async def test_large_pointing_error_strict_json_wire():
    """Cold-start tick (huge error) arrives as strict JSON with null loss."""
    import main as main_module
    from fastapi.testclient import TestClient

    main_module.simulation_manager = SimulationManager()
    client = TestClient(app)
    with client.websocket_connect("/ws/simulation") as websocket:
        websocket.send_json({"command": "start"})
        assert websocket.receive_json()["type"] == "ack"

        telemetries = []
        frames = 0
        for _ in range(30):
            raw = websocket.receive_text()
            msg = strict_loads(raw)  # raises on any Infinity/NaN token
            if msg["type"] == "telemetry":
                telemetries.append(msg)
            elif msg["type"] == "frame":
                frames += 1
            if len(telemetries) >= 3 and frames >= 1:
                break

        assert len(telemetries) >= 3
        assert frames >= 1
        for tele in telemetries:
            link = tele["link"]
            assert set(link) == {
                "received_power_w", "pointing_loss_db",
                "atmospheric_loss_db", "link_margin_db",
                "snr_db", "link_state",
            }
            # Unbounded loss is null on the wire, never Infinity.
            assert link["pointing_loss_db"] is None
            # Finite link values pass through exactly.
            assert link["atmospheric_loss_db"] == 5.0
            assert link["snr_db"] == -120.0
            assert link["link_margin_db"] == -130.0
            assert link["link_state"] == "OUTAGE"
            assert link["received_power_w"] == 0.0

        websocket.send_json({"command": "stop"})
    await main_module.simulation_manager.stop()


@pytest.mark.asyncio
async def test_finite_link_values_unchanged_on_wire():
    """A strongly aligned tick keeps exact finite values end to end."""
    import main as main_module
    from fastapi.testclient import TestClient

    main_module.simulation_manager = SimulationManager()
    manager = main_module.simulation_manager
    manager.dt = 1e-6
    for name in ("turbulence", "vibration", "camera_motion", "sensor_noise"):
        manager.set_disturbance(name, 0.0)
    # Aim the gimbal at the upcoming tick so the link stays finite.
    t_az, t_el = manager.target.angular_position(manager.simulation_time + manager.dt)
    manager.pan_tilt.azimuth = math.degrees(t_az)
    manager.pan_tilt.elevation = math.degrees(t_el)

    client = TestClient(app)
    with client.websocket_connect("/ws/simulation") as websocket:
        websocket.send_json({"command": "start"})
        assert websocket.receive_json()["type"] == "ack"
        tele = None
        for _ in range(30):
            msg = strict_loads(websocket.receive_text())
            if msg["type"] == "telemetry":
                tele = msg
                break
        assert tele is not None
        link = tele["link"]
        assert link["pointing_loss_db"] is not None
        assert math.isfinite(link["pointing_loss_db"])
        assert math.isfinite(link["link_margin_db"])
        assert math.isfinite(link["snr_db"])
        websocket.send_json({"command": "stop"})
    await main_module.simulation_manager.stop()
