"""
YOLO11n closed-loop PAT integration test.

Proves the complete automatic feedback loop with the existing simulator:
  YOLO11 detection -> beacon center -> pointing error -> PAT controller
  -> pan/tilt movement -> changed camera projection -> new frame
  -> YOLO detection again.

Uses unmodified PAT physics/control (same gains, thresholds, FOV).
Stationary target, ideal conditions (zero disturbances) - same setup as
the Classical baseline (test_pat_acquisition.py), so thresholds are NOT
weakened to force success.
"""
import math
import pytest

from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector
from simulation.target import Target
from simulation.pan_tilt import PanTilt
from simulation.pat_controller import PATController
from simulation.pat_state_machine import PATStateMachine
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

# Stationary target - same offset as Classical baseline test.
TARGET_AZ_DEG = 5.0
TARGET_EL_DEG = -8.0
DT = 1.0 / 30.0


def run_closed_loop(detector, ticks=150):
    """Run automatic PAT feedback loop with the given detector.

    Returns metrics dict. Uses existing Target/Camera/PAT/PanTilt only -
    no manual gimbal movement; all motion comes from PAT commands.
    """
    target = Target(width=1280, height=720)
    camera = VirtualCamera(width=1280, height=720)
    pan_tilt = PanTilt()
    controller = PATController()
    state_machine = PATStateMachine()

    target_az = math.radians(TARGET_AZ_DEG)
    target_el = math.radians(TARGET_EL_DEG)

    sim_time = 0.0
    errors, confidences, commands = [], [], []
    pixel_positions, gimbal_history, states = [], [], []
    detections = 0

    for _ in range(ticks):
        sim_time += DT
        gaz = math.radians(pan_tilt.azimuth)
        gel = math.radians(pan_tilt.elevation)

        # Changed camera projection from gimbal movement (existing physics).
        pos = target.pixel_from_angles(target_az, target_el, gaz, gel)
        assert pos is not None, "Stationary target must stay in FOV"
        pixel_positions.append(pos)

        # New frame -> detection again (same interface for both detectors).
        frame = camera.render(*pos)
        detection = detector.detect(frame)

        if detection is not None:
            detected = True
            dx, dy = detection["x"], detection["y"]
            conf = float(detection["confidence"])
            err = math.hypot(dx - camera.width / 2, dy - camera.height / 2)
            detections += 1
            errors.append(err)
            confidences.append(conf)
        else:
            detected = False
            dx, dy, conf, err = 0.0, 0.0, 0.0, 0.0

        # Pointing error -> PAT controller -> pan/tilt (automatic, unmodified).
        pat_state = state_machine.update(
            detected=detected, error_px=err,
            current_time=sim_time, simulation_running=True,
        )
        mode = state_machine.get_control_mode()
        az_cmd, el_cmd = controller.compute_command(
            detected=detected, beacon_x=dx, beacon_y=dy,
            control_mode=mode, dt=DT,
        )
        commands.append((az_cmd, el_cmd))
        pan_tilt.update(az_cmd, el_cmd, DT)
        gimbal_history.append((pan_tilt.azimuth, pan_tilt.elevation))
        states.append(pat_state.value)

    return {
        "errors": errors,
        "confidences": confidences,
        "commands": commands,
        "pixel_positions": pixel_positions,
        "gimbal_history": gimbal_history,
        "states": states,
        "detections": detections,
        "ticks": ticks,
        "final_gimbal": gimbal_history[-1],
    }


class TestYoloClosedLoop:
    """YOLO11n in the full automatic PAT feedback loop."""

    @requires_yolo
    def test_yolo_detector_is_active(self):
        """YOLO detector is active through the existing SimulationManager."""
        from main import SimulationManager

        manager = SimulationManager()
        assert manager.detector_type == "classical"  # default unchanged
        info = manager.set_detector("yolo")
        assert info["active"] == "yolo"
        assert manager.detector_type == "yolo"
        assert manager.detector.is_available, (
            f"YOLO not active: {manager.detector.load_error}"
        )
        frame = manager.camera.render(640, 360)
        detection = manager.detector.detect(frame)
        assert detection is not None
        for field in ("x", "y", "confidence", "bbox"):
            assert field in detection

    @requires_yolo
    def test_yolo_closed_loop_converges(self):
        """Full loop: YOLO -> error -> PAT -> gimbal -> new projection -> YOLO."""
        from main import SimulationManager

        # Prove integration through the manager's own components/detector.
        manager = SimulationManager()
        manager.set_detector("yolo")
        assert manager.detector_type == "yolo"
        yolo = manager.detector
        assert yolo.is_available

        m = run_closed_loop(yolo, ticks=150)

        errors = m["errors"]
        assert m["detections"] == m["ticks"], (
            f"YOLO should detect every frame, got {m['detections']}/{m['ticks']}"
        )
        assert len(errors) == m["ticks"]

        initial_error = errors[0]
        final_error = errors[-1]
        min_error = min(errors)
        avg_conf = sum(m["confidences"]) / len(m["confidences"])
        first_px = m["pixel_positions"][0]
        last_px = m["pixel_positions"][-1]
        px_shift = math.hypot(last_px[0] - first_px[0], last_px[1] - first_px[1])
        final_gimbal = m["final_gimbal"]
        gimbal_move = math.hypot(
            final_gimbal[0] - 0.0, final_gimbal[1] - 0.0
        )
        nonzero_cmds = sum(
            1 for azc, elc in m["commands"] if abs(azc) > 1e-9 or abs(elc) > 1e-9
        )

        print("\n=== YOLO11n Closed-Loop Results (stationary target, ideal) ===")
        print(f"Target: az={TARGET_AZ_DEG}deg el={TARGET_EL_DEG}deg | "
              f"frames/iterations: {m['detections']}/{m['ticks']}")
        print(f"Initial error: {initial_error:.2f}px | Final error: {final_error:.2f}px | "
              f"Min error: {min_error:.2f}px")
        print(f"Convergence: {initial_error:.2f} -> {final_error:.2f} "
              f"(x{initial_error / max(final_error, 1e-9):.1f} reduction)")
        print(f"Gimbal movement: (0.00,0.00) -> ({final_gimbal[0]:+.2f},{final_gimbal[1]:+.2f}) "
              f"(|move|={gimbal_move:.2f}deg)")
        print(f"Beacon pixel shift from gimbal: {px_shift:.1f}px "
              f"({first_px[0]:.1f},{first_px[1]:.1f}) -> ({last_px[0]:.1f},{last_px[1]:.1f})")
        print(f"Detection confidence: avg={avg_conf:.3f} min={min(m['confidences']):.3f}")
        print(f"Automatic PAT commands nonzero: {nonzero_cmds}/{m['ticks']}")
        print(f"States seen: {sorted(set(m['states']))}")

        # 1. Gimbal movement changes beacon pixel position (projection feedback).
        assert px_shift > 10.0, f"Gimbal must shift projection, got {px_shift:.1f}px"
        # 2. PAT error changes accordingly and converges (same bars as Classical).
        assert initial_error > 100.0, f"Initial error must be large, got {initial_error:.2f}"
        assert final_error < 8.0, f"Final error must converge <8px, got {final_error:.2f}"
        assert min_error < 5.0, f"Min error must reach <5px, got {min_error:.2f}"
        assert final_error < initial_error, "Error must decrease"
        assert m["errors"][len(errors) // 2] < initial_error, "Mid-run error must improve"
        # 3. Automatic pan/tilt without manual movement.
        assert nonzero_cmds > 0, "PAT must issue automatic gimbal commands"
        assert gimbal_move > 1.0, f"Gimbal must move automatically, got {gimbal_move:.2f}deg"
        assert abs(final_gimbal[0] - TARGET_AZ_DEG) < 2.0, (
            f"Gimbal az must approach target: {final_gimbal[0]:.2f} vs {TARGET_AZ_DEG}"
        )
        assert abs(final_gimbal[1] - TARGET_EL_DEG) < 2.0, (
            f"Gimbal el must approach target: {final_gimbal[1]:.2f} vs {TARGET_EL_DEG}"
        )
        # 4. Detection quality through the loop.
        assert avg_conf > 0.5, f"Avg confidence too low: {avg_conf:.3f}"
        for c in m["confidences"]:
            assert 0.0 <= c <= 1.0

    def test_classical_loop_unchanged(self):
        """Same harness with Classical CV still converges (no regression)."""
        classical = BeaconDetector()
        m = run_closed_loop(classical, ticks=150)
        errors = m["errors"]
        initial_error = errors[0]
        final_error = errors[-1]
        min_error = min(errors)
        final_gimbal = m["final_gimbal"]
        print("\n=== Classical CV Closed-Loop Reference (unchanged) ===")
        print(f"frames/iterations: {m['detections']}/{m['ticks']}")
        print(f"Initial error: {initial_error:.2f}px | Final error: {final_error:.2f}px | "
              f"Min error: {min_error:.2f}px")
        print(f"Gimbal: ({final_gimbal[0]:+.2f},{final_gimbal[1]:+.2f})")
        assert m["detections"] == m["ticks"]
        assert initial_error > 100.0
        assert final_error < 8.0
        assert min_error < 5.0
        assert abs(final_gimbal[0] - TARGET_AZ_DEG) < 2.0
        assert abs(final_gimbal[1] - TARGET_EL_DEG) < 2.0
