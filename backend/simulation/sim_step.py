"""Shared pure closed-loop simulation step (no I/O, no globals).

Performs the existing tick sequence exactly once, in the existing order:

    target geometry -> disturbance offsets -> render -> detect -> error ->
    PAT state machine -> search/PID control -> gimbal update ->
    link budget -> telemetry record fields.

This is a pure orchestration helper: every algorithm (projection,
rendering, detection, tracking, PAT control, gimbal physics, link
budget) is executed on the caller-supplied component objects. The only
state mutated is that of the passed-in components (gimbal angles,
controller integrals, tracker history, state-machine timers), exactly as
when callers inlined this sequence. No file, network, or logging I/O is
performed here, and no randomness is introduced (sensor noise comes from
the caller's DisturbanceModel, preserving its exact consumption order).

Per-caller behavior contract (all preserved bit-for-bit):

    - main.SimulationManager / mission_runner.tick:
        use_raster_search=True, feedforward=True, tracker given,
        link_config given, detect_when_out_of_fov=True.
        Search mode runs controller.reset() + RasterSearch.command;
        PID modes receive the measured gimbal rates (feedforward).
    - experiment_runner loop / benchmark run_scenario:
        use_raster_search=False, feedforward=False, tracker=None,
        link_config=None, detect_when_out_of_fov=False.
        Search/undetected ticks fall through to
        PATController.compute_command, whose internal hold-then-sweep
        search applies; out-of-FOV ticks skip rendering entirely.
    - mission_runner.tick additionally passes blank=True during forced
        loss: the frame is genuinely empty and a real detection on it
        raises RuntimeError (never faked, never silently accepted).

Callers keep ownership of timekeeping (sim_time is an input, never
advanced here), record-schema construction, broadcasting, and storage.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from .link_budget import LinkBudgetConfig, link_budget


def simulation_step(
    *,
    target: Any,
    target_override: Optional[Tuple[float, float]],
    disturbances: Any,
    camera: Any,
    detector: Any,
    tracker: Optional[Any],
    pan_tilt: Any,
    controller: Any,
    state_machine: Any,
    search: Optional[Any],
    link_config: Optional[LinkBudgetConfig],
    sim_time: float,
    dt: float,
    simulation_running: bool = True,
    blank: bool = False,
    detect_when_out_of_fov: bool = False,
    use_raster_search: bool = False,
    feedforward: bool = False,
) -> Dict[str, Any]:
    """Execute one closed-loop tick; return all telemetry fields.

    Args:
        target: Target instance (FOV geometry + angular_position).
        target_override: (az_rad, el_rad) to use instead of
            target.angular_position(sim_time), or None.
        disturbances: DisturbanceModel for this tick.
        camera: VirtualCamera instance.
        detector: detector instance with detect(frame).
        tracker: Tracker instance or None (skips tracking update).
        pan_tilt: PanTilt instance (read, then updated in place).
        controller: PATController instance.
        state_machine: PATStateMachine instance.
        search: RasterSearch instance (required if use_raster_search).
        link_config: LinkBudgetConfig or None (skips link evaluation).
        sim_time: current simulation time in seconds (not advanced here).
        dt: seconds per tick.
        simulation_running: forwarded to the state machine.
        blank: force a genuinely empty frame (forced-loss scripting).
        detect_when_out_of_fov: render (-1, -1) and run the real detector
            when the target is out of FOV/blanked; otherwise report no
            detection without rendering.
        use_raster_search: in search mode run controller.reset() +
            RasterSearch.command instead of compute_command.
        feedforward: pass measured gimbal rates to compute_command.

    Returns:
        Dict with frame, detection, detected, dx, dy, confidence,
        error_px, ground_truth_xy (or None when out of FOV), tracking
        (or None), pat_state (PATState enum), control_mode, az/el rate
        commands, post-update gimbal angles/velocities, target angles,
        pointing_error_rad, and link (or None).
    """
    if use_raster_search and search is None:
        raise ValueError("use_raster_search requires a search instance")

    # 1. Target angular position (absolute in space).
    if target_override is not None:
        target_az_rad, target_el_rad = target_override
    else:
        target_az_rad, target_el_rad = target.angular_position(sim_time)

    # 2. Gimbal pointing at tick start (closed loop: gimbal affects view).
    gimbal_az_rad = math.radians(pan_tilt.azimuth)
    gimbal_el_rad = math.radians(pan_tilt.elevation)

    # 3. Pointing error + optical link budget from the identical geometry
    # (PAT state plays no role in the link evaluation).
    pointing_error_rad = math.hypot(
        target_az_rad - gimbal_az_rad,
        target_el_rad - gimbal_el_rad,
    )
    link = link_budget(pointing_error_rad, link_config) \
        if link_config is not None else None

    # 4. Pixel projection, disturbances (same order as the inlined loops:
    # offsets are consumed only when the target projects in-FOV and the
    # frame is not forced blank).
    pixel_pos = target.pixel_from_angles(
        target_az_rad, target_el_rad,
        gimbal_az_rad, gimbal_el_rad,
    )

    if pixel_pos is not None and not blank:
        beacon_x, beacon_y = pixel_pos
        turb_x, turb_y = disturbances.turbulence_offset(sim_time)
        vib_x, vib_y = disturbances.vibration_offset(sim_time)
        cam_x, cam_y = disturbances.camera_offset(sim_time)
        noise_x, noise_y = disturbances.sensor_offset()
        beacon_x += turb_x + vib_x + cam_x + noise_x
        beacon_y += turb_y + vib_y + cam_y + noise_y
        ground_truth_xy: Optional[Tuple[float, float]] = pixel_pos
        frame = camera.render(beacon_x, beacon_y)
        detection = detector.detect(frame)
    else:
        ground_truth_xy = None
        if detect_when_out_of_fov:
            # Genuinely empty frame through the real detector (never faked).
            frame = camera.render(-1, -1)
            detection = detector.detect(frame)
            if blank and detection is not None:
                raise RuntimeError("blank frame produced a detection")
            detection = None
        else:
            frame = None
            detection = None

    # 5. Detection unpack + tracking error from camera center.
    if detection:
        detected = True
        dx = detection["x"]
        dy = detection["y"]
        confidence = detection["confidence"]
    else:
        detected = False
        dx, dy = 0.0, 0.0
        confidence = 0.0
        detection = None
    if detected:
        error_px = math.hypot(dx - camera.width / 2, dy - camera.height / 2)
    else:
        error_px = 0.0

    # 6. Tracker (PAT/tracker consume x/y/confidence only).
    tracking = tracker.update(detection) if tracker is not None else None

    # 7. PAT state machine.
    pat_state = state_machine.update(
        detected=detected,
        error_px=error_px,
        current_time=sim_time,
        simulation_running=simulation_running,
    )
    control_mode = state_machine.get_control_mode()

    # 8. Control: raster sweep in search mode, else PID (+feedforward).
    if use_raster_search and control_mode == "search":
        controller.reset()
        az_rate_cmd, el_rate_cmd = search.command(
            pan_tilt.azimuth,
            pan_tilt.elevation,
            sim_time,
            dt,
        )
    else:
        az_rate_cmd, el_rate_cmd = controller.compute_command(
            detected=detected,
            beacon_x=dx,
            beacon_y=dy,
            control_mode=control_mode,
            dt=dt,
            gimbal_az_rate_deg_s=pan_tilt.azimuth_velocity
            if feedforward else None,
            gimbal_el_rate_deg_s=pan_tilt.elevation_velocity
            if feedforward else None,
        )

    # 9. Gimbal update (closed loop: affects the next tick's view).
    pan_tilt.update(az_rate_cmd, el_rate_cmd, dt)

    return {
        "sim_time": sim_time,
        "target_az_rad": target_az_rad,
        "target_el_rad": target_el_rad,
        "pointing_error_rad": pointing_error_rad,
        "link": link,
        "ground_truth_xy": ground_truth_xy,
        "frame": frame,
        "detection": detection,
        "detected": detected,
        "dx": dx,
        "dy": dy,
        "confidence": confidence,
        "error_px": error_px,
        "tracking": tracking,
        "pat_state": pat_state,
        "control_mode": control_mode,
        "az_rate_cmd": az_rate_cmd,
        "el_rate_cmd": el_rate_cmd,
        "gimbal_az_deg": pan_tilt.azimuth,
        "gimbal_el_deg": pan_tilt.elevation,
        "gimbal_az_vel": pan_tilt.azimuth_velocity,
        "gimbal_el_vel": pan_tilt.elevation_velocity,
    }
