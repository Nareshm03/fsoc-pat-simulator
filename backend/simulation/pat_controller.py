"""
Point, Acquisition, and Tracking (PAT) controller with proper PID control.
"""
import logging
from typing import Tuple


class PATController:
    """
    PID-based PAT controller for gimbal pointing.
    
    Converts pixel error to angular velocity commands with proper gains.
    """
    
    def __init__(self):
        """Initialize PAT controller."""
        self.logger = logging.getLogger(__name__)
        
        # PID gains for different modes
        # Mode: 'acquire' - aggressive, fast convergence
        self.gains_acquire = {
            'kp': 0.10,   # Proportional: pixel error -> deg/sec (increased from 0.03)
            'ki': 0.002,  # Integral: accumulated error
            'kd': 0.02    # Derivative: rate of change (increased from 0.005)
        }
        
        # Mode: 'track' - smooth, stable
        self.gains_track = {
            'kp': 0.06,  # Increased from 0.02
            'ki': 0.001,
            'kd': 0.025  # Increased from 0.008
        }
        
        # Mode: 'locked' - very gentle, minimize overshoot
        self.gains_locked = {
            'kp': 0.03,  # Increased from 0.01
            'ki': 0.0005,
            'kd': 0.03   # Increased from 0.01
        }
        
        # Control state
        self.error_integral_x = 0.0
        self.error_integral_y = 0.0
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0

        # Anti-windup limits
        self.max_integral = 500.0  # pixels * seconds

        # Rate limits
        # 15 deg/s: the moving target peaks at 10 deg/s in azimuth, so a
        # 10 deg/s cap saturates exactly when trim headroom is needed most
        # (telemetry: 18-31% command saturation after feedforward). 15 gives
        # trim headroom while staying 2x below the PanTilt hardware limits
        # (30/20 deg/s), which still clamp actual motion.
        self.max_rate = 15.0  # deg/sec

        # Camera parameters
        self.camera_width = 1280
        self.camera_height = 720

        # Measured-rate velocity feedforward (ramp-tracking fix).
        # Telemetry evidence (see instrumented runs): with pure feedback the
        # steady-state lag err ~= target_rate / kp reaches ~100-170 px at the
        # target's 10 deg/s azimuth peak (track kp=0.06 -> 167 px), so LOCKED
        # (5 px sustained) is unreachable despite 100% detection, and commands
        # saturate at max_rate. The fix adds the *measured* target angular
        # rate to the command in the precision modes ('track', 'locked'):
        #   w_tgt = w_gimbal_measured + (beacon_px_velocity / px_per_deg)
        # using only real detections + the measured gimbal rate supplied by
        # the caller. No predicted/faked detections: without a previous real
        # detection (or without caller-supplied gimbal rates) the term is 0
        # and behavior is exactly the legacy PID.
        # Filter sizing (evidence-based sweep of alpha in 0.05..0.8 on
        # identical moving and moving+disturbed scenarios): alpha=0.5
        # maximizes lock retention in both (0.53 / 0.12) with stationary
        # bit-identical. Lower alpha starves accel response (filter lag bias
        # ~= tau*accel grows to several deg/s); higher alpha increasingly
        # chases the aliased 6-7 Hz vibration component and disturbed mean
        # error rises again. tau ~= dt*(1-alpha)/alpha ~= 0.03 s, residual
        # rate noise ~0.16 deg/s -> ~3 px trim in track mode.
        self.ff_gain = 1.0    # full measured-rate feedforward
        self.ff_alpha = 0.5   # EWMA smoothing for detector quantization noise
        self._ff_waz = 0.0    # filtered target-rate estimate, deg/s
        self._ff_wel = 0.0
        self._ff_prev_x = None  # previous real detection (None = no estimate)
        self._ff_prev_y = None

        # Deterministic bounded raster/sweep search (used when beacon lost).
        # Time-based triangular (zig-zag) scan: fast azimuth sweep combined
        # with a slow elevation drift. Velocity integration by the gimbal
        # yields a bounded excursion around the loss point:
        #   az amplitude  = az_rate  * az_period  / 4  = 10 deg
        #   el amplitude  = el_rate  * el_period  / 4  =  6 deg
        # Both rates are below max_rate (10 deg/s) and below the PanTilt
        # stage limits (30/20 deg/s); the PanTilt position clamp additionally
        # enforces gimbal limits. No randomness: pure function of accumulated
        # undetected time.
        self.search_hold_s = 0.1    # dwell (hold position) before sweeping
        self.search_az_rate = 5.0   # deg/s, azimuth sweep speed
        self.search_az_period = 8.0  # s, full back-and-forth azimuth cycle
        self.search_el_rate = 0.8   # deg/s, elevation drift speed
        self.search_el_period = 30.0  # s, full up-down elevation cycle
        self._search_time = 0.0     # accumulated continuous undetected time (s)

    def reset(self):
        """Reset controller state (clear integral and derivative terms)."""
        self.error_integral_x = 0.0
        self.error_integral_y = 0.0
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self._search_time = 0.0
        self._ff_waz = 0.0
        self._ff_wel = 0.0
        self._ff_prev_x = None
        self._ff_prev_y = None
        self.logger.info("PAT controller reset")

    def _reset_pid_only(self):
        """Clear PID integrals/derivatives but preserve search clock."""
        self.error_integral_x = 0.0
        self.error_integral_y = 0.0
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        # A stale rate estimate after a dropout would corrupt reacquisition,
        # and there is no measured motion without real detections.
        self._ff_waz = 0.0
        self._ff_wel = 0.0
        self._ff_prev_x = None
        self._ff_prev_y = None

    @property
    def _px_per_deg_az(self) -> float:
        return self.camera_width / 40.0  # FOV horizontal, deg

    @property
    def _px_per_deg_el(self) -> float:
        return self.camera_height / 30.0  # FOV vertical, deg

    def _update_ff_estimate(
        self,
        beacon_x: float,
        beacon_y: float,
        dt: float,
        gimbal_az_rate_deg_s,
        gimbal_el_rate_deg_s,
    ) -> None:
        """Fold one real detection into the filtered target-rate estimate.

        w_tgt_az = w_gimbal + (dx_px/dt)/px_per_deg  (pixel_x grows with rel_az)
        w_tgt_el = w_gimbal - (dy_px/dt)/px_per_deg  (pixel_y falls with rel_el)
        """
        if (
            dt is None or dt <= 0
            or gimbal_az_rate_deg_s is None or gimbal_el_rate_deg_s is None
            or self._ff_prev_x is None or self._ff_prev_y is None
        ):
            return
        raw_waz = gimbal_az_rate_deg_s + (beacon_x - self._ff_prev_x) / dt / self._px_per_deg_az
        raw_wel = gimbal_el_rate_deg_s - (beacon_y - self._ff_prev_y) / dt / self._px_per_deg_el
        self._ff_waz += self.ff_alpha * (raw_waz - self._ff_waz)
        self._ff_wel += self.ff_alpha * (raw_wel - self._ff_wel)

    @staticmethod
    def _triangle_rate(t: float, rate: float, period: float) -> float:
        """Piecewise-constant velocity of a symmetric triangle wave.

        Position integrates to a triangle bounded by +/- rate*period/4.
        Deterministic function of time only.
        """
        phase = (t % period) / period  # [0, 1)
        if phase < 0.25 or phase >= 0.75:
            return rate
        return -rate

    def _search_command(self) -> Tuple[float, float]:
        """Velocity command for the current accumulated search time."""
        t_scan = self._search_time - self.search_hold_s
        if t_scan <= 0.0:
            # Dwell: hold last position briefly to ride out single-frame
            # dropouts without sweeping.
            return 0.0, 0.0
        az = self._triangle_rate(t_scan, self.search_az_rate, self.search_az_period)
        el = self._triangle_rate(t_scan, self.search_el_rate, self.search_el_period)
        # Enforce controller rate limits (already within, clamp for safety).
        az = max(-self.max_rate, min(self.max_rate, az))
        el = max(-self.max_rate, min(self.max_rate, el))
        return az, el

    def get_search_info(self) -> dict:
        """Report deterministic search state (for tests/telemetry)."""
        t_scan = max(0.0, self._search_time - self.search_hold_s)
        return {
            "search_time": self._search_time,
            "scan_time": t_scan,
            "scanning": t_scan > 0.0,
            "az_amplitude_deg": self.search_az_rate * self.search_az_period / 4.0,
            "el_amplitude_deg": self.search_el_rate * self.search_el_period / 4.0,
        }
    
    def compute_command(
        self,
        detected: bool,
        beacon_x: float,
        beacon_y: float,
        control_mode: str,
        dt: float,
        gimbal_az_rate_deg_s=None,
        gimbal_el_rate_deg_s=None,
    ) -> Tuple[float, float]:
        """
        Compute gimbal angular velocity commands.

        Args:
            detected: Whether beacon is detected
            beacon_x: Detected beacon X position (pixels)
            beacon_y: Detected beacon Y position (pixels)
            control_mode: Control mode ('search', 'acquire', 'track', 'locked')
            dt: Time step (seconds)
            gimbal_az_rate_deg_s: Measured gimbal az rate (deg/s, optional).
                Enables measured-rate feedforward in 'track'/'locked'.
                None (default) preserves exact legacy PID behavior.
            gimbal_el_rate_deg_s: Measured gimbal el rate (deg/s, optional).

        Returns:
            (azimuth_rate_cmd, elevation_rate_cmd) in degrees/sec

        Notes:
            - When the beacon is genuinely undetected, a deterministic
              bounded raster/sweep is generated (see _search_command).
              Detection itself is never faked: this branch only runs when
              detected is False.
            - PID/threshold behavior for detected beacons is unchanged;
              feedforward only *adds* the measured target rate in the
              precision modes and is zero without measured data.
        """
        if not detected:
            # Genuine loss: hold briefly, then bounded raster scan.
            # Clear PID terms (avoid windup) but keep the search clock so
            # the sweep is continuous across ticks.
            self._reset_pid_only()
            if dt is not None and dt > 0:
                self._search_time += dt
            else:
                return 0.0, 0.0
            # Any genuine loss scans, regardless of the nominal mode label
            # (SEARCH and REACQUIRE both map to 'search' via the state
            # machine). Detection is never faked here.
            return self._search_command()

        # Beacon detected (real measurement): reset search clock and track.
        # If the state machine still reports 'search' on this exact tick
        # (e.g. IDLE->SEARCH transition), fall back to acquire gains rather
        # than ignoring the real detection.
        self._search_time = 0.0

        # Measured-rate feedforward state update (real detections only).
        self._update_ff_estimate(
            beacon_x, beacon_y, dt,
            gimbal_az_rate_deg_s, gimbal_el_rate_deg_s,
        )
        if control_mode in ('track', 'locked'):
            ff_az = self.ff_gain * self._ff_waz
            ff_el = self.ff_gain * self._ff_wel
        else:
            ff_az, ff_el = 0.0, 0.0
        self._ff_prev_x = beacon_x
        self._ff_prev_y = beacon_y

        # Calculate error from camera center
        center_x = self.camera_width / 2.0
        center_y = self.camera_height / 2.0
        
        error_x = beacon_x - center_x  # Positive = beacon to the right
        error_y = beacon_y - center_y  # Positive = beacon below center
        
        # Select gains based on mode
        if control_mode == 'acquire':
            gains = self.gains_acquire
        elif control_mode == 'track':
            gains = self.gains_track
        elif control_mode == 'locked':
            gains = self.gains_locked
        else:
            gains = self.gains_acquire
        
        # PID control for X (azimuth)
        # If beacon is RIGHT (+error_x), move gimbal RIGHT (+cmd) to shift view right
        p_x = gains['kp'] * error_x
        
        self.error_integral_x += error_x * dt
        self.error_integral_x = max(-self.max_integral, min(self.max_integral, self.error_integral_x))
        i_x = gains['ki'] * self.error_integral_x
        
        d_x = gains['kd'] * (error_x - self.previous_error_x) / dt if dt > 0 else 0.0
        
        az_rate_cmd = p_x + i_x + d_x
        
        # PID control for Y (elevation)
        # IMPORTANT: Different sign due to the minus in pixel_y = center - rel_el * scale
        # If beacon is DOWN (+error_y), we need rel_el to become more negative
        # which means gimbal_el must DECREASE (move UP), so command is NEGATIVE
        p_y = -gains['kp'] * error_y  # Negative feedback for elevation
        
        self.error_integral_y += error_y * dt
        self.error_integral_y = max(-self.max_integral, min(self.max_integral, self.error_integral_y))
        i_y = -gains['ki'] * self.error_integral_y  # Negative feedback
        
        d_y = -gains['kd'] * (error_y - self.previous_error_y) / dt if dt > 0 else 0.0  # Negative feedback
        
        el_rate_cmd = p_y + i_y + d_y

        # Measured-rate feedforward: carry the tracked target rate so the
        # PID only corrects the residual (fixes velocity lag).
        az_rate_cmd += ff_az
        el_rate_cmd += ff_el

        # Apply rate limits
        az_rate_cmd = max(-self.max_rate, min(self.max_rate, az_rate_cmd))
        el_rate_cmd = max(-self.max_rate, min(self.max_rate, el_rate_cmd))
        
        # Update previous error
        self.previous_error_x = error_x
        self.previous_error_y = error_y
        
        return az_rate_cmd, el_rate_cmd
    
    def set_gains(self, mode: str, kp: float, ki: float, kd: float):
        """
        Set PID gains for a specific mode.
        
        Args:
            mode: Control mode ('acquire', 'track', 'locked')
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        if mode == 'acquire':
            self.gains_acquire = {'kp': kp, 'ki': ki, 'kd': kd}
        elif mode == 'track':
            self.gains_track = {'kp': kp, 'ki': ki, 'kd': kd}
        elif mode == 'locked':
            self.gains_locked = {'kp': kp, 'ki': ki, 'kd': kd}
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.logger.info(f"PID gains for {mode} set: Kp={kp}, Ki={ki}, Kd={kd}")

