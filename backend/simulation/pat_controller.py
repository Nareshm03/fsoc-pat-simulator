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
        self.max_rate = 10.0  # deg/sec
        
        # Camera parameters
        self.camera_width = 1280
        self.camera_height = 720
        
    def reset(self):
        """Reset controller state (clear integral and derivative terms)."""
        self.error_integral_x = 0.0
        self.error_integral_y = 0.0
        self.previous_error_x = 0.0
        self.previous_error_y = 0.0
        self.logger.info("PAT controller reset")
    
    def compute_command(
        self,
        detected: bool,
        beacon_x: float,
        beacon_y: float,
        control_mode: str,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute gimbal angular velocity commands.
        
        Args:
            detected: Whether beacon is detected
            beacon_x: Detected beacon X position (pixels)
            beacon_y: Detected beacon Y position (pixels)
            control_mode: Control mode ('search', 'acquire', 'track', 'locked')
            dt: Time step (seconds)
            
        Returns:
            (azimuth_rate_cmd, elevation_rate_cmd) in degrees/sec
        """
        if not detected or control_mode == 'search':
            # No detection or searching - no correction
            # Could implement search pattern here
            self.reset()
            return 0.0, 0.0
        
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

