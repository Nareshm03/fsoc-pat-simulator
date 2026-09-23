"""
PAT (Point, Acquisition, and Tracking) State Machine.

States:
- IDLE: Simulation not running
- SEARCH: Looking for target (not detected)
- ACQUIRE: Target detected, large error, moving toward center
- FINE_TRACK: Target centered, small error, maintaining position
- LOCKED: Target locked, sustained low error, optimal tracking
- REACQUIRE: Lost target, attempting to reacquire

Transitions:
IDLE -> SEARCH (on start)
SEARCH -> ACQUIRE (target detected)
ACQUIRE -> FINE_TRACK (error < fine_track_threshold)
FINE_TRACK -> LOCKED (sustained error < lock_threshold for lock_duration)
LOCKED -> FINE_TRACK (error increases above lock_threshold)
FINE_TRACK/LOCKED -> REACQUIRE (target lost)
REACQUIRE -> SEARCH (after reacquire_timeout)
REACQUIRE -> ACQUIRE (target redetected)
"""

from enum import Enum
from typing import Optional


class PATState(Enum):
    """PAT system states."""
    IDLE = "IDLE"
    SEARCH = "SEARCH"
    ACQUIRE = "ACQUIRE"
    FINE_TRACK = "FINE_TRACK"
    LOCKED = "LOCKED"
    REACQUIRE = "REACQUIRE"


class PATStateMachine:
    """
    State machine for PAT system with proper state transitions
    and sustained lock requirements.
    """
    
    def __init__(self):
        """Initialize PAT state machine."""
        self.state = PATState.IDLE
        
        # Thresholds (pixels)
        self.acquire_threshold = 50.0  # Below this: ACQUIRE -> FINE_TRACK
        self.lock_threshold = 5.0      # Below this for lock_duration: FINE_TRACK -> LOCKED
        self.unlock_threshold = 10.0   # Above this: LOCKED -> FINE_TRACK
        self.reacquire_threshold = 100.0  # Above this after loss: trigger REACQUIRE
        
        # Timing
        self.lock_duration = 2.0  # Seconds of sustained low error to achieve LOCK
        self.reacquire_timeout = 3.0  # Seconds in REACQUIRE before returning to SEARCH
        
        # Internal state (all timestamps are simulation-time seconds)
        self._low_error_start_time: Optional[float] = None
        self._reacquire_start_time: Optional[float] = None
        self._last_detection_time: Optional[float] = None
        self._last_update_time: Optional[float] = None
        self._loss_count = 0
        
    def update(
        self,
        detected: bool,
        error_px: float,
        current_time: float,
        simulation_running: bool
    ) -> PATState:
        """
        Update state machine based on current conditions.
        
        Args:
            detected: Whether target is currently detected
            error_px: Tracking error in pixels
            current_time: Current simulation time (seconds)
            simulation_running: Whether simulation is active
            
        Returns:
            Current PAT state after update
        """
        self._last_update_time = current_time
        if not simulation_running:
            self.state = PATState.IDLE
            self._reset_timers()
            return self.state

        # Update last detection time
        if detected:
            self._last_detection_time = current_time
        
        # State machine logic
        if self.state == PATState.IDLE:
            if simulation_running:
                self.state = PATState.SEARCH
                self._reset_timers()
        
        elif self.state == PATState.SEARCH:
            if detected:
                self.state = PATState.ACQUIRE
                self._reset_timers()
        
        elif self.state == PATState.ACQUIRE:
            if not detected:
                self.state = PATState.REACQUIRE
                self._reacquire_start_time = current_time
                self._loss_count += 1
            elif error_px < self.acquire_threshold:
                self.state = PATState.FINE_TRACK
                self._low_error_start_time = None  # Reset lock timer
        
        elif self.state == PATState.FINE_TRACK:
            if not detected:
                self.state = PATState.REACQUIRE
                self._reacquire_start_time = current_time
                self._loss_count += 1
                self._low_error_start_time = None
            elif error_px >= self.acquire_threshold:
                # Error increased significantly
                self.state = PATState.ACQUIRE
                self._low_error_start_time = None
            elif error_px < self.lock_threshold:
                # Start/continue tracking time below lock threshold
                if self._low_error_start_time is None:
                    self._low_error_start_time = current_time
                elif (current_time - self._low_error_start_time) >= self.lock_duration:
                    # Sustained low error achieved!
                    self.state = PATState.LOCKED
        
        elif self.state == PATState.LOCKED:
            if not detected:
                self.state = PATState.REACQUIRE
                self._reacquire_start_time = current_time
                self._loss_count += 1
                self._low_error_start_time = None
            elif error_px >= self.unlock_threshold:
                # Error increased, drop back to fine tracking
                self.state = PATState.FINE_TRACK
                self._low_error_start_time = None
        
        elif self.state == PATState.REACQUIRE:
            if detected:
                # Reacquired!
                self.state = PATState.ACQUIRE
                self._reacquire_start_time = None
            elif self._reacquire_start_time is not None:
                elapsed = current_time - self._reacquire_start_time
                if elapsed >= self.reacquire_timeout:
                    # Timeout, go back to search
                    self.state = PATState.SEARCH
                    self._reacquire_start_time = None
        
        return self.state
    
    def _reset_timers(self):
        """Reset internal timers."""
        self._low_error_start_time = None
        self._reacquire_start_time = None
    
    def reset(self):
        """Reset state machine to initial state."""
        self.state = PATState.IDLE
        self._reset_timers()
        self._last_detection_time = None
        self._last_update_time = None
        self._loss_count = 0
    
    def get_control_mode(self) -> str:
        """
        Get control mode for PAT controller based on state.
        
        Returns:
            Control mode string: 'search', 'acquire', 'track', 'locked'
        """
        if self.state in (PATState.IDLE, PATState.SEARCH, PATState.REACQUIRE):
            return 'search'
        elif self.state == PATState.ACQUIRE:
            return 'acquire'
        elif self.state == PATState.FINE_TRACK:
            return 'track'
        elif self.state == PATState.LOCKED:
            return 'locked'
        return 'idle'
    
    def get_stats(self) -> dict:
        """Get statistics about state machine.

        All times are simulation-time seconds (matching the
        ``current_time`` passed to :meth:`update`), never wall-clock.
        """
        if (
            self._low_error_start_time is not None
            and self._last_update_time is not None
        ):
            time_below = self._last_update_time - self._low_error_start_time
        else:
            time_below = 0.0
        return {
            "current_state": self.state.value,
            "loss_count": self._loss_count,
            "time_below_lock_threshold": time_below,
        }
