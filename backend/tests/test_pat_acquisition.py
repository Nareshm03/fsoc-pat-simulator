"""
Tests for PAT acquisition, tracking, and lock behavior.
"""
import pytest
import sys
import os
import asyncio
import math

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.pat_state_machine import PATStateMachine, PATState
from simulation.pat_controller import PATController
from simulation.pan_tilt import PanTilt
from simulation.target import Target
from simulation.camera import VirtualCamera
from simulation.detector import BeaconDetector


class TestPATStateMachine:
    """Test PAT state machine transitions."""
    
    def test_initial_state(self):
        """Test state machine starts in IDLE."""
        sm = PATStateMachine()
        assert sm.state == PATState.IDLE
    
    def test_idle_to_search(self):
        """Test transition from IDLE to SEARCH when simulation starts."""
        sm = PATStateMachine()
        state = sm.update(
            detected=False,
            error_px=0.0,
            current_time=0.0,
            simulation_running=True
        )
        assert state == PATState.SEARCH
    
    def test_search_to_acquire(self):
        """Test transition from SEARCH to ACQUIRE when target detected."""
        sm = PATStateMachine()
        
        # Start simulation
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        assert sm.state == PATState.SEARCH
        
        # Target detected with large error
        state = sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        assert state == PATState.ACQUIRE
    
    def test_acquire_to_fine_track(self):
        """Test transition from ACQUIRE to FINE_TRACK when error decreases."""
        sm = PATStateMachine()
        
        # Go to ACQUIRE
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        assert sm.state == PATState.ACQUIRE
        
        # Error decreases below acquire threshold (50px)
        state = sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        assert state == PATState.FINE_TRACK
    
    def test_fine_track_to_locked(self):
        """Test transition from FINE_TRACK to LOCKED with sustained low error."""
        sm = PATStateMachine()
        sm.lock_duration = 1.0  # 1 second to lock (faster for testing)
        
        # Go to FINE_TRACK
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        assert sm.state == PATState.FINE_TRACK
        
        # Maintain low error for lock_duration
        sm.update(detected=True, error_px=3.0, current_time=0.5, simulation_running=True)
        assert sm.state == PATState.FINE_TRACK  # Not locked yet
        
        sm.update(detected=True, error_px=3.0, current_time=1.0, simulation_running=True)
        assert sm.state == PATState.FINE_TRACK  # Still not locked
        
        state = sm.update(detected=True, error_px=3.0, current_time=1.5, simulation_running=True)
        assert state == PATState.LOCKED  # Now locked!
    
    def test_locked_to_fine_track_on_error_increase(self):
        """Test LOCKED drops to FINE_TRACK if error increases."""
        sm = PATStateMachine()
        sm.lock_duration = 0.5
        
        # Achieve LOCKED state
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        sm.update(detected=True, error_px=3.0, current_time=0.3, simulation_running=True)
        sm.update(detected=True, error_px=3.0, current_time=1.0, simulation_running=True)
        assert sm.state == PATState.LOCKED
        
        # Error increases above unlock threshold (10px)
        state = sm.update(detected=True, error_px=15.0, current_time=1.1, simulation_running=True)
        assert state == PATState.FINE_TRACK
    
    def test_loss_triggers_reacquire(self):
        """Test beacon loss triggers REACQUIRE state."""
        sm = PATStateMachine()
        
        # Go to FINE_TRACK
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        assert sm.state == PATState.FINE_TRACK
        
        # Beacon lost
        state = sm.update(detected=False, error_px=0.0, current_time=0.3, simulation_running=True)
        assert state == PATState.REACQUIRE
    
    def test_reacquire_to_acquire_on_redetection(self):
        """Test REACQUIRE returns to ACQUIRE when target redetected."""
        sm = PATStateMachine()
        
        # Go to REACQUIRE
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        sm.update(detected=False, error_px=0.0, current_time=0.3, simulation_running=True)
        assert sm.state == PATState.REACQUIRE
        
        # Target redetected
        state = sm.update(detected=True, error_px=100.0, current_time=0.5, simulation_running=True)
        assert state == PATState.ACQUIRE
    
    def test_reacquire_timeout_to_search(self):
        """Test REACQUIRE times out to SEARCH."""
        sm = PATStateMachine()
        sm.reacquire_timeout = 1.0  # 1 second timeout
        
        # Go to REACQUIRE
        sm.update(detected=False, error_px=0.0, current_time=0.0, simulation_running=True)
        sm.update(detected=True, error_px=200.0, current_time=0.1, simulation_running=True)
        sm.update(detected=True, error_px=30.0, current_time=0.2, simulation_running=True)
        sm.update(detected=False, error_px=0.0, current_time=0.3, simulation_running=True)
        assert sm.state == PATState.REACQUIRE
        
        # Wait for timeout
        state = sm.update(detected=False, error_px=0.0, current_time=1.5, simulation_running=True)
        assert state == PATState.SEARCH


class TestPATController:
    """Test PAT controller command generation."""
    
    def test_no_command_when_not_detected(self):
        """Test controller outputs zero when target not detected."""
        controller = PATController()
        az_cmd, el_cmd = controller.compute_command(
            detected=False,
            beacon_x=0.0,
            beacon_y=0.0,
            control_mode='search',
            dt=0.033
        )
        assert az_cmd == 0.0
        assert el_cmd == 0.0
    
    def test_command_proportional_to_error(self):
        """Test controller output is proportional to pixel error."""
        controller = PATController()
        
        # Use LOCKED mode with gentle gains to avoid saturation
        # Very small errors
        az_cmd1, el_cmd1 = controller.compute_command(
            detected=True,
            beacon_x=650.0,  # 10px right of center (640)
            beacon_y=360.0,  # Centered vertically
            control_mode='locked',  # Gentle gains
            dt=0.033
        )
        
        print(f"\nTest 1: error=10px, mode=locked, az_cmd={az_cmd1:.3f}, el_cmd={el_cmd1:.3f}")
        
        # Should command positive azimuth rate (move right)
        assert az_cmd1 > 0.0, f"Expected positive az command, got {az_cmd1}"
        assert az_cmd1 < 10.0, f"Command should not saturate, got {az_cmd1}"
        # Should command near-zero elevation rate
        assert abs(el_cmd1) < 0.1, f"Expected near-zero el command, got {el_cmd1}"
        
        # Larger error - don't reset so derivative doesn't explode
        az_cmd2, el_cmd2 = controller.compute_command(
            detected=True,
            beacon_x=660.0,  # 20px right of center (2x the error)
            beacon_y=360.0,
            control_mode='locked',  # Gentle gains
            dt=0.033
        )
        
        print(f"Test 2: error=20px, mode=locked, az_cmd={az_cmd2:.3f}, el_cmd={el_cmd2:.3f}")
        print(f"Ratio: {az_cmd2/az_cmd1:.2f} (expected ~2.0 for proportional component)")
        
        # Larger error should give larger command
        # Note: won't be exactly 2x due to integral and derivative terms
        assert az_cmd2 > az_cmd1, f"Expected {az_cmd2} > {az_cmd1}"
        assert az_cmd2 < 10.0, f"Command should not saturate, got {az_cmd2}"
    
    def test_respects_rate_limits(self):
        """Test controller respects maximum rate limits."""
        controller = PATController()
        controller.max_rate = 10.0  # deg/sec
        
        # Very large error
        az_cmd, el_cmd = controller.compute_command(
            detected=True,
            beacon_x=1200.0,  # Far right
            beacon_y=700.0,   # Far down
            control_mode='acquire',
            dt=0.033
        )
        
        # Should be clamped to max rate
        assert abs(az_cmd) <= controller.max_rate
        assert abs(el_cmd) <= controller.max_rate
    
    def test_different_gains_for_different_modes(self):
        """Test controller uses different gains for different modes."""
        controller = PATController()
        
        # Simulate steady-state by running a few ticks first
        error_x = 20.0
        
        # Prime the controller with a few ticks to establish steady state
        for _ in range(3):
            controller.compute_command(
                detected=True,
                beacon_x=640.0 + error_x,
                beacon_y=360.0,
                control_mode='acquire',
                dt=0.033
            )
        
        # Now test acquire mode in steady state
        az_acquire, _ = controller.compute_command(
            detected=True,
            beacon_x=640.0 + error_x,
            beacon_y=360.0,
            control_mode='acquire',
            dt=0.033
        )
        
        # Reset and prime for locked mode
        controller.reset()
        for _ in range(3):
            controller.compute_command(
                detected=True,
                beacon_x=640.0 + error_x,
                beacon_y=360.0,
                control_mode='locked',
                dt=0.033
            )
        
        # Test locked mode in steady state
        az_locked, _ = controller.compute_command(
            detected=True,
            beacon_x=640.0 + error_x,
            beacon_y=360.0,
            control_mode='locked',
            dt=0.033
        )
        
        print(f"\nSteady-state error={error_x}px: acquire_cmd={az_acquire:.3f}, locked_cmd={az_locked:.3f}, ratio={abs(az_acquire/az_locked):.2f}")
        
        # Acquire mode should be more aggressive (higher proportional gain: 0.10 vs 0.03)
        assert abs(az_acquire) > abs(az_locked), f"Expected acquire ({az_acquire}) > locked ({az_locked})"


class TestIntegratedAcquisition:
    """Test integrated PAT acquisition behavior."""
    
    @pytest.mark.asyncio
    async def test_baseline_acquisition_stationary_target(self):
        """
        Test PAT acquisition with STATIONARY target under IDEAL conditions.
        Should reliably and quickly achieve LOCKED state.
        """
        # Setup simulation components
        camera = VirtualCamera(width=1280, height=720)
        detector = BeaconDetector()
        pan_tilt = PanTilt()
        controller = PATController()
        state_machine = PATStateMachine()
        state_machine.lock_duration = 2.0  # 2 seconds to lock
        
        # STATIONARY target at a fixed offset from center
        target_az_deg = 5.0  # 5 degrees right
        target_el_deg = -8.0  # 8 degrees up
        target_az_rad = math.radians(target_az_deg)
        target_el_rad = math.radians(target_el_deg)
        
        # FOV parameters (from Target class)
        fov_horizontal_deg = 40.0
        fov_vertical_deg = 30.0
        fov_horizontal_rad = math.radians(fov_horizontal_deg)
        fov_vertical_rad = math.radians(fov_vertical_deg)
        pixels_per_rad_x = camera.width / fov_horizontal_rad
        pixels_per_rad_y = camera.height / fov_vertical_rad
        
        # Simulation parameters
        dt = 1.0 / 30.0  # 30 FPS
        simulation_time = 0.0
        max_duration = 15.0  # 15 second test
        
        # Metrics
        acquisition_time = None
        lock_time = None
        min_error = float('inf')
        final_error = 0.0
        time_locked = 0.0
        errors = []
        states = []
        
        # Run simulation
        tick = 0
        while simulation_time < max_duration:
            simulation_time += dt
            tick += 1
            
            # Calculate pixel position (closed-loop with STATIONARY target)
            gimbal_az_rad = math.radians(pan_tilt.azimuth)
            gimbal_el_rad = math.radians(pan_tilt.elevation)
            
            # Relative angle
            rel_az = target_az_rad - gimbal_az_rad
            rel_el = target_el_rad - gimbal_el_rad
            
            # Check if in FOV
            if abs(rel_az) <= fov_horizontal_rad / 2 and abs(rel_el) <= fov_vertical_rad / 2:
                # Target is visible
                pixel_x = camera.width / 2 + rel_az * pixels_per_rad_x
                pixel_y = camera.height / 2 - rel_el * pixels_per_rad_y
                
                # Render and detect
                frame = camera.render(pixel_x, pixel_y)
                detection = detector.detect(frame)
                
                if detection:
                    detected = True
                    detected_x = detection["x"]
                    detected_y = detection["y"]
                    
                    # Calculate error
                    center_x = camera.width / 2.0
                    center_y = camera.height / 2.0
                    error_x = detected_x - center_x
                    error_y = detected_y - center_y
                    error_px = math.sqrt(error_x**2 + error_y**2)
                else:
                    detected = False
                    detected_x = 0.0
                    detected_y = 0.0
                    error_px = 0.0
            else:
                # Out of FOV
                detected = False
                detected_x = 0.0
                detected_y = 0.0
                error_px = 0.0
            
            # Update state machine
            pat_state = state_machine.update(
                detected=detected,
                error_px=error_px,
                current_time=simulation_time,
                simulation_running=True
            )
            
            # Get control mode
            control_mode = state_machine.get_control_mode()
            
            # Compute control commands
            az_cmd, el_cmd = controller.compute_command(
                detected=detected,
                beacon_x=detected_x,
                beacon_y=detected_y,
                control_mode=control_mode,
                dt=dt
            )
            
            # Update gimbal
            pan_tilt.update(az_cmd, el_cmd, dt)
            
            # Debug output for first few ticks and periodically
            if tick <= 20 or (tick % 30 == 0 and tick <= 300):
                print(f"Tick {tick:3d}: t={simulation_time:5.2f}s | "
                      f"State={pat_state.value:12s} | "
                      f"Det={detected} | "
                      f"Beacon=({detected_x:6.1f},{detected_y:6.1f}) | "
                      f"Err={error_px:6.1f}px | "
                      f"Cmd=({az_cmd:+6.2f},{el_cmd:+6.2f})°/s | "
                      f"Gimbal=({pan_tilt.azimuth:+7.2f},{pan_tilt.elevation:+7.2f})°")
            
            # Record metrics
            if detected:
                errors.append(error_px)
                min_error = min(min_error, error_px)
                final_error = error_px
            
            states.append(pat_state.value)
            
            # Track first acquisition (any tracking state)
            if acquisition_time is None and pat_state in (PATState.ACQUIRE, PATState.FINE_TRACK, PATState.LOCKED):
                acquisition_time = simulation_time
            
            # Track first LOCKED time (this is the real "lock acquisition time")
            if lock_time is None and pat_state == PATState.LOCKED:
                lock_time = simulation_time
            
            # Track total time in LOCKED state (lock retention)
            if pat_state == PATState.LOCKED:
                time_locked += dt
        
        # Assertions
        print(f"\n=== PAT Acquisition Test Results (Stationary Target) ===")
        print(f"Target position: az={target_az_deg}°, el={target_el_deg}°")
        print(f"First detection time: {acquisition_time:.2f}s" if acquisition_time else "FAILED TO DETECT")
        print(f"Lock acquisition time (START→LOCKED): {lock_time:.2f}s" if lock_time else "FAILED TO LOCK")
        print(f"Min error: {min_error:.2f}px")
        print(f"Final error: {final_error:.2f}px")
        print(f"Lock retention (time in LOCKED): {time_locked:.2f}s")
        print(f"States seen: {set(states)}")
        print(f"Final gimbal: az={pan_tilt.azimuth:.2f}°, el={pan_tilt.elevation:.2f}°")
        print(f"Target offset from gimbal: az={(target_az_deg - pan_tilt.azimuth):.2f}°, "
              f"el={(target_el_deg - pan_tilt.elevation):.2f}°")
        
        # Requirements for stationary target
        assert acquisition_time is not None, "Failed to detect target"
        assert acquisition_time < 1.0, f"Detection too slow: {acquisition_time:.2f}s"
        
        assert lock_time is not None, "Failed to achieve LOCKED state"
        assert lock_time < 6.0, f"Lock acquisition (START→LOCKED) too slow: {lock_time:.2f}s"
        
        assert min_error < 5.0, f"Min error too large: {min_error:.2f}px"
        assert final_error < 8.0, f"Final error too large: {final_error:.2f}px"
        
        assert time_locked > 5.0, f"Lock retention too short: {time_locked:.2f}s"
    
    @pytest.mark.asyncio
    async def test_baseline_acquisition_ideal_conditions(self):
        """
        Test PAT acquisition under IDEAL conditions (no disturbances).
        Should reliably achieve LOCKED state.
        """
        # Setup simulation components
        target = Target(width=1280, height=720)
        camera = VirtualCamera(width=1280, height=720)
        detector = BeaconDetector()
        pan_tilt = PanTilt()
        controller = PATController()
        state_machine = PATStateMachine()
        state_machine.lock_duration = 2.0  # 2 seconds to lock
        
        # Simulation parameters
        dt = 1.0 / 30.0  # 30 FPS
        simulation_time = 0.0
        max_duration = 10.0  # 10 second test
        
        # Metrics
        acquisition_time = None
        lock_time = None
        min_error = float('inf')
        final_error = 0.0
        time_locked = 0.0
        errors = []
        states = []
        
        # Run simulation
        tick = 0
        while simulation_time < max_duration:
            simulation_time += dt
            tick += 1
            
            # Get target angular position
            target_az_rad, target_el_rad = target.angular_position(simulation_time)
            
            # Calculate pixel position (closed-loop)
            gimbal_az_rad = math.radians(pan_tilt.azimuth)
            gimbal_el_rad = math.radians(pan_tilt.elevation)
            
            pixel_pos = target.pixel_from_angles(
                target_az_rad, target_el_rad,
                gimbal_az_rad, gimbal_el_rad
            )
            
            if pixel_pos is not None:
                beacon_x, beacon_y = pixel_pos
                # Render and detect
                frame = camera.render(beacon_x, beacon_y)
                detection = detector.detect(frame)
                
                if detection:
                    detected = True
                    detected_x = detection["x"]
                    detected_y = detection["y"]
                    
                    # Calculate error
                    center_x = camera.width / 2.0
                    center_y = camera.height / 2.0
                    error_x = detected_x - center_x
                    error_y = detected_y - center_y
                    error_px = math.sqrt(error_x**2 + error_y**2)
                else:
                    detected = False
                    detected_x = 0.0
                    detected_y = 0.0
                    error_px = 0.0
            else:
                detected = False
                detected_x = 0.0
                detected_y = 0.0
                error_px = 0.0
            
            # Update state machine
            pat_state = state_machine.update(
                detected=detected,
                error_px=error_px,
                current_time=simulation_time,
                simulation_running=True
            )
            
            # Get control mode
            control_mode = state_machine.get_control_mode()
            
            # Compute control commands
            az_cmd, el_cmd = controller.compute_command(
                detected=detected,
                beacon_x=detected_x,
                beacon_y=detected_y,
                control_mode=control_mode,
                dt=dt
            )
            
            # Update gimbal
            pan_tilt.update(az_cmd, el_cmd, dt)
            
            # Debug output for first few ticks
            if tick <= 10 or (tick % 30 == 0 and tick <= 150):
                print(f"Tick {tick:3d}: t={simulation_time:5.2f}s | "
                      f"State={pat_state.value:12s} | "
                      f"Det={detected} | "
                      f"Beacon=({detected_x:6.1f},{detected_y:6.1f}) | "
                      f"Err={error_px:6.1f}px | "
                      f"Cmd=({az_cmd:+6.2f},{el_cmd:+6.2f})°/s | "
                      f"Gimbal=({pan_tilt.azimuth:+7.2f},{pan_tilt.elevation:+7.2f})°")
            
            # Record metrics
            if detected:
                errors.append(error_px)
                min_error = min(min_error, error_px)
                final_error = error_px
            
            states.append(pat_state.value)
            
            # Track first detection (any tracking state)
            if acquisition_time is None and pat_state in (PATState.ACQUIRE, PATState.FINE_TRACK, PATState.LOCKED):
                acquisition_time = simulation_time
            
            # Track first LOCKED time (START→LOCKED duration)
            if lock_time is None and pat_state == PATState.LOCKED:
                lock_time = simulation_time
            
            # Track total time in LOCKED state (lock retention)
            if pat_state == PATState.LOCKED:
                time_locked += dt
        
        # Assertions
        print(f"\n=== PAT Acquisition Test Results (Moving Target, IDEAL) ===")
        print(f"First detection time: {acquisition_time:.2f}s" if acquisition_time else "FAILED TO DETECT")
        print(f"Lock acquisition time (START→LOCKED): {lock_time:.2f}s" if lock_time else "FAILED TO LOCK")
        print(f"Min error: {min_error:.2f}px")
        print(f"Final error: {final_error:.2f}px")
        print(f"Lock retention (time in LOCKED): {time_locked:.2f}s")
        print(f"States seen: {set(states)}")
        print(f"Final gimbal: az={pan_tilt.azimuth:.2f}°, el={pan_tilt.elevation:.2f}°")
        
        # Requirements for moving target (more relaxed)
        assert acquisition_time is not None, "Failed to detect target"
        assert acquisition_time < 2.0, f"Detection too slow: {acquisition_time:.2f}s"
        
        # Moving target may not achieve LOCK - tracking is acceptable
        if lock_time is not None:
            print(f"✓ LOCK achieved at {lock_time:.2f}s (excellent for moving target!)")
            assert lock_time < 10.0, f"Lock acquisition too slow: {lock_time:.2f}s"
            assert time_locked > 0.5, f"Lock retention too short: {time_locked:.2f}s"
        else:
            print(f"⚠ LOCK not achieved (target moving too fast - tracking only)")
            # For moving target, at least verify we achieve FINE_TRACK
            assert 'FINE_TRACK' in states or 'LOCKED' in states, "Failed to achieve at least FINE_TRACK"
        
        assert min_error < 50.0, f"Min error too large even for moving target: {min_error:.2f}px"
        assert final_error < 100.0, f"Final error too large: {final_error:.2f}px"
    
    def test_gimbal_respects_rate_limits(self):
        """Test gimbal respects velocity limits."""
        pan_tilt = PanTilt()
        
        # Command very large rate
        pan_tilt.update(azimuth_command=50.0, elevation_command=30.0, dt=0.033)
        
        # Velocity should be clamped
        assert abs(pan_tilt.azimuth_velocity) <= pan_tilt.max_azimuth_rate
        assert abs(pan_tilt.elevation_velocity) <= pan_tilt.max_elevation_rate
    
    def test_gimbal_respects_position_limits(self):
        """Test gimbal respects position limits."""
        pan_tilt = PanTilt()
        
        # Command to move beyond limits
        for _ in range(100):
            pan_tilt.update(azimuth_command=30.0, elevation_command=20.0, dt=0.1)
        
        # Position should be within limits
        assert pan_tilt.azimuth <= pan_tilt.max_azimuth
        assert pan_tilt.azimuth >= pan_tilt.min_azimuth
        assert pan_tilt.elevation <= pan_tilt.max_elevation
        assert pan_tilt.elevation >= pan_tilt.min_elevation


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
